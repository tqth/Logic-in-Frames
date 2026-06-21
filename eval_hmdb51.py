"""
eval_hmdb51.py
Evaluation script for HMDB51 action recognition with shard/resume/checkpoint support.
Adapted from eval_next_qa.py
"""

import os
import sys
import json
import argparse
import logging
import datetime
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("/kaggle/working/logic-in-frames")
sys.path.append("/kaggle/working/logic-in-frames/VSLS")

from VSLS.interface_llm import VSLSUniversalGrounder
from VSLS.interface_yolo import UltralyticsYOLOWorldInterface
from VSLS.VSLSFramework import VSLSFramework

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ── HMDB51 Classes ────────────────────────────────────────────────────────────
HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
    "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing",
    "field_hockey_penalty", "floor_gymnastics", "flic_flac", "golf", "handstand",
    "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
    "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
    "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit",
    "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword",
    "sword_exercise", "talk", "throw", "turn", "walk", "wave"
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HMDB51 action recognition")

    # Dataset
    parser.add_argument("--data_root",  type=str,
        default="/kaggle/input/datasets/nguyenbon/hmdb51/hmdb51",
        help="Root folder chứa các class subfolder")
    parser.add_argument("--split",      type=str, default="test",
        choices=["train", "test", "all"],
        help="Chạy split nào (nếu không có split file thì dùng all)")
    parser.add_argument("--split_dir",  type=str, default=None,
        help="Folder chứa split .txt files (optional)")

    # Models
    parser.add_argument("--yolo_ckpt", type=str, default="yolov8x-worldv2.pt")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Search
    parser.add_argument("--search_budget", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.05)
    parser.add_argument("--search_nframes", type=int,   default=8)

    # Eval mode
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "shard"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_samples_per_shard", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10)

    # Output
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/output/eval_hmdb51")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    data_root: str,
    split: str = "all",
    split_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Scan folder structure:
        data_root/
            brush_hair/
                video1.avi
                video2.avi
            catch/
                ...
    Trả về DataFrame với columns: [video_path, class_name, class_idx, row_idx]
    """
    # Lấy danh sách class từ folder
    class_names = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    logger.info(f"Found {len(class_names)} classes in {data_root}")

    # Load split files nếu có
    split_files = {}  # {class_name: set of video_name}
    if split_dir and split != "all":
        split_map = {"train": "1", "test": "2"}
        split_code = split_map.get(split, "2")
        for fname in os.listdir(split_dir):
            if not fname.endswith(".txt"):
                continue
            class_name = "_".join(fname.split("_")[:-2])  # remove _split1.txt
            filepath = os.path.join(split_dir, fname)
            videos_in_split = set()
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1] == split_code:
                        videos_in_split.add(parts[0])
            if class_name not in split_files:
                split_files[class_name] = set()
            split_files[class_name].update(videos_in_split)

    # Build records
    records = []
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        video_files = sorted([
            f for f in os.listdir(class_dir)
            if f.endswith((".avi", ".mp4", ".mkv"))
        ])

        for vf in video_files:
            # Filter by split nếu có split files
            if split_files and class_name in split_files:
                if vf not in split_files[class_name]:
                    continue

            records.append({
                "video_path" : os.path.join(class_dir, vf),
                "video_name" : vf,
                "class_name" : class_name,
                "class_idx"  : class_to_idx[class_name],
            })

    df = pd.DataFrame(records).reset_index(drop=True)
    df.index.name = "row_idx"
    df = df.reset_index()  # row_idx thành column

    if num_samples is not None:
        df = df.head(num_samples).reset_index(drop=True)
        df["row_idx"] = df.index

    logger.info(f"Dataset: {len(df)} videos | {len(class_names)} classes | split={split}")
    return df, class_names, class_to_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Shard management (giữ nguyên từ eval_next_qa.py)
# ═══════════════════════════════════════════════════════════════════════════════

def create_shards(df: pd.DataFrame, num_samples_per_shard: int) -> List[Dict]:
    shards, idx, shard_id = [], 0, 0
    total = len(df)
    while idx < total:
        end = min(idx + num_samples_per_shard, total)
        shards.append({
            "shard_id" : shard_id,
            "start_idx": idx,
            "end_idx"  : end - 1,
            "total"    : end - idx,
        })
        idx, shard_id = end, shard_id + 1
    logger.info(f"Created {len(shards)} shards ({num_samples_per_shard} samples/shard)")
    return shards


def _shard_dir(output_dir: str, shard_id: int) -> str:
    return os.path.join(output_dir, "shards", f"shard_{shard_id:03d}")


def load_shard_status(output_dir: str, shard_id: int) -> Dict:
    path = os.path.join(_shard_dir(output_dir, shard_id), "status.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"status": "pending", "processed_samples": 0}


def save_shard_status(output_dir: str, shard_id: int, status: Dict):
    d = _shard_dir(output_dir, shard_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump(status, f, indent=2)


def load_shard_results(output_dir: str, shard_id: int) -> List[Dict]:
    path = os.path.join(_shard_dir(output_dir, shard_id), "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def finished_rows_in_shard(results: List[Dict]) -> Set:
    """Dùng row_idx để identify sample đã chạy xong."""
    return {
        r["row_idx"] for r in results
        if r.get("pred_class") is not None
        or r.get("error") == "video not found"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Checkpoint (giữ nguyên)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    output_dir: str,
    shard_id: int,
    results: List[Dict],
    processed: int,
    total: int,
    status: str = "running",
):
    d = _shard_dir(output_dir, shard_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    save_shard_status(output_dir, shard_id, {
        "shard_id"         : shard_id,
        "status"           : status,
        "processed_samples": processed,
        "total_samples"    : total,
        "last_updated"     : datetime.datetime.now().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Evaluate single sample
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sample(
    row: pd.Series,
    grounder: VSLSUniversalGrounder,
    yolo: UltralyticsYOLOWorldInterface,
    class_names: List[str],
    args: argparse.Namespace,
    sample_out_dir: str,
) -> Dict:
    video_path  = row["video_path"]
    class_name  = row["class_name"]
    class_idx   = int(row["class_idx"])
    video_name  = row["video_name"]

    entry = {
        "row_idx"   : int(row["row_idx"]),
        "video_path": video_path,
        "video_name": video_name,
        "class_gt"  : class_name,
        "class_idx" : class_idx,
        "pred_class": None,
        "correct"   : False,
        "error"     : None,
        "timestamps": [],
        "search_mode": None,
    }

    if not os.path.exists(video_path):
        entry["error"] = "video not found"
        return entry

    try:
        classes_str = ", ".join(class_names)

        # ── Kiểm tra độ dài video ─────────────────────────────────────────
        import cv2
        cap = cv2.VideoCapture(video_path)
        raw_fps  = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / raw_fps if raw_fps > 0 else 0
        cap.release()

        SHORT_VIDEO_THRESHOLD = 8.1  # giây

        if duration < SHORT_VIDEO_THRESHOLD:
            # ── Uniform sampling: lấy đều search_nframes frame ───────────
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, args.search_nframes, dtype=int).tolist()
            all_frames = list(vr.get_batch(indices).asnumpy())
            timestamps = [round(idx / raw_fps, 2) for idx in indices]
            entry["search_mode"] = "uniform"
            logger.info(f"Short video ({duration:.1f}s) — uniform sampling {len(all_frames)} frames")

        else:
            # ── VSLS search bình thường ───────────────────────────────────
            framework = VSLSFramework(
                grounder=grounder,
                yolo_scorer=yolo,
                video_path=video_path,
                question="What action is being performed in this video?",
                options=classes_str,
                search_nframes=args.search_nframes,
                grid_rows=2,
                grid_cols=4,
                output_dir=sample_out_dir,
                confidence_threshold=args.confidence_threshold,
                search_budget=args.search_budget,
                prefix="hmdb51",
                device=args.device,
                update_method="spline",
            )
            target_objects, cue_objects, relations = framework.get_grounded_objects(
                prompt_type="cot", upload_video=1
            )
            video_searcher = framework.set_searching_targets(target_objects, cue_objects, relations)
            all_frames, timestamps = framework.perform_search(video_searcher)
            entry["search_mode"] = "vsls"

        # ── Action recognition ────────────────────────────────────────────
        pred_class = grounder.inference_action_recog(
            frames=all_frames,
            candidate_classes=class_names,
        )

        entry["pred_class"]  = pred_class
        entry["correct"]     = pred_class == class_name
        entry["timestamps"]  = [float(t) for t in timestamps]

    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        logger.warning(f"  Error {video_name}: {entry['error']}")
        logger.debug(traceback.format_exc())

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Evaluate shard
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_shard(
    shard_meta: Dict,
    df: pd.DataFrame,
    grounder: VSLSUniversalGrounder,
    yolo: UltralyticsYOLOWorldInterface,
    class_names: List[str],
    args: argparse.Namespace,
    total_shards: int,
    global_processed: int,
    global_total: int,
) -> Tuple[List[Dict], int]:
    shard_id    = shard_meta["shard_id"]
    start_idx   = shard_meta["start_idx"]
    end_idx     = shard_meta["end_idx"]
    shard_total = shard_meta["total"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Shard {shard_id+1}/{total_shards} | rows [{start_idx}:{end_idx+1}]")
    logger.info(f"{'='*60}")

    results      = load_shard_results(args.output_dir, shard_id)
    done_rows    = finished_rows_in_shard(results)
    shard_df     = df.iloc[start_idx : end_idx + 1]

    local_correct = sum(1 for r in results if r.get("correct"))
    local_valid   = sum(1 for r in results if r.get("pred_class") is not None)

    pbar = tqdm(
        shard_df.iterrows(),
        total=shard_total,
        desc=f"Shard {shard_id+1}/{total_shards}",
        dynamic_ncols=True,
    )

    for _, row in pbar:
        if int(row["row_idx"]) in done_rows:
            global_processed += 1
            continue

        sample_out_dir = os.path.join(args.output_dir, row["class_name"], row["video_name"])
        entry = evaluate_sample(row, grounder, yolo, class_names, args, sample_out_dir)
        results.append(entry)

        if entry.get("correct"):
            local_correct += 1
        if entry.get("pred_class") is not None:
            local_valid += 1
        global_processed += 1

        local_acc = local_correct / local_valid if local_valid > 0 else 0.0
        pbar.set_postfix({
            "acc"    : f"{local_acc:.3f}",
            "shard"  : f"{len(results)}/{shard_total}",
            "overall": f"{global_processed}/{global_total}",
            "pred"   : (entry.get("pred_class") or "err")[:12],
            "gt"     : entry.get("class_gt", "?")[:12],
        })

        if len(results) % args.save_every == 0:
            save_checkpoint(args.output_dir, shard_id, results,
                            processed=len(results), total=shard_total)

    pbar.close()

    actual_status = "completed" if len(results) >= shard_total else "running"
    save_checkpoint(args.output_dir, shard_id, results,
                    processed=len(results), total=shard_total,
                    status=actual_status)

    shard_correct = sum(1 for r in results if r.get("correct"))
    shard_valid   = sum(1 for r in results if r.get("pred_class") is not None)
    shard_acc     = shard_correct / shard_valid if shard_valid > 0 else 0.0

    summary = {
        "shard_id" : shard_id,
        "start_idx": start_idx,
        "end_idx"  : end_idx,
        "total"    : shard_total,
        "correct"  : shard_correct,
        "valid"    : shard_valid,
        "accuracy" : round(shard_acc, 4),
        "errors"   : sum(1 for r in results if r.get("error")),
    }
    with open(os.path.join(_shard_dir(args.output_dir, shard_id), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Shard {shard_id} done | acc={shard_acc:.3f} ({shard_correct}/{shard_valid})")
    return results, global_processed


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Merge & metrics
# ═══════════════════════════════════════════════════════════════════════════════

def merge_results(output_dir: str, shards: List[Dict]) -> List[Dict]:
    all_results = []
    for s in shards:
        all_results.extend(load_shard_results(output_dir, s["shard_id"]))
    merged_path = os.path.join(output_dir, "merged_results.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Merged {len(all_results)} results -> {merged_path}")
    return all_results


def compute_metrics(results: List[Dict], class_names: List[str], output_dir: str) -> Dict:
    # Overall
    by_class = defaultdict(lambda: {"correct": 0, "total": 0, "error": 0})

    for r in results:
        cls = r.get("class_gt", "unknown")
        if r.get("error") and r.get("pred_class") is None:
            by_class[cls]["error"] += 1
        else:
            by_class[cls]["total"]   += 1
            by_class[cls]["correct"] += int(r.get("correct", False))

    all_correct = sum(v["correct"] for v in by_class.values())
    all_total   = sum(v["total"]   for v in by_class.values())
    overall_acc = all_correct / all_total if all_total > 0 else 0.0

    # Mean class accuracy (quan trọng hơn overall cho HMDB51)
    per_class_acc = [
        v["correct"] / v["total"]
        for v in by_class.values() if v["total"] > 0
    ]
    mean_class_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0

    # Print per-class
    print(f"\n{'='*60}")
    print(f"{'Class':<25} | {'Correct':>8} | {'Total':>8} | {'Acc':>8}")
    print("-" * 60)
    for cls in sorted(by_class.keys()):
        c   = by_class[cls]["correct"]
        t   = by_class[cls]["total"]
        e   = by_class[cls]["error"]
        acc = c / t if t > 0 else 0.0
        print(f"{cls:<25} | {c:>8} | {t:>8} | {acc:>7.2%}  (errors: {e})")
    print("-" * 60)
    print(f"{'OVERALL (micro)':<25} | {all_correct:>8} | {all_total:>8} | {overall_acc:>7.2%}")
    print(f"{'MEAN CLASS ACC (macro)':<25} | {'':>8} | {'':>8} | {mean_class_acc:>7.2%}")
    print(f"{'='*60}")

    metrics = {
        "overall_accuracy"   : round(overall_acc, 4),
        "mean_class_accuracy": round(mean_class_acc, 4),
        "total_correct"      : all_correct,
        "total_samples"      : all_total,
        "total_errors"       : sum(v["error"] for v in by_class.values()),
        "by_class"           : {
            cls: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
                "correct" : v["correct"],
                "total"   : v["total"],
                "errors"  : v["error"],
            }
            for cls, v in by_class.items()
        },
        "computed_at": datetime.datetime.now().isoformat(),
    }

    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved -> {os.path.join(output_dir, 'final_metrics.json')}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, class_names, class_to_idx = load_dataset(
        data_root=args.data_root,
        split=args.split,
        split_dir=args.split_dir,
        num_samples=args.num_samples,
    )

    logger.info("Initializing models...")
    grounder = VSLSUniversalGrounder(
        backend="qwenvl",
        model_name=args.model_name,
        base_url=args.base_url,
    )
    yolo = UltralyticsYOLOWorldInterface(checkpoint_path=args.yolo_ckpt, device=args.device)
    logger.info("Models ready.")

    # ── Full mode ─────────────────────────────────────────────────────────────
    if args.eval_mode == "full":
        shards = [{"shard_id": 0, "start_idx": 0, "end_idx": len(df) - 1, "total": len(df)}]
        os.makedirs(_shard_dir(args.output_dir, 0), exist_ok=True)

        status = load_shard_status(args.output_dir, 0)
        if status.get("status") == "completed":
            logger.info("Full eval already completed. Recomputing metrics...")
        else:
            evaluate_shard(shards[0], df, grounder, yolo, class_names, args,
                           total_shards=1, global_processed=0, global_total=len(df))

        all_results = merge_results(args.output_dir, shards)
        compute_metrics(all_results, class_names, args.output_dir)

    # ── Shard mode ────────────────────────────────────────────────────────────
    else:
        shards           = create_shards(df, args.num_samples_per_shard)
        total_shards     = len(shards)
        global_total     = len(df)
        global_processed = 0

        for shard_meta in shards:
            shard_id = shard_meta["shard_id"]
            status   = load_shard_status(args.output_dir, shard_id)

            if status.get("status") == "completed":
                logger.info(f"Shard {shard_id+1}/{total_shards} already completed — skipping.")
                global_processed += shard_meta["total"]
                continue

            _, global_processed = evaluate_shard(
                shard_meta, df, grounder, yolo, class_names, args,
                total_shards=total_shards,
                global_processed=global_processed,
                global_total=global_total,
            )

        all_completed = all(
            load_shard_status(args.output_dir, s["shard_id"]).get("status") == "completed"
            for s in shards
        )
        if all_completed:
            logger.info("\nAll shards completed! Merging results...")
            all_results = merge_results(args.output_dir, shards)
            compute_metrics(all_results, class_names, args.output_dir)
        else:
            pending = [
                s["shard_id"] for s in shards
                if load_shard_status(args.output_dir, s["shard_id"]).get("status") != "completed"
            ]
            logger.info(f"Pending shards: {pending} — run again to resume.")


if __name__ == "__main__":
    main()
