"""
evaluate_nextqa_val.py
Robust evaluation script for NExT-QA with shard/resume/checkpoint support.
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

IDX2LETTER = {i: chr(65 + i) for i in range(5)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NExT-QA val set")

    # Dataset
    parser.add_argument("--csv_path",   type=str,
        default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/label/multi-choice/val.csv")
    parser.add_argument("--map_path",   type=str,
        default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/map_vid_vidorID.json")
    parser.add_argument("--video_root", type=str,
        default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/videos")

    # Models
    parser.add_argument("--yolo_ckpt",  type=str, default="yolov8x-worldv2.pt")
    parser.add_argument("--base_url",   type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device",     type=str, default="cuda:0")

    # Search
    parser.add_argument("--search_budget",        type=float, default=0.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.05)
    parser.add_argument("--search_nframes",       type=int,   default=8)

    # Eval mode
    parser.add_argument("--eval_mode",             type=str, default="full",
        choices=["full", "shard"])
    parser.add_argument("--num_samples",           type=int, default=None,
        help="Gioi han tong so sample (full mode)")
    parser.add_argument("--num_samples_per_shard", type=int, default=500,
        help="So sample moi shard (shard mode)")
    parser.add_argument("--save_every",            type=int, default=10,
        help="Checkpoint sau moi N sample")

    # Output
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/output/eval")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    csv_path: str,
    map_path: str,
    num_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(csv_path)
    with open(map_path) as f:
        vid_map = json.load(f)
    if num_samples is not None:
        df = df.head(num_samples)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    return df, vid_map


def get_video_path(video_id, vid_map: Dict, video_root: str) -> Optional[str]:
    key = str(video_id)
    if key not in vid_map:
        return None
    return os.path.join(video_root, vid_map[key] + ".mp4")


def build_options_str(row: pd.Series) -> str:
    cols = sorted([c for c in row.index if c.startswith("a") and c[1:].isdigit()])
    return "\n".join(f"{IDX2LETTER[i]}) {row[col]}" for i, col in enumerate(cols))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Shard management
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


def finished_qids_in_shard(results: List[Dict]) -> Set:
    return {
        r["row_idx"] for r in results
        if r.get("answer_pred") is not None
        or r.get("error") == "video not found"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Checkpoint
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
    vid_map: Dict,
    args: argparse.Namespace,
    sample_out_dir: str,
) -> Dict:
    video_id    = row["video"]
    question    = row["question"]
    options_str = build_options_str(row)
    answer_gt   = IDX2LETTER[int(row["answer"])]
    qtype       = row.get("type", "unknown")
    video_path  = get_video_path(video_id, vid_map, args.video_root)

    entry = {
        "row_idx"    : int(row.name),
        "video_id"   : str(video_id),
        "qid"        : int(row.get("qid", -1)),
        "type"       : qtype,
        "question"   : question,
        "options"    : options_str,
        "answer_gt"  : answer_gt,
        "answer_pred": None,
        "correct"    : False,
        "error"      : None,
        "timestamps" : [],
    }

    if video_path is None or not os.path.exists(video_path):
        entry["error"] = "video not found"
        return entry

    try:
        os.makedirs(sample_out_dir, exist_ok=True)

        framework = VSLSFramework(
            grounder=grounder,
            yolo_scorer=yolo,
            video_path=video_path,
            question=question,
            options=options_str,
            search_nframes=args.search_nframes,
            grid_rows=2,
            grid_cols=4,
            output_dir=sample_out_dir,
            confidence_threshold=args.confidence_threshold,
            search_budget=args.search_budget,
            prefix="nextqa",
            device=args.device,
            update_method="spline",
        )

        target_objects, cue_objects, relations = framework.get_grounded_objects(
            prompt_type="cot", upload_video=1
        )
        video_searcher = framework.set_searching_targets(target_objects, cue_objects, relations)
        all_frames, timestamps = framework.perform_search(video_searcher)
        answer_pred = framework.perform_qa(all_frames)
        answer_pred = answer_pred.strip().upper()[0]

        entry["answer_pred"] = answer_pred
        entry["correct"]     = answer_pred == answer_gt
        entry["timestamps"]  = [float(t) for t in timestamps]

    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        logger.warning(f"  Error qid={entry['qid']}: {entry['error']}")
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
    vid_map: Dict,
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

    # Resume: load existing results
    results   = load_shard_results(args.output_dir, shard_id)
    done_qids = finished_qids_in_shard(results)
    shard_df  = df.iloc[start_idx : end_idx + 1]

    local_correct = sum(1 for r in results if r.get("correct"))
    local_valid   = sum(1 for r in results if r.get("answer_pred") is not None)

    pbar = tqdm(
        shard_df.iterrows(),
        total=shard_total,
        desc=f"Shard {shard_id+1}/{total_shards}",
        dynamic_ncols=True,
    )

    for _, row in pbar:

        # Resume ở cấp sample
        if row.name in done_qids:
            continue

        sample_out_dir = os.path.join(args.output_dir, str(row["video"]))
        entry = evaluate_sample(row, grounder, yolo, vid_map, args, sample_out_dir)
        results.append(entry)

        if entry.get("correct"):
            local_correct += 1
        if entry.get("answer_pred") is not None:
            local_valid += 1
        global_processed += 1

        local_acc = local_correct / local_valid if local_valid > 0 else 0.0
        pbar.set_postfix({
            "acc"    : f"{local_acc:.3f}",
            "shard"  : f"{len(results)}/{shard_total}",
            "overall": f"{global_processed}/{global_total}",
            "pred"   : entry.get("answer_pred") or "err",
            "gt"     : entry.get("answer_gt", "?"),
        })

        # Checkpoint
        if len(results) % args.save_every == 0:
            save_checkpoint(args.output_dir, shard_id, results,
                            processed=len(results), total=shard_total)

    pbar.close()

    # Final save
    actual_status = "completed" if len(results) >= shard_total else "running"
    save_checkpoint(args.output_dir, shard_id, results,
                processed=len(results), total=shard_total,
                status=actual_status)

    # Shard summary
    shard_correct = sum(1 for r in results if r.get("correct"))
    shard_valid   = sum(1 for r in results if r.get("answer_pred") is not None)
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


def compute_metrics(results: List[Dict], output_dir: str) -> Dict:
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "error": 0})

    for r in results:
        qtype = r.get("type", "unknown")
        if r.get("error") and r.get("answer_pred") is None:
            stats[qtype]["error"] += 1
        else:
            stats[qtype]["total"]   += 1
            stats[qtype]["correct"] += int(r.get("correct", False))

    all_correct = sum(v["correct"] for v in stats.values())
    all_total   = sum(v["total"]   for v in stats.values())
    overall_acc = all_correct / all_total if all_total > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"{'Type':<10} | {'Correct':>8} | {'Total':>8} | {'Acc':>8}")
    print("-" * 55)
    for qtype in sorted(stats.keys()):
        c   = stats[qtype]["correct"]
        t   = stats[qtype]["total"]
        e   = stats[qtype]["error"]
        acc = c / t if t > 0 else 0.0
        print(f"{qtype:<10} | {c:>8} | {t:>8} | {acc:>7.2%}  (errors: {e})")
    print("-" * 55)
    print(f"{'OVERALL':<10} | {all_correct:>8} | {all_total:>8} | {overall_acc:>7.2%}")
    print(f"{'='*55}")

    metrics = {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct"   : all_correct,
        "total_samples"   : all_total,
        "total_errors"    : sum(v["error"] for v in stats.values()),
        "by_type"         : {
            qt: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
                "correct" : v["correct"],
                "total"   : v["total"],
                "errors"  : v["error"],
            }
            for qt, v in stats.items()
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

    df, vid_map = load_dataset(args.csv_path, args.map_path, args.num_samples)

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
            evaluate_shard(shards[0], df, grounder, yolo, vid_map, args,
                           total_shards=1, global_processed=0, global_total=len(df))

        all_results = merge_results(args.output_dir, shards)
        compute_metrics(all_results, args.output_dir)

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
                shard_meta, df, grounder, yolo, vid_map, args,
                total_shards=total_shards,
                global_processed=global_processed,
                global_total=global_total,
            )

        # Merge khi tất cả xong
        all_completed = all(
            load_shard_status(args.output_dir, s["shard_id"]).get("status") == "completed"
            for s in shards
        )
        if all_completed:
            logger.info("\nAll shards completed! Merging results...")
            all_results = merge_results(args.output_dir, shards)
            compute_metrics(all_results, args.output_dir)
        else:
            pending = [
                s["shard_id"] for s in shards
                if load_shard_status(args.output_dir, s["shard_id"]).get("status") != "completed"
            ]
            logger.info(f"Pending shards: {pending} — run again to resume.")


if __name__ == "__main__":
    main()
