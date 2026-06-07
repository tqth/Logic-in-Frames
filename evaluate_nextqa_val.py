import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import datetime

sys.path.append("/kaggle/working/logic-in-frames")
sys.path.append("/kaggle/working/logic-in-frames/VSLS")

from VSLS.interface_llm import VSLSUniversalGrounder
from VSLS.interface_yolo import UltralyticsYOLOWorldInterface
from VSLS.VSLSFramework import VSLSFramework

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate NExT-QA val set")
parser.add_argument("--csv_path",    type=str, default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/label/multi-choice/val.csv")
parser.add_argument("--map_path",    type=str, default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/map_vid_vidorID.json")
parser.add_argument("--video_root",  type=str, default="/kaggle/input/datasets/nguyenbon/next-qa/NExTVideo/videos")
parser.add_argument("--yolo_ckpt",   type=str, default="yolov8x-worldv2.pt")
parser.add_argument("--output_dir",  type=str, default="/kaggle/working/output/eval")
parser.add_argument("--num_samples", type=int, default=None, help="Số sample muốn eval, None = toàn bộ")
parser.add_argument("--device",      type=str, default="cuda:0")
parser.add_argument("--base_url",    type=str, default="http://localhost:8000/v1")
parser.add_argument("--model_name",  type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--search_budget",       type=float, default=0.5)
parser.add_argument("--confidence_threshold",type=float, default=0.05)
parser.add_argument("--search_nframes",      type=int,   default=8)
args = parser.parse_args()

CSV_PATH    = args.csv_path
MAP_PATH    = args.map_path
VIDEO_ROOT  = args.video_root
YOLO_CKPT   = args.yolo_ckpt
OUTPUT_DIR  = args.output_dir
NUM_SAMPLES = args.num_samples
DEVICE      = args.device
RESULT_PATH = os.path.join(OUTPUT_DIR, f"eval_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
with open(MAP_PATH) as f:
    vid_map = json.load(f)

if NUM_SAMPLES is not None:
    df = df.head(NUM_SAMPLES)

print(f"Evaluating on {len(df)} samples")

OPTIONS_COLS = sorted([c for c in df.columns if c.startswith("a") and c[1:].isdigit()])
IDX2LETTER   = {i: chr(65 + i) for i in range(len(OPTIONS_COLS))}  # {0:'A', 1:'B', ...}

def get_video_path(video_id):
    key = str(video_id)
    if key not in vid_map:
        return None
    return os.path.join(VIDEO_ROOT, vid_map[key] + ".mp4")

def build_options_str(row):
    return "\n".join(f"{IDX2LETTER[i]}) {row[col]}" for i, col in enumerate(OPTIONS_COLS))

# ── Init models (một lần duy nhất) ───────────────────────────────────────────
grounder = VSLSUniversalGrounder(
    backend="qwenvl",
    model_name=args.model_name,
    base_url=args.base_url,
)

yolo = UltralyticsYOLOWorldInterface(checkpoint_path=YOLO_CKPT, device=DEVICE)

# ── Evaluate ──────────────────────────────────────────────────────────────────
results   = []
stats     = defaultdict(lambda: {"correct": 0, "total": 0, "error": 0})

pbar = tqdm(df.iterrows(), total=len(df), desc="Evaluating")

for _, row in pbar:
    video_id   = row["video"]
    question   = row["question"]
    options_str = build_options_str(row)
    answer_gt  = IDX2LETTER[int(row["answer"])]   # "A"/"B"/...
    qtype      = row.get("type", "unknown")
    video_path = get_video_path(video_id)

    entry = {
        "video_id"   : str(video_id),
        "qid"        : int(row.get("qid", -1)),
        "type"       : qtype,
        "question"   : question,
        "options"    : options_str,
        "answer_gt"  : answer_gt,
        "answer_pred": None,
        "correct"    : False,
        "error"      : None,
    }

    # ── skip nếu video không tồn tại ─────────────────────────────────────────
    if video_path is None or not os.path.exists(video_path):
        entry["error"] = "video not found"
        stats[qtype]["error"] += 1
        results.append(entry)
        pbar.set_postfix({"skip": "no video"})
        continue

    try:
        sample_out_dir = os.path.join(OUTPUT_DIR, str(video_id))
        os.makedirs(sample_out_dir, exist_ok=True)

        framework = VSLSFramework(
            grounder=grounder,
            yolo_scorer=yolo,
            video_path=video_path,
            question=question,
            options=options_str,
            search_nframes=args.search_nframes,
            grid_rows=4,
            grid_cols=4,
            output_dir=sample_out_dir,
            confidence_threshold=args.confidence_threshold,
            search_budget=args.search_budget,
            prefix="nextqa",
            device=DEVICE,
            update_method="spline",
        )

        # Step 1: grounding
        target_objects, cue_objects, relations = framework.get_grounded_objects(
            prompt_type="cot", upload_video=1
        )

        # Step 2: search keyframes
        video_searcher = framework.set_searching_targets(target_objects, cue_objects, relations)
        all_frames, timestamps = framework.perform_search(video_searcher)

        # Step 3: QA
        answer_pred = framework.perform_qa(all_frames)
        answer_pred = answer_pred.strip().upper()[0]

        correct = answer_pred == answer_gt

        entry["answer_pred"] = answer_pred
        entry["correct"]     = correct
        entry["timestamps"]  = [float(t) for t in timestamps]

        stats[qtype]["total"]   += 1
        stats[qtype]["correct"] += int(correct)

    except Exception as e:
        entry["error"] = str(e)
        stats[qtype]["error"] += 1

    print("Predicted Answer: ", answer_pred)
    print("Ground truth Answer: ", answer_gt)

    results.append(entry)

    # ── cập nhật postfix tqdm ─────────────────────────────────────────────────
    total_done    = sum(v["total"] for v in stats.values())
    total_correct = sum(v["correct"] for v in stats.values())
    acc = total_correct / total_done if total_done > 0 else 0.0
    pbar.set_postfix({"acc": f"{acc:.3f}", "done": total_done})

pbar.close()

# ── Lưu kết quả ───────────────────────────────────────────────────────────────
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {RESULT_PATH}")

# ── In accuracy theo type ─────────────────────────────────────────────────────
print("\n{:<10} | {:>8} | {:>8} | {:>8}".format("Type", "Correct", "Total", "Acc"))
print("-" * 45)

all_correct = 0
all_total   = 0
for qtype in sorted(stats.keys()):
    c = stats[qtype]["correct"]
    t = stats[qtype]["total"]
    e = stats[qtype]["error"]
    acc = c / t if t > 0 else 0.0
    print("{:<10} | {:>8} | {:>8} | {:>7.2%}  (errors: {})".format(qtype, c, t, acc, e))
    all_correct += c
    all_total   += t

print("-" * 45)
overall = all_correct / all_total if all_total > 0 else 0.0
print("{:<10} | {:>8} | {:>8} | {:>7.2%}".format("OVERALL", all_correct, all_total, overall))
