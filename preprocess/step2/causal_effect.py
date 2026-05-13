import json, glob, os, re
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# IO helpers
# -----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_mode_from_path(path):
    return os.path.basename(os.path.dirname(path))

def extract_attn_layer(jsonl_path):
    basename = os.path.basename(jsonl_path)
    m = re.search(r"attn_layer(\d+)", basename)
    return int(m.group(1)) if m else None

def get_video_key(item):
    v = item.get("video", None)
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    if isinstance(v, str):
        return v
    return None

# -----------------------------
# Metrics helpers
# -----------------------------
def _clip_by_quantile(vals, q_low=0.00, q_high=1.0):
    arr = np.asarray(vals, dtype=np.float64)
    lo, hi = np.quantile(arr, [q_low, q_high])
    arr = arr[(arr >= lo) & (arr <= hi)]
    return arr.tolist()

def compute_means_and_layerlen(jsonl_path, target_videos=None, q_low=0.0, q_high=1.0,threshold=32256):
    diff_restored_correct = []
    print("threshold:", threshold)
    diff_corrupt_corrupt = []
    layer_lens = []

    for item in load_jsonl(jsonl_path):
        
        rcp  = item.get("restored_correct_word_prob")
        ccp  = item.get("corrupt_correct_word_prob")
        ccp2 = item.get("corrupt_corrupt_word_prob")
        rcp2 = item.get("restored_corrupt_word_prob")
        if None in (rcp, ccp, ccp2, rcp2):
            continue


        ll = item.get("layerlist_len", None)
        if ll is not None:
            if threshold is not None:
                if ll < threshold:
                    layer_lens.append(ll)
                    diff_restored_correct.append(rcp - ccp)
                    diff_corrupt_corrupt.append(ccp2 - rcp2)
            else:
                layer_lens.append(ll)
                diff_restored_correct.append(rcp - ccp)
                diff_corrupt_corrupt.append(ccp2 - rcp2)

    if len(diff_restored_correct) == 0:
        return None

    # (옵션) quantile clipping (네 코드 스타일 유지) :contentReference[oaicite:3]{index=3}
    diff_restored_correct_f = _clip_by_quantile(diff_restored_correct, q_low, q_high)
    diff_corrupt_corrupt_f  = _clip_by_quantile(diff_corrupt_corrupt,  q_low, q_high)

    if len(diff_restored_correct_f) == 0:
        diff_restored_correct_f = diff_restored_correct
    if len(diff_corrupt_corrupt_f) == 0:
        diff_corrupt_corrupt_f = diff_corrupt_corrupt

    # layerlist_len 평균 (유효 샘플 기준)
    mean_layerlist_len = float(np.mean(layer_lens)) if len(layer_lens) > 0 else float("nan")

    return {
        "mean_restored_correct_minus_corrupt_correct":
            float(sum(diff_restored_correct_f) / len(diff_restored_correct_f)),
        "mean_corrupt_corrupt_minus_restored_corrupt":
            float(sum(diff_corrupt_corrupt_f) / len(diff_corrupt_corrupt_f)),
        "num_samples_total_matching": int(len(diff_restored_correct)),   # ✅ 필터 후 남는 샘플 수
        "mean_layerlist_len": mean_layerlist_len,                       # ✅ layerlist_len 평균
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


def run_analysis(
    jsonl_path,
    threshold=None,
    plot_mode="plot",
    title=None,
    save_path=None,
):
    results = compute_means_and_layerlen(jsonl_path,threshold=threshold)
    txt_path=jsonl_path.replace(".json", f"_{threshold}_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis Results for {jsonl_path}\n")
        for key, value in results.items():
            if isinstance(value, (int, float)) and "_minus_" in key:
                f.write(f"{key}: {value * 100:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Saved analysis results to {txt_path}")
# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    base = "sports"#
    attn = "before_attn" #exp_bear1/final/qwen7b/step2/people/sinkk_divide_3
    # type = "object_high" #"all" #exp_bear1/final/qwen7b/step2/people/sinkk_divide_2
    threshold=None #exp_bear1/final/qwen3b/step2/sports/object
    type=None
    # exp_bear1/final/plus7b/step2/sports/object
    jsonl_glob = glob.glob(f"exp_bear1/appendix_layer2/qwen7b/step2/people/all/*.json")
    for jsonl_path in jsonl_glob:
        print("🍀 Processing:", jsonl_path)
        run_analysis(
            jsonl_path, 
            threshold=threshold,
            title=f"{base}"
        )
