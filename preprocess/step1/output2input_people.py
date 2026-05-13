
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
import csv
import json
import argparse
from tqdm import tqdm
from utils.utils_common import load_jsonl_as_video_dict,load_jsonl
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="exp_bear1/global2/qwen7b/step1/people/av/people_case_2.json")
    parser.add_argument("--output_path", type=str,
                        default="json_files/step2/people.json")
    return parser.parse_args()
def build_prompt(shuffled_categories: dict) -> str:
    """
    shuffled_categories: dict like
        {
            "A": "cattle mooing",
            "B": "zebra braying",
            ...
        }
    """
    if not shuffled_categories:
        raise ValueError("shuffled_categories is empty")

    lines = []
    lines.append("Classify the sounding object into one of the categories below:")
    lines.append("")

    for label, desc in shuffled_categories.items():
        lines.append(f"{label}. {desc}")

    last_letter = list(shuffled_categories.keys())[-1]
    lines.append("")
    lines.append(f"Respond ONLY with a single label letter (A–{last_letter}).")

    return "\n".join(lines)
#exp_bear1/real/qwen25_7b/step1/people_25_case_2/av/people_25_case_2_results.jsonl
args = parse_args()
input_file = args.input_file
v_file=input_file.replace("/av/","/v/")
v_dict=load_jsonl_as_video_dict(v_file)
output_path = args.output_path
converted=[]
for item in tqdm(load_jsonl(input_file)):
    user_instruction= build_prompt(item["choices"])
    entry = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": v_dict[item["video"]][0]["video"]},
                {"type": "text", "text": user_instruction},
            ],
            "label": item["label"],  # 첫 단어만 사용,"text_output_both": "monkey", "text_output_audio": "Dog", "text_output_video":
            "label_name": item["label_name"],
            "correct_pred": item["av"],
            "corrupt_pred": v_dict[item["video"]][0]["text_output"],
            "correct_correct_word_prob": item["av_prob"], #correct output에서 correct token의 확률
            "correct_corrupt_word_prob": item["v_prob"], #correct output에서 corrupt token의 확률
            "corrupt_correct_word_prob": v_dict[item["video"]][0]["av_prob"], #corrupt output에서 correct token의 확률
            "corrupt_corrupt_word_prob": v_dict[item["video"]][0]["v_prob"], #corrupt output에서 corrupt token의 확률
            "choices": item["choices"],
        }
    ]
    converted.append(entry)

# ========= 3) 저장 =========
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)

print(f"✅ 변환 완료! {len(converted)}개의 항목을 {output_path} 에 저장했습니다.")
