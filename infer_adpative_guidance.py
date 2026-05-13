import argparse
import json
import os
import time
import torch
from model.modeling_qwen2_5_omni_low import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_utils import process_mm_info
from tqdm import tqdm
import numpy as np
from logic.subsequence import *
import os, warnings, logging
from utils.utils_common import *
from logic.sink import *
from utils.logit import *
from logic.adaptive_guidance_complex import GuidanceLogits

from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)


# ----------------------------- #
#       Argument Parser         #
# ----------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=str, default="json_files/step1/genuine.json",
                        help="Path of dataset json (.json)")
    parser.add_argument("--save_path", type=str, default="exp_bear1/hall/qwen7b/step1",
                        help="Directory to save json result")
    parser.add_argument("--modality", type=str, default="av", choices=["a", "v", "av"])
    parser.add_argument("--ckpt_path", type=str, default="/mnt/bear3/users/jungji/ckpt/Qwen2.5-Omni-7B",
                        help="Path of model checkpoint")                    
    parser.add_argument("--k_divide", type=str, default=3)
    parser.add_argument("--cali", type=str, default="crossmodal")
    parser.add_argument("--guidance_strength", type=float, default=0.5)
    parser.add_argument("--adaptive_threshold", type=float, default=0.6)
    parser.add_argument("--text_mass_threshold", type=float, default=0.4)

    return parser.parse_args()

# ----------------------------- #
#           Main Code           #
# ----------------------------- #
if __name__ == "__main__":
    args = parse_args()
    device="cuda"
    # 모델 로드
    cali=args.cali
    ckpt_path=args.ckpt_path
    k_divide=int(args.k_divide)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto",device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path,use_fast=False if "3B" in ckpt_path else True)
    modality = args.modality
    adaptive_threshold=args.adaptive_threshold
    text_mass_threshold=args.text_mass_threshold

    save_result_filename = os.path.join(
        args.save_path,f"{cali}_str_{args.guidance_strength}_k_divide_{k_divide}_adaptive_{adaptive_threshold}_textmass_{text_mass_threshold}",
        os.path.basename(args.json_path).replace(".json", "_results.jsonl")  # jsonl형태 추천
    )
    if os.path.exists(save_result_filename):
        #지우기
        os.remove(save_result_filename)
    os.makedirs(os.path.dirname(save_result_filename), exist_ok=True)

    
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # [SPEED] Timing stats
    total_gen_tokens = 0
    wall_start = time.time()

    # [OPT] Open output file once (append mode) and line-buffer writes
    with open(save_result_filename, "a", encoding="utf-8", buffering=1) as out_file:
        # [OPT] Wrap entire loop in inference_mode to eliminate autograd overhead
        with torch.inference_mode():
            for idx, sample in enumerate(tqdm(data)):
                try:
                    if sample[1]["content"][1]["text"].startswith("Is "):
                        sample[1]["content"][1]["text"] = "Describe what you see and hear in a single sentence."
                    conversation = [sample[0], sample[1]]
                    ground_truth = sample[1]["label"] if "label" in sample[1] else None
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
                    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                    inputs = inputs.to(device).to(model.dtype)
                    video_idx = (inputs["input_ids"][0] == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
                    audio_idx = (inputs["input_ids"][0] == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
                    masking_token = {
                        "k_divide": k_divide,
                        "cali": cali
                    }
                    result = model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        masking_token=masking_token,
                        logits_processor=LogitsProcessorList([
                            GuidanceLogits(
                                guidance_strength=args.guidance_strength,
                                inputs=inputs,
                                adaptive_threshold=adaptive_threshold,
                                text_mass_threshold=text_mass_threshold,
                                model=model,
                                tokenizer=processor.tokenizer,
                                masking_token=masking_token,
                                video_idx=video_idx,
                                audio_idx=audio_idx
                            ),
                        ])
                    )
                    input_token_len = inputs["input_ids"].shape[1]

                    gen_tokens = result["sequences"].shape[-1] - input_token_len
                    total_gen_tokens += gen_tokens

                    # [OPT] Single batch_decode instead of duplicate calls
                    text_output = processor.batch_decode(
                        result["sequences"][0][input_token_len:].unsqueeze(0),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    print(text_output)
                    video_path = sample[1]["content"][0].get("video", "unknown_video")
                    audio_path = find_audio_path(video_path)

                    result_dict = {
                        "id": idx,
                        "video": video_path,
                        "audio_path": audio_path,
                        "label": sample[1]["label"] if "label" in sample[1] else None,
                        "original_label": sample[1].get("original_label", None),
                        "objects": sample[1].get("objects", None),
                        "text_output": text_output[0],
                        "single_label": sample[1].get("single_label", None),
                        "vggsounder_label": sample[1].get("vggsounder_label", None),
                        "single_object": sample[1].get("single_object", None),
                        "vggsounder_object": sample[1].get("vggsounder_object", None),
                    }
                    out_file.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
                    # Ensure intermediate results are persisted during long runs.
                    out_file.flush()

                    # [OPT] Free result tensors (no empty_cache — it forces slow GPU sync)
                    del result, inputs

                except Exception as e:
                    print(f"❗❗❗ Error at idx {idx}: {e}")
                    continue
    # [SPEED] Final summary
    wall_elapsed = time.time() - wall_start
    avg_ms = (wall_elapsed * 1000) / max(total_gen_tokens, 1)
    summary = {
        "total_samples": len(data),
        "total_tokens": total_gen_tokens,
        "total_time_s": round(wall_elapsed, 2),
        "avg_ms_per_token": round(avg_ms, 2),
    }
    summary_path = save_result_filename.replace("_results.jsonl", "_speed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n🎉 ALL DONE")
    print(f"📊 Speed: {total_gen_tokens} tokens in {wall_elapsed:.1f}s → {avg_ms:.2f} ms/token")
    print(f"📄 Summary saved → {summary_path}")
