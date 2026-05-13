import argparse
import json
import os
import torch
from model.modeling_qwen2_5_omni_low import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_utils import process_mm_info
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, warnings, logging
from utils.utils_common import *
from logic.sink import *
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

    parser.add_argument("--json_path", type=str, default="json_files/animal_25.json",
                        help="Path of dataset json (.json)")
    parser.add_argument("--save_path", type=str, default="exp_bear1/real/qwen25_7b/animal_25",
                        help="Directory to save json result")
    parser.add_argument("--modality", type=str, default="av", choices=["a", "v", "av"])
    parser.add_argument("--ckpt_path", type=str, default="/mnt/bear3/users/jungji/ckpt/Qwen2.5-Omni-7B",
                        help="Path of model checkpoint (.bin or .pt)")
    return parser.parse_args()

#case 1 : a+v 둘 다 사용
#case 2 : a+zero_out(v)
#case 3 : v+zero_out(a)
# ----------------------------- #
#           Main Code           #
# ----------------------------- #
if __name__ == "__main__":
    args = parse_args()
    device="cuda"
    ckpt_path = args.ckpt_path
    # 모델 로드
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto",device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path,use_fast=False if "3B" in ckpt_path else True)
    modality = args.modality
    
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    save_result_filename = os.path.join(
        args.save_path,modality,
        os.path.basename(args.json_path).replace(".json", "_results.jsonl")  # jsonl형태 추천
    )
    os.makedirs(os.path.dirname(save_result_filename), exist_ok=True)
    print(f"📁 Saving each output → {save_result_filename}")
    for idx, sample in enumerate(tqdm(data)):
        #try:
            conversation = [sample[0], sample[1]]
            ground_truth = sample[1]["label"] if "label" in sample[1] else None
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            if modality == "a":
                try:
                    videos = [v.clone() for v in videos]
                    videos[0]=torch.zeros_like(videos[0])
                    print("🍍🍍🍍 using only audio")
                except:
                    videos = [
                        [zero_pil_image(img) for img in img_list]
                        for img_list in videos   # images = [[PIL.Image, PIL.Image, ...], ...]
                    ]
                    print("🍍🍍🍍 using only audio")
            elif modality=="v":
                audios=[a.copy() for a in audios]
                audios[0]=np.zeros_like(audios[0])
                print("🍍🍍🍍 using only video")
            elif modality=="av":
                print("🍍🍍🍍 using both audio and video")
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            inputs = inputs.to(device).to(model.dtype)
            input_token_len = inputs["input_ids"].shape[1]
            result = model.generate(**inputs, use_audio_in_video=True)
            text_output = processor.batch_decode(result["sequences"][0][input_token_len:].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            video_path = sample[1]["content"][0].get("video", "unknown_video")
            audio_path= find_audio_path(video_path)
            result_dict = {
                "id": idx,
                "video": video_path,
                "audio_path": audio_path,
                "label": sample[1]["label"] if "label" in sample[1] else None,
                "label_name": sample[1].get("label_name", None),
                "text":sample[1]["content"][1].get("text", "unknown_video"),
                "choices": sample[1]["choices"] if "choices" in sample[1] else None,
                "text_output": text_output[0]
            }
            
            with open(save_result_filename, "a", encoding="utf-8") as w:
                w.write(json.dumps(result_dict, ensure_ascii=False) + "\n")  # append
            print("💚💚💚", idx, video_path, audio_path, text_output[0])
            #print("💚💚💚 gt", sample[1]["label"])
        # except Exception as e:
        #     print(f"❗❗❗ Error at idx {idx}: {e}")
        #     continue
    print("\n🎉 ALL DONE — incremental results saved safely!")
