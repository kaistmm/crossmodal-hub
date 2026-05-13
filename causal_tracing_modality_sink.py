import os, re, json
import torch, numpy
from collections import defaultdict
from logic import nethook
from logic.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)

from logic.causal_trace import (
    trace_with_patch,trace_with_patch_memory_optimized,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
import argparse
import json
import os
import torch
from model.modeling_qwen2_5_omni_low import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_utils import process_mm_info
from tqdm import tqdm
from logic.sink import get_layer_llm_sink_token
from logic.object import get_object_token,get_object_token_audio
from utils.logit import max_prob_for_word
from utils.utils_common import *
import os, warnings, logging
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper
import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.disable(logging.WARNING)
torch.set_grad_enabled(False)
import random
import torch

def build_layerlist(
    *,
    mode,                   # "animal" | "people"
    replace_mode,           # "all" | "sink" | "random"
    layer,                  # center layer
    window,                 # layer window size
    num_layers,
    video_idx,
    audio_idx,
    sink_token_data,   # dict: L -> {"video": [...], "audio": [...], ...}
    object_token_list,
    model,
    kind,
    seed=None               # random 재현성용 (optional)
):
    if seed is not None:
        random.seed(seed)

    layerlist = []

    # -----------------------------
    # 기준 token pool 결정
    # -----------------------------
    if mode == "sports":
        base_indices = audio_idx
        sink_key = "audio"
    elif mode == "people":
        base_indices = video_idx
        sink_key = "video"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # tensor → python list
    if isinstance(base_indices, torch.Tensor):
        base_indices = base_indices.detach().cpu().tolist()
    else:
        base_indices = list(base_indices)
    layer_range = range(max(layer, 0), min(num_layers, layer + window))
    base_set = set(base_indices)
    if sink_token_data is not None:
        
        base_sink_data = {
            int(k): v for k, v in sink_token_data.items()
            if int(k) in base_set
        }
        mds_mean_dict = dict(
            sorted(
                (
                    (token, float(np.nanmean(info["mds"])))
                    for token, info in base_sink_data.items()
                ),
                key=lambda x: x[1],
                reverse=True if mode=="people" else False   # 평균 mds 큰 것부터
            )
        )
        keys = list(mds_mean_dict.keys())
        n = len(keys)
        mid = n // 2
        uni_modal_sink_tokens = keys[:mid] #top 50% tokens
        cross_modal_sink_tokens = keys[mid:]
    if replace_mode == "all":
        for idx in base_indices:
            for L in layer_range:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode == "sink":
        for idx in base_sink_data.keys():
            if isinstance(idx, str):
                idx = int(idx)
            if idx not in base_indices:
                continue
            for L in layer_range:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode=="uni_modal_sink":
        for L in layer_range:
            for idx in uni_modal_sink_tokens:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode=="cross_modal_sink":
        for L in layer_range:
            for idx in cross_modal_sink_tokens:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode == "random_sink_num":
        k_mean = len(base_sink_data.keys())
        if k_mean <= 0:
            return layerlist
        if k_mean <= len(base_indices):
            rand_tokens = random.sample(base_indices, k_mean) 
        else:
            rand_tokens = random.choices(base_indices, k=k_mean)    
        for L in layer_range:
            for idx in rand_tokens:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode == "object":
        for L in layer_range:
            for idx in object_token_list:
                layerlist.append((idx, layername(model, L, kind)))
    elif replace_mode == "object_random_sink_num":
        k_mean = len(base_sink_data.keys())
        if len(object_token_list)==0:
            return layerlist
        if k_mean <= 0:
            return layerlist
        if k_mean <= len(object_token_list):
            rand_tokens = random.sample(object_token_list, k_mean) 
        else:
            rand_tokens = random.choices(object_token_list, k=k_mean)    
        for L in layer_range:
            for idx in rand_tokens:
                layerlist.append((idx, layername(model, L, kind)))
    else:
        raise ValueError(f"Unknown replace_mode: {replace_mode}")

    return layerlist

def parse_args():
    parser = argparse.ArgumentParser()
    valid_kinds = ["before_attn", "self_attn", "mlp"]
    valid_replace_modes = [
        "all",
        "sink",
        "object",
        "random_sink_num",
        "uni_modal_sink",
        "cross_modal_sink",
    ]

    parser.add_argument("--json_path", type=str, default=None,
                        help="Path of dataset json (.json)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save json result")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument(
        "--replace_mode",
        type=str,
        required=True,
        choices=valid_replace_modes,
        help="Replacement mode for causal tracing",
    )
    parser.add_argument("--ckpt_path", type=str, default="/mnt/bear3/users/jungji/ckpt/Qwen2.5-Omni-7B",
                        help="Path of model checkpoint")      
    parser.add_argument("--k_divide", type=int, default=2)
    parser.add_argument(
        "--kind",
        type=str,
        default="before_attn",
        choices=valid_kinds,
        help="Layer hook kind",
    )
    parser.add_argument("--sink_folder", type=str, default=None)
    args = parser.parse_args()
    if "sink" in args.replace_mode and args.k_divide not in [2, 3, 4]:
        parser.error("--k_divide must be one of [2, 3, 4] when replace_mode includes 'sink'.")
    return args
if __name__ == "__main__":
    args = parse_args()
    json_path=args.json_path
    save_path=args.save_path
    device="cuda"
    k_divide=args.k_divide
    kinds=[args.kind]
    window=50
    mode=args.mode 
    replace_mode=args.replace_mode
    layers=[0]
    ckpt_path=args.ckpt_path
    print(f"🍀 Loading model from {ckpt_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto",device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path,use_fast=False if "3B" in ckpt_path else True)
    less_audio=0
    num_layers=28 #.num_layers
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for kind in kinds:
        for layer in layers:
            try:
                if "sink" in replace_mode:
                    save_result_filename = os.path.join(save_path, replace_mode+f"k_divide_{k_divide}", kind+f"_layer{layer}_window{window}"+os.path.basename(json_path).replace(".jsonl", "_results.jsonl"))
                elif "object" not in replace_mode:
                    save_result_filename = os.path.join(save_path, replace_mode ,kind+f"_layer{layer}_window{window}"+os.path.basename(json_path).replace(".jsonl", "_results.jsonl"))
                else:
                    save_result_filename = os.path.join(save_path, replace_mode, kind+f"_layer{layer}_window{window}"+os.path.basename(json_path).replace(".jsonl", "_results.jsonl"))
                os.makedirs(os.path.dirname(save_result_filename), exist_ok=True)
                print(f"📁 Saving each output → {save_result_filename}")
                for idx, sample in enumerate(tqdm(data)):
                    USE_AUDIO_IN_VIDEO = True
                    conversation = [sample[0], sample[1]]# 
                    correct_pred_word=sample[1]["correct_pred"]
                    corrupt_pred_word=sample[1]["corrupt_pred"]
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                    video_name=os.path.basename(sample[1]["content"][0].get("video", "unknown_video")).split('.')[0]
                    if mode=="sports":
                        corrupt_videos = [v.clone() for v in videos]
                        corrupt_videos[0]=torch.zeros_like(videos[0])
                        print("🍍🍍🍍 Sports mode : Video corrupted!")
                        corrupt_inputs = processor(text=text, audio=audios, images=images, videos=corrupt_videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                        corrupt_inputs = corrupt_inputs.to(device).to(model.dtype)
                        sink_folder_path=f"{args.sink_folder}/k_divde_{k_divide}/{video_name}_history.json"
                    correct_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                    correct_inputs = correct_inputs.to(device).to(model.dtype)
                    input_token_len = correct_inputs["input_ids"].shape[1]
                    video_path=sample[1]["content"][0].get("video", "unknown_video")
                    video_idx = (correct_inputs["input_ids"][0] == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
                    audio_idx = (correct_inputs["input_ids"][0] == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
                    audio_ranges= get_continuous_ranges(audio_idx)
                    video_ranges= get_continuous_ranges(video_idx)
                    if len(audio_ranges)<5:
                        less_audio+=1
                        print(f"❌ Skipping sample idx {idx} due to insufficient audio tokens.")
                        continue
                    if "sink" in replace_mode:
                        with open(sink_folder_path, "r", encoding="utf-8") as f:
                            sink_token_data = json.load(f)
                    if "object" in replace_mode:
                        if mode=="people":
                            object_token_list, non_object_token_list = get_object_token(correct_inputs,video_idx, video_path,strict_level="high")
                            if object_token_list is None:
                                print(f"❌ Skipping sample idx {idx} due to object token extraction error.")
                                continue
                        elif mode=="sports":
                            object_token_list = get_object_token_audio(audio_idx, video_path,audio_ranges)
                            sink_token_data=None
                            if object_token_list is None:
                                print(f"❌ Skipping sample idx {idx} due to object token extraction error.")
                                continue
                    else:
                        object_token_list=None
                    layerlist=build_layerlist(mode=mode,replace_mode=replace_mode,layer=layer,window=window,
                                num_layers=num_layers,video_idx=video_idx,audio_idx=audio_idx,
                                sink_token_data=sink_token_data,object_token_list=object_token_list,model=model,kind=kind)
                    print("🍍",len(layerlist), "tokens to replace", replace_mode)
                    correct_output,restored_ouptut = trace_with_patch_memory_optimized(
                            model, correct_inputs, corrupt_inputs, layerlist)
                    restored_answer= processor.batch_decode(restored_ouptut["sequences"][0][input_token_len:].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    restored_probs = max_prob_for_word(restored_ouptut.logits,correct_pred_word, processor.tokenizer)
                    restored_probs_corrupt = max_prob_for_word(restored_ouptut.logits,corrupt_pred_word, processor.tokenizer)

                    result_dict = {
                        "id": idx,
                        "video": sample[1]["content"][0].get("video", "unknown_video"),
                        "audio_path": sample[1]["content"][0].get("audio", "unknown_audio"),
                        "label": sample[1]["label"] if "label" in sample[1] else None,
                        "label_name":sample[1].get("label_name", None),
                        "correct_pred": sample[1].get("correct_pred", None),
                        "corrupt_pred": sample[1].get("corrupt_pred", None),
                        "restored_pred": restored_answer[0],
                        "correct_correct_word_prob": sample[1].get("correct_correct_word_prob", None),
                        "correct_corrupt_word_prob": sample[1].get("correct_corrupt_word_prob", None),
                        "corrupt_correct_word_prob": sample[1].get("corrupt_correct_word_prob", None),
                        "corrupt_corrupt_word_prob": sample[1].get("corrupt_corrupt_word_prob", None),
                        "restored_correct_word_prob": restored_probs.item(),
                        "restored_corrupt_word_prob": restored_probs_corrupt.item(),
                        "layerlist_len": len(layerlist),
                    }
                    with open(save_result_filename, "a", encoding="utf-8") as w:
                        w.write(json.dumps(result_dict, ensure_ascii=False) + "\n")  # append
                    print(f"💚💚💚 {mode} {replace_mode}layer",layer,sample[1]["content"][0].get("video", "unknown_video"),sample[1]["content"][0].get("audio", "unknown_audio"))
                    correct_pred = sample[1].get("correct_pred")
                    corrupt_pred = sample[1].get("corrupt_pred")
                    correct_label = str(correct_pred) if correct_pred is not None else "None"
                    corrupt_label = str(corrupt_pred) if corrupt_pred is not None else "None"
                    print(f"{'TYPE':<9} | {'ANSWER':<10} | {(correct_label + ' PROBS'):<20} | {(corrupt_label + ' PROBS')}")
                    print("-" * 75)
                    print(
                        f"{'Correct':<9} | {correct_label:<10} | "
                        f"{fmt(sample[1].get('correct_correct_word_prob')):<20} | {fmt(sample[1].get('correct_corrupt_word_prob'))}"
                    )
                    print(
                        f"{'Corrupt':<9} | {corrupt_label:<10} | "
                        f"{fmt(sample[1].get('corrupt_correct_word_prob')):<20} | {fmt(sample[1].get('corrupt_corrupt_word_prob'))}"
                    )
                    print(
                        f"{'Restored':<9} | {str(restored_answer[0]):<10} | "
                        f"{fmt(restored_probs.item()):<20} | {fmt(restored_probs_corrupt.item())}"
                    )
            except Exception as e:
                print(f"❌ Error processing sample idx {idx}: {e}")
                continue
            finally:
                
                del correct_output
                del restored_ouptut
    print(f"\n🎉 ALL DONE — incremental results saved safely! Less audio cases skipped: {less_audio}")
