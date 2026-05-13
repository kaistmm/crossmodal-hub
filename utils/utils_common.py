import os
import torch
import json
from collections import defaultdict
from typing import Dict, List, Any
import os
from PIL import Image
import numpy as np
IMAGE_TOKEN_ID=151655
AUDIO_TOKEN_ID=151646
VIDEO_TOKEN_ID=151656

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
def load_jsonl_as_video_dict(
    jsonl_path: str,
    video_key: str = "video",
    allow_multiple: bool = True
) -> Dict[str, Any]:
    """
    Load a jsonl file and return a dict keyed by video (or video_path).

    Args:
        jsonl_path (str): path to .jsonl file
        video_key (str): primary key name ("video" or "video_path")
        allow_multiple (bool):
            - True  -> dict[video] = list of items
            - False -> dict[video] = single item (last one kept)

    Returns:
        Dict[str, Any]
    """
    if allow_multiple:
        video_dict = defaultdict(list)
    else:
        video_dict = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            key = item.get(video_key) or item.get("video_path")
            if key is None:
                continue
            if isinstance(key, list):
                key=os.path.basename(os.path.dirname(key[0]))
            if allow_multiple:
                video_dict[key].append(item)
            else:
                video_dict[key] = item

    return dict(video_dict)
def fmt(p, ndigits=3):
    if isinstance(p, (float, int)):
        return f"{p:.{ndigits}f}"
    else:
        return str(p)
def get_continuous_ranges(indices: torch.Tensor):
    if len(indices) == 0:
        return []
    indices = indices.tolist()
    ranges = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append([start, prev])
            start = i
            prev = i
    ranges.append([start, prev])
    return ranges

def image_to_audio_path(video_path: str) -> str:
    video_id = os.path.basename(os.path.dirname(video_path))
    audio_path = f"vggsound_with_mask/audio/{video_id}.wav"
    return audio_path
def zero_pil_image(img: Image.Image):
        w, h = img.size
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

def find_audio_path(video_path):
    if isinstance(video_path, list):
        path=image_to_audio_path(video_path[0])
        if os.path.exists(path):
            return path
    if "AVHBench" in video_path:
        base = video_path.replace(".mp4", ".wav").replace("videos", "audios")
        return base #vggsound_with_mask/sam2_mask_black
    elif "vggsound" in video_path and "sam2_mask_black" not in video_path:
        path=video_path.replace(".mp4",".wav").replace("video","audio")
        if os.path.exists(path):
            return path
    elif "vggsound" in video_path and "sam2_mask_black" in video_path:
        path=image_to_audio_path(video_path)
        if os.path.exists(path):
            return path
    else:
        path=video_path.replace(".mp4",".mp3")
        if os.path.exists(path):
            return path
        path=video_path.replace(".mp4",".wav")
        if os.path.exists(path):
            return path
    return None  # 모든 경로 실패