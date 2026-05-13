import cv2
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import string
from typing import List
import os
import matplotlib.pyplot as plt
import os, re, bisect, glob
from utils.utils_common import *
def _frame_id(p: str) -> int:
    """
    지원:
      001.jpg
      001_raw.jpg
      000123_raw.jpg
    """
    base = os.path.basename(p)
    m = re.search(r"(\d+)(?:_raw)?\.jpg$", base)
    return int(m.group(1)) if m else -1

def _uniform_targets(k=10, start=0, end=200):
    if k <= 1:
        return [start]
    return [int(round(start + i * (end - start) / (k - 1))) for i in range(k)]

def _snap(frames_sorted, t):
    """
    frames_sorted: 존재하는 프레임 번호 정렬 리스트
    t: 목표 프레임 번호
    return: t에 가장 가까운 프레임 번호(동률이면 왼쪽)
    """
    j = bisect.bisect_left(frames_sorted, t)
    if j == 0:
        return frames_sorted[0]
    if j == len(frames_sorted):
        return frames_sorted[-1]
    left = frames_sorted[j - 1]
    right = frames_sorted[j]
    return left if (t - left) <= (right - t) else right

def select_10_jpgs(image_paths, raw=False,max_frame=200):
    """
    - image_paths가 정확히 10개면 그대로 반환
    - 아니면 0~min(max_frame, 실제최대프레임) 범위에서 10개 등간격 목표 프레임 만들고,
      각 목표를 가장 가까운 존재 jpg로 스냅해서 반환 (중복 허용)
    """
    # 1) 필터 + 프레임 파싱
    items = []
    for p in image_paths:
        if raw:
            if "_raw" not in p:
                continue
        else:
            if "_raw" in p:
                continue
        fid = _frame_id(p)
        if fid >= 0:
            items.append((fid, p))

    if not items:
        return []

    # 2) 프레임 번호로 정렬
    items.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in items]

    # 3) 정확히 10장이면 그대로
    if len(sorted_paths) == 10:
        return sorted_paths

    # 4) 스냅 샘플링(중복 허용)
    frames = [fid for fid, _ in items]
    frame_to_path = {}
    for fid, p in items:
        frame_to_path[fid] = p  # 같은 fid가 여러개면 마지막으로 덮임

    end = min(max_frame, frames[-1])  # 목표 범위를 실제 존재 최대 프레임에 맞춰 클램프
    targets = _uniform_targets(k=10, start=0, end=end)

    chosen = []
    for t in targets:
        fid = _snap(frames, t)
        chosen.append(frame_to_path[fid])

    return chosen

def get_object_token(
    inputs,
    video_idx,
    video_path,         # 프레임 mask 경로 리스트 (gt_masks/*.png ...)
    spatial_merge_size=2,
    white_threshold=0.1,   # (지금은 안 쓰지만, 나중에 비율 threshold 쓸 때 대비로 둠)
    enlarge=False, 
    reconstruct=False, 
    save_dir=None
):
    try:
        video_grid_thw = inputs["video_grid_thw"][0]   # (3,) 형태 가정
        t = int(video_grid_thw[0].item())             # 예: 3 (temporal patches)
        H_grid_merged = int(video_grid_thw[1].item())
        W_grid_merged = int(video_grid_thw[2].item())
        h = H_grid_merged // spatial_merge_size   # 예: 12
        w = W_grid_merged // spatial_merge_size   # 예: 12
        object_token_list = []
        non_object_token_list = []
        import glob
        image_paths=glob.glob(video_path.replace("video","masks").replace(".mp4","")+"/*.jpg")
        image_paths=[p for p in image_paths if "raw" not in p]
        #image_paths=[p.replace("_raw","") for p in video_path if os.path.exists(p.replace("_raw",""))]
        image_paths = select_10_jpgs(image_paths, max_frame=200)
        flat_indexs=[]
        for i in range(t):
            png_path = image_paths[i]
            # 1. PNG 프레임(gt mask) 읽기
            frame = cv2.imread(png_path)  # (H, W, 3)
            # if frame is None or frame is None:
            #     print(f"[WARN] PNG frame {png_path} 읽기 실패, skip")
            #     continue
            # 흑백 mask로 변환
            if frame.ndim == 3:
                gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame
            H, W = gray1.shape[:2]
            cell_w = W / float(w)
            cell_h = H / float(h)
            # 이 프레임의 grid별 object 존재 여부 저장할 2D mask
            # object_cells[gy, gx] = True면 해당 grid는 object
            object_cells = np.zeros((h, w), dtype=bool)

            # 3. 각 grid cell마다 object 여부 판별 (조금이라도 흰색 포함되면 object)
            for gy in range(h):
                for gx in range(w):
                    # 해당 grid cell의 픽셀 영역 좌표 계산
                    x0 = int(round(gx * cell_w))
                    x1 = int(round((gx + 1) * cell_w))
                    y0 = int(round(gy * cell_h))
                    y1 = int(round((gy + 1) * cell_h))

                    # boundary 보정
                    x0 = max(0, min(x0, W))
                    x1 = max(0, min(x1, W))
                    y0 = max(0, min(y0, H))
                    y1 = max(0, min(y1, H))

                    if x1 <= x0 or y1 <= y0:
                        continue

                    patch1 = gray1[y0:y1, x0:x1]
                    has_white1 = (patch1 > 30).any()
                    
                    if has_white1:
                        object_cells[gy, gx] = True

            for gy in range(h):
                for gx in range(w):
                    flat_index = i * (h * w) + gy * w + gx
                    if flat_index >= len(video_idx):
                        continue

                    if object_cells[gy, gx]:
                        object_token_list.append(video_idx[flat_index].item())
                        flat_indexs.append(flat_index)
                    else:
                        non_object_token_list.append(video_idx[flat_index].item())
        if reconstruct:
            png_path_=png_path.replace(".jpg","_raw.jpg")
            if save_dir is not None:
                recon_dir=os.path.join(save_dir, "reconstruct_overlay")
            else:
                recon_dir="reconstruct_overlay"
            os.makedirs(recon_dir, exist_ok=True)
            frame = cv2.imread(png_path)  # (H, W, 3)
            vis = frame.copy() 
            obj_color = (0, 255, 0)
            non_color = (128, 128, 128)
            thickness_obj = 2
            thickness_non = 1
            cnt=0
            i=0
            for gy in range(h):
                for gx in range(w):
                    x0 = int(round(gx * cell_w))
                    x1 = int(round((gx + 1) * cell_w))
                    y0 = int(round(gy * cell_h))
                    y1 = int(round((gy + 1) * cell_h))

                    x0 = max(0, min(x0, W))
                    x1 = max(0, min(x1, W))
                    y0 = max(0, min(y0, H))
                    y1 = max(0, min(y1, H))

                    if x1 <= x0 or y1 <= y0:
                        continue
                    flat_index = i * (h * w) + gy * w + gx            
                    if flat_index in flat_indexs:#object_cells[gy, gx]:
                        cv2.rectangle(vis, (x0, y0), (x1, y1), obj_color, thickness_obj)
                    else:
                        cv2.rectangle(vis, (x0, y0), (x1, y1), non_color, thickness_non)

            # 저장 경로 설정
            base_name = os.path.basename(png_path)
            if recon_dir is None:
                out_path = png_path.replace(".png", "_overlay.png")
            else:
                out_path = os.path.join(recon_dir, base_name.replace(".png", "_overlay.png"))

            cv2.imwrite(out_path, vis)
            print(f"🔥🔥 Saved reconstruction overlay at: {out_path}")
        return object_token_list, non_object_token_list
    except Exception as e:
        return None, None
with open("vggsound_with_mask/genuine_flam_qwen.json", "r") as f:
    important_data1 = json.load(f)
with open("vggsound_with_mask/hallucination_flam_qwen.json", "r") as f:
    important_data2 = json.load(f)
important_data = important_data1 + important_data2  
audio2info = {
    item["audio"]: {k: v for k, v in item.items() if k != "audio"}
    for item in important_data
}

def get_object_token_audio(audio_idx, video_path, audio_ranges):
    audio_path=video_path.replace("/video/","/audio/").replace(".mp4",".wav")
    important_blocks=audio2info[audio_path]["important_blocks"]
    #len_=len(audio_ranges)
    selected_ranges = [audio_ranges[i] for i in important_blocks]
    selected_tokens = []
    for i in important_blocks:
        start, end = audio_ranges[i]
        selected_tokens.extend(range(start, end + 1))
    return selected_tokens
    #audio_ranges[]
    
    
    
def get_object_token_from_image_list(
    inputs,
    video_idx,
    image_list,         
    spatial_merge_size=2, 
    strict_level="very_high",
    reconstruct=False, 
    save_dir="figs"
):
    print("strict_level",strict_level)
    try:
        video_grid_thw = inputs["video_grid_thw"][0]
        t = int(video_grid_thw[0].item())
        H_grid_merged = int(video_grid_thw[1].item())
        W_grid_merged = int(video_grid_thw[2].item())
        h = H_grid_merged // spatial_merge_size
        w = W_grid_merged // spatial_merge_size
        
        object_token_list = []
        non_object_token_list = []
        image_paths = [p.replace("_raw.jpg", ".jpg") for p in image_list]
        flat_indexs = []  # 전체 비디오에서의 object 토큰 인덱스 모음

        # --- Loop Start ---
        for i in range(t):
            # 인덱스 범위 체크 (이미지 리스트보다 t가 길 경우 방지)
            if 2 * i + 1 >= len(image_paths):
                break

            png_path1 = image_paths[2 * i]
            png_path2 = image_paths[2 * i + 1]
            
            # 1. 이미지 읽기
            frame1 = cv2.imread(png_path1)
            frame2 = cv2.imread(png_path2)
            
            if frame1.ndim == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2
            
            H, W = gray1.shape[:2]
            cell_w = W / float(w)
            cell_h = H / float(h)
            
            object_cells = np.zeros((h, w), dtype=bool)

            # 2. Object 감지
            for gy in range(h):
                for gx in range(w):
                    x0 = int(round(gx * cell_w))
                    x1 = int(round((gx + 1) * cell_w))
                    y0 = int(round(gy * cell_h))
                    y1 = int(round((gy + 1) * cell_h))

                    x0 = max(0, min(x0, W))
                    x1 = max(0, min(x1, W))
                    y0 = max(0, min(y0, H))
                    y1 = max(0, min(y1, H))

                    if x1 <= x0 or y1 <= y0:
                        continue

                    patch1 = gray1[y0:y1, x0:x1]
                    patch2 = gray2[y0:y1, x0:x1]
                    if strict_level == "high":
                        white_ratio1 = np.sum(patch1 > 20) / patch1.size
                        white_ratio2 = np.sum(patch2 > 20) / patch2.size
                        if white_ratio1 > 0.2 and white_ratio2 > 0.2:
                            object_cells[gy, gx] = True
                    elif strict_level == "very_high":
                        white_ratio1 = np.sum(patch1 > 40) / patch1.size
                        white_ratio2 = np.sum(patch2 > 40) / patch2.size
                        if white_ratio1 > 0.4 and white_ratio2 > 0.4:
                            object_cells[gy, gx] = True
            # 3. 토큰 분류 및 저장
            current_frame_flat_indices = [] # 현재 프레임에서 object로 판별된 인덱스들 (Reconstruct용)

            for gy in range(h):
                for gx in range(w):
                    # 전체 비디오 기준 Flat Index 계산
                    flat_index = i * (h * w) + gy * w + gx
                    
                    if flat_index >= len(video_idx):
                        continue

                    if object_cells[gy, gx]:
                        object_token_list.append(video_idx[flat_index].item())
                        flat_indexs.append(flat_index)
                        current_frame_flat_indices.append(flat_index) # 현재 프레임 그리기용
                    else:
                        non_object_token_list.append(video_idx[flat_index].item())

            # --- [수정됨] Reconstruct가 Loop 안으로 들어옴 ---
            if reconstruct:
                if save_dir is not None:
                    recon_dir = os.path.join(save_dir, "reconstruct_overlay", strict_level)
                else:
                    recon_dir = "reconstruct_overlay"
                os.makedirs(recon_dir, exist_ok=True)
                
                vis = frame2.copy()  # 현재 Loop의 frame2 사용
                obj_color = (0, 255, 0)
                non_color = (128, 128, 128)
                
                for gy in range(h):
                    for gx in range(w):
                        x0 = int(round(gx * cell_w))
                        x1 = int(round((gx + 1) * cell_w))
                        y0 = int(round(gy * cell_h))
                        y1 = int(round((gy + 1) * cell_h))

                        x0 = max(0, min(x0, W))
                        x1 = max(0, min(x1, W))
                        y0 = max(0, min(y0, H))
                        y1 = max(0, min(y1, H))
                        
                        if x1 <= x0 or y1 <= y0:
                            continue
                        
                        # 현재 시점 i를 사용하여 인덱스 계산
                        flat_index = i * (h * w) + gy * w + gx
                        
                        # 현재 프레임 계산 결과인 object_cells를 직접 쓰거나 flat_index 비교
                        if object_cells[gy, gx]: 
                            cv2.rectangle(vis, (x0, y0), (x1, y1), obj_color, 2)
                        else:
                            cv2.rectangle(vis, (x0, y0), (x1, y1), non_color, 1)

                base_name = os.path.basename(png_path2)
                out_path = os.path.join(recon_dir, base_name.replace(".png", "_overlay.png"))
                
                cv2.imwrite(out_path, vis)
                print(f"Saved: {out_path}") # 너무 많이 찍히면 주석 처리

        return object_token_list, non_object_token_list
    except Exception as e:
        print(f"Error: {e}")
        return None, None