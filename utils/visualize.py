import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize






import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def split_and_sort_sinks_by_mean_mds(history, sinks, audio_idx, video_idx):
    if hasattr(audio_idx, "cpu"):
        audio_set = set(audio_idx.cpu().numpy().tolist())
    else:
        audio_set = set(list(audio_idx))

    if hasattr(video_idx, "cpu"):
        video_set = set(video_idx.cpu().numpy().tolist())
    else:
        video_set = set(list(video_idx))

    audio_sinks, video_sinks, unknown = [], [], []
    for t in sinks:
        if t in audio_set:
            audio_sinks.append(t)
        elif t in video_set:
            video_sinks.append(t)
        else:
            unknown.append(t)

    def mean_mds(t):
        m = history[t]["mds"]
        return float(np.mean(m)) if len(m) > 0 else np.nan

    # ✅ audio sink: mean MDS 작은 순 (ascending)
    audio_sinks_sorted = sorted(audio_sinks, key=mean_mds)

    # ✅ video sink: mean MDS 큰 순 (descending)
    video_sinks_sorted = sorted(video_sinks, key=mean_mds, reverse=True)

    return audio_sinks_sorted, video_sinks_sorted, unknown

def plot_mds_heatmap(history, sinks_sorted, num_layers, out_path,
                     audio_color="#DAE8FC", video_color="#F8CECC",
                     title=None):
    """
    x축: layer (0..num_layers-1)
    y축: sorted sink token positions (row index는 정렬된 순서)
    색: MDS 값 (작을수록 audio_color, 클수록 video_color)
    """
    if len(sinks_sorted) == 0:
        print(f"[skip] No sinks to plot for {out_path}")
        return

    # matrix: (num_sinks, num_layers)
    mat = np.zeros((len(sinks_sorted), num_layers), dtype=np.float32)
    for row, t in enumerate(sinks_sorted):
        mds_list = history[t]["mds"]
        # 안전: 길이 안 맞으면 pad/truncate
        if len(mds_list) < num_layers:
            mds_list = list(mds_list) + [mds_list[-1]] * (num_layers - len(mds_list))
        mat[row, :] = np.array(mds_list[:num_layers], dtype=np.float32)

    # custom cmap: audio -> video
    cmap = LinearSegmentedColormap.from_list("audio_to_video", [audio_color, video_color])

    # 범위 고정하면 비교가 쉬움 (MDS 정의상 [-1, 1])
    vmin, vmax = -1.0, 1.0

    plt.figure(figsize=(max(6, num_layers * 0.25), max(4, len(sinks_sorted) * 0.18)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Layer")
    plt.ylabel("Sink token index (sorted)")
    if title:
        plt.title(title)

    # y tick은 너무 많으면 생략/간소화
    if len(sinks_sorted) <= 40:
        plt.yticks(range(len(sinks_sorted)), sinks_sorted)
    else:
        # 너무 많으면 10개만 띄엄띄엄 표시
        step = max(1, len(sinks_sorted) // 10)
        yticks = list(range(0, len(sinks_sorted), step))
        ylabels = [str(sinks_sorted[i]) for i in yticks]
        plt.yticks(yticks, ylabels)

    plt.xticks(range(num_layers), range(num_layers))
    cbar = plt.colorbar(im)
    cbar.set_label("MDS")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Saved MDS heatmap to {out_path}")


