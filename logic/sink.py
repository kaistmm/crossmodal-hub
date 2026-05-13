import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
IMAGE_TOKEN_ID=151655
AUDIO_TOKEN_ID=151646
VIDEO_TOKEN_ID=151656
DIM_SINK=[458, 2570]
TAU=25
def rmsnorm(hidden_states, eps=1e-6):
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps)

def run_logic(hs):
    rms_norm_hs = torch.abs(rmsnorm(hs)) # [bsz, tok, dim]
    rms_values = torch.stack([rms_norm_hs[:, :, idx] for idx in DIM_SINK], dim=-1) # [bsz, tok, 2]
    max_rms_values = torch.max(rms_values, dim=-1)[0] # [bsz, tok]
    indices = torch.nonzero(max_rms_values > TAU)[:, 1] # [batch_axis, token_axis] -> [token_axis]
    return indices
def gen_mean_norm(hs):
    rms_norm_hs = torch.abs(rmsnorm(hs))  # [bsz, tok, dim]
    # rms_values = rms_norm_hs.mean(dim=-1)  # [bsz, tok]
    rms_values = torch.stack( [rms_norm_hs[:, :, idx] for idx in DIM_SINK], dim=-1 )
    max_rms_values = torch.max(rms_values, dim=-1)[0] # [bsz, tok]
    max_rms_values = max_rms_values.mean()  # [bsz, tok]
    return max_rms_values
def get_hidden_dim(result, verbose=False):
    layer_dict = {}
    per_layer_unique = []
    for layer_idx in range(28):
        mean_rms_values = gen_mean_norm(result["hidden_states"][0][layer_idx])
        layer_dict[f"layer_{layer_idx}"] = mean_rms_values
        if verbose:
            print(f"[Layer {layer_idx:02d}] mean rms values computed.")
    return layer_dict
def get_global_sink_token(result,video_idx,audio_idx,K=400):
    layer_dict={}
    for layer_idx in range(28):
        indices=run_logic(result["hidden_states"][0][layer_idx])
        layer_dict[f"layer_{layer_idx}"]=indices
    from collections import Counter
    all_indices = torch.cat([v for v in layer_dict.values() if v.numel() > 0]).tolist()
    freq = Counter(all_indices)
    most_common_indices = freq.most_common(K)
    #print(most_common_indices)
    visual_sink_list = torch.tensor([idx for idx, cnt in most_common_indices if idx in video_idx]).cuda() #sink list중에 audio token이 있을수도 있지.
    visual_non_sink_list = torch.tensor([idx for idx in video_idx if idx not in visual_sink_list]).cuda()
    audio_sink_list = torch.tensor([idx for idx, cnt in most_common_indices if idx in audio_idx]).cuda()
    audio_non_sink_list = torch.tensor([idx for idx in audio_idx if idx not in audio_sink_list]).cuda()
    return visual_sink_list, visual_non_sink_list, audio_sink_list, audio_non_sink_list

def analyze_sink_modality_causal_fast(
    attentions,
    layer_idx,
    audio_idx,
    video_idx,
    sink_tokens,
    direction="future_looks_sink"
):
    """
    Returns dict: token_idx -> (vid_score, aud_score, mds)
    """
    attn = attentions[layer_idx][0]   # (H, Q, K)
    device = attn.device
    H = attn.shape[0]

    sink_tokens = torch.as_tensor(sink_tokens, device=device)

    video_idx = torch.as_tensor(video_idx, device=device)
    audio_idx = torch.as_tensor(audio_idx, device=device)

    results = {}

    for sink in sink_tokens:
        if direction == "future_looks_sink":
            # Query > sink, Key = sink
            v_mask = video_idx > sink
            a_mask = audio_idx > sink

            v_idx = video_idx[v_mask]
            a_idx = audio_idx[a_mask]

            # (H, Nv), (H, Na)
            v_attn = attn[:, v_idx, sink] if v_idx.numel() > 0 else None
            a_attn = attn[:, a_idx, sink] if a_idx.numel() > 0 else None

        else:  # sink_looks_back
            v_mask = video_idx <= sink
            a_mask = audio_idx <= sink

            v_idx = video_idx[v_mask]
            a_idx = audio_idx[a_mask]

            v_attn = attn[:, sink, v_idx] if v_idx.numel() > 0 else None
            a_attn = attn[:, sink, a_idx] if a_idx.numel() > 0 else None

        v_score = v_attn.mean() if v_attn is not None else torch.tensor(0., device=device)
        a_score = a_attn.mean() if a_attn is not None else torch.tensor(0., device=device)

        mds = (v_score - a_score) / (v_score + a_score + 1e-9)

        results[int(sink.item())] = (
            v_score.item(),
            a_score.item(),
            mds.item()
        )

    return results

def collect_layer_stats_fast(
    attentions,num_layers,
    audio_idx,
    video_idx,
    target_tokens
):
    #num_layers = len(attentions)
    token_history = defaultdict(lambda: {
        "mds": [],
        "v_score": [],
        "a_score": []
    })

    for layer_idx in range(num_layers):
        layer_res = analyze_sink_modality_causal_fast(
            attentions,
            layer_idx,
            audio_idx,
            video_idx,
            target_tokens
        )

        for t, (v, a, m) in layer_res.items():
            token_history[t]["mds"].append(m)
            token_history[t]["v_score"].append(v)
            token_history[t]["a_score"].append(a)

    return token_history
def _to_cpu_set(x):
    """tensor/list -> python set(int)"""
    if isinstance(x, torch.Tensor):
        return set(x.detach().cpu().tolist())
    return set(list(x))

def _to_long_cpu_tensor(x):
    """tensor/list -> cpu long tensor (possibly empty)"""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.long)
    return torch.tensor(list(x), device="cpu", dtype=torch.long)

def _unique_cpu(indices):
    """indices를 unique한 cpu long tensor로 정규화 (빈 경우도 처리)"""
    if isinstance(indices, torch.Tensor):
        if indices.numel() == 0:
            return torch.tensor([], dtype=torch.long, device="cpu")
        return torch.unique(indices.detach().to("cpu", dtype=torch.long))
    # list/iterable
    lst = list(indices)
    if len(lst) == 0:
        return torch.tensor([], dtype=torch.long, device="cpu")
    return torch.unique(torch.tensor(lst, dtype=torch.long, device="cpu"))

def get_layer_llm_sink_token(result, video_idx, audio_idx, layer_limit=23, verbose=False):
    video_set = _to_cpu_set(video_idx)
    audio_set = _to_cpu_set(audio_idx)

    layer_dict = {}
    for layer_idx in range(layer_limit):
        indices = run_logic(result["hidden_states"][0][layer_idx])  # whatever run_logic returns
        indices_u = _unique_cpu(indices)  # cpu long tensor unique

        # 분류
        if indices_u.numel() == 0:
            v = a = o = torch.tensor([], dtype=torch.long, device="cpu")
        else:
            lst = indices_u.tolist()
            v_list = [i for i in lst if i in video_set]
            a_list = [i for i in lst if i in audio_set]
            # other: 둘 다 아닌 것 (또는 video/audio index 범위 밖인 것)
            o_list = [i for i in lst if (i not in video_set and i not in audio_set)]

            v = torch.tensor(v_list, dtype=torch.long, device="cpu")
            a = torch.tensor(a_list, dtype=torch.long, device="cpu")
            o = torch.tensor(o_list, dtype=torch.long, device="cpu")

        layer_dict[layer_idx] = {
            "video": v,
            "audio": a,
            "other": o,
            # 원하면 전체도 같이 저장 가능
            "all": indices_u,
        }

        if verbose:
            print(
                f"[Layer {layer_idx:02d}] #sink(all)={indices_u.numel()} | "
                f"video={v.numel()} audio={a.numel()} other={o.numel()}"
            )

    return layer_dict


def get_global_llm_sink_token(result, video_idx, audio_idx,min_layers=8, verbose=False):
    layer_dict = {}
    per_layer_unique = []
    for layer_idx in range(28):
        indices = run_logic(result["hidden_states"][0][layer_idx])
        if isinstance(indices, torch.Tensor) and indices.numel() > 0:
            indices_u = torch.unique(indices).cpu()
        else:
            indices_u = indices

        layer_dict[f"layer_{layer_idx}"] = indices_u
        if isinstance(indices_u, torch.Tensor) and indices_u.numel() > 0:
            per_layer_unique.append(indices_u)

        if verbose:
            lst = indices_u.tolist() if hasattr(indices_u, "tolist") else list(indices_u)
            print(f"[Layer {layer_idx:02d}] #sink={len(lst)}")

    from collections import Counter
    if len(per_layer_unique) == 0:
        # sink가 하나도 없으면 빈 텐서 반환
        empty = torch.tensor([], device="cuda", dtype=torch.long)
        return empty, torch.tensor(video_idx, device="cuda"), empty, torch.tensor(audio_idx, device="cuda")

    all_indices = torch.cat(per_layer_unique).tolist()
    freq = Counter(all_indices)
    # ✅ min_layers 이상인 토큰만
    filtered = [(idx, cnt) for idx, cnt in freq.items() if cnt >= min_layers]
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
    video_set = set(video_idx.detach().cpu().tolist() if isinstance(video_idx, torch.Tensor) else list(video_idx))
    audio_set = set(audio_idx.detach().cpu().tolist() if isinstance(audio_idx, torch.Tensor) else list(audio_idx))
    visual_sink = [idx for idx, cnt in filtered_sorted if idx in video_set]
    audio_sink  = [idx for idx, cnt in filtered_sorted if idx in audio_set]
    visual_sink_list = torch.tensor(visual_sink, device="cuda", dtype=torch.long)
    audio_sink_list  = torch.tensor(audio_sink,  device="cuda", dtype=torch.long)
    visual_sink_set = set(visual_sink)
    audio_sink_set  = set(audio_sink)
    visual_non_sink_list = torch.tensor([idx for idx in video_set if idx not in visual_sink_set], device="cuda", dtype=torch.long)
    audio_non_sink_list  = torch.tensor([idx for idx in audio_set if idx not in audio_sink_set], device="cuda", dtype=torch.long)
    return visual_sink_list, visual_non_sink_list, audio_sink_list, audio_non_sink_list

def plot_topk_sink_token_freq(
    layer_dict,
    topk=400,
    save_path=None,
    show_token_idx_every=20  # 몇 개마다 token idx를 표시할지
):
    """
    x축: rank (1~topk)
    y축: 해당 토큰이 sink로 선택된 layer 수
    """

    # 1) 모든 sink token 모으기
    all_idx = []
    for _, indices in layer_dict.items():
        if indices is None:
            continue
        if isinstance(indices, torch.Tensor):
            if indices.numel() > 0:
                all_idx.extend(indices.detach().cpu().tolist())
        else:
            all_idx.extend(list(indices))

    if len(all_idx) == 0:
        raise ValueError("sink token이 없습니다.")

    # 2) 빈도 계산
    freq = Counter(all_idx)
    most_common = freq.most_common(topk)

    token_indices = [t for t, _ in most_common]
    counts = [c for _, c in most_common]

    # 3) plot
    ranks = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(18, 6))
    plt.bar(ranks, counts)

    plt.xlabel("Token rank (by #layer selections)")
    plt.ylabel("Selected count (#layers)")
    plt.title(f"Top-{topk} sink tokens (rank-based view)")

    # x축 눈금은 rank 기준으로 듬성듬성
    xticks = np.arange(1, len(ranks) + 1, show_token_idx_every)
    plt.xticks(xticks, xticks)

    plt.tight_layout()

    # 4) 저장 / 출력
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Saved] {save_path}")
    else:
        plt.show()

    # (선택) rank ↔ token index 매핑 반환
    rank_to_token = {
        rank: token
        for rank, token in zip(ranks, token_indices)
    }
    return rank_to_token
