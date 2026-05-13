import torch
import numpy as np

def find_subsequence(haystack, needle):
    """
    haystack: List[int] (전체 토큰 id들)
    needle:   List[int] (찾을 토큰 id들)
    return:   List[Tuple[start, end]]  (end는 exclusive)
    """
    if len(needle) == 0 or len(haystack) < len(needle):
        return []
    hits = []
    L = len(needle)
    # 단순 O(nL) (keyword는 짧아서 보통 충분)
    for i in range(len(haystack) - L + 1):
        if haystack[i:i+L] == needle:
            hits.append((i, i+L))
    return hits


def get_keyword_token_indices_from_generate(
    sequences_1d,         # result["sequences"][0] (shape: [total_len])
    input_token_len,      # inputs 길이
    keyword,              # 찾고 싶은 문자열
    tokenizer,            # processor.tokenizer 또는 processor.tokenizer.backend_tokenizer 래핑
    try_variants=True,    # 공백/개행 변형도 시도
):
    """
    return:
      {
        "gen_range": (input_token_len, total_len),
        "matches_in_gen_local": [(s,e), ...],   # generated 부분 기준 인덱스 (0부터)
        "matches_in_full_global": [(S,E), ...], # sequences 전체 기준 인덱스
        "keyword_token_ids": [...],
      }
    """
    total_len = int(sequences_1d.shape[0])
    gen_ids = sequences_1d[input_token_len:].tolist()

    # keyword 토큰화 (중요: add_special_tokens=False)
    # tokenizer가 fast가 아니어도 token ids는 얻을 수 있음
    def encode_kw(kw):
        return tokenizer.encode(kw, add_special_tokens=False)

    candidates = [keyword]
    if try_variants:
        # 공백/개행 정규화 & 앞 공백 변형(많이 발생)
        kw = keyword.strip()
        candidates += [
            kw,
            " " + kw,
            kw.replace("\n", " ").replace("  ", " "),
            (" " + kw).replace("\n", " ").replace("  ", " "),
        ]

    best = None
    for cand in candidates:
        needle = encode_kw(cand)
        hits = find_subsequence(gen_ids, needle)
        if hits:
            best = (cand, needle, hits)
            break

    if best is None:
        return {
            "gen_range": (input_token_len, total_len),
            "matches_in_gen_local": [],
            "matches_in_full_global": [],
            "keyword_token_ids": [],
        }

    cand, needle, hits = best
    # generated 기준 -> full 기준으로 변환
    hits_global = [(input_token_len + s, input_token_len + e) for (s, e) in hits]

    return {
        "gen_range": (input_token_len, total_len),
        "matches_in_gen_local": hits,
        "matches_in_full_global": hits_global,
        "keyword_token_ids": needle,
        "matched_variant": cand,
    }
