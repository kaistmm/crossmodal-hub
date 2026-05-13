
import torch
import torch.nn.functional as F
def max_prob_for_word(logits_tuple, word, tokenizer):
    # 두 후보: 공백 없음 / 공백 있음
    cand_ids_list = [
        tokenizer.encode(word, add_special_tokens=False),
        tokenizer.encode(" " + word, add_special_tokens=False),
    ]
    # 단어가 여러 subtoken이면 "첫 subtoken" 기준으로 max를 보자 (네 의도랑 가장 유사)
    first_ids = [ids[0] for ids in cand_ids_list if len(ids) > 0]

    # 각 step에서 후보 id들의 확률 중 최대를 취함
    step_max = []
    for step_logits in logits_tuple:                 # (B, V)
        lp = F.log_softmax(step_logits[0], dim=-1)  # (V,)
        step_max.append(torch.exp(lp[first_ids]).max())     # scalar

    step_max = torch.stack(step_max)  # (T,)
    max_prob, best_k = step_max.max(dim=0)
    return max_prob