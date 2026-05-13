import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
import math

class GuidanceLogits(LogitsProcessor):
    def __init__(self, guidance_strength, inputs, model, tokenizer=None, 
                 video_idx=None, audio_idx=None, masking_token=None,
                 adaptive_threshold=0.6, text_mass_threshold=0.4,
                 target_layers=range(18, 26), # 분석할 레이어 범위
                 momentum=0.5 # [NEW] 관성 계수 (0.0 ~ 1.0, 클수록 변화가 느리고 부드러움)
                 ):
        self.base_guidance_strength = guidance_strength
        self.input = inputs
        self.masking_token = masking_token
        self.model = model
        self.tokenizer = tokenizer
        
        # State
        self.out = None        
        self.pkv_masked = None 
        self.pkv_orig = None   
        self.prev_strength = 0.0 # [NEW] 이전 스텝의 강도를 기억
        self.momentum = momentum # [NEW] 관성 설정
        
        # Indices
        self.video_idx = video_idx if video_idx is not None else []
        self.audio_idx = audio_idx if audio_idx is not None else []
        
        max_v = max(self.video_idx) #if self.video_idx else 0
        max_a = max(self.audio_idx)# if self.audio_idx else 0
        self.text_token_start = max(max_v, max_a) + 1
        
        # Thresholds
        self.adaptive_threshold = adaptive_threshold
        self.text_mass_threshold = text_mass_threshold
        self.target_layers = target_layers

    def _calculate_adaptive_metrics(self, attention_outputs,sink_data):
        cross_sum = 0.0
        uni_sum = 0.0
        text_mass_list = []
        av_sink = sink_data.get("av_sink", None)
        va_sink = sink_data.get("va_sink", None)
        aa_sink = sink_data.get("aa_sink", None)
        vv_sink = sink_data.get("vv_sink", None)
        for layer_idx in self.target_layers:
            last_layer_attn = attention_outputs[layer_idx]
            curr_token_attn = last_layer_attn[:, :, -1, :] 
            
            # 1. Text Mass
            t_mass = curr_token_attn[:, :, self.text_token_start:].sum(dim=-1).mean()
            text_mass_list.append(t_mass)
            
            # 2. Modality Sink Attention
            attn_av = curr_token_attn[:, :, av_sink].mean()
            attn_va = curr_token_attn[:, :, va_sink].mean()
            attn_aa = curr_token_attn[:, :, aa_sink].mean()
            attn_vv = curr_token_attn[:, :, vv_sink].mean()
            
            cross_sum += (attn_av + attn_va)
            uni_sum += (attn_aa + attn_vv)

        avg_text_mass = (sum(text_mass_list) / len(text_mass_list)).item()
        
        if isinstance(uni_sum, torch.Tensor):
            uni_sum = uni_sum.item()
        if isinstance(cross_sum, torch.Tensor):
            cross_sum = cross_sum.item()
            
        adaptive_weight = uni_sum / (cross_sum + uni_sum + 1e-8)
        return avg_text_mass, adaptive_weight

    def _determine_strength(self, attention_outputs,sink_data):
        """
        Momentum과 Sigmoid Decay를 사용하여 강도를 결정합니다.
        """
        text_mass, adaptive_weight = self._calculate_adaptive_metrics(attention_outputs,sink_data)
        print(f"🔍 Text Mass: {text_mass:.4f}, Adaptive Weight: {adaptive_weight:.4f}")
        if adaptive_weight < 0.6 or text_mass > 0.5:
                target_strength = 0.0
        else:
                target_strength = self.base_guidance_strength * adaptive_weight
        # 1. Base Calculation
        current_strength = (self.momentum * self.prev_strength) + ((1 - self.momentum) * target_strength)
        self.prev_strength = current_strength
        return current_strength
    def __call__(self, input_ids, logits, sink_data=None, model_input=None):
        # [Step 1] Initialization
        masking_token = sink_data #.get("masking_token", None) if sink_data else self.masking_token
        if self.out is None:
            safe_input = model_input.copy()
            safe_input.pop("masking_token", None)
            safe_input.pop("past_key_values", None) # 여기서 제거
            safe_input.pop("output_attentions", None)
            safe_input.pop("use_cache", None)
            out_orig_init = self.model(**safe_input, 
                                       
                                       use_cache=True)
            self.pkv_orig = out_orig_init.past_key_values
            final_strength = 0.0 #self._determine_strength(out_orig_init.attentions)
            
            self.out = self.model(**safe_input, 
                                  
                                  masking_token=masking_token, 
                                  use_cache=True)
            self.pkv_masked = self.out.past_key_values
            
        # [Step 2] Generation
        else:
            
            self.out_original = self.model(input_ids[:, -1:],
                                           use_audio_in_video=True,
                                           use_cache=True,
                                           output_attentions=True,
                                           past_key_values=self.pkv_orig,
                                           masking_token=None)
            self.pkv_orig = self.out_original.past_key_values
            final_strength = self._determine_strength(self.out_original.attentions,sink_data)
            print("🌀 Current Guidance Strength:", final_strength)
            self.out = self.model(input_ids[:, -1:],
                                  use_audio_in_video=True,
                                  use_cache=True,
                                  past_key_values=self.pkv_masked,
                                  masking_token=masking_token)
            self.pkv_masked = self.out.past_key_values

        # [Step 3] Apply Guidance
        if len(self.out.logits.shape) == 2:
             guidance_logits = self.out.logits
        else:
            guidance_logits = self.out.logits[:, -1, :]
        
        logits_log = F.log_softmax(logits, dim=-1)
        guidance_log = F.log_softmax(guidance_logits, dim=-1)
        
        # [NEW] Logit Difference Clamping
        # 두 모델의 격차가 너무 크면(-100, +100 등) Strength가 낮아도 이상한 값이 튈 수 있음
        # 차이를 -10 ~ +10 정도로 제한하여 안정성 확보
        diff = guidance_log - logits_log
        # diff = torch.clamp(diff, min=-10.0, max=10.0) # 필요시 주석 해제하여 사용
        
        # 수식: Original + Strength * (Masked - Original)
        # diff = Masked - Original 이므로 아래와 같음
        out = logits_log + final_strength * diff
        
        return out