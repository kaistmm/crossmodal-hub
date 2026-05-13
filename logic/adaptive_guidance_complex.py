import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
import math

class GuidanceLogits(LogitsProcessor):
    def __init__(self, guidance_strength, inputs, model, tokenizer=None, 
                 video_idx=None, audio_idx=None, masking_token=None,
                 adaptive_threshold=0.6, text_mass_threshold=0.4,
                 target_layers=range(18, 26),
                 momentum=0.5 
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
        self.target_layers = list(target_layers)
        
        # [OPT] Pre-convert sink indices to GPU tensors once (reused every step)
        self._sink_tensors_cached = False
        self._av_sink_t = None
        self._va_sink_t = None
        self._aa_sink_t = None
        self._vv_sink_t = None

    def _ensure_sink_tensors(self, sink_data, device):
        """Cache sink indices as GPU tensors to avoid repeated list→tensor conversion."""
        if self._sink_tensors_cached:
            return
        av_sink = sink_data.get("av_sink", None)
        va_sink = sink_data.get("va_sink", None)
        aa_sink = sink_data.get("aa_sink", None)
        vv_sink = sink_data.get("vv_sink", None)
        self._av_sink_t = torch.as_tensor(av_sink, device=device, dtype=torch.long) if av_sink is not None else None
        self._va_sink_t = torch.as_tensor(va_sink, device=device, dtype=torch.long) if va_sink is not None else None
        self._aa_sink_t = torch.as_tensor(aa_sink, device=device, dtype=torch.long) if aa_sink is not None else None
        self._vv_sink_t = torch.as_tensor(vv_sink, device=device, dtype=torch.long) if vv_sink is not None else None
        self._sink_tensors_cached = True

    def _calculate_adaptive_metrics(self, attention_outputs, sink_data):
        """Vectorized version: stack target layers, compute metrics in one pass."""
        device = attention_outputs[self.target_layers[0]].device
        self._ensure_sink_tensors(sink_data, device)
        
        num_target = len(self.target_layers)
        
        last_attns = []
        for li in self.target_layers:
            last_attns.append(attention_outputs[li][:, :, -1, :])  # [B, H, K]
        stacked = torch.stack(last_attns, dim=0)  # [num_layers, B, H, K]
        
        # 1. Text Mass — vectorized across layers
        text_mass_per_layer = stacked[:, :, :, self.text_token_start:].sum(dim=-1).mean(dim=(1, 2))  # [num_layers]
        avg_text_mass = text_mass_per_layer.mean().item()
        
        # 2. Modality Sink Attention — vectorized across layers
        cross_sum = (
            stacked[:, :, :, self._av_sink_t].mean(dim=(1, 2, 3)).sum() +
            stacked[:, :, :, self._va_sink_t].mean(dim=(1, 2, 3)).sum()
        ).item()
        uni_sum = (
            stacked[:, :, :, self._aa_sink_t].mean(dim=(1, 2, 3)).sum() +
            stacked[:, :, :, self._vv_sink_t].mean(dim=(1, 2, 3)).sum()
        ).item()
        
        # Free stacked tensor immediately
        del stacked, last_attns
        
        adaptive_weight = cross_sum / (cross_sum + uni_sum + 1e-8)
        return avg_text_mass, adaptive_weight

    def _determine_strength(self, attention_outputs, sink_data):
        """
        Momentum과 Sigmoid Decay를 사용하여 강도를 결정합니다.
        """
        text_mass, adaptive_weight = self._calculate_adaptive_metrics(attention_outputs, sink_data)
        
        # 1. Base Calculation
        target_strength = self.base_guidance_strength * adaptive_weight
        target_strength = min(0.6, target_strength) # Max Clamping

        # 2. [UPGRADE] Text Mass Decay using Cosine/Sigmoid (더 부드러움)
        if text_mass > self.text_mass_threshold:
            overage = text_mass - self.text_mass_threshold
            decay_margin = 0.2
            ratio = min(1.0, overage / decay_margin)
            penalty = 0.5 * (1 + math.cos(ratio * math.pi))
            target_strength *= penalty

        # 3. Adaptive Low Threshold (Soft Gating)
        if adaptive_weight < self.adaptive_threshold:
            soft_factor = max(0.0, (adaptive_weight / self.adaptive_threshold) ** 2)
            target_strength *= soft_factor

        # 4. [NEW] Momentum (관성) 적용
        final_strength = (self.momentum * self.prev_strength) + ((1 - self.momentum) * target_strength)
        self.prev_strength = final_strength


        return final_strength

    def __call__(self, input_ids, logits, sink_data=None, model_input=None):
        # [Step 1] Initialization
        masking_token = sink_data
        if self.out is None:
            safe_input = model_input.copy()
            safe_input.pop("masking_token", None)
            safe_input.pop("past_key_values", None)
            safe_input.pop("output_attentions", None)
            safe_input.pop("use_cache", None)
            
            # [OPT] No attention needed on init — strength is hardcoded to 0.0
            out_orig_init = self.model(**safe_input, 
                                       use_cache=True)
            self.pkv_orig = out_orig_init.past_key_values
            del out_orig_init  # [OPT] Free immediately
            final_strength = 0.0
            
            self.out = self.model(**safe_input, 
                                  masking_token=masking_token, 
                                  use_cache=True)
            self.pkv_masked = self.out.past_key_values
            
        # [Step 2] Generation
        else:
            out_original = self.model(input_ids[:, -1:],
                                       use_audio_in_video=True,
                                       use_cache=True,
                                       output_attentions=True,
                                       past_key_values=self.pkv_orig,
                                       masking_token=None)
            self.pkv_orig = out_original.past_key_values
            final_strength = self._determine_strength(out_original.attentions, sink_data)
            # [OPT] Free attention memory immediately after use
            del out_original
            
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
        
        diff = guidance_log - logits_log
        out = logits_log + final_strength * diff
        
        return out