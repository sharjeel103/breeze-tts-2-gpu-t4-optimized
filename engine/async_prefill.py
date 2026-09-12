"""
Decoupled Asynchronous Text Prefill Engine.
Executes text tokenization, template formatting, and the 600M Gemma-2 Text Encoder
on a background CUDA stream on GPU 0, eliminating prefill stutter for active streams.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import _left_pad_tensor

@dataclass
class PrefilledRequest:
    request_id: str
    prompt_text: str
    prefill_len: int
    max_frames: int
    effective_cfg: float
    is_dual_branch: bool
    audio_queue: asyncio.Queue
    
    # Prefilled state tensors
    initial_hidden: torch.Tensor
    initial_token: torch.Tensor
    branch_mask: torch.Tensor
    past_key_values: Any

class AsyncPrefillWorker:
    def __init__(self, model: Any, dev0: str = "cuda:0"):
        self.model = model
        self.dev0 = dev0
        self.stream = torch.cuda.Stream(device=self.dev0)
        self.ready_queue: asyncio.Queue = asyncio.Queue()
        self._reserved_tokens = [0, 1, 2, 3]

    def prefill_sync(
        self,
        request_id: str,
        text: str,
        instruction: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        speaker: str = "S0",
        guidance_scale: float = 4.0,
        max_frames: int = 1024,
        audio_queue: Optional[asyncio.Queue] = None,
    ) -> PrefilledRequest:
        t0 = time.time()
        
        request = {
            "text": text,
            "instruction": instruction,
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "speaker": speaker,
        }
        template_name = select_template_name(request)
        template = get_template(template_name)
        
        # Template preparation
        inputs = prepare_inputs(
            template=template,
            request=request,
            tokenizer=self.model.tokenizer,
            audio_tokenizer=self.model.audio_tokenizer,
            model_config=self.model.config,
        )
        
        with torch.cuda.device(self.dev0), torch.cuda.stream(self.stream):
            has_negative = 'cfg_negative_prompt_ids' in inputs
            effective_cfg = guidance_scale if has_negative else 1.0
            
            if has_negative:
                cond_ids = inputs['input_ids'].to(self.dev0)
                uncond_ids = inputs['cfg_negative_prompt_ids'].to(self.dev0)
                max_len = max(cond_ids.shape[1], uncond_ids.shape[1])
                
                input_ids = torch.cat([
                    _left_pad_tensor(cond_ids, max_len, 0),
                    _left_pad_tensor(uncond_ids, max_len, 0),
                ], dim=0)
                attention_mask = torch.cat([
                    _left_pad_tensor(inputs['attention_mask'].to(self.dev0), max_len, 0),
                    _left_pad_tensor(inputs['cfg_negative_prompt_attention_mask'].to(self.dev0), max_len, 0),
                ], dim=0)
                text_ids_mask = torch.cat([
                    _left_pad_tensor(inputs['text_ids_mask'].to(self.dev0), max_len, False),
                    _left_pad_tensor(inputs['cfg_negative_text_ids_mask'].to(self.dev0), max_len, False),
                ], dim=0)
                text_ids_len = torch.cat([
                    inputs['text_ids_len'].to(self.dev0),
                    inputs['cfg_negative_text_ids_len'].to(self.dev0),
                ], dim=0)
                
                cond_values = inputs.get('input_values')
                uncond_values = inputs.get('cfg_negative_input_values')
                if cond_values is not None and uncond_values is not None:
                    input_values = torch.cat([cond_values.to(self.dev0), uncond_values.to(self.dev0)], dim=0)
                elif cond_values is not None:
                    input_values = cond_values.to(self.dev0).repeat(2, 1, 1)
                else:
                    input_values = None
            else:
                cond_ids = inputs['input_ids'].to(self.dev0)
                input_ids = cond_ids.repeat(2, 1)
                attention_mask = inputs['attention_mask'].to(self.dev0).repeat(2, 1)
                text_ids_mask = inputs['text_ids_mask'].to(self.dev0).repeat(2, 1)
                text_ids_len = inputs['text_ids_len'].to(self.dev0).repeat(2)
                if inputs.get('input_values') is not None:
                    input_values = inputs['input_values'].to(self.dev0).repeat(2, 1, 1)
                else:
                    input_values = None
                    
            merged = self.model._merge_input_ids_with_input_values(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_ids_mask=text_ids_mask,
                text_ids_len=text_ids_len,
                input_values=input_values,
            )
            branch_embeds = merged['inputs_embeds'].contiguous()
            branch_mask = attention_mask.contiguous()
            
            position_ids = branch_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(branch_mask == 0, 1)
            
            # Execute text prefill forward pass on dev0
            backbone_out = self.model.backbone_model(
                inputs_embeds=branch_embeds,
                attention_mask=branch_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            hidden = backbone_out.last_hidden_state
            logits = self.model.lm_head(hidden[:, -1, :].float()).float()
            
            # Apply initial CFG to get initial token
            cond_logits = logits[:1]
            uncond_logits = logits[1:]
            guided_logits = uncond_logits + effective_cfg * (cond_logits - uncond_logits)
            
            from models.cudagraph.sampling import sample_logits
            token = sample_logits(
                guided_logits,
                suppress_tokens=self._reserved_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
            ).view(1)
            
            prefill_len = int(branch_mask.shape[1])
            
        return PrefilledRequest(
            request_id=request_id,
            prompt_text=text,
            prefill_len=prefill_len,
            max_frames=max_frames,
            effective_cfg=effective_cfg,
            is_dual_branch=has_negative,
            audio_queue=audio_queue or asyncio.Queue(),
            initial_hidden=hidden.detach(),
            initial_token=token.detach(),
            branch_mask=branch_mask.detach(),
            past_key_values=backbone_out.past_key_values,
        )

    async def submit_prefill(self, **kwargs) -> PrefilledRequest:
        loop = asyncio.get_running_loop()
        prefilled = await loop.run_in_executor(None, lambda: self.prefill_sync(**kwargs))
        await self.ready_queue.put(prefilled)
        return prefilled
