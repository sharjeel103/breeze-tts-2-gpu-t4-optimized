"""
Zero-Allocation In-Flight Continuous Batching Slot Manager.
Manages N independent virtual streams with in-place masking, immediate early-out,
and hot admission at each 80ms frame boundary.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import torch

class SlotState(Enum):
    IDLE = 0
    ACTIVE = 1
    FINISHED = 2

@dataclass
class Slot:
    slot_id: int
    state: SlotState = SlotState.IDLE
    request_id: Optional[str] = None
    prompt_text: str = ""
    prefill_len: int = 0
    current_step: int = 0
    max_frames: int = 1024
    effective_cfg: float = 1.0
    is_dual_branch: bool = False
    
    # Audio collection
    chunk_buffer: List[torch.Tensor] = field(default_factory=list)
    total_frames_generated: int = 0
    t_admitted: float = 0.0
    t_first_audio: Optional[float] = None
    
    # Delivery channel
    audio_queue: Optional[asyncio.Queue] = None

class SlotManager:
    def __init__(self, num_slots: int = 4, dev0: str = "cuda:0", dev1: str = "cuda:1"):
        self.num_slots = num_slots
        self.dev0 = dev0
        self.dev1 = dev1
        
        self.slots: List[Slot] = [Slot(slot_id=i) for i in range(num_slots)]
        
        # Pinned static masks (Zero allocation in hot loop)
        self.active_mask = torch.zeros(num_slots, dtype=torch.bool, device=dev0)
        self.active_mask_dev1 = torch.zeros(num_slots, dtype=torch.bool, device=dev1)
        
        # Lock for thread-safe slot allocation from async prefill worker
        self._lock = asyncio.Lock()
        
    def get_num_active(self) -> int:
        return sum(1 for s in self.slots if s.state == SlotState.ACTIVE)
        
    def find_free_slot(self) -> Optional[int]:
        for s in self.slots:
            if s.state == SlotState.IDLE:
                return s.slot_id
        return None

    def admit_request(
        self,
        slot_id: int,
        request_id: str,
        prompt_text: str,
        prefill_len: int,
        max_frames: int,
        effective_cfg: float,
        is_dual_branch: bool,
        audio_queue: asyncio.Queue,
    ) -> Slot:
        slot = self.slots[slot_id]
        slot.state = SlotState.ACTIVE
        slot.request_id = request_id
        slot.prompt_text = prompt_text
        slot.prefill_len = prefill_len
        slot.current_step = 0
        slot.max_frames = max_frames
        slot.effective_cfg = effective_cfg
        slot.is_dual_branch = is_dual_branch
        slot.chunk_buffer.clear()
        slot.total_frames_generated = 0
        slot.t_admitted = time.time()
        slot.t_first_audio = None
        slot.audio_queue = audio_queue
        
        # Update static hardware masks in-place
        self.active_mask[slot_id] = True
        self.active_mask_dev1[slot_id] = True
        return slot

    def retire_slot(self, slot_id: int) -> Tuple[str, int, float]:
        """
        Retires a slot immediately upon EOS or max tokens.
        Zeroes out mask in-place without triggering graph recapture.
        """
        slot = self.slots[slot_id]
        req_id = slot.request_id or f"slot_{slot_id}"
        total_frames = slot.total_frames_generated
        dur = total_frames * 0.08
        
        slot.state = SlotState.IDLE
        slot.request_id = None
        self.active_mask[slot_id] = False
        self.active_mask_dev1[slot_id] = False
        
        return req_id, total_frames, dur
