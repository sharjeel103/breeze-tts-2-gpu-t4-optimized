"""
Streaming Chunked Neural Vocoder for Sub-350ms Time-To-First-Audio (TTFA).
Executes on a dedicated CUDA stream on GPU 1, converting 16-codebook discrete frames
into 24,000 Hz 16-bit PCM audio chunks asynchronously.
"""
import asyncio
import time
from typing import Any, List, Optional
import numpy as np
import torch
import soundfile as sf
import io

class StreamingVocoderWorker:
    def __init__(
        self,
        audio_tokenizer: Any,
        device: str = "cuda:1",
        sample_rate: int = 24000,
        chunk_size: int = 5,  # 5 frames = 400ms
    ):
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Dedicated hardware stream on GPU 1
        self.stream = torch.cuda.Stream(device=self.dev1 if hasattr(self, 'dev1') else self.device)

    def warmup(self):
        """Warm up cuDNN and PyTorch JIT for the audio tokenizer decode."""
        print("[StreamingVocoder] Warming up neural vocoder on GPU 1...")
        for sz in [1, 2, self.chunk_size]:
            dummy_frames = [torch.zeros(16, dtype=torch.long, device=self.device) for _ in range(sz)]
            self.decode_chunk_sync(dummy_frames)
        torch.cuda.synchronize(self.device)
        print("[StreamingVocoder] Neural vocoder fully warmed up for all chunk sizes!")

    def decode_chunk_sync(self, frames: List[torch.Tensor]) -> bytes:
        """
        Decodes a list of 16-codebook frame tensors into 16-bit PCM WAV audio bytes.
        """
        if not frames:
            return b""
            
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            # Stack into [num_frames, 16]
            audio_codes_batch = torch.stack(frames, dim=0).to(self.device, non_blocking=True)
            codec_output = self.audio_tokenizer.decode({'audio_codes': [audio_codes_batch]})
            
            if hasattr(codec_output, 'audio_values'):
                audio_tensor = codec_output.audio_values
            else:
                audio_tensor = codec_output
                
            while isinstance(audio_tensor, (list, tuple)):
                audio_tensor = audio_tensor[0]
                
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.squeeze().detach().cpu().float().numpy()
            elif isinstance(audio_tensor, np.ndarray):
                audio_np = audio_tensor.squeeze()
            else:
                audio_np = np.zeros(1, dtype=np.float32)
                
            # Convert float32 [-1, 1] to 16-bit PCM WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, samplerate=self.sample_rate, format='WAV', subtype='PCM_16')
            return buffer.getvalue()

    async def emit_chunk(
        self,
        queue: asyncio.Queue,
        frames: List[torch.Tensor],
        is_final: bool = False,
    ):
        """
        Asynchronously runs decode_chunk_sync in threadpool and pushes to queue.
        """
        loop = asyncio.get_running_loop()
        wav_bytes = await loop.run_in_executor(None, self.decode_chunk_sync, frames)
        await queue.put({
            "type": "audio_chunk",
            "data": wav_bytes,
            "num_frames": len(frames),
            "is_final": is_final,
        })
        if is_final:
            await queue.put({"type": "eos"})
