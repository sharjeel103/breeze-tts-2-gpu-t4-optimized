import sys
import time
import traceback
from pathlib import Path
import torch

# Ensure breeze-tts is in path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from engine.config import EngineConfig
from engine.assembly_pipeline import AssemblyPipelineEngine

@torch.inference_mode()
def main():
    print("=== STARTING ADMISSION AND STEP DIAGNOSTIC ===", flush=True)
    cfg = EngineConfig(num_slots=2, chunk_size=5)
    engine = AssemblyPipelineEngine(cfg)
    
    print("[1] Initializing Engine & Capturing Graphs...", flush=True)
    engine.initialize()
    print("[1] Engine Initialized Successfully!", flush=True)

    print("[2] Running Prefill for Client 1...", flush=True)
    t0 = time.time()
    req = engine.prefill_worker.prefill_sync(
        request_id="Client_1_Short",
        text="Order confirmed, thank you.",
        instruction="Speak cleanly and politely.",
        guidance_scale=4.0,
    )
    print(f"[2] Prefill Complete in {(time.time()-t0)*1000:.1f}ms! prefill_len={req.prefill_len}", flush=True)

    print("[3] Testing Station Admission...", flush=True)
    st = engine.stations[0]
    
    print("  -> Setting guidance scale on backbone...", flush=True)
    st.backbone_graph.guidance_scale.fill_(req.effective_cfg)
    
    print("  -> Setting guidance scale on depth...", flush=True)
    st.depth_graph.set_guidance_scale(req.effective_cfg)
    
    print("  -> Hot-injecting KV cache into StaticCache...", flush=True)
    try:
        seq_len = st.backbone_graph.prefill_kv(req.past_key_values)
        print(f"  -> prefill_kv returned seq_len={seq_len}", flush=True)
    except Exception as e:
        print(f"  -> ERROR in prefill_kv: {e}", flush=True)
        traceback.print_exc()
        return

    print("  -> Setting generation state (position_ids, pad_lens)...", flush=True)
    try:
        st.backbone_graph.set_generation_state(req.branch_mask)
        print("  -> set_generation_state succeeded!", flush=True)
    except Exception as e:
        print(f"  -> ERROR in set_generation_state: {e}", flush=True)
        traceback.print_exc()
        return

    print("  -> Copying hidden and token buffers...", flush=True)
    st.hidden_dev0.copy_(req.initial_hidden[:, -1:, :])
    st.token_dev0.copy_(req.initial_token.view(-1).repeat(2))
    st.hidden_dev1.copy_(st.hidden_dev0)
    st.token_dev1.copy_(st.token_dev0)
    print("  -> Buffer copies complete!", flush=True)

    st.is_active = True
    st.bb_step = 0
    st.depth_step = 0
    st.max_frames = 25
    st.effective_cfg = req.effective_cfg
    st.request_id = req.request_id
    st.prompt_text = req.prompt_text
    
    print("[4] Testing step_assembly() for 5 cycles...", flush=True)
    for cycle in range(5):
        t_c = time.time()
        active = engine.step_assembly()
        dt = (time.time() - t_c) * 1000.0
        print(f"  Cycle {cycle}: active={active} | depth_step={st.depth_step} | bb_step={st.bb_step} | dt={dt:.2f}ms", flush=True)
        
    print("=== DIAGNOSTIC PASSED COMPLETELY! ===", flush=True)

if __name__ == "__main__":
    main()
