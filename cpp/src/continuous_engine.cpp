#include "breeze/continuous_engine.h"
#include "breeze/sampling.h"
#include "breeze/text_encoder.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace breeze {

struct EngineSeg {
    bool is_text = false;
    std::vector<int> tokens;
    std::vector<int> codes;
    int n_frames = 0;
    bool eos = false;
};

static EngineSeg eng_text_seg(BreezeModel & m, const std::string & s) {
    EngineSeg seg;
    seg.is_text = true;
    seg.tokens = m.tok.encode(s, true);
    return seg;
}

static std::vector<EngineSeg> eng_build_segments(BreezeModel & m, const GenRequest & r, const std::string & text,
                                                bool has_ref, const std::string & ref_text,
                                                const std::vector<int> & ref_codes, int ref_T, bool cond) {
    std::vector<EngineSeg> segs;
    const std::string spk = "[S0]";
    if (has_ref) {
        segs.push_back(eng_text_seg(m, spk + ref_text));
        EngineSeg a;
        a.is_text = false;
        a.codes = ref_codes;
        a.n_frames = ref_T;
        a.eos = true;
        segs.push_back(a);
    }
    std::string tail = cond ? spk + "<ins_bos>" + r.instruction + "<ins_eos>" + text : spk + text;
    segs.push_back(eng_text_seg(m, tail));
    return segs;
}

static std::vector<float> eng_assemble(BreezeModel & m, const std::vector<EngineSeg> & segs, int & total) {
    std::vector<float> out;
    total = 0;
    for (const EngineSeg & s : segs) {
        if (s.is_text) {
            std::vector<float> e = text_encoder_forward(m, s.tokens);
            out.insert(out.end(), e.begin(), e.end());
            total += (int) s.tokens.size();
        } else {
            std::vector<float> e = audio_embed_forward(m, s.codes, s.n_frames);
            out.insert(out.end(), e.begin(), e.end());
            total += s.n_frames;
            std::vector<int> eos_frame(m.cfg.num_codebooks, m.cfg.codebook_eos_token_id);
            std::vector<float> ee = audio_embed_forward(m, eos_frame, 1);
            out.insert(out.end(), ee.begin(), ee.end());
            total += 1;
        }
    }
    return out;
}

static std::vector<float> eng_combine_logits(const std::vector<float> & cond, const std::vector<float> & unc,
                                             bool use_cfg, float scale) {
    if (!use_cfg) return cond;
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = unc[i] + scale * (cond[i] - unc[i]);
    return out;
}

void EngineSlot::cleanup() {
    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();
    frames.clear();
    hist.clear();
    suppress.clear();
    active = false;
    cb0 = -1;
    step = 0;
    emitted = 0;
}

ContinuousEngine::ContinuousEngine(int device_id, int max_slots)
    : m_device_id(device_id), m_max_slots(max_slots) {
    m_slot_pool.resize(m_max_slots);
    for (int i = 0; i < m_max_slots; i++) {
        m_slot_pool[i].id = i;
        m_slot_pool[i].active = false;
    }
}

ContinuousEngine::~ContinuousEngine() {
    stop();
    for (auto & s : m_slot_pool) {
        s.cleanup();
    }
    m_model.free();
    if (m_has_overload) {
        m_model_overload.free();
    }
}

bool ContinuousEngine::load_model(const std::string & model_path, const std::string & overload_model_path) {
    printf("[Engine %d] Loading baseline model %s on GPU %d ...\n", m_device_id, model_path.c_str(), m_device_id);
    if (!m_model.load(model_path, true, m_device_id)) {
        fprintf(stderr, "[Engine %d] Failed to load baseline model on device %d\n", m_device_id, m_device_id);
        return false;
    }
    m_codec.init(m_model);
    printf("[Engine %d] Baseline model loaded. Backend: %s\n", m_device_id, m_model.backend.name());

    if (!overload_model_path.empty()) {
        printf("[Engine %d] Loading overload model %s on GPU %d ...\n", m_device_id, overload_model_path.c_str(), m_device_id);
        if (m_model_overload.load(overload_model_path, true, m_device_id)) {
            m_codec_overload.init(m_model_overload);
            m_has_overload = true;
            printf("[Engine %d] Overload model loaded successfully.\n", m_device_id);
        } else {
            fprintf(stderr, "[Engine %d] Warning: failed to load overload model, running baseline only.\n", m_device_id);
        }
    }
    return true;
}

void ContinuousEngine::enqueue(const GenRequest & req, const AudioCallback & cb) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.emplace(req, cb);
    }
    m_cv.notify_one();
}

void ContinuousEngine::start() {
    if (m_running.exchange(true)) return;
    m_worker = std::thread(&ContinuousEngine::step_loop, this);
    printf("[Engine %d] Continuous engine started with %d slots max.\n", m_device_id, m_max_slots);
}

void ContinuousEngine::stop() {
    if (!m_running.exchange(false)) return;
    m_cv.notify_all();
    if (m_worker.joinable()) {
        m_worker.join();
    }
    printf("[Engine %d] Continuous engine stopped.\n", m_device_id);
}

int ContinuousEngine::active_slots_count() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    int count = 0;
    for (const auto & s : m_slot_pool) {
        if (s.active) count++;
    }
    return count;
}

int ContinuousEngine::queue_depth() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (int) m_queue.size();
}

bool ContinuousEngine::init_slot(EngineSlot & slot, const GenRequest & req, const AudioCallback & cb) {
    slot.req = req;
    slot.cb = cb;
    slot.start_time = std::chrono::steady_clock::now();
    slot.step = 0;
    slot.emitted = 0;
    slot.frames.clear();
    slot.hist.clear();
    slot.suppress.clear();

    const bool prefer_overload = m_has_overload && (queue_depth() > 3);
    slot.using_overload = prefer_overload;
    BreezeModel & model = slot.using_overload ? m_model_overload : m_model;
    MimiCodec & codec = slot.using_overload ? m_codec_overload : m_codec;

    slot.rng.seed(req.seed != 0 ? req.seed : 42);
    slot.use_cfg = (req.cfg_scale != 1.0f);
    slot.max_new = req.max_new_tokens > 0 ? req.max_new_tokens : model.cfg.max_new_tokens;
    slot.chunk_max = std::max(1, req.chunk_max);
    slot.chunk = std::min(std::max(1, req.chunk_first), slot.chunk_max);

    slot.bp.temperature = req.temperature > 0.0f ? req.temperature : model.cfg.temperature;
    slot.bp.top_k = req.top_k > 0 ? req.top_k : model.cfg.top_k;
    slot.bp.top_p = req.top_p > 0.0f ? req.top_p : model.cfg.top_p;
    slot.bp.repetition_penalty = req.repetition_penalty > 0.0f ? req.repetition_penalty : model.cfg.repetition_penalty;

    for (int t = model.cfg.codec_codebook_size; t < model.cfg.audio_vocab_size; t++) {
        slot.suppress.push_back(t);
    }

    std::vector<int> ref_codes = req.ref_codes;
    int ref_T = req.ref_frames;
    if (ref_codes.empty() && !req.ref_audio.empty()) {
        ref_codes = codec.encode(req.ref_audio, ref_T);
    }
    const bool has_ref = !ref_codes.empty() && !req.ref_text.empty();

    int total_c = 0, total_u = 0;
    std::vector<float> emb_c = eng_assemble(model, eng_build_segments(model, req, req.text, has_ref, req.ref_text, ref_codes, ref_T, true), total_c);
    std::vector<float> emb_u;
    if (slot.use_cfg) {
        emb_u = eng_assemble(model, eng_build_segments(model, req, req.text, has_ref, req.ref_text, ref_codes, ref_T, false), total_u);
    }

    slot.st_c.init(model, total_c + slot.max_new + 8);
    if (slot.use_cfg) {
        slot.st_u.init(model, total_u + slot.max_new + 8);
    }

    slot.o_c = backbone_run(model, slot.st_c, emb_c, total_c);
    if (slot.use_cfg) {
        slot.o_u = backbone_run(model, slot.st_u, emb_u, total_u);
    }

    slot.depth.init(model, slot.use_cfg ? 2 : 1);

    std::vector<float> comb = eng_combine_logits(slot.o_c.logits, slot.o_u.logits, slot.use_cfg, req.cfg_scale);
    slot.cb0 = sample_token(comb, slot.bp, slot.rng, &slot.hist, &slot.suppress);
    slot.active = true;
    return true;
}

bool ContinuousEngine::flush_slot_audio(EngineSlot & slot, bool final_flush) {
    BreezeModel & model = slot.using_overload ? m_model_overload : m_model;
    MimiCodec & codec = slot.using_overload ? m_codec_overload : m_codec;
    const int nc = model.cfg.num_codebooks;
    const int spf = model.cfg.samples_per_frame;
    const int ctx = model.cfg.voc.sliding_window + 16;

    const int have = (int) slot.frames.size() / nc;
    while (have - slot.emitted >= slot.chunk || (final_flush && have > slot.emitted)) {
        const int start = slot.emitted;
        const int count = final_flush ? have - slot.emitted : slot.chunk;
        const int ctx_start = start > ctx ? start - ctx : 0;
        const int sub_T = start + count - ctx_start;
        std::vector<int> sub(slot.frames.begin() + (size_t) ctx_start * nc,
                             slot.frames.begin() + (size_t) (start + count) * nc);
        std::vector<float> audio = codec.decode(sub, sub_T);
        const int skip = (start - ctx_start) * spf;

        if (!slot.cb(audio.data() + skip, count * spf)) return false;
        slot.emitted += count;
        slot.chunk = std::min(slot.chunk + slot.chunk / 3 + 1, slot.chunk_max);
        if (!final_flush && have - slot.emitted < slot.chunk) break;
    }
    return true;
}

void ContinuousEngine::step_loop() {
    while (m_running.load()) {
        std::vector<EngineSlot*> active;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            // Check if all slots are idle and queue is empty
            bool has_active = false;
            for (auto & s : m_slot_pool) {
                if (s.active) { has_active = true; break; }
            }

            while (m_running.load() && !has_active && m_queue.empty()) {
                m_cv.wait(lock);
                for (auto & s : m_slot_pool) {
                    if (s.active) { has_active = true; break; }
                }
            }
            if (!m_running.load()) break;

            // Admit waiting requests into free slots
            while (!m_queue.empty()) {
                EngineSlot * free_slot = nullptr;
                for (auto & s : m_slot_pool) {
                    if (!s.active) { free_slot = &s; break; }
                }
                if (!free_slot) break; // All slots full

                auto item = m_queue.front();
                m_queue.pop();
                init_slot(*free_slot, item.first, item.second);
            }

            // Gather active slots for this step
            for (auto & s : m_slot_pool) {
                if (s.active) active.push_back(&s);
            }
        }

        if (active.empty()) continue;

        // Execute one 80ms frame step for each active slot
        for (EngineSlot * s : active) {
            BreezeModel & model = s->using_overload ? m_model_overload : m_model;
            std::vector<std::vector<float>> hiddens = { s->o_c.hidden };
            if (s->use_cfg) hiddens.push_back(s->o_u.hidden);

            std::vector<int> depth_codes = s->depth.run(model, hiddens, s->cb0, s->req.cfg_scale, s->rng);
            std::vector<int> frame = { s->cb0 };
            frame.insert(frame.end(), depth_codes.begin(), depth_codes.end());

            bool pad = true;
            for (int c : frame) {
                if (c != model.cfg.codebook_pad_token_id) { pad = false; break; }
            }
            if (!pad) {
                s->frames.insert(s->frames.end(), frame.begin(), frame.end());
                flush_slot_audio(*s, false);
            }
            s->hist.push_back(s->cb0);

            // Forward backbone
            std::vector<float> ae = audio_embed_forward(model, frame, 1);
            s->o_c = backbone_run(model, s->st_c, ae, 1);
            if (s->use_cfg) s->o_u = backbone_run(model, s->st_u, ae, 1);

            std::vector<float> comb = eng_combine_logits(s->o_c.logits, s->o_u.logits, s->use_cfg, s->req.cfg_scale);
            s->cb0 = sample_token(comb, s->bp, s->rng, &s->hist, &s->suppress);
            s->step++;

            // Check EOS or max tokens
            if (s->cb0 == model.cfg.backbone_eos_token_id || s->step >= s->max_new) {
                flush_slot_audio(*s, true);
                s->cb(nullptr, 0); // signal stream completion
                std::lock_guard<std::mutex> lock(m_mutex);
                s->cleanup();
            }
        }
    }
}

}
