#pragma once

#include "breeze/model.h"
#include "breeze/codec.h"
#include "breeze/generation.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace breeze {

struct EngineSlot {
    int id = -1;
    bool active = false;
    GenRequest req;
    AudioCallback cb;

    BackboneState st_c, st_u;
    DepthRunner depth;
    StepOut o_c, o_u;

    int cb0 = -1;
    std::vector<int> frames;
    std::vector<int> hist;
    std::vector<int> suppress;

    int emitted = 0;
    int step = 0;
    int max_new = 0;
    bool use_cfg = false;
    int chunk = 4;
    int chunk_max = 25;

    std::mt19937 rng;
    SampleParams bp;
    bool using_overload = false;
    std::chrono::steady_clock::time_point start_time;

    void cleanup();
};

class ContinuousEngine {
public:
    ContinuousEngine(int device_id = 0, int max_slots = 16);
    ~ContinuousEngine();

    bool load_model(const std::string & model_path, const std::string & overload_model_path = "");
    void enqueue(const GenRequest & req, const AudioCallback & cb);
    void start();
    void stop();

    int active_slots_count() const;
    int queue_depth() const;
    int device_id() const { return m_device_id; }
    bool is_running() const { return m_running.load(); }

    BreezeModel & model() { return m_model; }
    MimiCodec & codec() { return m_codec; }

private:
    int m_device_id;
    int m_max_slots;
    BreezeModel m_model;
    MimiCodec m_codec;

    bool m_has_overload = false;
    BreezeModel m_model_overload;
    MimiCodec m_codec_overload;

    std::vector<EngineSlot> m_slot_pool;
    std::queue<std::pair<GenRequest, AudioCallback>> m_queue;

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::atomic<bool> m_running{false};
    std::thread m_worker;

    void step_loop();
    bool init_slot(EngineSlot & slot, const GenRequest & req, const AudioCallback & cb, bool prefer_overload = false);
    bool flush_slot_audio(EngineSlot & slot, bool final_flush);
};

}
