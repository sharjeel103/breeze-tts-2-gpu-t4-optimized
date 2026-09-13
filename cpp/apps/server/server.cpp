#include "server.h"
#include "voices.h"
#include "ws.h"
#include "ws_api.h"

#include "breeze/audio.h"
#include "breeze/continuous_engine.h"
#include "breeze/generation.h"
#include "breeze/model.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#ifdef _WIN32
#include <windows.h> // after httplib, it pulls in winsock2 first
#endif

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

static std::string mmss(double s) {
    if (!(s >= 0) || s > 86399) s = 0;
    char buf[16];
    snprintf(buf, sizeof buf, "%02d:%02d", (int) s / 60, (int) s % 60);
    return buf;
}

static std::string bar(double frac, int width) {
    static const char * part[] = { " ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589" };
    if (!(frac > 0)) frac = 0;
    if (frac > 1) frac = 1;
    const double filled = frac * width;
    const int full = (int) filled;
    std::string s;
    for (int i = 0; i < full; i++) s += "\u2588";
    if (full < width) {
        s += part[(int) ((filled - full) * 8)];
        for (int i = full + 1; i < width; i++) s += " ";
    }
    return s;
}

int run_server(const ServerOptions & opts) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // otherwise the bar glyphs and any chinese text come out as mojibake
#endif
    BreezeModel model;
    printf("loading voice store model %s ...\n", opts.model.c_str());
    if (!model.load(opts.model, opts.use_gpu, 0)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const int sr = model.cfg.sample_rate;
    MimiCodec codec;
    codec.init(model);

    std::vector<std::unique_ptr<ContinuousEngine>> engines;
    engines.push_back(std::make_unique<ContinuousEngine>(0, opts.max_slots));
    if (!engines[0]->load_model(opts.model, opts.overload_model)) {
        fprintf(stderr, "failed to load engine 0\n");
        return 1;
    }
    engines[0]->start();

    if (opts.dual_gpu) {
        printf("Initializing Station B on GPU 1 (CUDA1) ...\n");
        auto eng1 = std::make_unique<ContinuousEngine>(1, opts.max_slots);
        if (eng1->load_model(opts.model, opts.overload_model)) {
            eng1->start();
            engines.push_back(std::move(eng1));
            printf("Station B on GPU 1 online and active!\n");
        } else {
            fprintf(stderr, "Warning: failed to load Station B on GPU 1, running single GPU\n");
        }
    }

    httplib::Server svr;
    auto mutex = std::make_shared<std::mutex>();

    VoiceStore store;
    store.load_dir(opts.voices_dir, model.cfg.num_codebooks);
    store.add_routes(svr, model, codec, *mutex, opts.voices_dir);

    WsServer ws;
    const int ws_port = opts.ws_port == 0 ? opts.port + 1 : opts.ws_port;
    if (ws_port > 0) {
        const bool up = ws.start(opts.host, ws_port, [&](WsConn & c) {
            ws_connection(c, model, codec, store, *mutex, opts.chunk_first, opts.chunk_max, opts.split_chars);
        });
        if (up) printf("websocket on ws://%s:%d\n", opts.host.c_str(), ws_port);
        else fprintf(stderr, "could not open the websocket port %d\n", ws_port);
    }

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        int active = 0, queued = 0;
        for (auto & e : engines) {
            active += e->active_slots_count();
            queued += e->queue_depth();
        }
        res.set_content("{\"status\":\"ok\",\"sample_rate\":" + std::to_string(sr) +
                        ",\"gpus\":" + std::to_string(engines.size()) +
                        ",\"active_slots\":" + std::to_string(active) +
                        ",\"queued_requests\":" + std::to_string(queued) +
                        ",\"ws_port\":" + std::to_string(ws_port > 0 ? ws_port : 0) + "}",
                        "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        GenRequest g;
        g.text = field(req, "text", "");
        g.instruction = field(req, "instruction", "Speak clearly and naturally.");
        g.ref_text = field(req, "ref_text", "");
        g.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        g.seed = atoi(field(req, "seed", "42").c_str());
        g.temperature = (float) atof(field(req, "temperature", "0").c_str());
        g.top_k = atoi(field(req, "top_k", "0").c_str());
        g.top_p = (float) atof(field(req, "top_p", "0").c_str());
        g.repetition_penalty = (float) atof(field(req, "repetition_penalty", "0").c_str());
        g.max_new_tokens = atoi(field(req, "max_new_tokens", "0").c_str());
        g.split_chars = atoi(field(req, "split_chars", std::to_string(opts.split_chars)).c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty())
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
        }
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");

        // Select engine with lower load
        ContinuousEngine * eng = engines[0].get();
        if (engines.size() > 1) {
            int load0 = engines[0]->active_slots_count() + engines[0]->queue_depth();
            int load1 = engines[1]->active_slots_count() + engines[1]->queue_depth();
            eng = (load0 <= load1) ? engines[0].get() : engines[1].get();
        }

        printf("[Route] Dispatching request (%d chars) to GPU %d (load: %d active, %d queued)\n",
               (int) g.text.size(), eng->device_id(), eng->active_slots_count(), eng->queue_depth());
        fflush(stdout);

        struct StreamQueue {
            std::mutex mtx;
            std::condition_variable cv;
            std::queue<std::vector<uint8_t>> chunks;
            bool finished = false;
        };
        auto sq = std::make_shared<StreamQueue>();

        eng->enqueue(g, [sq](const float * s, int n) -> bool {
            std::lock_guard<std::mutex> lock(sq->mtx);
            if (s == nullptr || n == 0) {
                sq->finished = true;
                sq->cv.notify_one();
                return true;
            }
            std::vector<uint8_t> pcm = to_pcm16(s, n);
            sq->chunks.push(std::move(pcm));
            sq->cv.notify_one();
            return true;
        });

        res.set_chunked_content_provider(
            "audio/pcm",
            [sq](size_t, httplib::DataSink & sink) -> bool {
                while (true) {
                    std::vector<uint8_t> chunk;
                    {
                        std::unique_lock<std::mutex> lock(sq->mtx);
                        while (sq->chunks.empty() && !sq->finished) {
                            sq->cv.wait(lock);
                        }
                        if (!sq->chunks.empty()) {
                            chunk = std::move(sq->chunks.front());
                            sq->chunks.pop();
                        } else if (sq->finished) {
                            sink.done();
                            return true;
                        }
                    }
                    if (!chunk.empty()) {
                        if (!sink.write((const char *) chunk.data(), chunk.size())) {
                            return false; // client disconnected
                        }
                    }
                }
            });
    });

    svr.Post("/v1/audio/convert", [&, mutex](const httplib::Request & req, httplib::Response & res) {
        std::unique_lock<std::mutex> lock(*mutex, std::try_to_lock);
        if (!lock) {
            res.status = 409;
            res.set_content("{\"error\":\"busy\"}", "application/json");
            return;
        }
        std::vector<float> src, ref;
        const auto load = [&](const char * name, std::vector<float> & out) {
            if (!req.has_file(name)) return false;
            const auto & f = req.get_file_value(name);
            return !f.content.empty() &&
                   read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, out);
        };
        if (!load("source", src)) {
            res.status = 400;
            res.set_content("{\"error\":\"a source wav file is required\"}", "application/json");
            return;
        }
        std::string ref_text = field(req, "ref_text", "");
        std::vector<int> vcodes;
        int vframes = 0;
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty()) {
            if (!store.take(vid, vcodes, vframes, ref_text)) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
                return;
            }
        } else if (!load("ref_audio", ref)) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio or voice_id is required\"}", "application/json");
            return;
        }
        if (ref_text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_text is required\"}", "application/json");
            return;
        }

        printf("conv %.2f s source, %s\n", (double) src.size() / sr,
               vframes > 0 ? "cached reference" : "uploaded reference");
        fflush(stdout);
        try {
            int T = 0;
            std::vector<int> codes = codec.encode(src, T);
            ConvertOptions copt;
            copt.src_text = field(req, "text", "");
            copt.temperature = (float) atof(field(req, "temperature", "0.3").c_str());
            copt.top_k = atoi(field(req, "top_k", "1").c_str());
            copt.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
            copt.keep_acoustic = atoi(field(req, "keep_acoustic", "0").c_str());
            copt.seed = atoi(field(req, "seed", "42").c_str());
            copt.ref_codes = vcodes;
            copt.ref_frames = vframes;
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> audio = convert_voice(model, codec, codes, T, ref, ref_text, copt);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double secs = (double) audio.size() / sr;
            printf("     %.2f s in %.2f s, %.2fx rt\n", secs, wall, wall > 0 ? secs / wall : 0.0);
            fflush(stdout);
            std::vector<uint8_t> pcm = to_pcm16(audio.data(), (int) audio.size());
            res.set_header("X-Sample-Rate", std::to_string(sr));
            res.set_header("X-Sample-Format", "s16le");
            res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
        } catch (const std::exception & e) {
            fprintf(stderr, "conversion error: %s\n", e.what());
            res.status = 500;
            res.set_content("{\"error\":\"conversion failed\"}", "application/json");
        }
    });

    if (opts.webui) {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::index_html, "text/html");
        });
        svr.Get("/style.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::style_css, "text/css");
        });
        svr.Get("/app.js", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::app_js, "application/javascript");
        });
        printf("web ui: http://%s:%d/\n", opts.host.c_str(), opts.port);
    }

    printf("listening on http://%s:%d\n", opts.host.c_str(), opts.port);
    if (!svr.listen(opts.host, opts.port)) {
        fprintf(stderr, "failed to bind %s:%d\n", opts.host.c_str(), opts.port);
        return 1;
    }
    model.free();
    return 0;
}

}
