#include "engine/cpu_prof.h"

#include "engine/log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace cpuprof {

namespace {
constexpr int kMax = 96;
Section     g_sections[kMax];
float       g_last_ms[kMax] = {};
std::atomic<int> g_count{0};
std::mutex  g_reg_mutex;
int         g_frames = 0;
uint64_t    g_total_frames = 0;
std::chrono::steady_clock::time_point g_last_end{};
}  // namespace

Section* section(const char* name) {
    std::lock_guard<std::mutex> lk(g_reg_mutex);
    const int n = g_count.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i)
        if (g_sections[i].name == name || std::strcmp(g_sections[i].name, name) == 0)
            return &g_sections[i];
    if (n >= kMax) return &g_sections[kMax - 1];
    g_sections[n].name = name;
    g_count.store(n + 1, std::memory_order_release);
    return &g_sections[n];
}

float last_ms(const char* name) {
    const int n = g_count.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        if (std::strcmp(g_sections[i].name, name) == 0) return g_last_ms[i];
    return 0.0f;
}

void end_frame() {
    const int n = g_count.load(std::memory_order_acquire);
    // Hitch attribution: a frame that took over 28 ms lists every section
    // that spent more than half a millisecond in it, plus the time no
    // section accounts for (driver, OS scheduling, another process).
    {
        const auto now = std::chrono::steady_clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(now - g_last_end).count();
        // (as in vk_engine.cpp: much longer than the frames around it, not merely over 28 ms)
        static double avg_ms = 16.0;
        const bool hitch = ++g_total_frames > 30 && frame_ms > std::max(28.0, 1.8 * avg_ms) && frame_ms < 5000.0;
        avg_ms += (std::min(frame_ms, 200.0) - avg_ms) * 0.03;
        g_last_end = now;
        std::string line;
        char hb[96];
        double covered = 0.0;
        for (int i = 0; i < n; ++i) {
            const double ms = static_cast<double>(g_sections[i].frame_ns.exchange(0, std::memory_order_relaxed)) * 1e-6;
            if (!hitch || ms < 0.5) continue;
            if (g_sections[i].name[0] != '+') covered += ms;   // '+': nested inside another section
            std::snprintf(hb, sizeof(hb), "%s%s %.1f", line.empty() ? "" : ", ", g_sections[i].name, ms);
            line += hb;
        }
        if (hitch)
            qlike::log::warnf("[hitch] frame %.1f ms: %s (unaccounted %.1f)", frame_ms,
                              line.empty() ? "no section over 0.5 ms" : line.c_str(),
                              std::max(0.0, frame_ms - covered));
    }
    if (++g_frames < 300) return;
    std::string line;
    char buf[96];
    for (int i = 0; i < n; ++i) {
        const uint64_t ns = g_sections[i].ns.exchange(0, std::memory_order_relaxed);
        g_sections[i].calls.store(0, std::memory_order_relaxed);
        g_last_ms[i] = static_cast<float>(static_cast<double>(ns) * 1e-6 / g_frames);
        if (g_last_ms[i] < 0.005f) continue;
        std::snprintf(buf, sizeof(buf), "%s%s %.2f", line.empty() ? "" : ", ",
                      g_sections[i].name, g_last_ms[i]);
        line += buf;
    }
    g_frames = 0;
    qlike::log::infof("[cpuprof] ms/frame: %s", line.c_str());
}

}  // namespace cpuprof
