#pragma once
// Tiny CPU section profiler for the frame loop.
//
//   CPU_SCOPE("name");          // times the enclosing block
//   cpuprof::end_frame();       // once per frame; logs averages every ~5 s,
//                               // and names the slow sections of any hitch frame
//
// Sections may be timed from any thread (worker jobs included): each sample
// is one atomic add. Names must be string literals (compared by pointer).
#include <atomic>
#include <chrono>
#include <cstdint>

namespace cpuprof {

struct Section {
    const char*           name = nullptr;
    std::atomic<uint64_t> ns{0};          // summed over the logging window
    std::atomic<uint64_t> frame_ns{0};    // this frame only (hitch attribution)
    std::atomic<uint32_t> calls{0};
};

Section* section(const char* name);          // find or register (thread-safe)
void     end_frame();                        // main thread, once per frame
// Averages of the last completed window, for the HUD / log. ms per frame.
float    last_ms(const char* name);

struct Scope {
    Section* s;
    std::chrono::steady_clock::time_point t0;
    explicit Scope(Section* sec) : s(sec), t0(std::chrono::steady_clock::now()) {}
    ~Scope() {
        const auto dt = std::chrono::steady_clock::now() - t0;
        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
        s->ns.fetch_add(ns, std::memory_order_relaxed);
        s->frame_ns.fetch_add(ns, std::memory_order_relaxed);
        s->calls.fetch_add(1, std::memory_order_relaxed);
    }
};

}  // namespace cpuprof

#define CPU_SCOPE_CAT2(a, b) a##b
#define CPU_SCOPE_CAT(a, b) CPU_SCOPE_CAT2(a, b)
#define CPU_SCOPE(name)                                                        \
    static cpuprof::Section* CPU_SCOPE_CAT(cpu_sec_, __LINE__) =               \
        cpuprof::section(name);                                                \
    cpuprof::Scope CPU_SCOPE_CAT(cpu_scope_, __LINE__)(CPU_SCOPE_CAT(cpu_sec_, __LINE__))
