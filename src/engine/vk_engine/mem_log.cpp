// How much memory the process holds (the periodic "[mem]" line: VIBE_LOG_MEM=1, vk_engine.cpp).
// In a file of its own: <windows.h> stays out of the engine's other sources.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace vibe {

// Megabytes: what is in RAM now (working set) and what is committed to the process (private).
void process_memory_mb(double& working, double& committed) {
    working = committed = 0.0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        working = double(pmc.WorkingSetSize) / 1048576.0;
        committed = double(pmc.PrivateUsage) / 1048576.0;
    }
#endif
}

}  // namespace vibe
