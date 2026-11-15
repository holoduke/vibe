#include "engine/jobs.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace jobs {

namespace {

struct Job {
    std::function<void()> fn;
    Group*                group = nullptr;
};

std::mutex               g_mutex;
std::condition_variable  g_cv;
std::deque<Job>          g_queue;
std::vector<std::thread> g_workers;
bool                     g_stop = false;
bool                     g_started = false;

bool try_pop(Job& out) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_queue.empty()) return false;
    out = std::move(g_queue.front());
    g_queue.pop_front();
    return true;
}

void execute(Job& j) {
    j.fn();
    j.group->pending.fetch_sub(1, std::memory_order_acq_rel);
}

void worker_loop() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(g_mutex);
            g_cv.wait(lk, [] { return g_stop || !g_queue.empty(); });
            if (g_queue.empty()) return;   // stop requested and drained
            j = std::move(g_queue.front());
            g_queue.pop_front();
        }
        execute(j);
    }
}

}  // namespace

void init() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_started) return;
    g_started = true;
    g_stop = false;
    // Leave the main thread and one core for the driver / audio / OS; more
    // than 6 workers does not pay for jobs this small.
    const int hc = static_cast<int>(std::thread::hardware_concurrency());
    const int n = std::clamp(hc - 2, 1, 6);
    g_workers.reserve(n);
    for (int i = 0; i < n; ++i) g_workers.emplace_back(worker_loop);
}

void shutdown() {
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (!g_started) return;
        g_stop = true;
    }
    g_cv.notify_all();
    for (auto& t : g_workers) t.join();
    g_workers.clear();
    std::lock_guard<std::mutex> lk(g_mutex);
    g_started = false;
}

int worker_count() { return static_cast<int>(g_workers.size()); }

void run(Group& g, std::function<void()> fn) {
    g.pending.fetch_add(1, std::memory_order_acq_rel);
    if (g_workers.empty()) {             // pool not running: do it now
        fn();
        g.pending.fetch_sub(1, std::memory_order_acq_rel);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_queue.push_back(Job{ std::move(fn), &g });
    }
    g_cv.notify_one();
}

void wait(Group& g) {
    // Help: run whatever is queued (ours or not) instead of sleeping; once
    // the queue is empty the remaining jobs are already on workers and
    // finish within microseconds, so a short spin is cheaper than a block.
    int spins = 0;
    while (g.busy()) {
        Job j;
        if (try_pop(j)) { execute(j); spins = 0; continue; }
        if (++spins < 64) { /* spin */ }
        else std::this_thread::yield();
    }
}

void parallel_for(int count, int grain, const std::function<void(int, int)>& fn) {
    if (count <= 0) return;
    grain = std::max(grain, 1);
    const int max_chunks = static_cast<int>(g_workers.size()) + 1;
    const int chunks = std::clamp(count / grain, 1, max_chunks);
    if (chunks == 1) { fn(0, count); return; }
    Group g;
    const int per = (count + chunks - 1) / chunks;
    for (int c = 1; c < chunks; ++c) {
        const int b = c * per, e = std::min(count, b + per);
        if (b >= e) break;
        run(g, [&fn, b, e] { fn(b, e); });
    }
    fn(0, std::min(count, per));         // the caller takes the first chunk
    wait(g);
}

}  // namespace jobs
