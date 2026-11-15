#pragma once
// Small worker pool for per-frame CPU work.
//
//   jobs::Group g;
//   jobs::run(g, [&] { ... });        // queued, runs on a worker
//   ... main thread does other work ...
//   jobs::wait(g);                    // helps with queued jobs until g is done
//
//   jobs::parallel_for(n, grain, [&](int begin, int end) { ... });
//
// The calling thread always takes part (in wait and in parallel_for), so the
// pool also works with zero workers and a job is never slower than inline
// execution by more than the queueing overhead.
#include <atomic>
#include <functional>

namespace jobs {

struct Group {
    std::atomic<int> pending{0};
    bool busy() const { return pending.load(std::memory_order_acquire) > 0; }
};

void init();                 // starts the workers (idempotent)
void shutdown();             // joins them; queued jobs are finished first
int  worker_count();

void run(Group& g, std::function<void()> fn);
void wait(Group& g);

// Splits [0, count) into chunks of at least `grain` items and runs
// fn(begin, end) on the workers plus the calling thread. Returns when done.
void parallel_for(int count, int grain, const std::function<void(int, int)>& fn);

}  // namespace jobs
