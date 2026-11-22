#pragma once

namespace vibe::crash {

// Install Windows SEH unhandled-exception filter + a C++ std::set_terminate
// hook. Both dump a symbolic stack trace into the vibe::log file (and stderr)
// before the process dies, so we get a real call stack for crashes that
// otherwise surface as a corrupted std::exception or a silent SEH abort.
//
// Call once, after vibe::log::open(), before any work that might crash.
// Idempotent.
void install();

} // namespace vibe::crash
