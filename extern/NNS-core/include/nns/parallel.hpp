// include/nns/parallel.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_PARALLEL_HPP
#define NNS_PARALLEL_HPP

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace nns {

namespace detail {

// 0 = automatic (NNS_NUM_THREADS environment variable, else the hardware
// concurrency). Shared by every translation unit (inline function static).
inline std::atomic<int>& thread_setting() {
  static std::atomic<int> setting{0};
  return setting;
}

inline int automatic_thread_count() {
  static const int count = [] {
    if (const char* env = std::getenv("NNS_NUM_THREADS")) {
      const long v = std::strtol(env, nullptr, 10);
      if (v > 0) return static_cast<int>(v);
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
  }();
  return count;
}

}  // namespace detail

/// Total estimated work (items x work_per_item, roughly in inner-loop
/// iterations) below which parallel_for runs serially. Spawning and joining a
/// thread costs tens of microseconds, which dominated small calls.
constexpr std::size_t kMinParallelWork = 200000;

/// Set the default number of worker threads used when a kernel is called with
/// n_threads <= 0 (the default everywhere). n <= 0 restores the automatic
/// choice: the NNS_NUM_THREADS environment variable if set to a positive
/// integer, else std::thread::hardware_concurrency(). Thread-safe; intended
/// for bindings that expose an `ncores` / `set_num_threads` option, and for
/// running inside process pools without oversubscribing the machine.
inline void set_num_threads(int n) { detail::thread_setting().store(n > 0 ? n : 0); }

/// The number of worker threads used for n_threads <= 0 (always >= 1).
inline int get_num_threads() {
  const int n = detail::thread_setting().load();
  return n > 0 ? n : detail::automatic_thread_count();
}

/// A lightweight lambda-driven static parallel iteration system.
/// Replaces RcppParallel::parallelFor. Every index is processed exactly once
/// by `func(chunk_begin, chunk_end)`; kernels write disjoint output slots, so
/// results never depend on the thread count. An exception thrown by `func`
/// on any thread is rethrown on the calling thread once every worker has
/// joined (the first one if several throw).
///
/// @param begin Starting loop index (inclusive).
/// @param end Ending loop index (exclusive).
/// @param func A callable matching void(std::size_t worker_begin, std::size_t worker_end).
/// @param n_threads Number of requested worker threads. If <= 0, get_num_threads() is used.
/// @param work_per_item Estimated cost of one index (e.g. the inner loop
///   length). When > 0 and (end - begin) * work_per_item < kMinParallelWork,
///   the loop runs serially on the calling thread. 0 means unknown (no
///   serial fallback beyond the thread count).
template <typename Func>
void parallel_for(std::size_t begin, std::size_t end, Func&& func, int n_threads = -1,
                  std::size_t work_per_item = 0) {
  if (end <= begin) return;
  const std::size_t total_elements = end - begin;

  std::size_t worker_count =
      static_cast<std::size_t>(n_threads > 0 ? n_threads : get_num_threads());

  // Prevent over-threading on tiny tasks
  if (worker_count > total_elements) worker_count = total_elements;
  if (work_per_item > 0 && total_elements < kMinParallelWork &&
      total_elements * work_per_item < kMinParallelWork) {
    worker_count = 1;
  }

  // Fallback cleanly to synchronous serial loop if single-threaded
  if (worker_count <= 1) {
    func(begin, end);
    return;
  }

  const std::size_t chunk_size = total_elements / worker_count;
  const std::size_t remainder = total_elements % worker_count;

  std::exception_ptr error;
  std::mutex error_mu;
  auto run = [&error, &error_mu](auto& f, std::size_t b, std::size_t e) {
    try {
      f(b, e);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mu);
      if (!error) error = std::current_exception();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(worker_count - 1);
  // A joinable std::thread terminates on destruction, so join on every exit path.
  struct Joiner {
    std::vector<std::thread>& threads;
    void join() {
      for (auto& t : threads)
        if (t.joinable()) t.join();
    }
    ~Joiner() { join(); }
  } joiner{threads};

  std::size_t current_begin = begin;
  for (std::size_t i = 0; i < worker_count; ++i) {
    const std::size_t current_end = current_begin + chunk_size + (i < remainder ? 1 : 0);
    // Main thread executes the final piece directly to eliminate thread spawn latency
    if (i == worker_count - 1) {
      run(func, current_begin, current_end);
    } else {
      // Each worker gets its own copy of func, as std::thread(func, ...) made.
      threads.emplace_back([&run, f = func, current_begin, current_end]() mutable {
        run(f, current_begin, current_end);
      });
      current_begin = current_end;
    }
  }

  joiner.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace nns

#endif  // NNS_PARALLEL_HPP
