// include/nns/parallel.hpp
//
// SPDX-License-Identifier: GPL-3.0-only
#ifndef NNS_PARALLEL_HPP
#define NNS_PARALLEL_HPP

#include <cstddef>
#include <thread>
#include <vector>

namespace nns {

/// A lightweight lambda-driven static parallel iteration system.
/// Replaces RcppParallel::parallelFor.
///
/// @param begin Starting loop index (inclusive).
/// @param end Ending loop index (exclusive).
/// @param func A callable matching void(std::size_t worker_begin, std::size_t worker_end).
/// @param n_threads Number of requested worker threads. If <= 0, hardware concurrency is used.
template <typename Func>
void parallel_for(std::size_t begin, std::size_t end, Func&& func, int n_threads = -1) {
  std::size_t total_elements = end - begin;
  if (total_elements == 0) return;

  // Determine available worker count
  unsigned int hw = std::thread::hardware_concurrency();
  std::size_t worker_count = (n_threads <= 0) ? (hw > 0 ? hw : 1) : static_cast<std::size_t>(n_threads);

  // Prevent over-threading on tiny tasks
  if (worker_count > total_elements) {
    worker_count = total_elements;
  }

  // Fallback cleanly to synchronous serial loop if single-threaded
  if (worker_count <= 1) {
    func(begin, end);
    return;
  }

  std::size_t chunk_size = total_elements / worker_count;
  std::size_t remainder  = total_elements % worker_count;

  std::vector<std::thread> threads;
  threads.reserve(worker_count - 1);

  std::size_t current_begin = begin;

  for (std::size_t i = 0; i < worker_count; ++i) {
    std::size_t current_end = current_begin + chunk_size + (i < remainder ? 1 : 0);

    // Main thread executes the final piece directly to eliminate thread spawn latency
    if (i == worker_count - 1) {
      func(current_begin, current_end);
    } else {
      threads.emplace_back(func, current_begin, current_end);
      current_begin = current_end;
    }
  }

  // Collect active workers
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

} // namespace nns

#endif // NNS_PARALLEL_HPP