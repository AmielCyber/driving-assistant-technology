/**
* GPT GENERATED CODE with Previous LaneStateLogger.h and LaneStateLogger.cpp passed as upload with the following
 * prompt:
 * "Can you reformat this class to do logging to a file without blocking code. Make it more general for other
 * classes to use by having a print (or a better name for writing to file), function that takes an std::string_view and
 * writes it to a file without blocking code. Use C++ 17 concurrency and add header string for the constructor for the
 * file's first line"
 */
#include "AsyncLogger.h"

AsyncLogger::AsyncLogger(const std::string& filename, const std::string_view header) {
  // std::ios::trunc forces the file to be truncated/overwritten if it exists
  log_file_.open(filename, std::ios::out | std::ios::trunc);

  // Write CSV header to the freshly cleared/created file
  if (log_file_.is_open() && !header.empty()) {
    log_file_ << header << '\n';
    log_file_.flush();
  }

  // Launch background thread to handle async writes
  worker_thread_ = std::thread(&AsyncLogger::process_queue, this);
}

AsyncLogger::~AsyncLogger() {
  // Signal worker thread to wrap up and notify
  running_ = false;
  cv_.notify_one();

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void AsyncLogger::write(std::string_view message) {
  enqueue(std::string(message));
}

void AsyncLogger::write(std::string&& message) {
  enqueue(std::move(message));
}

void AsyncLogger::enqueue(std::string msg) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(msg));
  }
  cv_.notify_one(); // Wake up background thread
}

void AsyncLogger::process_queue() {
  while (running_ || !queue_.empty()) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until queue has work or logger is shutting down
    cv_.wait(lock, [this] {
        return !queue_.empty() || !running_;
    });

    while (!queue_.empty()) {
      std::string msg = std::move(queue_.front());
      queue_.pop();

      // Unlock while performing file I/O so writers aren't blocked
      lock.unlock();
      if (log_file_.is_open()) {
        log_file_ << msg << '\n';
      }
      lock.lock();
    }
  }

  if (log_file_.is_open()) {
    log_file_.flush();
  }
}
