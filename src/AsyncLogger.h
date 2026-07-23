/**************************** LLM GENERATED CODE ********************************************************/
/**
* GPT GENERATED CODE with Previous LaneStateLogger.h and LaneStateLogger.cpp passed as upload with the following
 * prompt:
 * "Can you reformat this class to do logging to a file without blocking code. Make it more general for other
 * classes to use by having a print (or a better name for writing to file), function that takes an std::string_view and
 * writes it to a file without blocking code. Use C++ 17 concurrency and add header string for the constructor for the
 * file's first line"
 */
#ifndef ASYNC_LOGGER_H
#define ASYNC_LOGGER_H

#include <string>
#include <string_view>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <atomic>

class AsyncLogger {
public:
    // Optional header parameter writes CSV/log header during setup
    explicit AsyncLogger(const std::string& filename, std::string_view header = "");
    ~AsyncLogger();

    // Prevent copying to ensure safe single-thread file ownership
    AsyncLogger(const AsyncLogger&) = delete;
    AsyncLogger& operator=(const AsyncLogger&) = delete;

    // Accepts std::string_view, string literals, and std::string without unnecessary copies
    void write(std::string_view message);

    // Overload for temporary strings (rvalues) to allow move semantics
    void write(std::string&& message);

private:
    void enqueue(std::string msg);
    void process_queue();

    std::ofstream log_file_;
    std::queue<std::string> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{true};
    std::thread worker_thread_;
};

#endif // ASYNC_LOGGER_H
/**************************** END LLM GENERATED CODE ********************************************************/