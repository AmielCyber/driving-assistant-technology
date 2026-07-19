#include "ADASFeatureThreadPool.h"
ADASFeatureThreadPool::ADASFeatureThreadPool(size_t thread_count){
  if (thread_count == 0) {
    thread_count = 1;
  }
  worker_threads.reserve(thread_count);
  for (size_t i = 0; i < thread_count; ++i) {
    worker_threads.emplace_back([this]{worker_loop();});
  }
}
ADASFeatureThreadPool::~ADASFeatureThreadPool() {
  {
    std::lock_guard lock(tasks_mutex);
    stop = true;
  }
  tasks_condition.notify_all(); // Notify all threads
  for (auto& worker : worker_threads) {
    if (worker.joinable()) {
      // Blocking code to stop all threads
      worker.join();
    }
  }
}
std::future<cv::Mat> ADASFeatureThreadPool::submit(std::function<cv::Mat()> task) {
  std::packaged_task job(std::move(task));
  std::future<cv::Mat> annotation_result = job.get_future();
  {
    //  Use RAII mutex lock to unlock at end of scope
    std::lock_guard lock(tasks_mutex);
    if (stop) {
      throw std::runtime_error("submit_task() called on stopped ADASFeatureThreadPool");
    }
    worker_queue.push(std::move(job));
  }
  // Notify one thread that is their turn to take the next task.
  tasks_condition.notify_one();
  return annotation_result;
}

size_t ADASFeatureThreadPool::size() const noexcept {return worker_threads.size(); }

void ADASFeatureThreadPool::worker_loop() {
  for (;;) {
    // ADASFeature process task that returns a cv::Mat
    std::packaged_task<cv::Mat()> job;
    {
      std::unique_lock lock(tasks_mutex);
      // Wait queue until unlocked
      tasks_condition.wait(lock,[this] {
        return stop || !worker_queue.empty();
      });
      if (stop && worker_queue.empty()) {
        // No more future work, lets quit
        return;
      }
      // Assign the job
      job = std::move(worker_queue.front());
      worker_queue.pop();
    }
    // Time to do the job, in our case process the image and provide annotations
    // Call the callable aka process(cv::Mat)
    job();
  }
}

