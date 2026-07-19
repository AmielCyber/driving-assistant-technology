#ifndef ADAS_FEATURE_THREAD_POOL_H
#define ADAS_FEATURE_THREAD_POOL_H
#include <future>
#include <opencv2/core/mat.hpp>
#include <thread>

/**
 * Using reference "C++ Concurrency in Action" 2nd edition by Anthony Williams Pg.302 to implement
 * our own simple custom thread for functions that return a cv::Mat. In our case the process implementations
 * of the ADASFeature interfaces. Gives approximately 5FPS more compare to OpenMP and reduces build overhead,
 * but at the cost with more code.
 */
class ADASFeatureThreadPool {
public:
  /**
   *
   * @param thread_count The amount of threads to do parallel work. Pass the number of ADASFeature implementations.
   */
  explicit ADASFeatureThreadPool(size_t thread_count = 1);
  /**
   * Stop all jobs that are not in progress, wake up all thread and join all threads.
   */
  ~ADASFeatureThreadPool();
  // Rule of three, Since we overrode the default deconstructor
  ADASFeatureThreadPool(const ADASFeatureThreadPool&) = delete;
  ADASFeatureThreadPool& operator=(const ADASFeatureThreadPool&) = delete;

  /**
   * Submit a process image task to the thread pool.
   * @param task The task to be done. In our case the ADASFeature process implementation method.
   * @return The result of the task. In our case the image with the annotations.
   */
  std::future<cv::Mat> submit(std::function<cv::Mat()> task);
  /**
   * Get number of threads in the thread pool.
   * @return The number of worker threads in the thread pool
   */
  [[nodiscard]] size_t size() const noexcept;
private:
  std::vector<std::thread> worker_threads;
  // ADASFeature implementation process tasks: cv::Mat process(cv::Mat&)
  std::queue<std::packaged_task<cv::Mat()>> worker_queue;
  std::mutex tasks_mutex;
  std::condition_variable tasks_condition; // For efficient thread sleep
  bool stop{false};
  /**
   * A thread's iteration while waiting for tasks.
   */
  void worker_loop();
};

#endif // ADAS_FEATURE_THREAD_POOL_H
