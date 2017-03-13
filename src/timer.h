#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>
#include <time.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

struct CPUTimer
{
	struct timespec startCPUtime;
	struct timespec stopCPUtime;
	
	void Start()
	{
		clock_gettime(CLOCK_REALTIME, &startCPUtime);
	}
	
	void Stop()
	{
		clock_gettime(CLOCK_REALTIME, &stopCPUtime);
	}

	 unsigned long long int Runtime()
	{
		unsigned long long int runtime;
		runtime = 1000000000 * (stopCPUtime.tv_sec - startCPUtime.tv_sec) + (stopCPUtime.tv_nsec - startCPUtime.tv_nsec);
		return runtime;
	}

};

#endif  /* GPU_TIMER_H__ */
