#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

class CudaStreamControler {
	static const int kStreamNum = 8;
	static cudaStream_t stream[kStreamNum];
	static void initialize();
	static void close();
public:
	static cudaStream_t getStream();
};
