#include "stdafx.h"
#include "cuda_guard.cuh"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "index_.h"

Index::Cell* CudaGuard::pd_grid_;
Object* CudaGuard::pd_objects_;
curandState* CudaGuard::pd_curand_state_;

CudaGuard::CudaGuard() {
	cudaSetDevice(0);

	Index::Initialize();
	cudaMalloc((void**)&pd_grid_, sizeof(Index::grid_));
	cudaMemcpy(pd_grid_, Index::grid_, sizeof(Index::grid_), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&pd_objects_, Objects::kTotalObjectNum * sizeof(Object));
	cudaMemset(pd_objects_, 0, Objects::kTotalObjectNum * sizeof(Object));

	cudaMalloc((void**)&pd_curand_state_, Objects::kTotalObjectNum * sizeof(curandState));
}

CudaGuard::~CudaGuard() {
	cudaFree(pd_objects_);
	cudaFree(pd_grid_);
	cudaFree(pd_curand_state_);

	cudaDeviceReset();
}
