#include "stdafx.h"
#include "cuda_guard.cuh"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "index_.h"

G_Grid::Cell* CudaGuard::pd_grid_;
Object* CudaGuard::pd_objects_;
curandState* CudaGuard::pd_curand_state_;

CudaGuard::CudaGuard() {
	cudaSetDevice(0);

	G_Grid::Generate_Randomly();
	cudaMalloc((void**)&pd_grid_, sizeof(G_Grid::grid_));
	cudaMemcpy(pd_grid_, G_Grid::grid_, sizeof(G_Grid::grid_), cudaMemcpyHostToDevice);

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
