#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "object.cuh"
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "index_.h"
#include "cuda_guard.cuh"

Object Objects::objects_[Objects::kTotalObjectNum];

__global__ void devStep(curandState* s, Object* objects_, const G_Grid::Cell* __restrict__ grid_) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto& o = objects_[idx];
	o.position_ += o.speed_;

	auto& edge = grid_[o.cell_id].vertex_[o.vertex_pos_].edges_[o.edge_pos_];
	auto edge_length = edge.length_;
	while (o.position_ >= edge_length) {
		o.position_ -= edge_length;
		o.cell_id = edge.to_cell_;
		o.vertex_pos_ = edge.to_vertex_pos_;
		int new_edge_pos = curand_uniform(s + idx) * grid_[o.cell_id].vertex_[o.vertex_pos_].edge_num_;
		auto& new_edge = grid_[o.cell_id].vertex_[o.vertex_pos_].edges_[new_edge_pos];
		o.edge_pos_ = new_edge_pos;
		o.edge_id_ = new_edge.id_;
	}
}

void Objects::Step() {
	int size = Objects::kTotalObjectNum;

	devStep<<<16, size / 16>>>(CudaGuard::pd_curand_state_, CudaGuard::pd_objects_, CudaGuard::pd_grid_);
	cudaDeviceSynchronize();
	cudaMemcpy(Objects::objects_, CudaGuard::pd_objects_, size * sizeof(Object), cudaMemcpyDeviceToHost);
}

__global__ void devInitialize(curandState* s, Object* objects, const G_Grid::Cell* __restrict__ grid_) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(1232, idx, 0, &s[idx]);

	objects[idx].id_ = idx;
	objects[idx].cell_id = curand_uniform(s + idx) * G_Grid::kCellNum;

	objects[idx].vertex_pos_ = curand_uniform(s + idx) * grid_[objects[idx].cell_id].vertex_num;
	auto& vertex = grid_[objects[idx].cell_id].vertex_[objects[idx].vertex_pos_];

	objects[idx].edge_pos_ = curand_uniform(s + idx) * vertex.edge_num_;
	auto& edge = vertex.edges_[objects[idx].edge_pos_];
	objects[idx].edge_id_ = edge.id_;

	objects[idx].position_ = curand_uniform(s + idx) * edge.length_;
	objects[idx].speed_ = Objects::kMinSpeed + curand_uniform(s + idx) * (Objects::kMaxSpeed - Objects::kMinSpeed);
}

void Objects::Initialize() {
	memset(Objects::objects_, 0, sizeof(Object) * kTotalObjectNum);

	int size = Objects::kTotalObjectNum;

	devInitialize<<<16, size / 16>>>(CudaGuard::pd_curand_state_, CudaGuard::pd_objects_, CudaGuard::pd_grid_);
	cudaDeviceSynchronize();
	cudaMemcpy(Objects::objects_, CudaGuard::pd_objects_, size * sizeof(Object), cudaMemcpyDeviceToHost);
}
