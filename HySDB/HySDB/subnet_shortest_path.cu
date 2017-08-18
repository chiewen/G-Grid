#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "cuda_guard.cuh"
#include "subnet_shortest_path.cuh"
#include "index_.h"

void SubnetBF::BfOnCpu(int* cells, int cell_num, int start_cell, int start_vertex, int start_edge, int distance, SBfResult* result, int edge_num) {
	int grid_to_result[Index::kCellNum];
	for (int i = 0; i < Index::kCellNum; ++i) {
		grid_to_result[i] = -1;
	}

	int result_idx = 0;
	for (int i = 0; i < cell_num; ++i) {
		grid_to_result[cells[i]] = i;
		for (int j = 0; j < Index::kMaxVerticesPerCell; ++j) {
			auto& r = result[result_idx++];
			r.id_ = Index::grid_[cells[i]].vertex_[j].id_;
			r.distance_ = std::numeric_limits<int>::max();
			r.previous_vertex_id = 0;
		}
	}
	auto& edge = Index::grid_[start_cell].vertex_[start_vertex].edges_[start_edge];
	result[Index::kMaxVerticesPerCell * grid_to_result[edge.to_cell_] + edge.to_vertex_pos_].distance_ = edge.length_ - distance;
	for (int i = 0; i < edge_num; ++i) {
		for (int j = 0; j < cell_num; ++j) {
			auto& c = Index::grid_[cells[j]];
			for (int k = 0; k < Index::kMaxVerticesPerCell; ++k) {
				auto& v = c.vertex_[k];
				int v_dist = result[Index::kMaxVerticesPerCell * grid_to_result[cells[j]] + k].distance_;
				if (v_dist < std::numeric_limits<int>::max()) {
					for (int l = 0; l < Index::kMaxEdgesPerVertex; ++l) {
						auto& e = v.edges_[l];
						if (e.id_ != 0 && grid_to_result[e.to_cell_] != -1) {
							auto& r = result[Index::kMaxVerticesPerCell * grid_to_result[e.to_cell_] + e.to_vertex_pos_];
							if (r.distance_ > v_dist + e.length_) {
								r.distance_ = v_dist + e.length_;
								r.previous_vertex_id = v.id_;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void devBfOnGpu(int* cells, int cell_num, int * grid_to_result, SubnetBF::SBfResult* result, Index::Cell* grid_, int edge_num, int int_max) {
	int cells_per_loop = SubnetBF::kMaxThreadsPerBlock / Index::kMaxVerticesPerCell;
	int loop_num = (cell_num + cells_per_loop - 1) / cells_per_loop;

	for (int i = 0; i < edge_num; ++i) {
		for (int j = 0; j < loop_num; ++j) {
			int current_cell = threadIdx.y + cells_per_loop * j;
			if (current_cell < cell_num) {
				auto& cell = grid_[cells[current_cell]];
				auto& vertex = cell.vertex_[threadIdx.x];
				int v_dist = result[Index::kMaxVerticesPerCell * grid_to_result[cells[current_cell]] + threadIdx.x].distance_;
				if (v_dist < int_max) {
					for (int l = 0; l < Index::kMaxEdgesPerVertex; ++l) {
						auto& e = vertex.edges_[l];
						if (e.id_ != 0 && grid_to_result[e.to_cell_] != -1) {
							auto& r = result[Index::kMaxVerticesPerCell * grid_to_result[e.to_cell_] + e.to_vertex_pos_];
							if (r.distance_ > v_dist + e.length_) {
								r.distance_ = v_dist + e.length_;
								r.previous_vertex_id = vertex.id_;
							}
						}
					}
				}
			}
		}
		__syncthreads();
	}
}

void SubnetBF::BfOnGpu(int* cells, int cell_num, int start_cell, int start_vertex, int start_edge, int distance, SBfResult* result, int edge_num) {
	int grid_to_result[Index::kCellNum];
	for (int i = 0; i < Index::kCellNum; ++i) {
		grid_to_result[i] = -1;
	}

	int result_idx = 0;
	for (int i = 0; i < cell_num; ++i) {
		grid_to_result[cells[i]] = i;
		for (int j = 0; j < Index::kMaxVerticesPerCell; ++j) {
			auto& r = result[result_idx++];
			r.id_ = Index::grid_[cells[i]].vertex_[j].id_;
			r.distance_ = std::numeric_limits<int>::max();
			r.previous_vertex_id = 0;
		}
	}

	auto& edge = Index::grid_[start_cell].vertex_[start_vertex].edges_[start_edge];
	result[Index::kMaxVerticesPerCell * grid_to_result[edge.to_cell_] + edge.to_vertex_pos_].distance_ = edge.length_ - distance;
	
	int * d_cells;
	cudaMalloc((void**)&d_cells, sizeof(int) * cell_num);
	cudaMemcpy(d_cells, cells, sizeof(int) * cell_num, cudaMemcpyHostToDevice);

	int * d_grid_to_result;
	cudaMalloc((void**)&d_grid_to_result, sizeof(int) * Index::kCellNum);
	cudaMemcpy(d_grid_to_result, grid_to_result, sizeof(int) * Index::kCellNum, cudaMemcpyHostToDevice);

	SBfResult * d_result;
	cudaMalloc((void**)&d_result, sizeof(SBfResult) * cell_num * Index::kMaxVerticesPerCell);
	cudaMemcpy(d_result, result, sizeof(SBfResult) * cell_num * Index::kMaxVerticesPerCell, cudaMemcpyHostToDevice);

	dim3 block(Index::kMaxVerticesPerCell, kMaxThreadsPerBlock / Index::kMaxVerticesPerCell);
	devBfOnGpu <<<1, block>>>(d_cells, cell_num, d_grid_to_result, d_result, CudaGuard::pd_grid_, edge_num, std::numeric_limits<int>::max());

	cudaMemcpy(result, d_result, sizeof(SBfResult) * cell_num * Index::kMaxVerticesPerCell, cudaMemcpyDeviceToHost);

	cudaFree(d_grid_to_result);
	cudaFree(d_result);
	cudaFree(d_cells);
}
