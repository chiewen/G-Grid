#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "index_.h"
#include "object.cuh"
#include "message_processor.cuh"
#include "now.h"
#include "cuda_stream_controller.h"

__global__ void xsfl_msg_knl(int n, MessageC* A, MessageC* T) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int bundle_id = id / MessageBucket::kRhoB;

	MessageC cached_messages[5];

	for (int i = id * MessageBucket::kRhoB; i < MessageBucket::kRhoB; ++i) {
		MessageC m = A[i];
		for (int j = 4; j >= 0; --j) {
			for (int k = 0; k < 5; ++k) {
				if (cached_messages[k].oid == m.oid) {
					if (cached_messages[k].t < m.t) {
						cached_messages[k] = m;
					}
					break;
				}
				if (cached_messages[k].oid == 0) {
					cached_messages[k] = m;
					break;
				}
			}

			m.oid = __shfl_xor_sync(0xFFFFFFFF, m.oid, 1 << j);
			m.eid = __shfl_xor_sync(0xFFFFFFFF, m.eid, 1 << j);
			m.rho = __shfl_xor_sync(0xFFFFFFFF, m.rho, 1 << j);
			m.t = __shfl_xor_sync(0xFFFFFFFF, m.t, 1 << j);
		}
		for (int j = 0; j < 4; ++j) {
			T[m.oid * n + bundle_id] = m;
		}
	}
}

__global__ void clct_knl(int n, int* o_num, MessageC *R, MessageC* T) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int bundle_id = id / MessageBucket::kRhoB;

	MessageC m;
	for (int i = id * n; i < id * n + n; ++i) {
		if (T[i].oid == id && T[i].t > m.t) {
			m = T[i];
		}
		if (m.oid != 0) {
			for (int j = m.cid * G_Grid::kMaxObjectsPerCell; j < m.cid * G_Grid::kMaxObjectsPerCell + G_Grid::kMaxObjectsPerCell ; ++j) {
				if (R[j].oid == 0) R[j] = m;
			}
		}
	}
}

MessageBucket* MessageLists::lists_[G_Grid::kCellNum];

void MessageLists::MessageCleaning(std::vector<int> lists, int message_out_num, MessageC* messages) {
	int n_to_clean = std::accumulate(lists.begin(), lists.end(), 0,
	                                 [](int t, int i) -> int {
	                                 auto pm = lists_[i];
	                                 int total = 0;
	                                 while (pm != nullptr && (Now::now() - pm->t < kTimeDelta)) {
		                                 //TODO delete if obsolete
		                                 total++;
		                                 pm = pm->p;
	                                 }
	                                 return t + total;
                                 });

	MessageC *h_buckets, *d_buckets, *d_T, *d_R, *h_R;
	int n_message_out, *d_m;
	cudaMallocHost(&h_buckets, sizeof(MessageC) * MessageBucket::kRhoB * n_to_clean);
	cudaMalloc(&d_buckets, MessageBucket::kRhoB * n_to_clean);
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_T, sizeof(MessageC) * Objects::kTotalObjectNum * n_to_clean);
	cudaMalloc(&d_R, sizeof(MessageC) * G_Grid::kCellNum * G_Grid::kMaxObjectsPerCell);
	cudaMallocHost(&h_R, sizeof(MessageC) * G_Grid::kCellNum * G_Grid::kMaxObjectsPerCell);
	cudaMemcpy(d_buckets, h_buckets, sizeof(MessageC) * MessageBucket::kRhoB * n_to_clean, cudaMemcpyHostToDevice);

	cudaSetDevice(0);

	dim3 block(128);
	dim3 grid(n_to_clean / 128);

	xsfl_msg_knl << <grid, block, 0, CudaStreamControler::getStream()>> >(n_to_clean, d_buckets, d_T);

	clct_knl << <dim3(Objects::kTotalObjectNum / 128), block, 0, CudaStreamControler::getStream()>> >(n_to_clean, d_m, d_R, d_T);

	cudaDeviceSynchronize();

	cudaMemcpy(d_R, h_R, sizeof(MessageC) * MessageBucket::kRhoB * n_to_clean, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_m, &n_message_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_buckets);
	cudaFree(d_m);
	cudaFree(d_T);
	cudaFree(d_R);
	cudaFreeHost(h_buckets);
	cudaFreeHost(h_R);
}
