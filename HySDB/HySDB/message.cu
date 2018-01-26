#include "stdafx.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifndef __THROW
#define __THROW
#endif

#include <device_functions.h>
#include "message.h"
#include "algorithm"
#include "now.h"

__global__ void xsfl_msg_knl(int t, int* o_num, Message* A, Message* T) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int bundle_id = id / MessageBucket::kRhoB;

	Message cached_messages[5];

	for (int i = id * MessageBucket::kRhoB; i < MessageBucket::kRhoB; ++i) {
		Message m = A[i];
		for (int j = 4; j >= 0; --j) {
			for (int k = 0; k < 5; ++k) {
				if (cached_messages[k].oid == m.oid) {
					if (cached_messages[k].t < m.t) {
						cached_messages[k] = m;
					}
					break;
				}
				if(cached_messages[k].oid == 0) {
					cached_messages[k] = m;
					break;
				}
			}				

			m.oid = __shfl_xor_sync(0xFFFFFFFF, m.oid, 1 << j);
//			m.eid = __shfl_xor(m.eid, 1 << j);
//			m.rho = __shfl_xor(m.rho, 1 << j);
//			m.t = __shfl_xor(m.t, 1 << j);
		}
		for (int j = 0; j < 4; ++j) {
			T[bundle_id] = m;
		}
	}
}

MessageBucket* MessageLists::lists_[Index::kCellNum];

void MessageLists::MessageCleaning(std::vector<int> lists, int message_out_num, Message* messages) {
	int n_to_clean = std::accumulate(lists.begin(), lists.end(), 0,
	                                 [](int t, int i) ->int {
	                                 auto pm = lists_[i];
	                                 int total = 0;
	                                 while (pm != nullptr && (Now::now() - pm->t < kTimeDelta)) {
		                                 //TODO delete if obsolete
		                                 total++;
		                                 pm = pm->p;
	                                 }
	                                 return t + total;
                                 });

	Message *h_buckets, *d_buckets, *d_T;
	int n_message_out, *d_m;
	cudaMallocHost(&h_buckets, sizeof(Message) * MessageBucket::kRhoB * n_to_clean);
	cudaMalloc(&d_buckets, MessageBucket::kRhoB * n_to_clean);
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_T, sizeof(Message) * n_to_clean);
	cudaMemcpy(d_buckets, h_buckets, sizeof(Message) * MessageBucket::kRhoB * n_to_clean, cudaMemcpyHostToDevice);

	cudaSetDevice(0);

	dim3 block(128);
	dim3 grid(n_to_clean / 128);

	//	xsfl_msg_knl << <grid, block, 0, CudaStreamControler::getStream()>> >(Now::now(), d_m, d_buckets, d_T);

	cudaDeviceSynchronize();

	cudaMemcpy(h_buckets, d_buckets, sizeof(Message) * MessageBucket::kRhoB * n_to_clean, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_m, &n_message_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_buckets);
	cudaFree(d_m);
	cudaFree(d_T);
	cudaFreeHost(h_buckets);
}
