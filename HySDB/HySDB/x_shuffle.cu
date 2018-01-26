#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "x_shuffle.h"

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
				else if(cached_messages[k].oid == 0) {
					cached_messages[k] = m;
					break;
				}
			}				

			m.oid = __shfl_xor(m.oid, 1 << j);
			m.eid = __shfl_xor(m.eid, 1 << j);
			m.rho = __shfl_xor(m.rho, 1 << j);
			m.t = __shfl_xor(m.t, 1 << j);
		}
		for (int j = 0; j < 4; ++j) {
			T[bundle_id] = m;
		}
	}
}
