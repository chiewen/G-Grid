#pragma once

#include "stdafx.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

class SubnetBF {
public:
	static const int kMaxThreadsPerBlock = 64;

	struct SBfResult {
		int id_;
		int distance_;
		int previous_vertex_id;
	};

	SubnetBF() {}

	static void BfOnCpu(int* cells, int cell_num, int start_cell,
	                    int start_vertex, int start_edge, int distance,
	                    SBfResult* result_num, int edge_num);

	static void BfOnGpu(int* cells, int cell_num, int start_cell,
	                    int start_vertex, int start_edge, int distance,
	                    SBfResult* result_num, int edge_num);

};


