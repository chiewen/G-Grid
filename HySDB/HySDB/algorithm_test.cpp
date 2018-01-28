#include "stdafx.h"

#include <gtest/gtest.h>
#include "index_.h"
#include "object.cuh"
#include "cuda_guard.cuh"
#include "subnet_shortest_path.cuh"

TEST(Algorithm, SubnetBfCpu) {
	CudaGuard guard;
	Objects::Initialize();
	int cells[]{0,1,2,3,4,6};
	const int result_length = 6 * Index::kMaxVerticesPerCell;
	SubnetBF::SBfResult result[result_length];

	SubnetBF::BfOnCpu(cells, 6, 0, 0, 0, 172, result, Index::grid_[0].edge_num + Index::grid_[1].edge_num + Index::grid_[2].edge_num + Index::grid_[3].edge_num + Index::grid_[4].edge_num + Index::grid_[6].edge_num);

//	for (auto& bf_result : result) {
//		if (bf_result.id_ != 0)
//			std::cout << bf_result.id_ << " :" << bf_result.previous_vertex_id << " " << bf_result.distance_ << " " << std::endl;
//	}
//	std::cout << std::endl << std::endl;

	SubnetBF::SBfResult result_g[result_length];
	SubnetBF::BfOnGpu(cells, 6, 0, 0, 0, 172, result_g, Index::grid_[0].edge_num + Index::grid_[1].edge_num + Index::grid_[2].edge_num + Index::grid_[3].edge_num + Index::grid_[4].edge_num + Index::grid_[6].edge_num);

//	for (auto& bf_result : result_g) {
//		if (bf_result.id_ != 0)
//			std::cout << bf_result.id_ << " :" << bf_result.previous_vertex_id << " " << bf_result.distance_ << " " << std::endl;
//	}
//	std::cout << std::endl << std::endl;
	for (int i = 0; i < result_length; ++i) {
		ASSERT_EQ(result[i].distance_, result_g[i].distance_);
		ASSERT_EQ(result[i].id_, result_g[i].id_);
		ASSERT_EQ(result[i].previous_vertex_id, result_g[i].previous_vertex_id);
	}
}
