#include "stdafx.h"

#include <gtest/gtest.h>
#include "index_.h"
#include "z_order.h"
#include "object.cuh"
#include "cuda_guard.cuh"

using namespace std;

TEST(Index, Common) {
	EXPECT_EQ(Index::kCellNum, 16);
	EXPECT_EQ(Index::kMaxVerticesPerCell, 8);
	EXPECT_EQ(Index::kMaxEdgesPerVertex, 8);
	EXPECT_EQ(sizeof(int), 4);
	EXPECT_EQ(sizeof(Index::grid_),
		sizeof(int) *
		Index::kCellNum * (2 +
			Index::kMaxVerticesPerCell * (2 +
				Index::kMaxEdgesPerVertex * 4)));
}

TEST(Index, ZOrder) {
	EXPECT_EQ(ZOrder::z_order_x[6] + ZOrder::z_order_y[4], 52);
	auto xy = ZOrder::GetXYFromCellId(52);
	EXPECT_EQ(xy.first, 6);
	EXPECT_EQ(xy.second, 4);
	EXPECT_EQ(ZOrder::z_order_x[5] + ZOrder::z_order_y[6], 57);
	xy = ZOrder::GetXYFromCellId(57);
	EXPECT_EQ(xy.first, 5);
	EXPECT_EQ(xy.second, 6);

	auto neighbors = ZOrder::NeighborsOf(50);
	vector<int> should_be{37, 45, 49, 57, 48, 56, 39, 51};
	EXPECT_EQ(neighbors, should_be);

	neighbors = ZOrder::NeighborsOf(18);
	should_be = {5, 13, 17, 25, 16, 24, 7, 19};
	EXPECT_EQ(neighbors, should_be);
}

TEST(Index, Initialization) {
	Index::Initialize();

	//vertex number
	int vertex_sum = 0;
	for (auto& cell : Index::grid_) {
		int count = count_if(begin(cell.vertex_), end(cell.vertex_),
		                     [=](const Index::Cell::Vertex& v) {
		                     return v.id_ != 0;
	                     });
		EXPECT_EQ(cell.vertex_num, count);
		vertex_sum += count;
	}
	EXPECT_EQ(vertex_sum, Index::vertex_num);

	//edges not pointing to their containing vertices
	for (int j = 0; j < Index::kCellNum; ++j) {
		auto& cell = Index::grid_[j];
		for (int i = 0; i < cell.vertex_num; ++i) {
			auto& vertex = cell.vertex_[i];
			if (vertex.id_ != 0)
				for (auto& edge : vertex.edges_) {
					if (edge.id_ != 0)
					EXPECT_TRUE(edge.to_vertex_pos_ != i || edge.to_cell_ != j);
				}
		}
	}

	//edges number 
	for (auto& cell : Index::grid_) {
		int edge_num_c = std::accumulate(std::begin(cell.vertex_), std::end(cell.vertex_), 0, [](int sum, Index::Cell::Vertex& vertex) {
		                                 return sum + vertex.edge_num_;
	                                 });
		EXPECT_EQ(edge_num_c, cell.edge_num);
		for (auto& vertex : cell.vertex_) {
			int edge_num_v = std::count_if(std::begin(vertex.edges_), std::end(vertex.edges_), [=](Index::Cell::Vertex::Edge& edge) {
			                               return edge.id_ != 0;
		                               });
			EXPECT_EQ(vertex.edge_num_, edge_num_v);
		}
	}
}

class ObjectsTest: public ::testing::Test {
public:
	void test_edge() const {
		for (int i = 0; i < Objects::kTotalObjectNum; ++i) {
			auto& o = Objects::objects_[i];
			auto& e = Index::grid_[o.cell_id].vertex_[o.vertex_pos_].edges_[o.edge_pos_];
			ASSERT_EQ(Objects::objects_[i].edge_id_, e.id_);
			ASSERT_GE(e.length_, o.position_);
			ASSERT_EQ(o.cell_id, Index::edge_cell_map_[e.id_]);
		}
	}
};

TEST_F(ObjectsTest, Initialize) {
	CudaGuard guard;
	Objects::Initialize();

	//edge id should be correct
	test_edge();

	for (int i = 0; i < 10; ++i) {
		Objects::Step();
		test_edge();
	}

}
