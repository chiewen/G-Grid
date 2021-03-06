﻿#pragma once
#include "stdafx.h"
#include "gtest_prod.h"

//This class is for the main index of G-Grid. It also provides an initialize method for generating index randomly.
class G_Grid {
	FRIEND_TEST(Index, Initialization);

	//These methods are for generating random cells.
	static int VertexNumInCell();
	static int EdgeNumInVertex();
	static int EdgeLength();
	static int EdgeNumToNeighbors();

public:
	static const int kCellNum = 1 << 4;
	static const int kMaxVerticesPerCell = 1 << 3;
	static const int kMaxEdgesPerVertex = 1 << 3;
	static const int kMaxObjectsPerCell = 32;
	
	static int vertex_num;

	struct Cell {
		int vertex_num;
		int edge_num;

		struct Vertex {
			int id_;
			int edge_num_;

			struct Edge {
				int id_;
				int length_;
				int to_vertex_pos_;
				int to_cell_;
			} edges_[kMaxEdgesPerVertex];
		} vertex_[kMaxVerticesPerCell];
	};

	static Cell grid_[kCellNum];
	static std::vector<int> edge_cell_map_;

	static std::vector<int> Neighbors(int cell_id);
	static void Generate_Randomly();
};
