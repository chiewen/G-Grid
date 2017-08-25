#pragma once
#include "index_.h"
#include "object.cuh"

class ObjectIndex {
public:
	static const int kMapSize = std::max(static_cast<int>((Index::kCellNum / sizeof(int)) / 8), 1);
	struct CellMap {
		int edge_id_;
		int position_;
		int map[kMapSize];
	};
	static CellMap index_[Objects::kTotalObjectNum];

	static void initialize();
	static void update(int object_id, int edge_id);
};

