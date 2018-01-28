#pragma once
#include "index_.h"
#include "object.cuh"

class ObjectIndex {
public:
	struct CellMap {
		int edge_id_;
		int position_;
		int cell_id_;
	};
	static CellMap index_[Objects::kTotalObjectNum];

	static void initialize();
	static void update(int object_id, int edge_id);
};

