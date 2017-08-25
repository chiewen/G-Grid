#include "stdafx.h"
#include "object_index.h"

ObjectIndex::CellMap ObjectIndex::index_[];

void ObjectIndex::initialize() {
	memset(index_, 0, sizeof(index_));
}

void ObjectIndex::update(int object_id, int edge_id) {
	CellMap& cm = index_[object_id];
	cm.map[cm.edge_id_ / sizeof(int) / 8] = 0;
	int cell_id = Index::edge_cell_map_[edge_id];
	cm.map[cell_id / sizeof(int) / 8] = 1 << cell_id % (sizeof(int) * 8);
}
