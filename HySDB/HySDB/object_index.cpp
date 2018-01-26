#include "stdafx.h"
#include "object_index.h"

ObjectIndex::CellMap ObjectIndex::index_[];

void ObjectIndex::initialize() {
	memset(index_, 0, sizeof(index_));
}

void ObjectIndex::update(int object_id, int edge_id) {
	CellMap& cm = index_[object_id];
	cm.edge_id_ = edge_id;
	cm.cell_id_ = G_Grid::edge_cell_map_[edge_id];
}
