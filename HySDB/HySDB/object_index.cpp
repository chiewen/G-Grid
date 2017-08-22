#include "stdafx.h"
#include "object_index.h"

ObjectIndex::CellMap ObjectIndex::object_index_[];

void ObjectIndex::initialize() {
	memset(object_index_, 0, sizeof(object_index_));
}

void ObjectIndex::update(int object_id, int cell_id) {
	CellMap& cm = object_index_[object_id];
	cm.map[cm.edge_id_ / sizeof(int)] = 0;

	cm.map[cell_id / sizeof(int)] = 1;
}
