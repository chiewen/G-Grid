#include "stdafx.h"
#include "index_.h"
#include "z_order.h"

using namespace std;

G_Grid::Cell G_Grid::grid_[G_Grid::kCellNum];

std::vector<int> G_Grid::edge_cell_map_;

int G_Grid::vertex_num;

int G_Grid::VertexNumInCell() { return kMaxVerticesPerCell - rand() % (kMaxVerticesPerCell / 2); }

int G_Grid::EdgeNumInVertex() { return kMaxEdgesPerVertex - rand() % (kMaxEdgesPerVertex / 2); }

int G_Grid::EdgeLength() { return rand() % 200 + 30; }

int G_Grid::EdgeNumToNeighbors() { return (rand() % kMaxEdgesPerVertex / 3); }

//This methods provides a way to generate an G_Grid randomly.
void G_Grid::Generate_Randomly() {
	//clear all data in G_Grid
	memset(&grid_, 0, sizeof(grid_));
	vertex_num = 0;

	//generate vertices
	int vertex_id = 1;
	for (int i = 0; i < kCellNum; ++i) {
		int vertex_num_in_cell = VertexNumInCell();
		vertex_num += vertex_num_in_cell;
		grid_[i].vertex_num = vertex_num_in_cell;
		for (int j = 0; j < vertex_num_in_cell; ++j) {
			grid_[i].vertex_[j].id_ = vertex_id++;
		}
	}

	//generate edges
	int edge_id = 1;
	for (int i = 0; i < kCellNum; ++i) {
		for (int j = 0; j < grid_[i].vertex_num; ++j) {

			int edge_num_in_vertex = EdgeNumInVertex();

			grid_[i].edge_num += edge_num_in_vertex;
			grid_[i].vertex_[j].edge_num_ = edge_num_in_vertex;

			for (int k = 0; k < edge_num_in_vertex; ++k) {
				auto& edge = grid_[i].vertex_[j].edges_[k];
				edge.id_ = edge_id++;
				edge.length_ = EdgeLength();
				edge.to_cell_ = i;
				int pos = rand() % grid_[i].vertex_num;
				if (pos == j) {
					//requires more than one vertex in a cell
					pos = (j + 1) % grid_[i].vertex_num;
				}
				edge.to_vertex_pos_ = pos;
			}
		}
	}

	//generate edges to neighbor cells
	for (int i = 0; i < kCellNum; ++i) {
		auto neighbors = Neighbors(i);
		for (int j = 0; j < grid_[i].vertex_num; ++j) {
			auto& vertex = grid_[i].vertex_[j];

			int edge_to_neighbors = std::min(EdgeNumToNeighbors(), (kMaxEdgesPerVertex - vertex.edge_num_));

			for (int k = vertex.edge_num_; k < vertex.edge_num_ + edge_to_neighbors; ++k) {
				vertex.edges_[k].id_ = edge_id++;
				vertex.edges_[k].length_ = EdgeLength();
				int to_cell = rand() % neighbors.size();
				vertex.edges_[k].to_cell_ = neighbors[to_cell];
				vertex.edges_[k].to_vertex_pos_ = rand() % grid_[neighbors[to_cell]].vertex_num;
			}
			vertex.edge_num_ += edge_to_neighbors;
			grid_[i].edge_num += edge_to_neighbors;
		}
	}

	//indexing edge to cell
	edge_cell_map_.assign(edge_id, 0);
	for (int i = 0; i < kCellNum; ++i) {
		for (int j = 0; j < grid_[i].vertex_num; ++j) {
			for (int k = 0; k < grid_[i].vertex_[j].edge_num_; ++k) {
				edge_cell_map_[grid_[i].vertex_[j].edges_[k].id_] = i;
			}	
		}
	}
}

std::vector<int> G_Grid::Neighbors(int cell_id) {
	auto result = ZOrder::NeighborsOf(cell_id);
	result.erase(std::remove_if(result.begin(), result.end(), [](int r) {
	                            return r < 0 || r >= kCellNum;
                            }), result.end());

	return result;
}
