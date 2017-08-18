#pragma once
#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "index_.h"

struct Object {
	friend class Objects;
	friend class ObjectsTest;

	int id_;
	int speed_;

	int edge_id_;
	int position_;

	int cell_id;
	int vertex_pos_;
	int edge_pos_;
};

class Objects {
public:
	static const int kTotalObjectNum = 1024;
	static const int kMaxSpeed = 10;
	static const int kMinSpeed = 2;

	static Object objects_[kTotalObjectNum];

	static void Initialize();

	static void Step();
};

