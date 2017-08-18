#include "stdafx.h"
#include "index_.h"
#include "object.cuh"
#include <curand_kernel.h>

struct CudaGuard {
	static Index::Cell* pd_grid_;
	static Object* pd_objects_;
	static curandState* pd_curand_state_;

	CudaGuard();
	~CudaGuard();
};
