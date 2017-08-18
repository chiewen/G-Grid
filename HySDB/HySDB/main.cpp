#include "stdafx.h"
#include <gtest/gtest.h>
#include "object.cuh"
#include "index_.h"
#include "cuda_guard.cuh"

void google_test(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
}

void normal_run() {
	CudaGuard guard;
	Objects::Initialize();
}

int main(int argc, char* argv[])
{
	google_test(argc, argv);
//	normal_run();
	return 0;
}

