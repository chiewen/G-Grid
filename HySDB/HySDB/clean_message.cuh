#pragma once

#include "stdafx.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <algorithm>

__global__ void clct_knl(int* c, const int* a, const int* b, long start, int cal_num);
