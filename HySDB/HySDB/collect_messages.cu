#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "collect_messages.cuh"
