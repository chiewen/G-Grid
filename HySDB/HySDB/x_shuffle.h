
#pragma once

#include "stdafx.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include "message.h"

__global__ void xsfl_msg_knl(int t, int * o_num, Message * A, Message* T);
