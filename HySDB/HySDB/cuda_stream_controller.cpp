#include "stdafx.h"
#include "cuda_stream_controller.h"

cudaStream_t CudaStreamControler::stream[CudaStreamControler::kStreamNum];

void CudaStreamControler::initialize() {
	for (int i = 0; i < kStreamNum; ++i) {
		cudaStreamCreate(stream + i);
	}
}

void CudaStreamControler::close() {
	for (int i = 0; i < kStreamNum; ++i) {
		cudaStreamDestroy(stream[i]);
	}
}

cudaStream_t CudaStreamControler::getStream() {
	static int i = 0;
	return stream[i++ % kStreamNum];
}
