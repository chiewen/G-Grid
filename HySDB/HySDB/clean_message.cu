

#include "clean_message.cuh"

const int stream_num = 8;
const int num_per_kernel = 16 * 128 * 16 * 10;
const int repeat_num = 4096;
const int transfer_num = 64;
const int thread_num = 1024;
const long long arraySize = thread_num * stream_num * transfer_num * (num_per_kernel / repeat_num);

cudaError_t clean_calc_message(int* c, const int* a, const int* b);

__global__ void clct_knl(int* c, const int* a, const int* b, long start, int cal_num) {
	int i = threadIdx.x + start;
	for (int j = 0; j < cal_num; j++)
		c[i] = a[i] + b[i];
}

__global__ void xsfl_msg_knl(int* c, const int* a, const int* b, long start, int cal_num) {
	int i = threadIdx.x + start;
	for (int j = 0; j < cal_num; j++)
		c[i] = a[i] + b[i];
}

__global__ void sdst_knl(int* c, const int* a, const int* b, long start, int cal_num) {
	int i = threadIdx.x + start;
	for (int j = 0; j < cal_num; j++)
		c[i] = a[i] + b[i];
}

__global__ void fst_k_knl(int* c, const int* a, const int* b, long start, int cal_num) {
	int i = threadIdx.x + start;
	for (int j = 0; j < cal_num; j++)
		c[i] = a[i] + b[i];
}

__global__ void ursvd_knl(int* c, const int* a, const int* b, long start, int cal_num) {
	int i = threadIdx.x + start;
	for (int j = 0; j < cal_num; j++)
		c[i] = a[i] + b[i];
}

int clean_calc() {

	int *a, *b, *c;

	cudaMallocHost(&a, arraySize * sizeof(int));
	cudaMallocHost(&c, arraySize * sizeof(int));
	b = a;

	//std::generate(a, a + arraySize, []() { return 0; });
	//std::generate(b, b + arraySize, []() { return 0; });
	for (int i = 0; i < arraySize; ++i) {
		a[i] = 0;
	}

	cudaError_t cudaStatus = clean_calc_message(c, a, b);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "clean_calc_message failed!");
		return 1;
	}


	cudaFreeHost(a);
	cudaFreeHost(c);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//	system("pause");
	return 0;
}

cudaError_t clean_calc_message(int* c, const int* a, const int* b) {
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	cudaStream_t stream[stream_num];

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_b, arraySize * sizeof(int));

	for (int i = 0; i < stream_num; ++i) {
		cudaStreamCreate(stream + i);
	}

	dim3 block(128);
	dim3 block2(512);
	dim3 grid(thread_num / 4 / 128, 4);

	for (int i = 0; i < transfer_num; ++i) {
		for (int j = 0; j < stream_num; ++j) {
			int num = num_per_kernel * (stream_num * i + j);
			long start = num % (arraySize - num_per_kernel - 302400);
			cudaMemcpyAsync(dev_a + start, a + start, num_per_kernel - 102400 + (rand()*10) % 302400, cudaMemcpyHostToDevice, stream[j]);
			//			cudaMemcpyAsync(dev_b + start, b + start, num_per_kernel, cudaMemcpyHostToDevice, stream[j]);
			xsfl_msg_knl << <grid, block, 0, stream[j] >> >(dev_c, dev_a, dev_b, start, 150 + rand() % 100);
			clct_knl << <grid, block, 0, stream[j] >> >(dev_c, dev_a, dev_b, start, 100 + rand() % 50);
			sdst_knl << <grid, block, 0, stream[j] >> >(dev_c, dev_a, dev_b, start, 50 + rand() % 30);
			fst_k_knl << <grid, block2, 0, stream[j] >> >(dev_c, dev_a, dev_b, start, 200 + rand() % 100);
			ursvd_knl << <grid, block2, 0, stream[j] >> >(dev_c, dev_a, dev_b, start, 100 + rand() % 80);
			cudaMemcpyAsync(c + start, dev_c + start, num_per_kernel / 8, cudaMemcpyDeviceToHost, stream[j]);
			//			int tt = 3;
			//			for (int j = 0; j < 1; ++j) {
			//				tt += tan(0.1) * tan(0.1);
			//			}
		}
	}
	cudaError_t status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching xsfl_msg_knl!\n", status);
	}

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);


	for (int i = 0; i < stream_num; ++i) {
		cudaStreamDestroy(stream[i]);
	}

	return status;
}
