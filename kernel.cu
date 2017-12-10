#include "kernel.cuh"

__global__ void sum_row(float *img, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height) // width of image
{
	// Shared memory for four tables
	__shared__ float buffer_l1[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_l2[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_lx[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_ly[BLOCK_HEIGHT][BLOCK_WIDTH];

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int m, cursor;
	// Temporal values for prefix sum
	float temp_l1, temp_l2, temp_lx, temp_ly;
	float last_l1, last_l2, last_lx, last_ly;

	for (m = 0; m < (I_width / BLOCK_WIDTH + (I_width%BLOCK_WIDTH>0)); m++)
	{
		int index_in = row * I_width + blockDim.x * m + threadIdx.x;
		int thread_width = BLOCK_WIDTH;
		// thread_width is the width of blocks with activated threads
		if ((m == I_width / BLOCK_WIDTH) && I_width%BLOCK_WIDTH > 0)
			thread_width = I_width%BLOCK_WIDTH;

		if (index_in < (I_width * I_height) && (threadIdx.x < thread_width))
		{
			// Dealing with images whose width is larger than the buffer width
			if (m > 0) {
				last_l1 = buffer_l1[threadIdx.y][BLOCK_WIDTH - 1];
				last_l2 = buffer_l2[threadIdx.y][BLOCK_WIDTH - 1];
				last_lx = buffer_lx[threadIdx.y][BLOCK_WIDTH - 1];
				last_ly = buffer_ly[threadIdx.y][BLOCK_WIDTH - 1];
			}

			// Copy from global memory to shared memory
			buffer_l1[threadIdx.y][threadIdx.x] = img[index_in];
			buffer_l2[threadIdx.y][threadIdx.x] = powf(img[index_in], 2);
			buffer_lx[threadIdx.y][threadIdx.x] = img[index_in] * (blockDim.x * m + threadIdx.x);
			buffer_ly[threadIdx.y][threadIdx.x] = img[index_in] * row;

			// Prefix sum for current array
			for (cursor = 1; cursor <= ceilf(log2f(thread_width)); cursor++)
			{
				if ((threadIdx.x >= __float2int_rd(powf(2, cursor - 1))) && (threadIdx.x < thread_width))
				{
					temp_l1 = buffer_l1[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_l2 = buffer_l2[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_lx = buffer_lx[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_ly = buffer_ly[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
				}
				__syncthreads();
				if ((threadIdx.x >= __float2int_rd(powf(2, cursor - 1))) && (threadIdx.x < thread_width))
				{
					buffer_l1[threadIdx.y][threadIdx.x] += temp_l1;
					buffer_l2[threadIdx.y][threadIdx.x] += temp_l2;
					buffer_lx[threadIdx.y][threadIdx.x] += temp_lx;
					buffer_ly[threadIdx.y][threadIdx.x] += temp_ly;
				}
				__syncthreads();
			}

			// Dealing with images whose width is larger than the buffer width
			if (m > 0) {
				buffer_l1[threadIdx.y][threadIdx.x] += last_l1;
				buffer_l2[threadIdx.y][threadIdx.x] += last_l2;
				buffer_lx[threadIdx.y][threadIdx.x] += last_lx;
				buffer_ly[threadIdx.y][threadIdx.x] += last_ly;
			}

			// Copy from shared memory to global memory
			l1_dev[index_in] = buffer_l1[threadIdx.y][threadIdx.x];
			l2_dev[index_in] = buffer_l2[threadIdx.y][threadIdx.x];
			lx_dev[index_in] = buffer_lx[threadIdx.y][threadIdx.x];
			ly_dev[index_in] = buffer_ly[threadIdx.y][threadIdx.x];
		}
	}
}

__global__ void sum_col(float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height)
{
	// Shared memory for four tables
	__shared__ float buffer_l1[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_l2[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_lx[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_ly[BLOCK_HEIGHT2][BLOCK_WIDTH2];

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int m, cursor;
	// Temporal values for prefix sum
	float temp_l1, temp_l2, temp_lx, temp_ly;
	float last_l1, last_l2, last_lx, last_ly;

	for (m = 0; m < (I_height / BLOCK_HEIGHT2 + (I_height%BLOCK_HEIGHT2>0)); m++)
	{

		int index_in = (blockDim.y * m + threadIdx.y) * I_width + col;
		int thread_height = BLOCK_HEIGHT2;
		if ((m == I_height / BLOCK_HEIGHT2) && I_height%BLOCK_HEIGHT2 > 0)
			thread_height = I_height%BLOCK_HEIGHT2;

		if ((index_in < I_width * I_height) && col < I_width)
		{
			// for array length longer than block width. Each element has to add the
			// previous sum.
			if (m > 0) {
				last_l1 = buffer_l1[BLOCK_HEIGHT2 - 1][threadIdx.x];
				last_l2 = buffer_l2[BLOCK_HEIGHT2 - 1][threadIdx.x];
				last_lx = buffer_lx[BLOCK_HEIGHT2 - 1][threadIdx.x];
				last_ly = buffer_ly[BLOCK_HEIGHT2 - 1][threadIdx.x];
			}

			// Copy from global memory to shared memory (? if buffer length longer than img)
			buffer_l1[threadIdx.y][threadIdx.x] = l1_dev[index_in];
			buffer_l2[threadIdx.y][threadIdx.x] = l2_dev[index_in];
			buffer_lx[threadIdx.y][threadIdx.x] = lx_dev[index_in];
			buffer_ly[threadIdx.y][threadIdx.x] = ly_dev[index_in];

			// Prefix sum for current array
			for (cursor = 1; cursor <= ceilf(log2f(thread_height)); cursor++) {
				/*prefix sum*/
				if (threadIdx.y >= __float2int_rd(powf(2, cursor - 1))) {
					temp_l1 = buffer_l1[threadIdx.y - __float2int_rd(powf(2, cursor - 1))][threadIdx.x];
					temp_l2 = buffer_l2[threadIdx.y - __float2int_rd(powf(2, cursor - 1))][threadIdx.x];
					temp_lx = buffer_lx[threadIdx.y - __float2int_rd(powf(2, cursor - 1))][threadIdx.x];
					temp_ly = buffer_ly[threadIdx.y - __float2int_rd(powf(2, cursor - 1))][threadIdx.x];
				}
				__syncthreads();
				if (threadIdx.y >= __float2int_rd(powf(2, cursor - 1))) {
					buffer_l1[threadIdx.y][threadIdx.x] += temp_l1;
					buffer_l2[threadIdx.y][threadIdx.x] += temp_l2;
					buffer_lx[threadIdx.y][threadIdx.x] += temp_lx;
					buffer_ly[threadIdx.y][threadIdx.x] += temp_ly;
				}
				__syncthreads();
			}

			// Dealing with images whose height is larger than the buffer height
			if (m > 0) {
				buffer_l1[threadIdx.y][threadIdx.x] += last_l1;
				buffer_l2[threadIdx.y][threadIdx.x] += last_l2;
				buffer_lx[threadIdx.y][threadIdx.x] += last_lx;
				buffer_ly[threadIdx.y][threadIdx.x] += last_ly;
			}
			// Copy from shared memory to global memory
			l1_dev[index_in] = buffer_l1[threadIdx.y][threadIdx.x];
			l2_dev[index_in] = buffer_l2[threadIdx.y][threadIdx.x];
			lx_dev[index_in] = buffer_lx[threadIdx.y][threadIdx.x];
			ly_dev[index_in] = buffer_ly[threadIdx.y][threadIdx.x];
		}
	}
}

__global__ void compute_feature(float vt1value, float vt2value, float vt3value, float vt4value, float *v1_dev, float *v2_dev, float *v3_dev, float *v4_dev, float *X_dev, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int K, int M, int N) {
	float S1value = 0;
	float S2value = 0;
	float Sxvalue = 0;
	float Syvalue = 0;
	float v1value = 0;
	float v2value = 0;
	float v3value = 0;
	float v4value = 0;


	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if ((col <= (M - K)) && (row <= (N - K))) {
		// Compute four sum of pixel values within every patch.
		S1value = (l1_dev[(row + K - 1)*M + (col + K - 1)] - l1_dev[(row + K - 1)*M + (col)] - l1_dev[(row)*M + (col + K - 1)] + l1_dev[(row)*M + (col)]);
		S2value = (l2_dev[(row + K - 1)*M + (col + K - 1)] - l2_dev[(row + K - 1)*M + (col)] - l2_dev[(row)*M + (col + K - 1)] + l2_dev[(row)*M + (col)]);
		Sxvalue = (lx_dev[(row + K - 1)*M + (col + K - 1)] - lx_dev[(row + K - 1)*M + (col)] - lx_dev[(row)*M + (col + K - 1)] + lx_dev[(row)*M + (col)]);
		Syvalue = (ly_dev[(row + K - 1)*M + (col + K - 1)] - ly_dev[(row + K - 1)*M + (col)] - ly_dev[(row)*M + (col + K - 1)] + ly_dev[(row)*M + (col)]);

		// Compute four features for every patch and place them in right place.
		v1value = S1value / K / K;
		v2value = S2value / K / K - v1value*v1value;
		v3value = 4.0 * (Sxvalue - (col + 1.0 * (K - 1) / 2) * S1value) / K / K / K;
		v4value = 4.0 * (Syvalue - (row + 1.0 * (K - 1) / 2) * S1value) / K / K / K;
		v1_dev[row * (M - K + 1) + col] = v1value;
		v2_dev[row * (M - K + 1) + col] = v2value;
		v3_dev[row * (M - K + 1) + col] = v3value;
		v4_dev[row * (M - K + 1) + col] = v4value;

		// Compute the square of Euclidean distance between the template and every patch and place the results in right place.
		X_dev[row * (M - K + 1) + col] = powf(v1value - vt1value, 2) + powf(v2value - vt2value, 2) + powf(v3value - vt3value, 2) + powf(v4value - vt4value, 2);

	}

}