#include "kernel.cuh"

__global__ void sum_row(float *img, float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height) // width of image
{
	
	__shared__ float buffer_l1[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_l2[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_lx[BLOCK_HEIGHT][BLOCK_WIDTH];
	__shared__ float buffer_ly[BLOCK_HEIGHT][BLOCK_WIDTH];

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int m, cursor;
	float temp_l1, temp_l2, temp_lx, temp_ly;
	float last_l1, last_l2, last_lx, last_ly;

	for (m = 0; m < (I_width / BLOCK_WIDTH + (I_width%BLOCK_WIDTH>0)); m++)
	{
		int index_in = row * I_width + blockDim.x * m + threadIdx.x;
		int thread_width = BLOCK_WIDTH;
		if ((m == I_width / BLOCK_WIDTH) && I_width%BLOCK_WIDTH > 0)
			thread_width = I_width%BLOCK_WIDTH;

		if (index_in < (I_width * I_height) && (threadIdx.x < thread_width))
		{
			// Copy from global memory to shared memory (? if buffer length longer than img)
			
			// Dealing with images whose width is larger than the buffer width
			if (m > 0) {
				last_l1 = buffer_l1[threadIdx.y][BLOCK_WIDTH - 1];
				last_l2 = buffer_l2[threadIdx.y][BLOCK_WIDTH - 1];
				last_lx = buffer_lx[threadIdx.y][BLOCK_WIDTH - 1];
				last_ly = buffer_ly[threadIdx.y][BLOCK_WIDTH - 1];

				//printf("m is %d, Last_l1 is %f\n", m, last_l1);
			}
			
			buffer_l1[threadIdx.y][threadIdx.x] = img[index_in];
			buffer_l2[threadIdx.y][threadIdx.x] = powf(img[index_in], 2);
			buffer_lx[threadIdx.y][threadIdx.x] = img[index_in] * (blockDim.x * m + threadIdx.x);
			buffer_ly[threadIdx.y][threadIdx.x] = img[index_in] * row;

			// Prefix sum for current array
			for (cursor = 1; cursor <= ceilf(log2f(thread_width)); cursor++) 
			{
				if ((threadIdx.x >= __float2int_rd(powf(2,cursor-1))) && (threadIdx.x < thread_width))
				{
					temp_l1 = buffer_l1[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_l2 = buffer_l2[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_lx = buffer_lx[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					temp_ly = buffer_ly[threadIdx.y][threadIdx.x - __float2int_rd(powf(2, cursor - 1))];
					
					//printf("temp l1 is %f\n", temp_l1);
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
			// Option 2
			if (m > 0) {
				buffer_l1[threadIdx.y][threadIdx.x] += last_l1;
				buffer_l2[threadIdx.y][threadIdx.x] += last_l2;
				buffer_lx[threadIdx.y][threadIdx.x] += last_lx;
				buffer_ly[threadIdx.y][threadIdx.x] += last_ly;
			}

			l1_dev[index_in] = buffer_l1[threadIdx.y][threadIdx.x];
			l2_dev[index_in] = buffer_l2[threadIdx.y][threadIdx.x];
			lx_dev[index_in] = buffer_lx[threadIdx.y][threadIdx.x];
			ly_dev[index_in] = buffer_ly[threadIdx.y][threadIdx.x];
		}
	}
}

__global__ void sum_col(float *l1_dev, float *l2_dev, float *lx_dev, float *ly_dev, int I_width, int I_height)
{
	__shared__ float buffer_l1[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_l2[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_lx[BLOCK_HEIGHT2][BLOCK_WIDTH2];
	__shared__ float buffer_ly[BLOCK_HEIGHT2][BLOCK_WIDTH2];

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int m, cursor; 
	float temp_l1, temp_l2, temp_lx, temp_ly;
	float last_l1, last_l2, last_lx, last_ly;

	for (m = 0; m < (I_height / BLOCK_HEIGHT2 + (I_height%BLOCK_HEIGHT2>0)); m++)
	{
		//printf("m is %d\n", m);
		//int index_in = row * I_width + blockDim.x * m + threadIdx.x;
		int index_in = (blockDim.y * m + threadIdx.y) * I_width + col;
		int thread_height = BLOCK_HEIGHT2;
		if ((m == I_height / BLOCK_HEIGHT2) && I_height%BLOCK_HEIGHT2 > 0)
			thread_height = I_height%BLOCK_HEIGHT2;
		//if ((I_width % BLOCK_WIDTH2 > 0) && (col >= I_width))
		//printf("col is %d\n", col);
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

			// printf("Value is %f\n", l1_dev[index_in]);
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

			if (m > 0) {
				buffer_l1[threadIdx.y][threadIdx.x] += last_l1;
				buffer_l2[threadIdx.y][threadIdx.x] += last_l2;
				buffer_lx[threadIdx.y][threadIdx.x] += last_lx;
				buffer_ly[threadIdx.y][threadIdx.x] += last_ly;
			}
				l1_dev[index_in] = buffer_l1[threadIdx.y][threadIdx.x];
				l2_dev[index_in] = buffer_l2[threadIdx.y][threadIdx.x];
				lx_dev[index_in] = buffer_lx[threadIdx.y][threadIdx.x];
				ly_dev[index_in] = buffer_ly[threadIdx.y][threadIdx.x];
				
		}
	}
}