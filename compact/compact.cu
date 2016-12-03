#include <stdio.h>

__global__ void init_numbers(int *d_in) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  d_in[index] = index+1;
}

__global__ void is_even_predicate(const int *d_in, int *d_result) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  d_result[index] = d_in[index] % 2 == 0;
}

__global__ void primitive_sum_scan(const int *d_in, int *d_out) {
  extern __shared__ int sdata[];
  int index = threadIdx.x;

  // Make it exclusive
  sdata[index] = (index > 0) ? d_in[index-1] : 0;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset*=2) {
    int value = sdata[index];
    int neighbour = (offset <= index) ? sdata[index - offset] : 0;
    __syncthreads();
    sdata[index] = value + neighbour;
    __syncthreads();
  }

  d_out[index] = sdata[index];
}

__global__ void scatter(const int *d_numbers, const int *d_predicates, const int *d_positions, int *d_out) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (d_predicates[index]) {
    int position = d_positions[index];
    d_out[position] = d_numbers[index];
  }
}


int main(int argc, char const **argv) {
  int THREAD_COUNT = 1024;
  int BLOCK_COUNT = 1;

  int *d_numbers, *d_predicates, *d_scatter_positions, *d_result;
  cudaMalloc((void **) &d_numbers, sizeof(int) * BLOCK_COUNT * THREAD_COUNT);
  cudaMalloc((void **) &d_predicates, sizeof(int) * BLOCK_COUNT * THREAD_COUNT);
  cudaMalloc((void **) &d_scatter_positions, sizeof(int) * BLOCK_COUNT * THREAD_COUNT);
  cudaMalloc((void **) &d_result, sizeof(int) * BLOCK_COUNT * THREAD_COUNT);


  init_numbers<<<BLOCK_COUNT, THREAD_COUNT>>>(d_numbers);
  is_even_predicate<<<BLOCK_COUNT, THREAD_COUNT>>>(d_numbers, d_predicates);
  primitive_sum_scan<<<BLOCK_COUNT, THREAD_COUNT, BLOCK_COUNT * THREAD_COUNT * sizeof(int)>>>(d_predicates, d_scatter_positions);
  scatter<<<BLOCK_COUNT, THREAD_COUNT>>>(d_numbers, d_predicates, d_scatter_positions, d_result);

  int h_result[BLOCK_COUNT * THREAD_COUNT];
  cudaMemcpy(h_result, d_result, sizeof(int) * BLOCK_COUNT * THREAD_COUNT, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 512; i++) {
    printf("%i\n", h_result[i]);
  }

  cudaFree(d_numbers);
  cudaFree(d_predicates);
  cudaFree(d_scatter_positions);
  return 0;
}
