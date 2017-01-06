#include <stdio.h>


__global__ void init_numbers(int *d_numbers, int value, int size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    d_numbers[index] = value;
  }
}


__global__ void local_blelloch_prefix_sum(
  int *d_input,
  int input_size,
  int *d_output
) {
  extern __shared__ int temp[];
  int tid = threadIdx.x;
  int index = tid + blockIdx.x * blockDim.x;

  // Copy to shared memory.
  // If index exceeds the input size use zero, so we keep a length which is
  // a power of two.
  temp[tid] = (index < input_size) ? d_input[index] : 0;
  __syncthreads();

  // Create a binary tree, that reduces the (local) elements
  for (int k = 2; k <= blockDim.x; k <<= 1) {
    if ((tid+1) % k == 0) {
      int offset = k >> 1;
      temp[tid] = temp[tid] + temp[tid - offset];
    }
    __syncthreads();
  }

  // Set the last (local) element to zero
  if (tid == (blockDim.x-1)) {
    temp[tid] = blockIdx.x > 0 ? temp[0] : 0;
  }
  __syncthreads();

  // Perform downsweep
  for (int k = blockDim.x; k > 1; k >>= 1) {
    if ((tid+1) % k == 0) {
      int offset = k >> 1;
      float old_value = temp[tid];
      int left_child_index = tid - offset;
      temp[tid] = temp[tid] + temp[left_child_index];
      temp[left_child_index] = old_value;
   }
   __syncthreads();
  }

  // Copy the results if the index does not execeed the input size
  if (index < input_size) {
    d_output[index] = temp[tid];
  }
}


__global__ void add_block_sums(
  int *d_input,
  int input_size,
  int *const d_block_sums
) {
  // Load the corresponding sum into shared memory
  __shared__ int block_sum;
  if (threadIdx.x == 0) {
    block_sum = d_block_sums[blockIdx.x];
  }
  __syncthreads();

  // Assert that the index does not execeed the input size
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input_size) {
    // Increment all elements in this block by the corresponding block sum
    d_input[index] = d_input[index] + block_sum;
  }
}


__global__ void gather_every_nth(
  int *const d_input,
  int *d_output,
  int n
) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int element_index = (n-1) + (index * n);
  d_output[index] = d_input[element_index];
}

/**
 * A two staged exclusive sum scan that can handle an arbitrary number of
 * elements between 0 and 1024^2.
 *
 * Stage 1: 1024 single Blelloch scans with 1024 elements each
 * Stage 2: 1 Blelloch scan of the 1024 maximums (as required)
 */
void prefix_sum(
  int *d_input,
  int input_size,
  int *d_output
) {
  // Always use 1024 threads due to Blelloch only works on lenghts that are a
  // power of two.
  const int THREAD_COUNT = 1024;
  const int BLOCK_COUNT = ceil((double) input_size / (double) THREAD_COUNT);

  // Execute stage 1
  int shared_size = THREAD_COUNT * sizeof(int);
  local_blelloch_prefix_sum<<<BLOCK_COUNT, THREAD_COUNT, shared_size>>>(d_input, input_size, d_output);

  // Execute state 2 (if necessary)
  if (BLOCK_COUNT > 1) {
    // Allocate memory for the block sums
    int *d_block_sums;
    cudaMalloc((void **) &d_block_sums, sizeof(int) * BLOCK_COUNT);

    // Gather local maximums
    gather_every_nth<<<1, BLOCK_COUNT>>>(d_output, d_block_sums, THREAD_COUNT);

    // Scan the final sums of the blocks
    shared_size = THREAD_COUNT * sizeof(int);
    local_blelloch_prefix_sum<<<1, THREAD_COUNT, shared_size>>>(d_block_sums, BLOCK_COUNT, d_block_sums);

    // Add the block sums to the input items
    add_block_sums<<<BLOCK_COUNT, THREAD_COUNT>>>(d_output, input_size, d_block_sums);

    // Free block sums
    cudaFree(d_block_sums);
  }
}


int main(int argc, char **argv) {
  int ELEMENT_COUNT = 1030;

  // Initialization
  int *d_numbers, *d_scan_result;
  cudaMalloc((void **) &d_numbers, ELEMENT_COUNT * sizeof(int));
  cudaMalloc((void **) &d_scan_result, ELEMENT_COUNT * sizeof(int));
  init_numbers<<<2, 1024>>>(d_numbers, 1, ELEMENT_COUNT);

  // Scan
  // simple_blelloch_scan<<<1, ELEMENT_COUNT, ELEMENT_COUNT * sizeof(float)>>>(d_numbers, d_scan_result, ELEMENT_COUNT);
  prefix_sum(d_numbers, ELEMENT_COUNT, d_scan_result);

  // Copy result
  int result[ELEMENT_COUNT];
  cudaMemcpy(result, d_scan_result, ELEMENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < ELEMENT_COUNT; i++) {
    printf("%i%s", result[i], (i % 20 == 19) ? "\n" : "\t");
  }
  printf("\n");

  cudaFree(d_numbers);
  cudaFree(d_scan_result);

  cudaDeviceSynchronize();

  return 0;
}
