#include <stdio.h>


__global__ void init_numbers(float *d_numbers, float value) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  d_numbers[index] = value;
}


__global__ void simple_blelloch_scan(float *d_in, float *d_out, int n) {
  extern __shared__ float temp[];
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  // Copy to shared memory
  temp[index] = d_in[index];
  __syncthreads();

  // Build tree
  for (int k = 2; k <= n; k <<= 1) {
    if ((index + 1) % k == 0) {
      int offset = k>>1;
      temp[index] = temp[index] + temp[index - offset];
    }
    __syncthreads();
  }

  // Set element n to zero
  if (index == (n-1)) {
    temp[index] = 0;
  }
  __syncthreads();

  // Perform downsweep
  for (int k = n; k > 1; k >>= 1) {
    if ((index + 1) % k == 0) {
      int offset = k>>1;
      float old_value = temp[index];
      int left_child_index = index - offset;
      temp[index] = temp[index] + temp[left_child_index];
      temp[left_child_index] = old_value;
    }
    __syncthreads();
  }

  __syncthreads();
  d_out[index] = temp[index];
}


int main(int argc, char **argv) {
  int ELEMENT_COUNT = 1024;

  // Initialization
  float *d_numbers, *d_scan_result;
  cudaMalloc((void **) &d_numbers, ELEMENT_COUNT * sizeof(float));
  cudaMalloc((void **) &d_scan_result, ELEMENT_COUNT * sizeof(float));
  init_numbers<<<1, ELEMENT_COUNT>>>(d_numbers, 1.0);

  // Scan
  simple_blelloch_scan<<<1, ELEMENT_COUNT, ELEMENT_COUNT * sizeof(float)>>>(d_numbers, d_scan_result, ELEMENT_COUNT);

  // Copy result
  float result[ELEMENT_COUNT];
  cudaMemcpy(result, d_scan_result, ELEMENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < ELEMENT_COUNT; i++) {
    printf("%.1f%s", result[i], (i % 20 == 19) ? "\n" : "\t");
  }
  printf("\n");

  cudaFree(d_numbers);
  cudaFree(d_scan_result);

  cudaDeviceSynchronize();

  return 0;
}
