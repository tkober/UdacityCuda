// Radix Test

#include <stdio.h>

__global__ void binary_histogram(
  unsigned int *const d_input,
  int input_size,
  int item_count,
  int *d_histograms,
  int bit
) {
  // Initialize lokal histogram
  int bins[2];
  bins[0] = 0;
  bins[1] = 0;

  // The first item this thread is responsible for
  int start_index = threadIdx.x * item_count;
  // Process all items, this thread is thread is responsible for
  for (int index = start_index; index < start_index + item_count; index++) {
    // If the index exceeds the number of items, stop
    if (index >= input_size) {
      break;
    }

    // Otherwise...
    unsigned int mask = 1 << bit;
    unsigned int item = d_input[index];
    // ...check whether the requested bit is set...
    int bin = (mask & item) > 0;
    // ...and increment the appropriate bin
    bins[bin] = bins[bin] + 1;
  }

  // Finally copy the local histogram to the global list
  int histogram_index = threadIdx.x * 2;
  d_histograms[histogram_index] = bins[0];
  d_histograms[histogram_index+1] = bins[1];
}


__global__ void simple_reduce_historgrams(
  int *d_histograms,
  int *d_global_histogram,
  int n,
  int bins
) {
  extern __shared__ int sdata[];
  int thread = threadIdx.x;

  // Copy all histograms to shared memory
  for (int bin = 0; bin < bins; bin++) {
    sdata[thread * bins + bin] = d_histograms[thread * bins + bin];
  }
  __syncthreads();

  // Process list as a binary tree, bottom up
  for (int k = 2; k <= n; k <<= 1) {
    // If the current thread is part of the current level
    // update the value by adding the value of the
    // left child
    if ((thread+1) % k == 0) {
      // The distance to left child is the node distance on the current level
      // devided by two
      int offset = k>>1;
      // Add all bins of the left child tot the corresponding bins of the thread
      for (int bin = 0; bin < bins; bin++) {
        sdata[thread * bins + bin] = sdata[thread * bins + bin] + sdata[(thread - offset) * bins + bin];
      }
    }
    __syncthreads();
  }

  // The result in the root is at the last position of the list
  if (thread == (n-1)) {
    // Copy all bins
    for (int bin = 0; bin < bins; bin++) {
      d_global_histogram[bin] = sdata[thread * bins + bin];
    }
  }
}


void historgram_for_bit(
  unsigned int *const d_values,
  int *d_result,
  int value_count,
  int bit
) {
  // Due to primitive global histogram implementation,
  // only 1 block with 1024 threads
  const int MAX_THREADS = 1024;
  const int HISTOGRAM_COUNT = MAX_THREADS;
  // The number of items each thread needs to process
  const int ITEMS_PER_HISTOGRAM = (value_count / HISTOGRAM_COUNT) + 1;
  // Number of bins, for binary digits --> 2
  const int BIN_COUNT = 2;

  int *d_histograms;
  // Calculate HISTOGRAM_COUNT histograms, each containing
  // the bins for ITEMS_PER_HISTOGRAM items
  cudaMalloc((void **) &d_histograms, sizeof(int) * HISTOGRAM_COUNT * BIN_COUNT);
  binary_histogram<<<1, HISTOGRAM_COUNT>>>(d_values, value_count, ITEMS_PER_HISTOGRAM, d_histograms, bit);

  // Reduce the local histograms to one global
  int shared_size = sizeof(int) * HISTOGRAM_COUNT * BIN_COUNT;
  simple_reduce_historgrams<<<1, HISTOGRAM_COUNT, shared_size>>>(d_histograms, d_result, HISTOGRAM_COUNT, BIN_COUNT);

  cudaFree(d_histograms);
}


__global__ void calculate_predicates(
  unsigned int *const d_values,
  int value_count,
  int *d_result,
  unsigned int mask,
  int matching_value
) {
  // The index this thread is responsible for
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // Check whether index is valid (last block)
  if (index < value_count) {
    // Check if the corresponding bit is set
    int bit_set = (mask & d_values[index]) > 0;
    // Invert the result if interested in 0s
    d_result[index] = (matching_value == 0) ? bit_set^1 : bit_set;
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


void calculate_relativ_positions(
  unsigned int *const d_values,
  int value_count,
  int *d_output,
  int bit,
  int matching_value
) {
  // Create a bit mask to check the requested bit
  const unsigned int MASK = 1<<bit;
  const int THREAD_COUNT = 1024;
  const int BLOCK_COUNT = (value_count / THREAD_COUNT) + 1;

  // Calculate the predicates for the values
  calculate_predicates<<<BLOCK_COUNT, THREAD_COUNT>>>(d_values, value_count, d_output, MASK, matching_value);
  ///
  // int h_predicates[4];
  // cudaMemcpy(h_predicates, d_output, sizeof(int) * 4, cudaMemcpyDeviceToHost);
  // printf("pred. b=%i --> [%i, %i, %i, %i]\n", matching_value, h_predicates[3], h_predicates[2], h_predicates[1], h_predicates[0]);
  ///

  // Execute an exclusive (prefix-)sum scan to get the relative positions
  prefix_sum(d_output, value_count, d_output);
}


__global__ void reorder_scatter(
  unsigned int* const from_values,
  unsigned int* const from_positions,
  unsigned int* const to_values,
  unsigned int* const to_positions,
  int size,
  int start_position,
  int *d_relative_positions,
  int bit,
  int bit_set
) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    int is_bit_set = (from_values[index] & (1 << bit)) > 0;
    if (is_bit_set == bit_set) {
      int new_index = start_position + d_relative_positions[index];
      to_values[new_index] = from_values[index];
      to_positions[new_index] = from_positions[index];
    }
  }
}


///
int get_bit(int n, int bit) {
  return (n & (1<<bit)) > 0;
}
///


int main(int argc, char const **argv) {
  const size_t numElems = 4;
  unsigned int h_numbers[] = { 7, 14, 4, 1};

  unsigned int *d_inputVals;
  unsigned int *d_inputPos;
  unsigned int *d_outputVals;
  unsigned int *d_outputPos;

  cudaMalloc((void **) &d_inputVals, sizeof(unsigned int) * numElems);
  cudaMalloc((void **) &d_inputPos, sizeof(unsigned int) * numElems);
  cudaMalloc((void **) &d_outputVals, sizeof(unsigned int) * numElems);
  cudaMalloc((void **) &d_outputPos, sizeof(unsigned int) * numElems);

  cudaMemcpy(d_inputVals, h_numbers, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inputPos, h_numbers, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice);
  printf("INPUT:\n");
  printf("i\tval\tbits\n");
  printf("-------------------\n");
  for (int j = 0; j < numElems; j++) {
    printf("%i\t%i\t%i %i %i %i\n", j, h_numbers[j], get_bit(h_numbers[j], 3), get_bit(h_numbers[j], 2), get_bit(h_numbers[j], 1), get_bit(h_numbers[j], 0));
  }
  printf("\n\n");

  const int THREAD_COUNT = min((int) numElems, 1024);
  const int BLOCK_COUNT = ceil(numElems / THREAD_COUNT);
  // The number if steps in the Radix sort,
  // corresponding to the number of bits/digits
  const int LENGHT = 4;//8 * sizeof(unsigned int);

  // Allocate device memory for a binary histogram
  int *d_histogram;
  cudaMalloc((void **) &d_histogram, sizeof(int) * 2);

  // Allocate device memory for the relative positions
  int *d_relative_positions;
  cudaMalloc((void **) &d_relative_positions, sizeof(int) * numElems);

  int h_histogram[2];
  h_histogram[0] = 0;

  // Process one radix step per bit/digit
  int i = 0;
  for (i = 0; i < LENGHT; i++) {
    // Alternate on every step:
    // step even: I -> O
    // step odd:  O -> I
    int is_step_even = (i % 2) == 0;

    unsigned int *const d_from_val  = is_step_even ? d_inputVals : d_outputVals;
    unsigned int *const d_from_pos  = is_step_even ? d_inputPos : d_outputPos;
    unsigned int *const d_to_val    = is_step_even ? d_outputVals : d_inputVals;
    unsigned int *const d_to_pos    = is_step_even ? d_outputPos : d_inputPos;

    // Create a histogram for the occurrences of 0 and 1
    // for the current bit/digit
    historgram_for_bit(d_from_val, d_histogram, numElems, i);
    cudaMemcpy(h_histogram+1, d_histogram, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate relative positions for bits/digit that are equal to 0
    calculate_relativ_positions(d_from_val, numElems, d_relative_positions, i, 0);

    // Scatter the items to their new position
    reorder_scatter<<<BLOCK_COUNT, THREAD_COUNT>>>(d_from_val, d_from_pos,
      d_to_val, d_to_pos, numElems, h_histogram[0], d_relative_positions, i, 0);

    // Calculate relative positions for bits/digit that are equal to 1
    calculate_relativ_positions(d_from_val, numElems, d_relative_positions, i, 1);

    // Scatter the items to their new position
    reorder_scatter<<<BLOCK_COUNT, THREAD_COUNT>>>(d_from_val, d_from_pos,
      d_to_val, d_to_pos, numElems, h_histogram[1], d_relative_positions, i, 1);


    unsigned int h_result_vals[numElems];
    unsigned int h_result_pos[numElems];
    cudaMemcpy(h_result_vals, d_to_val, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_pos, d_to_pos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost);

    printf("i\tval\tbits\n");
    printf("-------------------\n");
    for (int j = 0; j < numElems; j++) {
      printf("%i\t%i\t%i %i %i %i\n", j, h_result_vals[j], get_bit(h_result_vals[j], 3), get_bit(h_result_vals[j], 2), get_bit(h_result_vals[j], 1), get_bit(h_result_vals[j], 0));
    }
    printf("\n");
  }

  // if radix step count is even copy once again, so the result is in d_output
  if (i % 2 == 0) {
    cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  }

  cudaFree(d_histogram);
  cudaFree(d_relative_positions);
  cudaFree(d_inputVals);
  cudaFree(d_inputPos);
  cudaFree(d_outputVals);
  cudaFree(d_outputPos);

  return 0;
}
