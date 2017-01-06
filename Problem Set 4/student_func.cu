//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


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
  const int BLOCK_COUNT = ceil(input_size / THREAD_COUNT);

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
    shared_size = BLOCK_COUNT * sizeof(int);
    local_blelloch_prefix_sum<<<1, BLOCK_COUNT, shared_size>>>(d_block_sums, input_size, d_block_sums);

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
  int *relative_positions,
  int bit,
  int bit_set
) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    int is_bit_set = (from_values[index] & (1 << bit)) > 0;
    if (is_bit_set == bit_set) {
      to_values[index] = from_values[index];
      to_positions[index] = from_positions[index];
    }
  }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  const int THREAD_COUNT = min((int) numElems, 1024);
  const int BLOCK_COUNT = ceil(numElems / THREAD_COUNT);
  // The number if steps in the Radix sort,
  // corresponding to the number of bits/digits
  const int LENGHT = 8 * sizeof(unsigned int);

  // Allocate device memory for a binary histogram
  int *d_histogram;
  cudaMalloc((void **) &d_histogram, sizeof(int) * 2);

  // Allocate device memory for the relative positions
  int *d_relative_positions;
  cudaMalloc((void **) &d_relative_positions, sizeof(int) * numElems);

  // Process one radix step per bit/digit
  int i;
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

    // Calculate relative positions for bits/digit that are equal to 0
    calculate_relativ_positions(d_from_val, numElems, d_relative_positions, i, 0);

    // Scatter the items to their new position
    reorder_scatter<<<BLOCK_COUNT, THREAD_COUNT>>>(d_from_val, d_from_pos,
      d_to_val, d_to_pos, numElems, d_histogram[0], d_relative_positions, i, 0);

    // Calculate relative positions for bits/digit that are equal to 1
    calculate_relativ_positions(d_inputVals, numElems, d_relative_positions, i, 1);

    // Scatter the items to their new position
    reorder_scatter<<<BLOCK_COUNT, THREAD_COUNT>>>(d_from_val, d_from_pos,
      d_to_val, d_to_pos, numElems, d_histogram[1], d_relative_positions, i, 1);
  }

  // if radix step count is even copy once again, so the result is in d_output
  if (i % 2 == 0) {
    cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  }

  cudaFree(d_histogram);
  cudaFree(d_relative_positions);
}
