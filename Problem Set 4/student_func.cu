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
  unsigned int * const d_input,
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
  unsigned int *d_values,
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


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  // The number if steps in the Radix sort,
  // corresponding to the number of bits/digits
  const int LENGHT = 8 * sizeof(unsigned int);

  // Allocate device memory for a binary histogram
  int *d_histogram;
  cudaMalloc((void **) &d_histogram, sizeof(int) * 2);

  // Process one radix step per bit/digit
  for (int i = 0; i < LENGHT; i++) {

    // Create a histogram for the occurrences of 0 and 1
    // for the current bit/digit
    historgram_for_bit(d_inputVals, d_histogram, numElems, i);

  }

  cudaFree(d_histogram);
}
