/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>

#include "utils.h"


__global__ void reduce_max_min(const float *d_in, float *d_out, bool useMax) {
  extern __shared__ float sdata[];
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int myId = threadIdx.x;

  // Put whole block in shared memory
  sdata[myId] = d_in[index];
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i>>=1) {
    if (myId < i) {
      sdata[myId] = useMax ?
      (max(sdata[myId], sdata[myId+i])) :
      (min(sdata[myId], sdata[myId+i])) ;
    }
    __syncthreads();
  }

  if (myId == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}


__global__ void init_bins(int *d_bins) {
  d_bins[threadIdx.x + blockIdx.x * blockDim.x] = 0;
}


__global__ void primitive_histogram(
  const float *d_in,
  int *d_bins,
  const float min,
  const float range,
  const int binCount) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int bin = (d_in[index] - min) / range * binCount;
    atomicAdd(&(d_bins[bin]), 1);
}


__global__ void primitive_prefix_sum_scan(
  const int *d_in,
  unsigned int *d_out
) {
  // Using Hillis-Steel algorithm
  extern __shared__ int sd[];
  int index = threadIdx.x;

  sd[index] = index > 0 ? d_in[index-1] : 0;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset*=2) {
    int value = sd[index];
    int neighbour = index >= offset ? sd[index-offset] : 0;
    __syncthreads();
    sd[index] = value + neighbour;
    __syncthreads();
  }

  d_out[index] = sd[index];
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  const int BLOCK_SIZE = numCols;
  const int BLOCK_COUNT = numRows;

  float *d_intermediate, *d_logLumMax, *d_logLumMin;

  cudaMalloc((void **) &d_intermediate, sizeof(float) * BLOCK_COUNT * BLOCK_SIZE);
  cudaMalloc((void **) &d_logLumMax, sizeof(float));
  cudaMalloc((void **) &d_logLumMin, sizeof(float));

  // Reduce to maximum
  reduce_max_min<<<BLOCK_COUNT, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_logLuminance, d_intermediate, true);
  reduce_max_min<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_intermediate, d_logLumMax, true);

  // Reduce to minimum
  reduce_max_min<<<BLOCK_COUNT, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_logLuminance, d_intermediate, false);
  reduce_max_min<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_intermediate, d_logLumMin, false);

  cudaMemcpy(&max_logLum, d_logLumMax, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&min_logLum, d_logLumMin, sizeof(float), cudaMemcpyDeviceToHost);

  float logLumRange = max_logLum - min_logLum;

  // Create histogram
  int *d_bins;
  cudaMalloc((void **) &d_bins, numBins * sizeof(int));
  init_bins<<<1, numBins>>>(d_bins);
  primitive_histogram<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_logLuminance, d_bins, min_logLum, logLumRange, numBins);

  // CDF scan
  primitive_prefix_sum_scan<<<1, numBins, numBins * sizeof(int)>>>(d_bins, d_cdf);

  cudaFree(d_intermediate);
  cudaFree(d_logLumMax);
  cudaFree(d_logLumMin);
  cudaFree(d_bins);

  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you) TODO       */


}
