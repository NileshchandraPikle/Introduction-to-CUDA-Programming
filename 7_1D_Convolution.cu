/*
   Program name: 1D_Convolution.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418   
         Webpage: https://piklenileshchandra.wixsite.com/personal

    Purpose: To perform 1D convolution using mask.
             This type of computation is performed in heat conduction or image convolution. Ofcourse 
             the aforementioned applications are in higher dimensions such as 2D or 3D. The same idea 
             of 1D convolution can be extended for higher dimensions as well.

    Description: 1D convolution is performed on an array of size array_size by using a 1D mask of size 
                 mask_size 
                 For simplicity both input array and mask are initialized to 1 as it will be easy to 
                 validate the results 
           Link: https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

__global__ void OneDconvolutionKernel(int *d_input, int *d_output, int *d_mask, int array_size, int mask_size)
{
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, sum = 0;
  if(gid < array_size)
  {
    for(i = -(mask_size/2); i <= (mask_size/2); i++)
    {
      if(  ((gid + i) >= 0) && ((gid + i) < array_size))
      {
       sum += d_input[gid + i] * d_mask[i + (mask_size/2)];
      }
    }
    d_output[gid] = sum;
  } 
}

int main()
{
  int mask_size = 3;        // size of mask
  unsigned int array_size = 2<<24;   // size of input array

  int i,h_mask[mask_size],j=0,sum=0;
  
  for(i = 0; i < mask_size; i++)
  {
    h_mask[i] = 1;                     // Initialize mask
  }
  //for(i = 0; i < mask_size; i++)
  //printf(" h_mask[%d] = %d \n ",i,mask[i]);

  int *h_input = (int *)malloc(array_size * sizeof(int)); // input array memory allocation
  int *h_output = (int *)malloc(array_size * sizeof(int)); // Output array to store the results
  int *hg_output = (int *)malloc(array_size * sizeof(int)); // Output array to store the results from GPU output

  if(!h_input)
  {
   printf("\n ERROR: Memory allocation to array h_input!!! \n"); // Error handling code memory allocation
  }
  if(!h_output)
  {
   printf("\n ERROR: Memory allocation to array h_output!!! \n"); // Error handling code memory allocation
  }

  if(!hg_output)
  {
   printf("\n ERROR: Memory allocation to array h_output!!! \n"); // Error handling code memory allocation
  }

  for(i = 0; i < array_size; i++)
  {
   h_input[i] = 1;   // input array initialization to 1
   h_output[i] = 0;
   hg_output[i] = 0;
  }
  /**********************************************************************************************/
  /* Sequential Program to perform convolution operation on input vector h_input and store results
     in h_output vector. Mask size 'mask_size' determines range of neighboring numbers considered 
     to perform convolution.*/
  /**********************************************************************************************/
   clock_t t;       // C programming clock functions
   t = clock();     // Record start time 
  for(i = 0; i < array_size; i++)
  {
    sum = 0;
    for(j = -(mask_size/2); j <= (mask_size/2); j++)
    {
      if( ((i + j) >= 0) && ((i + j) < array_size) )  // To handle boundary conditions
      {
       sum += h_input[i+j] * h_mask[j+(mask_size/2)];
      }
    }
   h_output[i] = sum;
  }
   t = clock() - t;  // Record end time and find difference
   double CPUtime_taken = ((double)t)/CLOCKS_PER_SEC;
  
  //for(i = 0; i < array_size; i++)
 // printf(" h_output[%d] = %d \n ",i,h_output[i]);


  /*
     Parallel program using CUDA
     1. Declare GPU side variables
     2. Allocate memory to GPU side variables
     3. Copy data from CPU to GPU
     4. Launch Kernel
     5. Copy results back from GPU to CPU
     6. Free GPU memories  
  */

 /* 
    1. Declare GPU side variables :- 
       d_input to store input array d_output to store result
       d_mask to store mask on GPU
 */
  int *d_input,*d_output, *d_mask; 

 /* 
    2. Allocate memory for GPU variables
 */

  cudaMalloc((void **)&d_input, array_size*sizeof(int));
  cudaMalloc((void **)&d_output, array_size*sizeof(int));
  cudaMalloc((void **)&d_mask, mask_size*sizeof(int));

 /* 
    3. Copy data from CPU to GPU
 */

  cudaMemcpy(d_input, h_input, array_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, array_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask, mask_size*sizeof(int), cudaMemcpyHostToDevice);

  /*
     4.1 Decide thread configuration for kerenl launch using variables numThreads and numBlocks
         numThreads indicates number of threads per thread block and numBlocks indicates number of
         blocks in a grid. Note that threads within a thread block and thread blocks are both organized
         in 1D fashion. The number of blocks depends on numThreads and input array size i.e. "array_size" 
         The ceil function guarantees that "atleast" array_size number of threads will be launched
  */
  int numThreads, numBlocks; 
  numThreads = 32;
  numBlocks = ceil(array_size/(double)numThreads);
  /*
     4.2 OneDconvolutionKernel is launched using thread configuration as numBlocks and numThreads
         all GPU side variables are passed as call by reference by passing the address/pointer to the
         memory locations
  */

  float GPUelapsed = 0.0;  // To store final kernel execution time 
  cudaEvent_t start, stop; // Variables to record start and stop of kernel
  cudaEventCreate(&start); // Event create start
  cudaEventCreate(&stop);  // Event create stop

  cudaEventRecord(start, 0); // Record time at start variables
  OneDconvolutionKernel<<<numBlocks, numThreads>>>(d_input, d_output, d_mask, array_size, mask_size);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPUelapsed, start, stop);

  cudaEventDestroy(start); // Event destroy start
  cudaEventDestroy(stop);  // Event destroy stop
  printf("\n**************************************************************************\n");
  printf("\n Program name: 1D Convolution");
  printf("\n Elapsed Time on CPU is %f ms", CPUtime_taken*1000);
  printf("\n Elapsed Time on GPU is %0.10f ms",GPUelapsed);
  printf("\n Speedup = (CPU time)/(GPU time) = %f\n", (CPUtime_taken*1000)/GPUelapsed);
  printf("\n**************************************************************************\n");
  /*
     5. Copy results back from GPU to CPU. The GPU side results are stored in 'd_output' array are 
        copied back to CPU in 'hg_output'. Here we are using 'hg_output' array to store copy back 
        results from GPU output.
  */

   cudaMemcpy(hg_output, d_output, array_size * sizeof(int), cudaMemcpyDeviceToHost);

  /*
    Finally in order to validate the convolution results obtained from the device are validated by
    comparinf with the results obtained by sequential computations.
    The output generated from sequential convolution code is stored in 'h_output' array.
    The output generated from device convolution kernel is stored in 'hg_output' array.
    All entries are compared and if atleast one of the entry is not matching then results are wrong
  */
  int flag = 1;
  for(i = 0; i < array_size; i++)
  {
    //printf("\n h_output[%d] = %d    hg_output[%d] = %d ", i,h_output[i],i,hg_output[i]);
    if(h_output[i] != hg_output[i])
    {
      flag = 0;
    }
  }
  if(flag == 0)
  {
   printf("\n Results are wrong!!! \n Solution not matching!!!\n");
  }

// 6.1 Free GPU side memories
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_mask);

// 6.2 Free CPU side memories
free(h_input);
free(h_output);
free(hg_output);

  return(0);
}

