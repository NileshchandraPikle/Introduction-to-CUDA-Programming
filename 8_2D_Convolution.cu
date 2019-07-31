/*
   Program name: 2D_Convolution.cu
    Author name: Dr. Nileshchandra Pikle
    
        Purpose: To perform 2D convolution using mask.
                 This type of computation is performed in heat conduction or image convolution.

    Description: 2D convolution is performed on an array of size array_size by using a 2D mask of size 
                 mask_size * mask_size  
                 For simplicity both input array and mask are initialized to 1 as it will be easy to 
                 validate the results 
      Functions:
                 1. printMat() :- Used to print matrix to debug and validate the resulst 
                 2. checkResulst() :- Used to compare results from CPU and GPU side output matrices
                 3. TwoDconvolutionKernel() :- Device kernel to execute 2D convolution operation in 
                    parallel
           Link: https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

__global__ void TwoDconvolutionKernel(int *d_input, int *d_output, int *d_mask, int array_size, int mask_size)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int k,l, sum;

   if( (row < array_size) && (col < array_size))
   { 
     sum = 0;
     for(k = -(mask_size/2); k <= (mask_size/2); k++)
     {
       for(l = -(mask_size/2); l <= (mask_size/2); l++)
       {
         if( ((row + k) >= 0) && ((row + k) < array_size) && ((col + l) >= 0) && ((col + l) < array_size)) 
         {
            sum += d_mask[(k+mask_size/2)*mask_size + (l+mask_size/2)] * d_input[(row+k)*array_size + (col +l)];
         }
       }
      d_output[row*array_size+col] = sum;
     } 
   }
}
 void printMat(int *matrix, int size)
 {
  int i,j;
      for(i = 0; i < size; i++)
      {
      for(j = 0; j < size; j++)
      {
        printf(" %d ",matrix[i*size+j]);
      }
      printf("\n");
    }
    printf("\n");
 }

 int checkResulst(int *h_output, int *hg_output, int array_size)
 {
  int i,j,flag = 1;
  for(i = 0; i < array_size;i++)
  {
    for(j = 0; j < array_size;j++)
    {
      if(h_output[i * array_size + j] != hg_output[i * array_size + j])
      {
       flag = 0;
      }
    }
  }
  return flag;
 }
int main()
{
  int mask_size = 5;        // size of mask
  unsigned int array_size = 2<<12;   // size of input array

  int i,j,k,l,h_mask[mask_size][mask_size],sum=0;
  
  for(i = 0; i < mask_size; i++)
  {
   for(j = 0; j < mask_size; j++)
   {
    h_mask[i][j] = 1;           // Initialize mask
   }
  }
  //for(i = 0; i < mask_size; i++)
  // for(j = 0; j < mask_size; j++)
  //printf(" h_mask[%d][%d] = %d \n ",i,j,mask[i][j]);

  int *h_input = (int *)malloc(array_size *array_size * sizeof(int)); // input array memory allocation
  int *h_output = (int *)malloc(array_size *array_size * sizeof(int)); // Output array to store the results
  int *hg_output = (int *)malloc(array_size *array_size * sizeof(int)); // Output array to store the results from GPU output

  if(!h_input)
  {
   printf("\n ERROR: Memory allocation to array h_input!!! \n");    // Error handling code memory allocation
  }
  if(!h_output)
  {
   printf("\n ERROR: Memory allocation to array h_output!!! \n");  // Error handling code memory allocation
  }

  if(!hg_output)
  {
   printf("\n ERROR: Memory allocation to array h_output!!! \n");  // Error handling code memory allocation
  }

  for(i = 0; i < array_size; i++)
  {
   for(j = 0; j < array_size; j++)
   {
     h_input[i*array_size+j]   = 1;   // input array initialization to 1
     h_output[i*array_size+j]  = 0;
     hg_output[i*array_size+j] = 0;
   }
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
   for(j = 0; j < array_size; j++)
   {
     sum = 0;
     for(k = -(mask_size/2); k <= (mask_size/2); k++)
     {
       for(l = -(mask_size/2); l <= (mask_size/2); l++)
       {
         if( ((i + k) >= 0) && ((i + k) < array_size) && ((j + l) >= 0) && ((j + l) < array_size)) 
         {
            sum += h_mask[k+mask_size/2][l+mask_size/2] * h_input[(i+k)*array_size + (j +l)];
         }
       }
      h_output[i*array_size+j] = sum;
     }
   }
  }
   
   //printMat(h_output,array_size);
   t = clock() - t;  // Record end time and find difference
   double CPUtime_taken = ((double)t)/CLOCKS_PER_SEC;
   printf("\n Time required for sequential execution is %f ms\n", CPUtime_taken*1000);

  /**********************************************************************************************/
  /* Parallel Program to perform convolution operation on input matrix h_input and store results
     in h_output matrix. Mask size 'mask_size x mask_size' determines range of neighboring numbers
     considered to perform convolution.*/
  /**********************************************************************************************/
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

    int *d_input, *d_output, *d_mask;
   /* 
    2. Allocate memory for GPU variables
   */
  cudaMalloc((void**)&d_input, array_size*array_size*sizeof(int));
  cudaMalloc((void**)&d_output, array_size*array_size*sizeof(int));
  cudaMalloc((void**)&d_mask, mask_size*mask_size*sizeof(int));
 /* 
    3. Copy data from CPU to GPU
 */
  cudaMemcpy(d_input, h_input, array_size*array_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, array_size*array_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask, mask_size*mask_size*sizeof(int), cudaMemcpyHostToDevice);
   
 /*
     4.1 Decide thread configuration for kerenl launch using variables threads and blocks
         threads indicates number of threads per thread block and blocks indicates number of
         blocks in a grid. Note that threads within a thread block and thread blocks are both organized
         in 2D fashion. The number of blocks depends on numThreads and input array size i.e.
         "array_size" 
         The ceil function guarantees that "atleast" array_size number of threads will be launched
  */

  dim3 threads(16,16,1);
  dim3 blocks( ceil(array_size/(double)threads.x), ceil(array_size/(double)threads.y),1);
  /*
     4.2 TwoDconvolutionKernel is launched using thread configuration as blocks and threads
         all GPU side variables are passed as call by reference by passing the address/pointer to the
         memory locations
  */

  float GPUelapsed = 0.0;  // To store final kernel execution time 
  cudaEvent_t start, stop; // Variables to record start and stop of kernel
  cudaEventCreate(&start); // Event create start
  cudaEventCreate(&stop);  // Event create stop

  cudaEventRecord(start, 0); // Record time at start variables

  TwoDconvolutionKernel<<<blocks,threads>>>(d_input, d_output, d_mask, array_size, mask_size);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPUelapsed, start, stop);

  cudaEventDestroy(start); // Event destroy start
  cudaEventDestroy(stop);  // Event destroy stop
  printf("\n Elapsed Time on GPU was %0.10f ms\n",GPUelapsed);
  printf("\n Speedup = (CPU time)/(GPU time) = %f\n", (CPUtime_taken*1000)/GPUelapsed);

  /*
     5. Copy results back from GPU to CPU. The GPU side results are stored in 'd_output' array are 
        copied back to CPU in 'hg_output'. Here we are using 'hg_output' array to store copy back 
        results from GPU output.
  */
  cudaMemcpy(hg_output, d_output, array_size *array_size * sizeof(int), cudaMemcpyDeviceToHost);


   /*
    Finally in order to validate the convolution results obtained from the device are validated by
    comparinf with the results obtained by sequential computations.
    The output generated from sequential convolution code is stored in 'h_output' matrix.
    The output generated from device convolution kernel is stored in 'hg_output' matrix.
    All entries are compared and if atleast one of the entry is not matching then results are wrong
  */
   
   int flag = checkResulst(h_output, hg_output, array_size);
   if(flag == 0)
   {
    printf("\n Caution: Something went wrong...Results are not matching!!!");
   }
  //printMat(hg_output,array_size);

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
