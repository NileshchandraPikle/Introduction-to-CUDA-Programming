/*
   Program name: HelloGPU_ThreadOrganization.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418 
         Webpage: https://piklenileshchandra.wixsite.com/personal
    
    Purpose: To demonstarte 
             1. How to write CUDA program
             2. Calling CUDA kernel  
             3. How to compile & run CUDA program
             4. How to retrieve thread attributes such as thread Ids, blockIds and block dimension

    Discrition: 
                * Given two functions helloCPU() and helloGPU()
                  helloCPU() function is executed on CPU and prints message
                  "Hello from the CPU."
                
                * helloGPU() function is executed on GPU and prints message
                  "Hello from the GPU." 
                * __global__ keyword before a function indicates that function to be executed on GPU  
                * <<<numT, numB>>> this specifies the number of threads per block (numT) and
                  number of thread blocks (numB) lunched for the function/kernel helloGPU() 
                * threadIdx.x gives thread identification number BUT LOCAL TO THREAD BLOCK,
                * blockIdx.x  gives block identification number, 
                * blockDim.x gives number of threads in block

                 To compile nvcc -arch=sm_35 1_HelloGPU_ThreadOrganization.cu
                 To Run     ./a.out
*/
#include <stdio.h>

void helloCPU()
{
 printf("Hello from the CPU.\n");
}
__global__ void helloGPU()
{
  int tid  = threadIdx.x; // thread number 
  int bid  = blockIdx.x;  // block number 
  int bdim = blockDim.x;  // number of threads per block
  printf("threadID = %d   blockId = %d  block Dimension = %d \n", tid, bid, bdim);
}

int main()
{
  helloCPU();
  // helloGPU<<<Num_Thread_Blocks, Num_Threads_Per_Block>>>(); 
  helloGPU<<<3,3>>>();     // kernel launch with 
  cudaDeviceSynchronize(); // To synchronize CPU and GPU
}
