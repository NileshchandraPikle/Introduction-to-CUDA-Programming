/*
   Program name: HelloGPU.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418  
         Webpage: https://piklenileshchandra.wixsite.com/personal
    
    Purpose: To demonstarte 
             1. How to write a simple CUDA program
             2. Calling CUDA kernel  
             3. How to compile & run CUDA program

    Discrition: Given two functions helloCPU() and helloGPU()
                helloCPU() function is executed on CPU and prints message
                "Hello from the CPU."
                
                helloGPU() function is executed on GPU and prints message
                "Hello from the GPU." 

                 To compile nvcc -arch=sm_35 1_HelloGPU.cu
                 To Run     ./a.out
*/
#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}
__global__ void helloGPU()
{
  printf("Hello also from the GPU.\n");
}

int main()
{

  helloCPU();
  // First #thread blocks Second = # threads per block 
  helloGPU<<<2,32>>>();
  cudaDeviceSynchronize();
  return 0;
}

