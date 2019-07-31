/*
   Program name: 2vecAddUsingOneBlock(N>numT).cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418 
         Webpage: https://piklenileshchandra.wixsite.com/personal
    
    Purpose: To perform vector addition using CUDA             

    Description: Vector addition using a single thread block (N > # threads per block)
 
                * Three vectors h_a, h_b, h_c are defined on Host with size N
                * Three vectors d_a, d_b, d_c are defined on Device with size N
                * Addition of h_a + h_b vectors performed on Host is stored in h_c 
                * Addition of d_a + d_b vectors performed on Device is stored in d_c
                * Result of Device (d_c) copied back to host in vector hr_c
                * To validate results h_c and hr_c are compared

                 To compile nvcc -arch=sm_35 2vecAddUsingOneBlock(N>numT).cu
                 To Run     ./a.out
*/
#include <stdio.h>
#include<time.h>
#include<math.h>


__global__ void vecAdd_kernel_UsingManyBlocks(int *d_a, int *d_b, int *d_c, int N)
{
  int tid = threadIdx.x;// threadIdx.x returns thread IDs local to each thread block
  int i;
  for(i = tid; i < N; i += blockDim.x) // To avoid out of order memory access by thread
  {
     d_c[i] = d_a[i] + d_b[i];
  }
}
void vecAddCPU(int *h_a, int *h_b, int *h_c, int N)
{
  int i;
  for(i = 0; i < N; i++)
  {
    h_c[i] = h_a[i] + h_b[i];
  }
}

int main()
{
 int i,N = 2<<24; 
  
  /*********************** Memory Allocation on CPU **********************************/
  
  int *h_a = (int *)malloc(N * sizeof(int)); // Memory allocation on CPU for vector h_a  input
  int *h_b = (int *)malloc(N * sizeof(int)); // Memory allocation on CPU for vector h_b  input
  int *h_c = (int *)malloc(N * sizeof(int)); // Memory allocation on CPU for vector h_c  output
  int *hr_c = (int *)malloc(N * sizeof(int));// Memory allocation on CPU for vector hr_c  output from GPU

  int *d_a, *d_b, *d_c; // Decleration of GPU variables
   
  /****************************** Error Handling ***********************************/
   if(!h_a)
   {
    printf("\n CPU: Error occured while allocating memory to h_a!");
   }
   if(!h_b)
   {
    printf("\n CPU: Error occured while allocating memory to h_b!");
   }
   if(!h_c)
   {
    printf("\n CPU: Error occured while allocating memory to h_c!");
   }

  /************************** Data initialization on CPU *****************************/

   for(i = 0; i < N; i++)
   {
     h_a[i] = 2;
     h_b[i] = 2;
     h_c[i] = 0;
     hr_c[i] = 0;
   }   
  /************************** Vector addition on CPU *********************************/
   clock_t t;
   t = clock();
   vecAddCPU(h_a, h_b, h_c,N); // CPU vector addition function
   t = clock() - t;
   double CPUtime_taken = ((double)t)/CLOCKS_PER_SEC;
   
   /*
   for(i = 0; i < N; i++)
   {
     printf("\n h_c[%d] = %d",i,h_c[i]);
   }
   */

  /*
    1. To perform vector addition on GPU first we have to declare variables on GPU.
       These vaiables have been already declared as d_a, d_b, d_c where d_ stands for
       these varaibles are declared on device. Note that d_ is NOT a keyword it used 
       to separate device and host variables.

    2. As program will be executed on GPU, memory should be allocated for device variables as well
       The memory is allocated using cudaMalloc() function which takes two arguments address of device
       variable and size of memory.

    3. Once memory is allocated om GPU, data should be initialized. Either you can initialize data by 
       using separate kernel or transfer it from host to device. In this assignment we will use
       data transfer because it is often used in real life applications.
       
       To transfer data cudaMemcpy() function is used which has 4 parameters as follows
       
       cudaMemcpy(dest_addr, src_addr, size, direction_of_copy);
  
       a. dest_addr = Destination variable address
       b. src_addr  = Source variable address
       c. size      = Size of memory to be transfered
       b. direction_of_copy = data can be transfered from Host to Device of Device to Host

    4. Kernel lauch is similar to C-programming function. Only difference is thread configuraion is added.
 
       vecAdd_kernel<<<NumberOfThreadBlocks, NumberOfThreadsPerBlock>>>(d_a, d_b, d_c, N);

       NumberOfThreadBlocks   : Integer value determines number of thread blocks to be launched
       NumberOfThreadsPerBlock: Integer value determines number of threads per block to be launched

    5. Data is copiedback from device to host using cudaMemcpy() function 
  
    6. Finally both CPU and GPU memories are freed     
  */

 
  /************************** 2. Memory allocation on GPU ******************************/

   cudaMalloc((void **)&d_a, N*sizeof(int)); // Allocate memory on GPU for variable d_a  input
   cudaMalloc((void **)&d_b, N*sizeof(int)); // Allocate memory on GPU for variable d_b  input
   cudaMalloc((void **)&d_c, N*sizeof(int)); // Allocate memory on GPU for variable d_c  output

  /******************** 3. Transfer data from Host to Device ***************************/

  cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice); // Copy data from Host to Device for vector a
  cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice); // Copy data from Host to Device for vector b
  cudaMemcpy(d_c, hr_c, N*sizeof(int), cudaMemcpyHostToDevice); // Copy data from Host to Device for vector c

  /******************** 4. Kernel lauch to execute vector addition on Device ***************************/

  /* Declaring Time variables to measure GPU time*/
  float GPUelapsed = 0.0;  // To store final kernel execution time 
  cudaEvent_t start, stop; // Variables to record start and stop of kernel
  cudaEventCreate(&start); // Event create start
  cudaEventCreate(&stop);  // Event create stop

  cudaEventRecord(start, 0); // Record time at start variables

  /*Vector Addition using many thread blocks*/
  
  vecAdd_kernel_UsingManyBlocks<<<1, 1024>>>(d_a, d_b, d_c, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPUelapsed, start, stop);

  cudaEventDestroy(start); // Event destroy start
  cudaEventDestroy(stop);  // Event destroy stop

  printf("\n *************************************************************"); 
  printf("\n Program Name: vector addition when N > # threads per block");
  printf("\n N = %d  # threads per block = %d",N,N);
  printf("\n Elapsed Time on CPU is %f ms", CPUtime_taken*1000);
  printf("\n Elapsed Time on GPU is %0.10f ms",GPUelapsed);
  printf("\n Speedup = (CPU time)/(GPU time) = %f", (CPUtime_taken*1000)/GPUelapsed);
  printf("\n *************************************************************");


/************************* 5. Copy results back from GPU to CPU***************************************/
cudaMemcpy(hr_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);//Copy data from GPU to Host for vector c
  
/**************** Validate whether CPU and GPU results are matching or NOT ***************************/
  for(i = 0; i < N; i++)
  {
   //printf("\n hr_c[%d] = %d", i, hr_c[i]);
   if(hr_c[i] != h_c[i])
   {
    printf("\n Results are wrong!!!\n");
   }  

   }
  /* 6. Free CPU memory*/
  free(h_a); 
  free(h_b);
  free(h_c);
  free(hr_c);
  /* 6. Free GPU memory*/
  cudaFree(d_a); 
  cudaFree(d_b);
  cudaFree(d_c);
return(0);
}


