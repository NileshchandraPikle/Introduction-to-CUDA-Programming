/*
   Program name: gld_throughput.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418 
    
    Purpose: Program to demonstrate effect on performance for global memory access pattern

    Description: Two vector addition kernels are implemented, one with coalesced global memory access 
                 and other with strided global memory access. The total number of threads are less than  
                 array size.
                 1.  nvprof --devices 0 --metrics gld_transactions ./a.out
                     This metrics returns total number of global memory load transactions
                 2. nvprof --devices 0 --metrics gld_efficiency ./a.out
                     This metrics returns global memory load efficiency
*/

#include<stdio.h>
#include<stdlib.h>

#define CHECK(call)                                                     \
{									\
  const cudaError_t error = call;					\
  if(error != cudaSuccess)						\
  {									\
    printf("Error %s %d", __FILE__, __LINE__);				\
    printf("\n Code %d Reason %s \n",error, cudaGetErrorString(error));	\
    exit(1);								\
  }									\
}


   __global__ void vecAddUncoalesced(float *d_A, float *d_B, float *d_C, int N, int offset)
   {
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     int idx = gid*offset;
     for(int i = idx; i < (gid + offset); i++)
     {  
       d_C[i] = d_A[i] + d_B[i];
     }
   }

   __global__ void vecAddCoalesced(float *d_A, float *d_B, float *d_C, int N, int offset)
   {
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     int stride = gridDim.x*blockDim.x;
     for(int i = gid; i < N; i+= stride)
     {  
       d_C[i] = d_A[i] + d_B[i];
     }
   }

   __global__ void initDataGpu(float *d_C, int N)
   {
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     d_C[gid] = 0.0;
   }

  void initData(float *h_A, float *h_B, float *hostRef, float *gpuRef, int N)
  {
   for(int i = 0; i < N; i++)
   {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
    hostRef[i] = 0.0;
    gpuRef[i] = 0.0;
   }
  }

  void hostFunc(float *h_A, float *h_B, float *hostRef, int N)
  { 
   for(int k = 0; k < N; k++)
   {
     hostRef[k] = h_A[k] + h_B[k];
   }
  }

  int compareResults(float *hostRef,float *gpuRef, int N)
  {
   for(int i = 0; i < N; i++)
   {
    printf("\n %f %f ",hostRef[i],gpuRef[i]);
    if(hostRef[i] != gpuRef[i])
    {
     return 1;
    }
   }
    return 0;
  }  
int main()
 {
  int N;
  N = 32; // # elements in arrays
  size_t nBytes = N * sizeof(float); // Size reqiored to stoare arrays

  int offset = 2;// To determine the stride size in kernel
  printf("\n Enter offset value (2/4/8): ");
  scanf("%d",&offset);
  if(offset == 2 || offset == 8 || offset == 8)
  {

  }else{ 
        printf("\n Wrong input! Run again! Have a nice day :-)");
        return 0;
       }
  

  /********************* Memory allocation at host side ****************************/
  float *h_A = (float *)malloc(nBytes); // Host side input array h_A
  float *h_B = (float *)malloc(nBytes); // Host side input array h_B
  float *hostRef = (float *)malloc(nBytes); // Host side reference output array hostRef
  float *gpuRef = (float *)malloc(nBytes);  // Device side reference output array gpuRef

  initData(h_A,h_B,hostRef, gpuRef, N); // Data initialization function
  
  hostFunc(h_A, h_B, hostRef, N);
  /********************* Memory allocation at device side ****************************/

  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc((void **)&d_A, nBytes)); // Device side input array d_A
  CHECK(cudaMalloc((void **)&d_B, nBytes)); // Device side input array d_B
  CHECK(cudaMalloc((void **)&d_C, nBytes)); // Device side output array d_C

  /********************* Data transfer from host to device ****************************/
  
  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
  
  /******************************* Kernel1 Launch **************************************/

  int numT, numB, numB1;
  numT = 256; // # threads per block
  numB = ceil((N/offset)/(float)numT); // # blocks
  numB1 = ceil(N/(float)numT); // # blocks
  vecAddCoalesced<<<numB,numT>>>(d_A, d_B, d_C, N, offset); // Kernel with coalesced memory access

  /********************* Data transfer from host to device ****************************/
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)); // Transfer data from device to host  
  
  int check = compareResults(hostRef,gpuRef, N);  
  if(check!= 0)
  {
    printf("\n ALERT!!! CPU and GPU side results are not matching After coalesced kernel!!!");
  }

  /******************************* Kernel2 Launch **************************************/
  initData(h_A,h_B,hostRef, gpuRef, N); // Data initialization function Host
  hostFunc(h_A, h_B, hostRef, N);
  initDataGpu<<<numB1,numT>>>(d_C,N);  // Data initialization kernel Device
  vecAddUncoalesced<<<numB,numT>>>(d_A, d_B, d_C, N, offset);  // Kernel with uncoalesced memory access
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost)); // Transfer data from device to host  
  printf("\n Uncoalesced\n");
  check = compareResults(hostRef,gpuRef, N);  
  if(check!= 0)
  {
    printf("\n ALERT!!! CPU and GPU side results are not matching After uncoalesced kernel!!!");
  }
  /******************************* Free device and Host Memories ********************************/
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return(0);
}
