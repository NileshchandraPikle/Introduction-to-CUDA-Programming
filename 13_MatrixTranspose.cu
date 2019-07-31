/*
   Program name: MatrixTranspose.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418  
         Webpage: https://piklenileshchandra.wixsite.com/personal
    
    Purpose: To perform Matrix Transpose using CUDA
             
    Description: Matrix transpose program is considered to demonstrate 2D thread block and 2D grid.

    Three versions of matrix transpose are created here
    1. Matrix transpose using single thread (sequential)
    2. Matrix transpose using N threads (1D thread block) where N is #rows in matrix
    3. Matrix transpose using N x N threads (2D thread block) where N is #rows in matrix
    
   *In first version a single thread on GPU responsible to perform entire transpose operation hence it 
    is a sequetial operation. This version performs worst than sequential CPU as GPU core is lightweight
   *In second version each thread is responsible to take a row of matrix and store into new matrix in 
    column. Consider this as coarse grained. 
   *In third version each thread is responsible to take a single element of matrix and store into new 
    matrix at transposed position. Consider this as coarse grained. 

    Note: 3rd version is NOT optimized. However it outperforms first and second version. Shared memory 
          level optimizations can be performed to accelerate execution even further. To see how to 
          optimize second version refer link below
          Link: https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/


*/

#include<stdio.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

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
 __global__ void transposeKernel_single_thread(int *d_input, int *d_output, int MAT_SIZE)
 {
   int i,j;
   for(i = 0; i < MAT_SIZE; i++)
   {
     for(j = 0; j < MAT_SIZE; j++)
     {
       d_output[j * MAT_SIZE + i] = d_input[i * MAT_SIZE + j];
     }
   } 
 }

 __global__ void transposeKernel_thread_per_row(int *d_input, int *d_output, int MAT_SIZE)
 {
    int j;
    int gid = threadIdx.x + blockIdx.x *blockDim.x;
    if(gid < MAT_SIZE)
    {
      for(j = 0; j < MAT_SIZE; j++)
      {
        d_output[gid * MAT_SIZE + j] = d_input[j * MAT_SIZE + gid];
      }
    }
 }

  __global__ void transposeKernel_thread_per_element(int *d_input, int *d_output, int MAT_SIZE)
  {
    int row,col;
    col = threadIdx.x + blockIdx.x * blockDim.x;
    row = threadIdx.y + blockIdx.y * blockDim.y;

    if( (row < MAT_SIZE) && (col < MAT_SIZE) )
    { 
     d_output[row * MAT_SIZE + col] = d_input[col * MAT_SIZE + row];
    }
  }

  void initMatrix(int *matrix, int MAT_SIZE, int flag)
  {
    int i,j;
    if(flag == 0)
    {
      for(i = 0; i < MAT_SIZE; i++)
      {
        for(j = 0; j < MAT_SIZE; j++)
        {
          matrix[i * MAT_SIZE + j] = 0;
        }
      }
     }else{
            for(i = 0; i < MAT_SIZE; i++)
            {
             for(j = 0; j < MAT_SIZE; j++)
             {
               matrix[i * MAT_SIZE + j] = i;
               //printf(" %d ",matrix[i * MAT_SIZE + j]);
             }
            //printf("\n");
            }
           }
    }

    int checkResult(int *hg_output, int *h_output, int MAT_SIZE)
    {
       int i,j, flag = 1;
       for(i = 0; i < MAT_SIZE; i++)
       {
         for(j = 0; j < MAT_SIZE; j++)
         {
           if(hg_output[i*MAT_SIZE+j] != h_output[i*MAT_SIZE+j])
           {
             flag = 0;
           }
         }
       }
       return flag;
    }
    
    __global__ void init_kernel(int *d_output, int MAT_SIZE)
    {
       int i,j;
       for(i = 0; i < MAT_SIZE; i++)
       {
         for(j = 0; j < MAT_SIZE; j++)
         {
            d_output[i * MAT_SIZE + j] = 0;
         }
       }
    }

  int main()
  {
    int i,j; 
    int MAT_SIZE = 2048;   // Matrix size MAT_SIZE x MAT_SIZE
    int *h_input  = (int *)malloc(MAT_SIZE*MAT_SIZE*sizeof(int));
    int *h_output = (int *)malloc(MAT_SIZE*MAT_SIZE*sizeof(int));
    int *hg_output = (int *)malloc(MAT_SIZE*MAT_SIZE*sizeof(int));
    
    if(!h_input)
    {
     printf("\n Error: Allocating memory to h_input!!!");
    }
    if(!h_output)
    {
     printf("\n Error: Allocating memory to h_output!!!");
    }
    if(!hg_output)
    {
     printf("\n Error: Allocating memory to h_output!!!");
    }
    /*
      Data initialization for input matrix h_input and h_output matrix
       
    */

     initMatrix(h_output,MAT_SIZE,0);  // matrix initialization function 0 indicates initialize with 0
     initMatrix(hg_output,MAT_SIZE,0); // matrix initialization function 0 indicates initialize with 0
     initMatrix(h_input,MAT_SIZE,1);  // matrix initialization function 0 indicates initialize with row index (refer function)

    /*
      Matrix transpose code: h_output stores the transposed output of matrix of h_input       
    */
    clock_t t;     // clock function to calculate execution time of sequential program
    t = clock();   // record start time

    for(i = 0; i < MAT_SIZE; i++)
    {
      for(j = 0; j < MAT_SIZE; j++)
      {
       h_output[i * MAT_SIZE + j] = h_input[j * MAT_SIZE + i];
       //printf(" %d ", h_output[i * MAT_SIZE + j]);
      }
      //printf("\n");
    }
   t = clock() - t;
   double CPUtime_taken = ((double)t)/CLOCKS_PER_SEC;
   printf("\n     Time required for sequential execution is %.3f ms\n", CPUtime_taken*1000);

    /*
       Parallel CUDA programs start here
    */

    /* Allocate memory to d_input and d_output array on device */
    int *d_input, *d_output;
    CHECK(cudaMalloc((void **)&d_input, MAT_SIZE*MAT_SIZE*sizeof(int)));
    CHECK(cudaMalloc((void **)&d_output, MAT_SIZE*MAT_SIZE*sizeof(int)));
    
    /* Transfer data from host memory to device memory h_input to d_input and h_output to d_output array on device*/ 

    CHECK(cudaMemcpy(d_input, h_input, MAT_SIZE*MAT_SIZE*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output, h_output, MAT_SIZE*MAT_SIZE*sizeof(int), cudaMemcpyHostToDevice));

/**********************************************************************************************/
    /* 1. Transpose kerene launch using single thread
          A thread block of single thread is configured to perform matrix transpose
          This is exactly like sequential version of matrix transpose as only one 
          thread is responsible to perform entire matrix transpose operation          
    */
     
  /* Declaring Time variables to measure GPU time*/
  float GPUelapsed = 0.0;  // To store final kernel execution time 
  cudaEvent_t start, stop; // Variables to record start and stop of kernel
  cudaEventCreate(&start); // Event create start
  cudaEventCreate(&stop);  // Event create stop
  cudaEventRecord(start, 0); // Record time at start variables

/**/transposeKernel_single_thread<<<1,1>>>(d_input,d_output,MAT_SIZE); // krenel for matrix transpose

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPUelapsed, start, stop);
  cudaEventDestroy(start); // Event destroy start
  cudaEventDestroy(stop);  // Event destroy stop
  printf("\n 1. Parallel runtime Single thread %0.3f ms",GPUelapsed);
  printf("\n    Parallel Speedup Single thread = (CPU time)/(GPU time) = %0.3f\n", (CPUtime_taken*1000)/GPUelapsed);
    
  CHECK(cudaMemcpy(hg_output, d_output, MAT_SIZE*MAT_SIZE*sizeof(int), cudaMemcpyDeviceToHost)); // copy results back on host

    int check = checkResult(hg_output, h_output, MAT_SIZE); // This function checks two matrices are same or not
    if(check == 0)printf("\n Something went wrong => Results are not matching for 'transposeKernel_single_thread' !!!");
    initMatrix(hg_output,MAT_SIZE,0); // Initialize hg_output matrix to 0
    init_kernel<<<1,1>>>(d_output, MAT_SIZE); // Initialize d_output matrix to 0
/**********************************************************************************************/

    /* 2. Transpose kerene launch using thread per row
          In this configuration, # threads launched are equalt to # rows of matrix i.e MAT_SIZE
          Where each thread is responsible to perform transpose operation.
          Each thread reads the matrix entries row major order from d_input matrix and stores in 
          d_output matrix in column major order.    

    # threads launched = # rows in matrix
    numT = # threads per thread block
    numB = # thread blocks in the grid
  */

   int numT, numB;
   numT = 128;
   numB = ceil(MAT_SIZE/(float)numT);
  /* Declaring Time variables to measure GPU time*/
  float GPUelapsed2 = 0.0;  // To store final kernel execution time 
  cudaEvent_t start2, stop2; // Variables to record start and stop of kernel
  cudaEventCreate(&start2); // Event create start
  cudaEventCreate(&stop2);  // Event create stop
  cudaEventRecord(start2, 0); // Record time at start variables

/**/ transposeKernel_thread_per_row<<<numB,numT>>>(d_input,d_output,MAT_SIZE); // krenel for matrix transpose

  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&GPUelapsed2, start2, stop2);
  cudaEventDestroy(start2); // Event destroy start
  cudaEventDestroy(stop2);  // Event destroy stop
  printf("\n 2. Parallel runtime thread per row is %0.3f ms",GPUelapsed2);
  printf("\n    Parallel Speedup thread per row = (CPU time)/(GPU time) = %0.3f\n", (CPUtime_taken*1000)/GPUelapsed2);
    
  CHECK(cudaMemcpy(hg_output, d_output, MAT_SIZE*MAT_SIZE*sizeof(int), cudaMemcpyDeviceToHost)); // copy results back on host

    check = checkResult(hg_output, h_output, MAT_SIZE); // This function checks two matrices are same or not
    if(check == 0)printf("\n Something went wrong => Results are not matching for 'transposeKernel_thread_per_row'!!!");
    initMatrix(hg_output,MAT_SIZE,0); // Initialize hg_output matrix to 0
    init_kernel<<<1,1>>>(d_output, MAT_SIZE); // Initialize d_output matrix to 0


/**********************************************************************************************/
    /* 3. Transpose kerene launch using thread per element of the matrix
          In this configuration, # threads launched are equalt to # of elements in the matrix i.e MAT_SIZE x MAT_SIZE
          Where each thread is responsible to perform transpose operation on a single element.
          Each thread reads a single corresponding matrix element from d_input matrix and stores in 
          d_output matrix in transposed indices.    

    # threads launched = # elements in matrix
    numT = # threads per thread block in 2D
    numB = # thread blocks in the grid in 2D
  */

   dim3 num2T(8, 8,1);
   dim3 num2B(ceil(MAT_SIZE/(float)num2T.x), ceil(MAT_SIZE/(float)num2T.y ),1 );
   //printf("\n num2B.x = %d num2B.y = %d ", num2B.x,num2B.x );
  /* Declaring Time variables to measure GPU time */
  float GPUelapsed3 = 0.0;  // To store final kernel execution time 
  cudaEvent_t start3, stop3; // Variables to record start and stop of kernel
  cudaEventCreate(&start3); // Event create start
  cudaEventCreate(&stop3);  // Event create stop
  cudaEventRecord(start3, 0); // Record time at start variables

/**/ transposeKernel_thread_per_element<<<num2B,num2T>>>(d_input,d_output,MAT_SIZE); // krenel for matrix transpose

  cudaEventRecord(stop3, 0);
  cudaEventSynchronize(stop3);
  cudaEventElapsedTime(&GPUelapsed3, start3, stop3);
  cudaEventDestroy(start3); // Event destroy start
  cudaEventDestroy(stop3);  // Event destroy stop
  printf("\n 3. Parallel runtime thread per element is %0.3f ms",GPUelapsed3);
  printf("\n    Parallel Speedup thread per element = (CPU time)/(GPU time) = %0.3f\n", (CPUtime_taken*1000)/GPUelapsed3);
    
  CHECK(cudaMemcpy(hg_output, d_output, MAT_SIZE*MAT_SIZE*sizeof(int), cudaMemcpyDeviceToHost)); // copy results back on host

    check = checkResult(hg_output, h_output, MAT_SIZE); // This function checks two matrices are same or not
    if(check == 0)
   {
    printf("\n Something went wrong => Results are not matching for 'transposeKernel_thread_per_element'!!!");
   }

   free(h_input);
   free(h_output);
   free(hg_output);
   cudaFree(d_input);
   cudaFree(d_output);
   return(0);    
  }
