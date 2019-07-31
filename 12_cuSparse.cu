/*
   Program name: cuSParse.cu
    Author name: Dr. Nileshchandra Pikle
           Email: nilesh.pikle@gmail.com
  Contact Number: 7276834418 
         webpage: https://piklenileshchandra.wixsite.com/personal
    
    Purpose: Program to demonstrate CUDA library use

    Description: A simple program to demonstrate how to use CUDA libraries is developed
        * Library: cuSparse
        * Use: To perform operations on sparse matrices in compressed format
        
        To compile: nvcc -arch=sm_35 -lcusparse cuSparse.cu 
        To run: ./a.out
      Detailed steps and library function parameters are mentioned in the program

*/


#include<stdio.h>
#include<stdlib.h>
#include <cusparse.h>

int M = 2048;
int N = 2048;

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
/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void print_partial_matrix(float *M, int nrows, int ncols, int max_row,
        int max_col)
{
    int row, col;

    for (row = 0; row < max_row; row++)
    {
        for (col = 0; col < max_col; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc, char **argv)
{
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    int *dANnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalANnz;
    float alpha = 3.0f;
    float beta = 4.0f;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t Adescr = 0;

    // Generate input
    srand(9384);
    int trueANnz = generate_random_dense_matrix(M, N, &A);
    int trueBNnz = generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);

    printf("A:\n");
    print_partial_matrix(A, M, N, 10, 10);
    printf("B:\n");
    print_partial_matrix(B, N, M, 10, 10);

    // Create the cuSPARSE handle
    cusparseCreate(&handle);

    // Allocate device memory for vectors and the dense form of the matrix A
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));
    CHECK(cudaMalloc((void **)&dANnzPerRow, sizeof(int) * M));

    /*  Construct a descriptor of the matrix
        cusparseSetMatType defines matrix type as follows
        CUSPARSE_MATRIX_TYPE_GENERAL = 0, 
        CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,     
        CUSPARSE_MATRIX_TYPE_HERMITIAN = 2, 
        CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3 

        cusparseSetMatIndexBase sets matrix index base either 0 or 1
        CUSPARSE_INDEX_BASE_ZERO = 0, 
        CUSPARSE_INDEX_BASE_ONE = 1
    */
    /* sparse matrix descriptor */
    /* When the matrix descriptor is created, its fields are initialized to: 
       CUSPARSE_MATRIX_TYPE_GENERAL
       CUSPARSE_INDEX_BASE_ZERO
       All other fields are uninitialized
    */   
    cusparseCreateMatDescr(&Adescr);
    cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0x00, sizeof(float) * M * M));

    // Compute the number of non-zero elements in A\
    
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
                                dA, M, dANnzPerRow, &totalANnz);
    /*
      1. handle: handle to the cuSPARSE library context.
      2. CUSPARSE_DIRECTION_ROW: direction that specifies to count nonzero elements by
         CUSPARSE_DIRECTION_ROW 
      3. M: number of rows of matrix A.
      4. N: number of columns of matrix A.
      5. Adescr: The descriptor of matrix A. The supported matrix type is CUSPARSE_MATRIX_TYPE_GENERAL. 
         Also, the supported index bases are CUSPARSE_INDEX_BASE_ZERO and CUSPARSE_INDEX_BASE_ONE
      6. dA: array of dimensions (lda, N). lda -> Leading dimensions of dense array dA 
      7. M: lda i.e. Leading dimensions of dense array dA 
      8. dANnzPerRow: Array of size m or n containing the number of nonzero elements per row or column, 
         respectively. 
      9. totalANnz: Address of variable that stores value of total non zero entries in matrix dA

    */
    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }

    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    cusparseSdense2csr(handle, M, N, Adescr, dA, M, dANnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA);
     /*
        1. dCsrValA: Array of nonzero elements of matrix dA
        2. dCsrRowPtrA: integer array of M + 1 elements that contains the start of every row and the 
           end of the last row plus one
        3. dCsrColIndA: integer array of nnz (csrRowPtrA(m) - csrRowPtrA(0) ) column indices of the non-
           zero elements of matrix A
     */
    
    float milliseconds = 0; 
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M,
                                  M, N, totalANnz, &alpha, Adescr, dCsrValA,
                                  dCsrRowPtrA, dCsrColIndA, dB, N, &beta, dC,
                                  M);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);  
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("\n Parallel execution time = %f\n",milliseconds*0.001);
    /*
         C = α ∗ op ( A ) ∗ B + β ∗ C 
      CUSPARSE_OPERATION_NON_TRANSPOSE:  
       A if trans == CUSPARSE_OPERATION_NON_TRANSPOSE 
       A^T if trans == CUSPARSE_OPERATION_TRANSPOSE 
       A^H if trans == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE 
       alpha: constant mutiplier
       beta: constant mutiplier If beta is zero, C does not have to be a valid input. 
         
    */
    // Copy the result vector back to the host
    CHECK(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));

   // printf("C:\n");
    //print_partial_matrix(C, M, M, 10, 10);

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(dANnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    cusparseDestroyMatDescr(Adescr);
    cusparseDestroy(handle);

    return 0;
}
