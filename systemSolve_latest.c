/*
 * Copyright (C) 2009-2012 EM Photonics, Inc.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to EM Photonics ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code may
 * not redistribute this code without the express written consent of EM
 * Photonics, Inc.
 *
 * EM PHOTONICS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  EM PHOTONICS DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL EM
 * PHOTONICS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as that
 * term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of "commercial
 * computer  software"  and "commercial computer software documentation" as
 * such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the
 * U.S. Government only as a commercial end item.  Consistent with 48
 * C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the source code with only those rights set
 * forth herein. 
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code, the
 * above Disclaimer and U.S. Government End Users Notice.
 *
 */

/*
 * CULA Example: systemSolve
 *
 * This example shows how to use a system solve for multiple data types.  Each
 * data type has its own example case for clarity.  For each data type, the
 * following steps are done:
 *
 * 1. Allocate a matrix on the host
 * 2. Initialize CULA
 * 3. Initialize the A matrix to the Identity
 * 4. Call gesv on the matrix
 * 5. Verify the results
 * 6. Call culaShutdown
 *
 * After each CULA operation, the status of CULA is checked.  On failure, an
 * error message is printed and the program exits.
 *
 * Note: CULA Premium and double-precision GPU hardware are required to run the
 * double-precision examples
 *
 * Note: this example performs a system solve on an identity matrix against a
 * random vector, the result of which is that same random vector.  This is not
 * true in the general case and is only appropriate for this example.  For a
 * general case check, the product A*X should be checked against B.  Note that
 * because A is modifed by GESV, a copy of A would be needed with which to do
 * the verification.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cula_lapack.h>


void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}


void culaFloatExample()
{
#ifdef NDEBUG
    int N = 10;
#else
    int N = 1024;
#endif
    int NRHS = 1;
    int i, j;

    culaStatus status;
    
    culaFloat* A = NULL;
    culaFloat* A_bak = NULL;
    culaFloat* B = NULL;
    culaFloat* X = NULL;
    culaInt* IPIV = NULL;


    culaFloat* Ad = NULL;
    culaFloat* Ad_bak = NULL;
    culaFloat* Bd = NULL;
    culaFloat* Xd = NULL;
    culaInt* IPIVd = NULL;

     
     
    // culaFloat one = 2.0f;
    culaFloat thresh = 1e-6f;
    culaFloat diff;

    printf("-------------------\n");
    printf("       SGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices on host\n");
    A = (culaFloat*)malloc(N*N*sizeof(culaFloat));
    A_bak = (culaFloat*)malloc(N*N*sizeof(culaFloat));
    B = (culaFloat*)malloc(N*sizeof(culaFloat));
    X = (culaFloat*)malloc(N*sizeof(culaFloat));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV || !A_bak)
        exit(EXIT_FAILURE);

    printf("Allocating Matrices on device\n");
    cudaMalloc((void**)&Ad,N*N*sizeof(culaFloat));
    cudaMalloc((void**)&Ad_bak,N*N*sizeof(culaFloat));
    cudaMalloc((void**)&Bd,N*sizeof(culaFloat));
    cudaMalloc((void**)&Xd,N*sizeof(culaFloat));
    cudaMalloc((void**)&IPIVd,N*sizeof(culaInt));
    
    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaFloat));
    memset(A_bak, 0, N*N*sizeof(culaFloat));
    for(i = 0; i < N; ++i) {
      culaFloat sum = 0.f;
      for (j = 0; j < N; j++) {
        A_bak[i * N + j] = A[i * N + j] = rand() % 10 + 1;
        sum += A[i * N + j];
      }
      A[i * N + i] = A_bak[i * N + i] = 2.f * sum;
        // printf("%g, %g\n", one, A[i * N + i]);
    }
  /* A[0]=3;
   A[1]=3;
   A[2]=0;
   A[3]=0;
   A[4]=2;
   A[5]=2;
   A[6]=1;
   A[7]=0;
   A[8]=1;*/
    //Printing the matix
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
        B[i] = (culaFloat)rand();
    memcpy(X, B, N*sizeof(culaFloat));

    memset(IPIV, 0, N*sizeof(culaInt));

//Copy from Host to Device
    cudaMemcpy(Ad,A, N*N*sizeof(culaFloat),cudaMemcpyHostToDevice);
    cudaMemcpy(Ad_bak,A_bak, N*N*sizeof(culaFloat),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,B, N*sizeof(culaFloat),cudaMemcpyHostToDevice);
    cudaMemcpy(Xd,X, N*sizeof(culaFloat),cudaMemcpyHostToDevice);
    cudaMemcpy(IPIVd,IPIV, N*sizeof(culaInt),cudaMemcpyHostToDevice);


/*    //Printing the matix
    printf("\n\n");
      for (i=0; i< N*N; ++i)
      {
      printf("%f,",A[i]);
      if((i+1)%N==0)
        printf("\n");
      }*/
    printf("Calling culaSgesv\n");
    status = culaSgesv(N, NRHS, A, N, IPIV, X, N);
    checkStatus(status);
//Copy result from Device to Host

    cudaMemcpy(A,Ad, N*N*sizeof(culaFloat),cudaMemcpyDeviceToHost);
    cudaMemcpy(A_bak,Ad_bak, N*N*sizeof(culaFloat),cudaMemcpyDeviceToHost);
    cudaMemcpy(B,Bd, N*sizeof(culaFloat),cudaMemcpyDeviceToHost);
    cudaMemcpy(Xd,X, N*sizeof(culaFloat),cudaMemcpyDeviceToHost);
    cudaMemcpy(IPIVd,IPIV, N*sizeof(culaInt),cudaMemcpyDeviceToHost);




    printf("Verifying Result\n");
    int success = 1;
    for(i = 0; i < N; ++i)
    {
        culaFloat sum = 0.f;
        for (j = 0; j < N; j++)
          sum += A_bak[i * N + j] * X[j];
        diff = sum - B[i];
        if(diff < 0.0f)
            diff = -diff;
        if(!(diff < thresh)) {
            printf("Result check:  i=%d  sum=%f  B[i]=%f", i, sum, B[i]);
            success = 0;
        }
    }
    if (success)
      printf("Success\n");
    else
      printf("Failed\n");
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(A_bak);
    free(X);
    free(B);
    free(IPIV);
    cudaFree(Ad);
    cudaFree(Ad_bak);
    cudaFree(Bd);
    cudaFree(Xd);
    cudaFree(IPIVd);

}


void culaFloatComplexExample()
{
#ifdef NDEBUG
    int N = 4096;
#else
    int N = 512;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaFloatComplex* A = NULL;
    culaFloatComplex* B = NULL;
    culaFloatComplex* X = NULL;
    culaInt* IPIV = NULL;

    culaFloatComplex one = { 1.0f, 0.0f };
    culaFloat thresh = 1e-6f;
    culaFloat diffr;
    culaFloat diffc;
    culaFloat diffabs;

    printf("-------------------\n");
    printf("       CGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaFloatComplex*)malloc(N*N*sizeof(culaFloatComplex));
    B = (culaFloatComplex*)malloc(N*sizeof(culaFloatComplex));
    X = (culaFloatComplex*)malloc(N*sizeof(culaFloatComplex));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaFloatComplex));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
    {
        B[i].x = (culaFloat)rand();
        B[i].y = (culaFloat)rand();
    }
    memcpy(X, B, N*sizeof(culaFloatComplex));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaCgesv\n");
    status = culaCgesv(N, NRHS, A, N, IPIV, X, N);
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diffr = X[i].x - B[i].x;
        diffc = X[i].y - B[i].y;
        diffabs = (culaFloat)sqrt(X[i].x*X[i].x+X[i].y*X[i].y)
                - (culaFloat)sqrt(B[i].x*B[i].x+B[i].y*B[i].y);
        if(diffr < 0.0f)
            diffr = -diffr;
        if(diffc < 0.0f)
            diffc = -diffc;
        if(diffabs < 0.0f)
            diffabs = -diffabs;
        if(diffr > thresh || diffc > thresh || diffabs > thresh)
            printf("Result check failed:  i=%d  X[i]=(%f,%f)  B[i]=(%f,%f)", i, X[i].x, X[i].y, B[i].x, B[i].y);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();


    free(A);
    free(B);
    free(IPIV);
}


// Note: CULA Premium is required for double-precision
#ifdef CULA_PREMIUM
void culaDoubleExample()
{
#ifdef NDEBUG
    int N = 10;
#else
    int N = 512;
#endif
    int NRHS = 1;
    int i,j;

    culaStatus status;
    
    culaDouble* A = NULL;
    culaDouble* A_bak = NULL;
    culaDouble* B = NULL;
    culaDouble* X = NULL;
    culaInt* IPIV = NULL;

    culaDouble *work = NULL;
    // culaDouble *swork = NULL;
    int *info;

    culaDouble one = 1.0;
    culaDouble thresh = 1e-6;
    culaDouble diff;
    
    printf("-------------------\n");
    printf("       DGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaDouble*)malloc(N*N*sizeof(culaDouble)); 
    A_bak = (culaDouble*)malloc(N*N*sizeof(culaDouble));
    B = (culaDouble*)malloc(N*sizeof(culaDouble));
    X = (culaDouble*)malloc(N*sizeof(culaDouble));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
  //  work = (culaDouble*)malloc(N * NRHS * sizeof(culaDouble));
    //swork = (culaDouble*)malloc(N * (N+NRHS) * sizeof(culaDouble));
    info = (int *)malloc(N * sizeof(int));
    if(!A || !B || !IPIV || !A_bak)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaDouble));
    memset(A_bak, 0, N*N*sizeof(culaDouble));
    for(i = 0; i < N; ++i){
      culaDouble sum = 0.0;
      for (j=0; j<N; j++) {
        A_bak[i*N+j] = A[i * N + j] = rand() % 10 + 1;
        sum+= A[i * N + j];
      }
      A[i * N + i] = A_bak[i*N + i] = 2.0 * sum;
    }
    // Set B to a random matrix (see note at top
    for(i = 0; i < N; ++i)
        B[i] = (culaDouble)rand();
    memcpy(X, B, N*sizeof(culaDouble));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaDgesv\n");
    int iter = 0;
    status = culaDgesv(N, NRHS, A, N, IPIV, X, N);
   // printf("iter = %d\n", iter);
    if(status == culaInsufficientComputeCapability)
    {
        printf("No Double precision support available, skipping example\n");
        free(A);
        free(B);
        free(IPIV);
        culaShutdown();
        return;
    }
    checkStatus(status);

    printf("Verifying Result\n");
    int success = 1;
    double max_b = 0.0;
    for (i = 0; i < N; i++)
      if (max_b < fabs(B[i]))
        max_b = fabs(B[i]);

    for(i = 0; i < N; ++i)
    {
      culaDouble sum = 0.f;
      for (j= 0; j< N; j++)
        sum += A_bak[i*N  + j] * X[j];
        diff = sum - B[i];
        if(diff < 0.0)
            diff = -diff;
        diff /= max_b;
        if(!(diff < thresh)){
            printf("Result check:  i=%d, diff = %g\n", i,diff);
    success = 0;
        }
    }
    if(success)
      printf("Success\n");
    else
      printf("Failed\n");

    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}


void culaDoubleComplexExample()
{
#ifdef NDEBUG
    int N = 1024;
#else
    int N = 128;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaDoubleComplex* A = NULL;
    culaDoubleComplex* B = NULL;
    culaDoubleComplex* X = NULL;
    culaInt* IPIV = NULL;

    culaDoubleComplex one = { 1.0, 0.0 };
    culaDouble thresh = 1e-6;
    culaDouble diffr;
    culaDouble diffc;
    culaDouble diffabs;

    printf("-------------------\n");
    printf("       ZGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaDoubleComplex*)malloc(N*N*sizeof(culaDoubleComplex));
    B = (culaDoubleComplex*)malloc(N*sizeof(culaDoubleComplex));
    X = (culaDoubleComplex*)malloc(N*sizeof(culaDoubleComplex));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaDoubleComplex));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
    {
        B[i].x = (culaDouble)rand();
        B[i].y = (culaDouble)rand();
    }
    memcpy(X, B, N*sizeof(culaDoubleComplex));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaZgesv\n");
    status = culaZgesv(N, NRHS, A, N, IPIV, X, N);
    if(status == culaInsufficientComputeCapability)
    {
        printf("No Double precision support available, skipping example\n");
        free(A);
        free(B);
        free(IPIV);
        culaShutdown();
        return;
    }
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diffr = X[i].x - B[i].x;
        diffc = X[i].y - B[i].y;
        diffabs = (culaDouble)sqrt(X[i].x*X[i].x+X[i].y*X[i].y)
                - (culaDouble)sqrt(B[i].x*B[i].x+B[i].y*B[i].y);
        if(diffr < 0.0)
            diffr = -diffr;
        if(diffc < 0.0)
            diffc = -diffc;
        if(diffabs < 0.0)
            diffabs = -diffabs;
        if(diffr > thresh || diffc > thresh || diffabs > thresh)
            printf("Result check failed:  i=%d  X[i]=(%f,%f)  B[i]=(%f,%f)", i, X[i].x, X[i].y, B[i].x, B[i].y);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}
#endif


int main(int argc, char** argv)
{
    culaFloatExample();
    // culaFloatComplexExample();
    
    // Note: CULA Premium is required for double-precision
#ifdef CULA_PREMIUM
    culaDoubleExample();
   // culaDoubleComplexExample();
#endif

    return EXIT_SUCCESS;
}

