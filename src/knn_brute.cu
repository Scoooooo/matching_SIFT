#include "curand_kernel.h"
#include "knn_brute.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
#include "knn_brute.h"
#include "helper.h"
#include <algorithm>

// thrust 
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie or just manged 
struct abs2 {
    __host__ __device__ float operator()(const float &x) const { return x * x; }
};

__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, const float * __restrict__ d_norms_y_2, float * __restrict__ d_dots,
                                      const int NX, const int NY) {

    const int i = threadIdx.x + blockIdx.x * gridDim.x;
    const int j = threadIdx.y + blockIdx.y * gridDim.y;

    if ((i < NY) && (j < NX)) d_dots[i * NX+ j] = d_norms_x_2[j] + d_norms_y_2[i] - 2 * d_dots[i * NX+ j];

}

void cublas_2nn_brute(des_t * q_points, des_t * r_points, int q_n, int r_n, float4  * sorted)
{
// steps ||x,y||^2 = ||x||^2 + ||y||^2 - 2x*y^T
// notes cublas works in colum major,, c++ is in row major :(  
// meaning our input q_points and r_poins are already transpoed 
// first we need to get the norms of x and y 
// then we need to multiply every element with itself 
// then we need to 
    const int Ndims = 128;        // --- Number of rows
    const int NQ    = q_n;        // --- Number of columns
    const int NR    = r_n;        // --- Number of columns

    //cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float a = 1.f;
    float b = 0.f;
 
    
    // norm of q 
   // thrust::device_vector<float> q_N(NX);
    float * q_N; 
    cudaMallocManaged((void **)&q_N, q_n * sizeof(float)) ; 

    thrust::device_vector<float> q_2(Ndims * NQ);
    thrust::transform((float * )q_points, (float * )q_points + q_n, q_2.begin(), abs2());
    thrust::device_vector<float> d_ones(Ndims, 1.f);

    cublasSgemv(handle, CUBLAS_OP_N, Ndims, NQ, &a, thrust::raw_pointer_cast(q_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &b, q_N, 1);
   // cublasSgemv(handle, CUBLAS_OP_N, 1, q_n, 128, &a, thrust::raw_pointer_cast(d_ones.data()), 1, (float *)q_points, 128, &b,  q_N,1 );

    // norm of r 
//    thrust::device_vector<float> r_N(NX);
    float * r_N; 
    cudaMallocManaged((void **)&r_N, r_n * sizeof(float)) ; 


    thrust::device_vector<float> r_2(Ndims * NR);
    thrust::transform((float * )r_points, (float * )r_points + r_n, r_2.begin(), abs2());

    cublasSgemv(handle, CUBLAS_OP_N, Ndims, NR, &a, thrust::raw_pointer_cast(r_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &b, r_N, 1);
    //cublasSgemv(handle,  CUBLAS_OP_N, 1, r_n, 128, &a, thrust::raw_pointer_cast(d_ones.data()), 1, (float *)r_points, 128, &b,  r_N,1 );
//
    /***********************************/
    /* CALCULATING THE SCALAR PRODUCTS */
    /***********************************/
//  thrust::device_vector<float> d_dots(NX * NY);
// array of the distances between all q and r points 
    float * dev_dist_f ; 
    cudaMallocManaged((void **)&dev_dist_f, q_n* r_n * sizeof(float)) ; 
//
//
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NQ, NR, Ndims, &a,
                               (float * )q_points, Ndims, (float * )r_points, Ndims, &b,
                               dev_dist_f, NQ);
    float * dev_dist ; 
    //array of dist from each q point to every r point 
    cudaMallocManaged((void **)&dev_dist, q_n* r_n * sizeof(float)) ; 

    dim3 grid_size(q_n, r_n, 1) ;
    dim3 block_size(32, 1, 1) ;   

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size>>>(q_points,r_points, dev_dist);
    
    cudaDeviceSynchronize() ; 
    for (size_t i = 0; i < 10; i++)
    {

        printf("%f \n", r_N[i]); 
        printf("%f \n", q_N[i]); 
        printf("%f \n", dev_dist_f[i]); 
        printf("%f == %f\n", dev_dist[i], r_N[i] + q_N[i] - 2*dev_dist_f[i]); 
    }
    


    cudaFree(r_N); 
    cudaFree(q_N); 
    cudaFree(dev_dist_f); 
}

void device_brute(des_t * q_points, des_t * r_points, int q_n, int r_n, float4  * sorted)
{
    // array of the distances between all q and r points 
    float * dev_dist ; 
    //array of dist from each q point to every r point 
    cudaMallocManaged((void **)&dev_dist, q_n* r_n * sizeof(float)) ; 

    dim3 grid_size(q_n, r_n, 1) ;
    dim3 block_size(32, 1, 1) ;   

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size>>>(q_points,r_points, dev_dist);
    cudaDeviceSynchronize();

    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,1,1) ; 

    min_dist<<<gridSize,blockSize>>>(dev_dist, r_n , sorted) ; 
    cudaDeviceSynchronize();
    cudaFree(dev_dist) ; 
}

//kernels
//finds the sqr euclidan distance between two 128 vector arrays
__global__ void sqrEuclidianDist(des_t * q_points, des_t * r_points, float * dist_array)   
{   
    float dist = 0.0f ;
    // find dist 
    for (size_t i = 0; i < 4; i++)
    {
        float a = ((float *)q_points[blockIdx.x])[threadIdx.x+(i*32)]; 
        float b = ((float *)r_points[blockIdx.y])[threadIdx.x+(i*32)]; 
        float c = a - b ; 
        dist += c * c ; 
    }

    dist += __shfl_down_sync( 0xffffffff, dist, 16 );
    dist += __shfl_down_sync( 0xffffffff, dist, 8 ); 
    dist += __shfl_down_sync( 0xffffffff, dist, 4 ); 
    dist += __shfl_down_sync( 0xffffffff, dist, 2 );
    dist += __shfl_down_sync( 0xffffffff, dist, 1 );   
    if(threadIdx.x == 0)
    {
        dist_array[blockIdx.x * gridDim.y + blockIdx.y] = dist; 
    }
}
//find smallest vlaue in the warp and index  
__device__ inline void best_in_warp(float4  &min_2)
{    
    for (int i = 16; i > 0; i/= 2)
    {          
        float x_dist = __shfl_down_sync( 0xffffffff, min_2.x, i );
        float y_dist = __shfl_down_sync( 0xffffffff, min_2.y, i );
        float w_value = __shfl_down_sync( 0xffffffff, min_2.w, i );
        float z_value = __shfl_down_sync( 0xffffffff, min_2.z, i );
        if(x_dist < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = x_dist ;  
                
            min_2.w = min_2.z ; 
            min_2.z = z_value;  
        }
        else{
            if(x_dist < min_2.y)
            {
                min_2.y = x_dist ; 
                min_2.w = z_value;  
                continue ; 
            }
        } 
        if(y_dist < min_2.y)
        {
            min_2.y = y_dist ; 
            min_2.w = w_value;  
        }
    }
}
 
// x warps per dist 
__global__ void min_dist(float *  dist, int size ,float4 * sorted)
{
    //           finds the dist array         x dim       y dim pos                      
    int offset = (blockIdx.x * size)+ threadIdx.y * blockDim.x ;
   
    float4 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 

    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+=(blockDim.x * blockDim.y) )
    {
        if(dist[i + offset + threadIdx.x] < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = dist[i + offset + threadIdx.x ] ;  
            min_2.w = min_2.z ;  
            min_2.z = i + threadIdx.x + threadIdx.y * blockDim.x  ; 
        }
        else{
            if(dist[i + offset + threadIdx.x ] < min_2.y)
            {
                min_2.y = dist[i + offset + threadIdx.x ] ;  
                min_2.w = i + threadIdx.x + threadIdx.y * blockDim.x   ;  
            }
        }
    }
    best_in_warp(min_2) ;   

    __shared__ float4 best[32] ; 
    
    if(threadIdx.y == 0)
    {
        best[threadIdx.x].x = MAXFLOAT ;
        best[threadIdx.x].y = MAXFLOAT ; 
    }
    __syncthreads() ; 
    if(threadIdx.x == 0)
    {
        best[threadIdx.y] = min_2 ;
    }
    
    __syncthreads() ; 
    if(threadIdx.y == 0){

        min_2 = best[threadIdx.x] ; 
        best_in_warp(min_2) ; 
        if(threadIdx.x == 0)
        { 
            sorted[blockIdx.x ] = min_2 ; 
        }
    }
}

//host brute

void host_brute(des_t * q_points, des_t * r_points, int q_points_size, int r_points_size, float4  * sorted)
{
    
    float * lenght;
    cudaMallocHost((void **)&lenght, r_points_size * q_points_size * sizeof(float)) ; 
    for (size_t i = 0; i < q_points_size; i++)
    {
        for (size_t ii = 0; ii < r_points_size; ii++)
        {
            lenght[(i * r_points_size) + ii ] = host_lenght(q_points[i], r_points[ii]) ;  
        }
    }
    
    host_sort(lenght,r_points_size, q_points_size, sorted) ; 
    cudaFree(lenght) ; 
}

//given an array of arrays of lenghts it sorts the array and returns a sroted array 
//containing the 2 shortests lenghts from each array 
void host_sort(float * dist, int size, int array_size, float4 * sorted)
{   
    for (int i = 0; i < array_size ; i++)
    {
        float4 min_2 ;  
        min_2.x = MAXFLOAT ; 
        min_2.y = MAXFLOAT ;
        min_2.z = MAXFLOAT ;  
        min_2.w = MAXFLOAT ;  
        int offset = i * size ; 
        for (int ii = 0; ii < size; ii++)
        {
            if(dist[ii + offset] < min_2.x)
            {
                min_2.y = min_2.x ; 
                min_2.x = dist[ii + offset] ;  
                
                min_2.w = min_2.z ; 
                min_2.z = ii ;  
                
            }
            else{
                if(dist[ii + offset] < min_2.y)
                {
                    min_2.y = dist[ii + offset] ; 
                    min_2.w = ii ;  
                }
            }
        }
        sorted[i] = min_2 ;  
    }
}

// given 2 points find the euclidina lenght before taking the root
float host_lenght(des_t x, des_t y){
    float * vec1 = (float * )x ; 
    float * vec2 = (float * )y ;
    float dist =  0.0f  ;  
    for (size_t i = 0; i < 128; i++)
    {
        float a = vec1[i] ; 
        float b = vec2[i] ;  
        float c = a - b ; 
        dist += c * c ; 
    }
    return dist ; 
}



