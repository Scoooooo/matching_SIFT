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

// makes sure that we have enough memory  
void cublas_2nn_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle)
{
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
     used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    int i = 1 ; 
    size_t need_byte = (size_t)q_n *  (size_t)r_n * 4 ;  

    int temp = q_n ; 
    size_t dont_use = 10000000000 ; 
    while ( need_byte > (free_byte - dont_use ))
    {
        printf("%i, %zu, %zu is \n", i, need_byte, free_byte - dont_use) ; 
        i ++ ; 
        temp = q_n / i ; 
        need_byte = (size_t)temp* (size_t) r_n * 4 ;  
    }
    printf("%i, %zu, %zu is \n", i, need_byte, free_byte - dont_use) ; 
    float * dist ; 
    cudaMalloc((void **)&dist, (size_t)temp*  (size_t)r_n * 4) ; 
    int ii; 
    for (ii = 0; ii < i; ii++)
    {
        cublas_2nn_brute_f(q_points + (ii * temp), r_points, temp, r_n, sorted + (ii * temp), dist, handle); 
        printf("%i \n", ii) ; 
    }
    if((q_n % temp ) > 0 )
    {
        int left =  q_n % temp ; 
        printf("left %i \n", left) ; 
        cublas_2nn_brute_f(q_points + (ii * temp), r_points, left, r_n, sorted + (ii * temp), dist, handle); 
    }

    cudaFree(dist); 
}
// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie or just manged 
void cublas_2nn_brute_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, float * dist,cublasHandle_t handle)
{
// steps d(x,y)^2 = ||x||^2 + ||y||^2 - 2x*y^T
// for sift we only need to solve d(x,y)^2 = 2 + -2x*y^t
// or -x*y^t
// notes cublas works in colum major,, c++ is in row major :(  
// meaning our input q_points and r_poins are already transpoed 
    // add minus 
    float a = -1.f;
    float b = 0.f;
 
   // we are in row major so we want our output from cublas to be in row major as well 
    // d^t = r * q^t is what cublas dose if see from cloum major
    // which is d = r^t * q   
    
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (float *)r_points, 128, (float *)q_points, 128, &b, dist, r_n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("dot failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    } 
    // want to find min value for each dist array 
    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,4,1) ; 
    min_dist_f<<<gridSize,blockSize>>>(dist, r_n , sorted) ; 
    cudaError_t cudaStat = cudaDeviceSynchronize();

    if (cudaStat != CUDA_SUCCESS) {
        printf ("min dist failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
}

void device_brute(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted)
{
    // array of the distances between all q and r points 
    float * dev_dist ; 
    //array of dist from each q point to every r point 
    cudaMalloc((void **)&dev_dist, q_n* r_n * sizeof(float)) ; 

    dim3 grid_size(q_n, r_n, 1) ;
    dim3 block_size(32, 1, 1) ;   

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size>>>(q_points,r_points, dev_dist);
    cudaDeviceSynchronize();

    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,1,1) ; 

    min_dist_f<<<gridSize,blockSize>>>(dev_dist, r_n , sorted) ; 
    cudaDeviceSynchronize();

    cudaFree(dev_dist) ; 
}

//kernels
//finds the sqr euclidan distance between two 128 vector arrays
__global__ void sqrEuclidianDist(des_t_f * q_points, des_t_f * r_points, float * dist_array)   
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
__global__ void min_dist_f(float *  dist, int size ,float4 * sorted)
{
    //            finds the dist array         x dim       y dim pos                      
    int offset = (blockIdx.x * size)+ threadIdx.y * blockDim.x ;
   
    float4 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 
    //maybe better to read values first, however compiler may just be doing it for me
    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+=(blockDim.x * blockDim.y) )
    {

        float temp = dist[i + offset + threadIdx.x]; 
        if(temp < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = temp ;  
            min_2.w = min_2.z ;  
            min_2.z = i + threadIdx.x + threadIdx.y * blockDim.x  ; 
        }
        else{
            if(temp< min_2.y)
            {
                min_2.y = temp;  
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

void host_brute(des_t_f * q_points, des_t_f * r_points, int q_points_size, int r_points_size, float4  * sorted)
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
float host_lenght(des_t_f x, des_t_f y){
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



