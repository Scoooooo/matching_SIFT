
#include <iostream>
#include <string>
#include "knn_brute.h"
int des_t_dim = 128 ;
// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie
void device_brute(des_t * q_points, des_t * r_points, int q_n, int r_n, float2  * sorted)
{
    // array of the distances between all q and r points 
    float * dev_dist ; 

    //array of dist from each q point to every r point 
    cudaMallocManaged((void **) &dev_dist, q_n* r_n * sizeof(float)) ; 

    dim3 block_size(32, 3, 1) ;   
    dim3 grid_size(q_n, r_n, 1) ;

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size, 0>>>(q_points,r_points, dev_dist);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < q_n * r_n; i++)
    {
        printf("len dev %f \n", dev_dist[i]) ; 
    }
     
    dim3 blockSize(32,3,1) ; 
    dim3 gridSize(q_n,1,1) ;

    min_2_4<<<gridSize,blockSize>>>(dev_dist, r_n , sorted) ; 
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
        dist_array[blockIdx.y * gridDim.x + blockIdx.x] = dist; 
    }
}
//find smallest vlaue in the warp 
__device__ inline void best_in_warp(float2  &min_2)
{    
    for (int i = 16; i > 0; i/= 2)
    {          
        float temp = __shfl_down_sync( 0xffffffff, min_2.x, i );
        float temp2 = __shfl_down_sync( 0xffffffff, min_2.y, i );
        if(temp < min_2.x)
        {
          min_2.y = min_2.x ;  
          min_2.x = temp ;
        }
        else{
            if(temp < min_2.y)
            {
                min_2.y = temp ;  
            }
        }
        if(temp2< min_2.y)
        {
            min_2.y = temp2 ;  
        }
    }
}


// 32 threads for each dist_array using shlf  
__global__ void min_2_2(float *  dist, int size ,float2 * sorted)
{
    float2 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 
   
    int offset = (threadIdx.y + blockIdx.x * 32) * size ;

    for (int i = 0; (i + threadIdx.x) < size; i += 32)
    {
        if(dist[i + offset + threadIdx.x] < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = dist[i + offset + threadIdx.x] ;  
        }
        else{
            if(dist[i + offset + threadIdx.x] < min_2.y)
            {
                min_2.y = dist[i + offset + threadIdx.x] ;  
            }
        }
    }

    best_in_warp(min_2) ;

    if(threadIdx.x == 0)
    { 
        sorted[threadIdx.y + blockIdx.x * 32] = min_2 ; 
    }
}

// reading float vs float2 3 4 ?  
// 32 wraps per dist  
__global__ void min_2_3(float *  dist, int size ,float2 * sorted)
{
    float2 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 
   
    int offset = (blockIdx.x) * size + (threadIdx.y * 32) ; 
    if(threadIdx.x == 0){
        //printf("offset %d  \n", offset) ; 
    
    }
    for (int i = 0; (i + threadIdx.x) < size ; i+= 1024)
    {
        if(dist[i + offset + threadIdx.x] < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = dist[i + offset + threadIdx.x] ;  
        }
        else{
            if(dist[i + offset + threadIdx.x] < min_2.y)
            {
                min_2.y = dist[i + offset + threadIdx.x] ;  
            }
        }
    }
    // find best in warp 
    best_in_warp(min_2) ; 
    // find best in all the 32 warps  
    __shared__ float2 best[32] ; 
    if(threadIdx.x == 0)

    {
        best[threadIdx.y] = min_2 ;
    }
    __syncthreads() ; 

    if(threadIdx.y == 0){
        min_2 = best[threadIdx.x] ; 
        best_in_warp(min_2) ; 
    }
    
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        sorted[blockIdx.x] = min_2 ; 
    }
}
 
// x warps per dist 
__global__ void min_2_4(float *  dist, int size ,float2 * sorted)
{
    float2 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 
    //           finds the dist array         x dim       y dim pos                         z dim
    int offset = (blockIdx.x * size)+ threadIdx.y * blockDim.x ;
    
    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+=(blockDim.x * blockDim.y) )
    {
        // float2 temp = 
        if(dist[i + offset + threadIdx.x ] < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = dist[i + offset + threadIdx.x ] ;  
        }
        else{
            if(dist[i + offset + threadIdx.x ] < min_2.y)
            {
                min_2.y = dist[i + offset + threadIdx.x ] ;  
            }
        }
    }

    best_in_warp(min_2) ;   

    __shared__ float2 best[32] ; 
    
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

void host_brute(des_t * q_points, des_t * r_points, int q_points_size, int r_points_size, float2  * sorted)
{
    
    float * lenght;
    cudaMallocHost(&lenght, r_points_size * q_points_size * sizeof(float)) ; 
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
void host_sort(float * dist, int size, int array_size, float2 * sorted)
{   
    for (int i = 0; i < array_size ; i++)
    {
        float2 min_2 ;  
        min_2.x = 0xffffff; 
        min_2.y = 0xffffff; 
        int offset = i * size ; 
        for (int ii = 0; ii < size; ii++)
        {
            if(dist[ii + offset] < min_2.x)
            {
                min_2.y = min_2.x ; 
                min_2.x = dist[ii + offset] ;  
            }
            else{
                if(dist[ii + offset] < min_2.y)
                {
                min_2.y = dist[ii + offset] ;  
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
    float dist = 0 ;  
    for (size_t i = 0; i < des_t_dim; i++)
    {
        float a = vec1[i] ; 
        float b = vec2[i] ;  
        float c = a - b ; 
        dist += c * c ; 
    }
    return dist ; 
}



