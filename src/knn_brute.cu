#include "knn_brute.h"
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



