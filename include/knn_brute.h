//change to change dim of the points to be compared 
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "stdint.h"
typedef  float des_t_f[128]; 
typedef __half2 des_t_h2[64]; 

// gpu functions 
__global__ void sqrEuclidianDist(des_t_f * q_points, des_t_f * r_points, float * dist_array)   ;
__device__ inline void best_in_warp_float(float4  &min_2) ; 
__device__ inline void best_in_warp(__half2  &min_2, int2 &index) ; 
__global__ void min_dist_f(float *  dist, int size ,float4 * sorted);
__global__ void min_dist_h(__half2 *  dist, int size , int * matches);
void device_brute(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted);

// float 
void cublas_2nn_brute_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, float * dist,cublasHandle_t handle); 
void cublas_2nn_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle); 

//half float 
int cublas_2nn_sift(void * q_points, void * r_points, int type, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t * handle, cudaStream_t * stream, int stream_n) ; 
int cublas_2nn_sift_batch(void * q_points, des_t_h2 * Q, int type, des_t_h2 * R, uint32_t q_n, uint32_t r_n, half2 * dist, uint32_t * matches, float threshold, cublasHandle_t handle, cudaStream_t stream); 
__global__ void find_matches(half2 *  dist, int size , uint32_t * matches, float threshold); 
__global__ void float2half(float * points, half2 * output) ; 
__device__ inline void min_half(__half2  &min_2, __half2 temp, int2 &index, int2 temp_index); 

//cpu
void host_brute(des_t_f * q_points, des_t_f * r_points, int q_points_size, int r_points_size, float4  * sorted);
float host_lenght(des_t_f x, des_t_f y);
void host_sort(float * dist, int size, int array_size, float4 * sorted);