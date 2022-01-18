//change to change dim of the points to be compared 
#include "cublas_v2.h"
#include "cuda_fp16.h"
typedef  float des_t_f[128]; 
typedef  __half des_t_h[128]; 

// gpu functions 
__global__ void sqrEuclidianDist(des_t_f * q_points, des_t_f * r_points, float * dist_array)   ;
__device__ inline void best_in_warp(float4  &min_2) ; 
__global__ void min_dist_f(float *  dist, int size ,float4 * sorted);
__global__ void min_dist_h(__half *  dist, int size ,float4 * sorted);
void device_brute(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted);

// float 
void cublas_2nn_brute_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, float * dist,cublasHandle_t handle); 
void cublas_2nn_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle); 

//half float 
void cublas_2nn_brute_h(des_t_h * q_points, des_t_h * r_points, int q_n, int r_n, float4  * sorted, float * dist,cublasHandle_t handle); 
void cublas_2nn_h(des_t_h * q_points, des_t_h * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle); 

//cpu
void host_brute(des_t_f * q_points, des_t_f * r_points, int q_points_size, int r_points_size, float4  * sorted);
float host_lenght(des_t_f x, des_t_f y);
void host_sort(float * dist, int size, int array_size, float4 * sorted);