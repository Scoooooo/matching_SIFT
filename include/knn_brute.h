//change to change dim of the points to be compared 
typedef  float des_t[128];  
// gpu functions 
__global__ void sqrEuclidianDist(des_t * q_points, des_t * r_points, float * dist_array)   ;
__global__ void min_2_2(float *  dist, int size ,float2 * sorted);
__device__ inline void best_in_warp(float4  &min_2) ; 
__global__ void min_2_3(float *  dist, int size ,float2 * sorted);
__global__ void min_dist(float *  dist, int size ,float4 * sorted);
void device_brute(des_t * q_points, des_t * r_points, int q_n, int r_n, float4  * sorted);
//cpu

void host_brute(des_t * q_points, des_t * r_points, int q_points_size, int r_points_size, float4  * sorted);
float host_lenght(des_t x, des_t y);
void host_sort(float * dist, int size, int array_size, float4 * sorted);