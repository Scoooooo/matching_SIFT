typedef  float des_t[128];  

__global__ void sqrEuclidianDist(des_t * q_points, des_t * r_points, float * dist_array)   ;
__global__ void min_2_2(float *  dist, int size ,float2 * sorted);
__device__ inline void best_in_warp(float2  &min_2) ; 
__global__ void min_2_3(float *  dist, int size ,float2 * sorted);
__global__ void min_2_4(float *  dist, int size ,float2 * sorted);