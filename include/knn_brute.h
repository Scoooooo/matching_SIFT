//change to change dim of the points to be compared 
int des_t_dim = 128 ; 
typedef  float des_t[128];  
// gpu functions 
__global__ void sqrEuclidianDist(des_t * q_points, des_t * r_points, float * dist_array)   ;
__global__ void min_2_2(float *  dist, int size ,float2 * sorted);
__device__ inline void best_in_warp(float2  &min_2) ; 
__global__ void min_2_3(float *  dist, int size ,float2 * sorted);
__global__ void min_2_4(float *  dist, int size ,float2 * sorted);
//cpu
void sort_host(float * dist, int size, int dim, float2 * sorted) ; 