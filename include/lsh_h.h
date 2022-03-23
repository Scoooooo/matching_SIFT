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
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>


typedef  float des_t_f[128];  
typedef __half2 des_t_h2[64]; 
typedef __half des_t_h[128]; 

typedef struct{
   half2 x;
   half2 y;
   half2 z;
   half2 w;
} half8;

typedef struct 
{
    half2 a ; 
    half2 b ; 
} half4;


// first 32 bits are for the half2 
// next 32 bits are for the index 

//typedef struct 
//{
//      half2 min2 ; 
//      uint32_t index ; 
//} min2_index;
//
typedef struct 
{
      half2 min2 ; 
      uint32_t index ; 
} m2 ;

typedef union 
{
    unsigned long long ulong ;  
    m2 min_index ; 
}min2_index;


void make_vec_h(int nbits, des_t_h2 * vec) ; 

__global__ void set_bit_h(int * buckets, int nbits, half2 * dot, int size) ; 


__device__ inline void reduce_h(half &var); 
//make bucket  
void lsh_thread_dot_sort_reduce(des_t_h2 * points, uint32_t size, des_t_h2 * rand_array, int rand_array_size, half2 * dot_res, int * code_host, int * code_dev, int * index_host, int * index_dev, int2 * buckets, 
                                cudaStream_t stream, int &buckets_after_reduce) ; 

int lsh_gpu(void * q_points, void * r_points, int type, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t handle, int stream_n, int l, int lsh_type, int nbits) ; 

__global__ void find_all_neigbours_dist_1(int to_read, int * neighbouring_buckets, int nbits, int * bucket, int n_buckets ) ; 

__global__ void brute_2nn_h(min2_index * min_2_index, int shared_size, int * index_r, int * index_q, int4 * start_size, des_t_h2 * Q, des_t_h2 *  R, uint32_t * matches) ;


__device__  inline void set_sorted_h(min2_index * min2, int q_index, half2 temp, u_int32_t best_idx, uint32_t  * mathces); 

__device__ inline void best_in_warp_h(__half2  &min_2, uint32_t &index); 

__global__ void set_matches(min2_index * min2, uint32_t * matches, int size); 

void test_alloc(const char * s, cudaError_t stat) ; 
void test_kernel(const char *  s,cudaError_t stat) ; 
