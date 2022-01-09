typedef  float des_t[128];  

void make_vec(int dim, des_t  &vec);

__device__ inline void reduce(float &var); 

//make bucket  
__global__ void dot_gpu(des_t *  rand, des_t * points, float *dot); 
__global__ void set_bit(int *buckets, int nbits, float * dot); 

void lsh_test(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist) ; 

__global__ void find_all_neigbours_dist_1(int to_read, int * neighbouring_buckets, int nbits, int * bucket, int n_buckets ) ; 

__global__ void find_all_neigbours_dist_2_odd(int to_read, int * neighbouring_buckets, int nbits, int * bucket ) ;

__global__ void find_all_neigbours_dist_2_pair(int to_read, int * neighbouring_buckets, int nbits, int * bucket )  ; 

__global__ void brute_2nn(float4 * sorted, int * index_r, int * index_q, int4 * start_size, des_t * r_p, des_t *  q_p) ; 

__device__ inline float4 set_sorted(float4 sorted , float4 min); 

__device__ inline void best_in_warp(float4  &min_2); 

void test_alloc(const char * s, cudaError_t stat) ; 
void test_kernel(const char *  s,cudaError_t stat) ; 
