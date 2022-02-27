#include <iostream>
#include "curand_kernel.h"
#include <string>
#include "cublas_v2.h"
#include "lsh_h.h"
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

// makes random vectors used to hash 
// atm this is not a very good soulution
void make_vec_h(int nbits, des_t_h2 * &vec)
{
    des_t_h2 * vector = vec ; 
    for (int i = 0; i < nbits; i++)
    {
        for (int ii = 0; ii < 64;  ii++)
        {
             //   if(i == rand_vec_to_zero)
             //   {

             //   vector[i] =  0 ;
             //   }
             //   else{

             //   vector[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) -0.5 ;
            ((half2 *) vector[i])[ii].x = __float2half((float) ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) -0.5)) ;
            ((half2 *) vector[i])[ii].y = __float2half((float) ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) -0.5)) ;
        }
        /* code */
    }
}

class IndexCompare_int
{
    thrust::counting_iterator<int> _index_copy;
    int* _code ;

public:
    IndexCompare_int( thrust::counting_iterator<int> index_copy, int* code)
        : _index_copy( index_copy)
        , _code( code)
    { }

    __host__ __device__
    inline bool operator()( int left, int right ) const
    {
        return (_code[_index_copy[left]] < _code[_index_copy[right]]); 
    }
};

void test_alloc(const char * s, cudaError_t stat)
{        
    if (stat != cudaSuccess) 
    {
        printf ("%s device memory allocation failed \n");
        exit(EXIT_FAILURE) ; 
    }
}

void test_kernel(const char *  s, cudaError_t stat)
{
    if (stat != cudaSuccess) 
    {
        printf ("%s kernel failed \n");
        exit(EXIT_FAILURE) ; 
    }
}

// n_r, l, 1 
// nbits ,1 ,1 for now we use 32 threads but could also use 16 8 4 2 or 64 / 128  
// set the bit for the one random array then combine whitin warp to form 32 bit buckets 
// want buckets to have layaout 
// bucket[0][0] ......... bucket[0][l]
// bucket[n][0] --------- bucket[n][l]
//
__global__ void set_bit(int * buckets, int nbits, float * dot)
{
    uint32_t var = 0 ; 
    // index of the dot prouduct we need 
    //int dot_idx =  blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y * blockDim.x + threadIdx.x ;    
    int dot_idx =  blockIdx.x * nbits  + threadIdx.x ;    
    // only care if its a relevent thread 
    if((threadIdx.x ) < nbits )
    {
        if((dot[dot_idx] )  >= 0 ) 
        {
            var |= 1UL << threadIdx.x;
        }
    }
    var += __shfl_down_sync( 0xffffffff, var, 16 );
    var += __shfl_down_sync( 0xffffffff, var, 8 ); 
    var += __shfl_down_sync( 0xffffffff, var, 4 ); 
    var += __shfl_down_sync( 0xffffffff, var, 2 );
    var += __shfl_down_sync( 0xffffffff, var, 1 );   
    if(threadIdx.x == 0)
    {
       //buckets[blockIdx.x * gridDim.y + blockIdx.y ] = var ;     
       buckets[blockIdx.x] = var ;     
    }  
}
typedef struct{
   half2 x;
   half2 y;
   half2 z;
   half2 w;
} half8;


// what i want is that each thread will read 4 half2, this will lead to 4 128 reads per warp  
// 3 casese 8 16 32 
// 8 -> 4 * 8 byte per -> meaning we would read for 32 dots at a time gving us peak perforamce for both read and write this is easy 
// 16 -> 4 * 16 byte per -> need to read 2 points to get 128 byte per read this is not so easy, write directly to shared.... what to do for the one out of bounds ? hmmm  
// 32 -> 

//each thread will make its own value, this means we get both coalsed reads and writes 
//one check will be done by  amout of bytes we have to read
//so for 8 each thread will read 8 values or 16 bytes. 
__global__ void set_bit_h(int * buckets, int nbits, half2 * dot)
{
    uint32_t var = 0 ; 
    extern __shared__ half2 dots_shared[];  
    //index of the dot prouduct we need 
    //int dot_idx = blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y * blockDim.x + threadIdx.x ;    
    // x, 1, 1
    // 32, y, 1                       start of block             start of warp  
    int warp_idx  = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x ; 
    int idx = warp_idx + threadIdx.x ; 

    // the number of threads in the warp which will write to buckets 
    int active_threads_warp = (( warp_idx + 32) < gridDim.x) ? 32 : (gridDim.x - warp_idx)  ;     
    // how many half2s this warp will read 
    //will be minus if we are out of bounds 
    int to_read_warp = (nbits / 2) * active_threads_warp ;   

    //offset within the block
    int offset_block = threadIdx.y * blockDim.x * (nbits/2) ; 
    // read to shared 
    for (int i = threadIdx.x ; i < to_read_warp; i+=(32))
    {
        dots_shared[offset_block + i] = dot[warp_idx * (nbits/2) + i] ; 
    }
    // a thread will only read from the part which belongs to its own warp -> no need to sync block 
    __syncwarp() ; 
    if(threadIdx.x < active_threads_warp) 
    {
        half2 temp ; 
        for (int i = 0; i < (nbits/2); i++)
        {
            temp = dots_shared[offset_block + threadIdx.x * (nbits/2) + i] 
            if(__hgt(0.0, temp.x))
            {
                var |= 1 << i;
            }
            if(__hgt(0.0, temp.y))
            {
                var |= 1 << i;
            }
        }
        buckets[warp_idx + threadIdx.x] = var ; 
    }
}

// reduce float 
__device__ inline void reduce(float &var)
{
    var += __shfl_down_sync( 0xffffffff, var, 16 );
    var += __shfl_down_sync( 0xffffffff, var, 8 ); 
    var += __shfl_down_sync( 0xffffffff, var, 4 ); 
    var += __shfl_down_sync( 0xffffffff, var, 2 );
    var += __shfl_down_sync( 0xffffffff, var, 1 );   
}


// not usable atm
// called with 
// block, nbits, x, 1 x = is what ever number we need to make 3 warps nbits = 32 -> 3 31 -> 3 16 -> 6 8-> 12       
// grid, (bucket_n * 2 /number of sm), 1, 1.  
//
// want to maximize use of shared memory so there is max one read from global memory per bucket 
//    
__global__ void find_all_neigbours_dist_1(int to_read, int * neighbouring_buckets, int nbits, int * bucket, int n_buckets ) 
{
    // read all the buckets from global memory
    // read n per sm / block 
    __shared__ int buckets ;
    if(threadIdx.x == 0)
    {
        buckets = bucket[blockIdx.x] ; 
    }
    __syncthreads() ; 

    int neigbour = buckets ; 
    neigbour ^= 1UL << threadIdx.x ; 
    neighbouring_buckets[threadIdx.x + n_buckets * blockIdx.x] = neigbour ; 
}

__global__ void find_all_neigbours_dist_2_odd(int to_read, int * neighbouring_buckets, int nbits, int * bucket ) 
{
     // read all the buckets from global memory
    // read n per sm / block 
    __shared__ int buckets ;
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        buckets = bucket[blockIdx.x] ; 
    }
    __syncthreads() ; 

    int neigbour = buckets ; 
    neigbour  ^= 1UL << threadIdx.x ; 
    neigbour  ^= 1UL << ((threadIdx.x + 1 + threadIdx.y) % nbits)  ; 

    neighbouring_buckets[blockIdx.x * (blockDim.x * blockDim.y + nbits) + nbits + threadIdx.x * blockDim.y + threadIdx.y ] = neigbour ; 
}
__global__ void find_all_neigbours_dist_2_pair(int to_read, int * neighbouring_buckets, int nbits, int * bucket ) 
{
    // read all the buckets from global memory
    // read n per sm / block 
    __shared__ int buckets ;
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        buckets = bucket[blockIdx.x] ; 
    }
    __syncthreads() ; 

    int neigbour = buckets ; 
    int val = 0  ; 
    if(threadIdx.y + 1 == blockDim.y )
    {
        val = blockDim.x ; 
    }
    else
    {
        val = ((threadIdx.x + 1 + threadIdx.y) % nbits) ;  
    }
    neigbour  ^= 1UL << threadIdx.x ; 
    neigbour  ^= 1UL << val; 

    neighbouring_buckets[blockIdx.x * (blockDim.x * blockDim.y + nbits) + nbits + threadIdx.x * blockDim.y + threadIdx.y ] = neigbour ; 
}

//find smallest vlaue in the warp and index  
__device__ inline void best_in_warp_float(float4  &min_2)
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


__device__ inline float4 set_sorted(float4 sorted , float4 min)
{
    if(sorted.x > min.x)
    {
        if(sorted.x > min.y)
        {
            sorted.y = min.y ; 
            sorted.x = min.x ; 
            sorted.w = min.w ;  
            sorted.z = min.z ;  
        }
        else
        {
            sorted.y = sorted.x ; 
            sorted.w = sorted.z ; 
            sorted.x = min.x ; 
            sorted.z = min.z ; 
        }
    }
    else
    {
        if (sorted.y > min.x)
        {
            sorted.y = min.x ; 
            sorted.w = min.z ; 
        }
    }
    return sorted ; 
}
// takes 2 buckets and find the 2nns for each point from the query bucket 
// called with 
// grid, y, 1, 1 
// block 32, x, 1 
__global__ void brute_2nn(float4 * sorted, int * index_r, int * index_q, int4 * start_size, des_t_f * r_p, des_t_f *  q_p) 
{
    // use a int for now to test 
    int r_size = 4 ; 
    int4 start_size_q_r = start_size[blockIdx.x] ; 

    __shared__ float4 r_points[32 * 4]   ;
    // dose not need to be shared hmm 
    float4 best ; 
     
    // for each q point 
    for (int i = 0; i < start_size_q_r.y; i += r_size) 
    {
        float4 a ; 
        int count = 0 ; 
        // set shared value and read in q point 
        if((i + threadIdx.y) < start_size_q_r.y)
        {
            // read new q point 
            a = ((float4 * )q_p[index_q[start_size_q_r.w + (threadIdx.y + i)]])[threadIdx.x]; 
            best.x = MAXFLOAT ; 
            best.y = MAXFLOAT ; 
        } 
        // for every r point find dist to q points we have read in  
        for (int ii = 0; ii < start_size_q_r.z ; ii += r_size)
        {
            // read to shared ? 
            __syncthreads(); 
            if((ii + threadIdx.y) < start_size_q_r.z)
            {
                r_points[32 * threadIdx.y + threadIdx.x] = ((float4 * )r_p[index_r[start_size_q_r.x + (threadIdx.y + ii)]])[threadIdx.x];  
                // add if stamtemnt and cal dist for this here ?_? maybe
            }
            __syncthreads() ; 
            if((i + threadIdx.y) < start_size_q_r.y)
            {
                
                int iii = 0 ; 
                while (((iii + ii) < start_size_q_r.z ) && iii < 4)
                {
                    
                    float res = 0.f ; 
                    float4 b = r_points[threadIdx.x + iii * 32]  ; 
                    float4 c ; 
                    c.x = a.x - b.x ; 
                    c.y = a.y - b.y ; 
                    c.z = a.z - b.z ; 
                    c.w = a.w - b.w ; 

                    res =
                    (c.x )*(c.x ) + (c.y )*(c.y ) +
                    (c.z )*(c.z ) + (c.w )*(c.w )  ;  
                    reduce(res) ;    
                    res = __shfl_sync(0xFFFFFFFF, res, 0 ) ;

                    // set value 
                    if(threadIdx.x == count)
                    {
                        if(best.x == MAXFLOAT)
                        {
                            best.x = res ;  
                            best.z = index_r[iii + start_size_q_r.x + ii] ; 
                        }
                        // will never be reached atm 
                        else
                        {
                            best.y = res ;  
                            best.w = index_r[iii + start_size_q_r.x + ii] ; 
                        }
                    } 

                    iii ++ ; 
                    count +=1 ;  

                    if(count == 32)
                    {
                        best_in_warp_float(best) ; 
                        count = 0 ; 
                        if(threadIdx.x == 0)
                        {
                            // could also keep the valus in shared so there is no need to read from sorted more than once hmmm 
                            sorted[index_q[start_size_q_r.w + (threadIdx.y + i)]] = set_sorted(sorted[index_q[start_size_q_r.w + (threadIdx.y + i)]], best ); 
                        }
                        best.x = MAXFLOAT ;
                        best.y = MAXFLOAT ;  
                    }
                }
            }
            __syncthreads() ; 
        }

        if((i + threadIdx.y) < start_size_q_r.y)
        {
            best_in_warp_float(best) ; 
            count = 0 ; 
            if(threadIdx.x == 0)
            {
                sorted[index_q[start_size_q_r.w + (threadIdx.y + i)]] = set_sorted(sorted[index_q[start_size_q_r.w + (threadIdx.y + i)]], best ); 
            }
            best.x = MAXFLOAT ;
            best.y = MAXFLOAT ;  
        }
    }   
}

typedef enum {
    FLOAT_HOST=0, 
    FLOAT_DEVICE=1,
    HALF_DEVICE=2,
} type_mem  ;  

typedef enum {
    DOT_INT_BUCKET=0, 
} lsh_type;  




// main function 
// should be called with 2/4 handels and streams 
int lsh_gpu(void * q_points, void * r_points, int type, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t handle, cudaStream_t * stream, int stream_n, int l, int lsh_type, int nbits)
{
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    // step 1: 
    // fix data
    // will be done in 2 streams  
    des_t_h2 * R;  
    des_t_h2 * Q;

    // we have 2 possible inputs floats/half_floats if half it has to be in device memory if floats it can be device / host 
    // half will be fastest as we do not need to convert 
    // R needs to be in device memory and of type half2 since we use the whole array for every cublas call
    // for R size cublas wants r_n % 8 == 0 

    // floats in host memory  
    // not very optimal 
    // will just use zero copy for now assumese that memory is mapped and pinned 
    // note for some reason this is faster than float_device some times hmmmmm   
    if (type == FLOAT_HOST)
    {
        float * r_copy ; 
        float * q_copy ; 
        // pointer for zero copy
        cudaHostGetDevicePointer(&r_copy, r_points, 0);
        cudaHostGetDevicePointer(&q_copy, q_points, 0);
        
        // malloc R 
        cudaMallocAsync((void **)&R, r_n * sizeof(des_t_h2), stream[0]);
        float2half<<<r_n, 64, 0, stream[0]>>>((float * )r_copy, (half2 * )R) ; 

        // malloc Q  
        cudaMallocAsync((void **)&Q, q_n * sizeof(des_t_h2), stream[1]);
        float2half<<<q_n, 64, 0, stream[1]>>>((float * )q_copy, (half2 * )Q) ; 
    }
    // floats in device memory 
    else if (type == FLOAT_DEVICE )
    {
        // malloc R 
        cudaMallocAsync((void **)&R, r_n * sizeof(des_t_h2), stream[0]);
        float2half<<<r_n, 64, 0, stream[0]>>>((float * )r_points, (half2 * )R) ; 

        // malloc Q  
        cudaMallocAsync((void **)&Q, q_n * sizeof(des_t_h2), stream[1]);
        float2half<<<q_n, 64, 0, stream[1]>>>((float * )q_points, (half2 * )Q) ; 
                
    }
    // halfs in device memory, no need to do anything
    else if(type == HALF_DEVICE)
    {
        // check if we need to pad if we do we need to remake 
        R = (des_t_h2 * )r_points ; 
        Q = (des_t_h2 *)q_points ; 
    }

    // 4 streams is all we need to be 100% sure that we use all the resorces we possibly can
    // if we only run one iteration 4 streams are usless 
    int stream_n;
    if(l == 1)
    {
        stream_n = 2 ;
    }
    else stream_n = 4 ; 

    // cublas want size which is size % 8 = 0 
    // it also makes progrmaing easier 
    // make % 4 or % 2 == 0 insted ? 
    int rand_array_size ; 
    if(nbits <= 8)
    {
        rand_array_size = 8 ; 
    }
    else if(nbits <= 16)
    {
        rand_array_size = 16 ; 
    }
    else
    {
        rand_array_size = 32 ; 
    }
    
       
    des_t_h2 *rand_array[stream_n/2] ;
    // dot from random vector to q / r points 
    half2 * dot_res_r[stream_n/2], * dot_res_q[stream_n/2];
 
    // hash codes  
    int *code_r[stream_n/2], *code_q[stream_n/2];

    // index into bucket array and copy to sort 
    int *index_r[stream_n/2], * index_q[stream_n/2]; 

    // all buckets in use 
    int *buckets_r[stream_n/2], *buckets_q[stream_n/2]; 

    // size of each of the buckets 
    int * buckets_r_size[stream_n/2], * buckets_q_size[stream_n/2] ;  

    // used to reduce by key 
    int * code_by_index_r[stream_n/2], * code_by_index_q[stream_n/2] ; 

    // will give us index_copy[0 -> N] = 0 -> N  
    // used to index into the buckets 
    thrust::counting_iterator<int> index_copy(0);
    
    // will always give us 1 
    // is used to both find number of elemets in each bucket and make an array of all in use buckets 
    thrust::constant_iterator<int> array_of_ones(1) ; 

    // a bucket of all the points each q has to check   
    //int * neighbouring_buckets;

    // thrust pointers for q r 
    // todo check if we need thrust pointers 
    thrust::device_ptr<int> ptr_q_index[stream_n/2];
    thrust::device_ptr<int> ptr_r_index[stream_n/2];

    thrust::device_ptr<int> ptr_code_by_index_q[stream_n/2] ; 
    thrust::device_ptr<int> ptr_code_by_index_r[stream_n/2] ; 

    thrust::device_ptr<int> ptr_code_q[stream_n/2];
    thrust::device_ptr<int> ptr_code_r[stream_n/2];

    thrust::device_ptr<int> ptr_buckets_q[stream_n/2]; 
    thrust::device_ptr<int> ptr_buckets_r[stream_n/2]; 

    thrust::device_ptr<int> ptr_buckets_q_size[stream_n/2] ; 
    thrust::device_ptr<int> ptr_buckets_r_size[stream_n/2] ; 
    
    //malloc
    for (int i = 0; i < stream_n/2; i++)
    {
        // called either by 0 or by 2  
        //will use manged for now
        //cudaMallocAsync((void **)&rand_array[i], sizeof(des_t_h2) * rand_array_size,stream[ i * 2] ) ; 
        cudaMallocManaged((void **)&rand_array[i], sizeof(des_t_h2) * rand_array_size) ; 
        cudaMemsetAsync(rand_array[i], 0, sizeof(des_t_h2) * rand_array_size, stream[i * 2]) ;  

        cudaMallocAsync((void **)&index_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&index_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&code_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&code_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&dot_res_q[i], sizeof(half) *q_n * rand_array_size , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&dot_res_r[i], sizeof(half) *r_n * rand_array_size , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&buckets_q_size[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&buckets_r_size[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&buckets_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&buckets_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&code_by_index_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&code_by_index_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 
        
        thrust::device_ptr<int> ptr_q_index[i] =  thrust::device_pointer_cast(index_q[i]) ;
        thrust::device_ptr<int> ptr_r_index[i] =  thrust::device_pointer_cast(index_r[i]) ;

        thrust::device_ptr<int> ptr_code_by_index_q[i] = thrust::device_pointer_cast(code_by_index_q[i]) ; 
        thrust::device_ptr<int> ptr_code_by_index_r[i] = thrust::device_pointer_cast(code_by_index_r[i]) ; 

        thrust::device_ptr<int> ptr_code_q[i] = thrust::device_pointer_cast(code_q[i]);
        thrust::device_ptr<int> ptr_code_r[i] = thrust::device_pointer_cast(code_r[i]);

        thrust::device_ptr<int> ptr_buckets_q[i] = thrust::device_pointer_cast(buckets_q[i]); 
        thrust::device_ptr<int> ptr_buckets_r[i] = thrust::device_pointer_cast(buckets_r[i]); 
 
        thrust::device_ptr<int> ptr_buckets_q_size[i] = thrust::device_pointer_cast(buckets_q_size[i]);
        thrust::device_ptr<int> ptr_buckets_r_size[i] = thrust::device_pointer_cast(buckets_r_size[i]);
    }

    // for cublas  
    half a = 1.0f;
    half b = 0.0f;
    
    // use to keep track of the streams 

    int stream_counter = 0 ; 
    for (int L = 0; L < l; L++)
    {
        
        // set index arrays  
        thrust::copy(thrust::cuda::par.on(stream[stream_counter * 2]), index_copy, index_copy + q_n, index_q[stream_counter]) ;  
        thrust::copy(thrust::cuda::par.on(stream[stream_counter * 2 + 1]), index_copy, index_copy + r_n, index_r[stream_counter]) ;  

        //todo do on gpu 
        for (int i = 0; i < nbits; i++)
        {
            make_vec_h(nbits, rand_array[stream_counter]);
        }

        cublasSetStream(handle, stream[stream_counter * 2]) ; 
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, q_n, 128, &a, (half *)rand_array, 128, (half *)Q, 128, &b, (half *)dot_res_q[stream_counter], rand_array_size);
        dim3 grid_bit_q(q_n,1,1) ; 
        dim3 block_bit_q(32,1,1) ; 
        set_bit_h<<<grid_bit_q, block_bit_q, 0, stream[stream_counter * 2]>>>(code_q[stream_counter], nbits, dot_res_q[stream_counter]) ; 

        cublasSetStream(handle, stream[stream_counter * 2 + 1]) ; 
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, r_n, 128, &a, (half *)rand_array, 128, (half *)R, 128, &b, (half *)dot_res_r[stream_counter], rand_array_size);
        dim3 grid_bit_r(r_n,1,1) ; 
        dim3 block_bit_r(32,1,1) ; 
        set_bit_h<<<grid_bit_r, block_bit_r, 0, stream[stream_counter * 2 + 1]>>>(code_r[stream_counter], nbits, dot_res_r[stream_counter]) ; 
        
        IndexCompare_int code_r_sort(index_copy, code_r[stream_counter]);
        IndexCompare_int code_q_sort(index_copy, code_q[stream_counter]);



        // will have to use c++ threads because i need the output of reduce by key 
        // sort and reduce for q buckets 
        thrust::sort(thrust::cuda::par.on(stream[stream_counter * 2]), ptr_q_index[stream_counter], ptr_q_index[stream_counter] + q_n, code_q_sort);
        thrust::gather(thrust::cuda::par.on(stream[stream_counter * 2]), ptr_q_index[stream_counter], ptr_q_index[stream_counter] + q_n, ptr_code_q[stream_counter], ptr_code_by_index_q[stream_counter]) ; 
        auto new_end_q = thrust::reduce_by_key(thrust::cuda::par.on(stream[stream_counter * 2]), ptr_code_by_index_q[stream_counter], ptr_code_by_index_q[stream_counter]+ q_n,
                        array_of_ones, ptr_buckets_q[stream_counter], ptr_buckets_q_size[stream_counter]) ; 

        // sort and reduce for r buckets  
        thrust::sort(thrust::cuda::par.on(stream[stream_counter * 2 + 1]), ptr_r_index[stream_counter], ptr_r_index[stream_counter] + r_n, code_r_sort);
        thrust::gather(thrust::cuda::par.on(stream[stream_counter * 2 + 1]), ptr_r_index[stream_counter], ptr_r_index[stream_counter] + r_n, ptr_code_q[stream_counter], ptr_code_by_index_r[stream_counter]) ; 
        auto new_end_r = thrust::reduce_by_key(thrust::cuda::par.on(stream[stream_counter * 2 + 1]), ptr_code_by_index_r[stream_counter], ptr_code_by_index_r[stream_counter]+ r_n, 
                        array_of_ones, ptr_buckets_r[stream_counter], ptr_buckets_r_size[stream_counter]) ; 


        // hmm is this safe lets hope so 

        int n_r_buckets = new_end_r.first - (ptr_buckets_r[stream_counter]) ; 
        int n_q_buckets = new_end_q.first - (ptr_buckets_q[stream_counter]) ; 


        // use thrust lower bound 
        //will give us the first value where we can insert without destroying order  



        // what we need size / start of both  
        //waiting for new end hmmm 



    
    }
    

    return 0 ; 
} 

void lsh_test(des_t_f *q_points, des_t_f *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist, cublasHandle_t handle) 
{  
   
    int size_bucket = 0 ;
    if(max_dist == 1)
    {
        size_bucket = nbits ; 
    }
    else{
        size_bucket = ((nbits * (nbits -1 )) / 2) + nbits ;  
    }

   // thrust pointers for q 
   for (int L = 0; L < l; L++)
    {
        // memsetstuff
        cudaMemset(neighbouring_buckets, 0, sizeof(int) * n_q * size_bucket);

        // set index arrays  
       // dot random vectors with n_r
        // using cublas
        //dim3 grid_dot_r(n_r, nbits, 1) ;
        //dim3 block_dot_r(32, 1, 1) ;   
        //dot_gpu<<<grid_dot_r, block_dot_r>>>(rand_array, r_points, dot_res_r); 

        //cublas dot
        // note the rand array is read as colum major not row major 

        
        // set bit for code_r 
        dim3 grid_bit_r(n_r,1,1) ; 
        dim3 block_bit_r(32,1,1) ; 
        set_bit<<<grid_bit_r, block_bit_r>>>(code_r, nbits, dot_res_r) ; 

        // dot random vectors with q
        //dim3 grid_dot_q(n_q, nbits, 1) ;
        //dim3 block_dot_q(32, 1, 1) ;   
        //dot_gpu<<<grid_dot_q, block_dot_q>>>(rand_array, q_points, dot_res_q); 

        //cublas dot
        // note the rand array is read as colum major not row major 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nbits, n_q, 128, &a, (float *)rand_array, nbits, (float *)q_points, 128, &b, dot_res_q, nbits);

        // set bit for hash values for code_q 
        dim3 grid_bit_q(n_r,1,1) ; 
        dim3 block_bit_q(32,1,1) ; 
        set_bit<<<grid_bit_q, block_bit_q>>>(code_q, nbits, dot_res_q) ; 

        // todo fix dist only need dist one  
       // if(max_dist > 0){
       //      
       //     dim3 grid_bucket(n_q_buckets, 1, 1) ; 
       //     dim3 block_bucket(nbits, 1 ,1) ;                
       //     find_all_neigbours_dist_1<<<grid_bucket, block_bucket>>>(1, neighbouring_buckets, nbits, buckets_q, size_bucket) ; 

       //     if(max_dist == 2) 
       //     {
       //         if(nbits % 2)
       //         {
       //             dim3 grid_bucket(n_q, 1, 1) ; 
       //             dim3 block_bucket(nbits,(nbits - 1) / 2 ,1) ;                
       //             find_all_neigbours_dist_2_odd<<<grid_bucket, block_bucket>>>(1, neighbouring_buckets, nbits, code_q) ; 
       //         }
       //         else
       //         {
       //             dim3 grid_bucket(n_q, 1, 1) ; 
       //             dim3 block_bucket(nbits - 1,(nbits) / 2 ,1) ;                
       //             find_all_neigbours_dist_2_pair<<<grid_bucket, block_bucket>>>(1, neighbouring_buckets, nbits, code_q) ; 
       //         }
       //     }
       // }

        int count_r = 0 ;  
        int count_q = 0 ;  

        int start_index_r = 0 ; 
        int start_index_q = 0 ; 
        
        // what each block in our kernel needs is 
        // number of r and q points to read 
        // start index 
        // index of each q point into the soreted array 

        // we can use block number to index into the array
        int4 * index_size_start ; 
        cudaMallocManaged((void **) &index_size_start, sizeof(int) * n_q_buckets) ; 
        int counter = 0 ;  
        // can this be done on gpu ? todo 

        while (count_q < n_q_buckets && count_r < n_r_buckets)
        {
            if(buckets_q[count_q] == buckets_r[count_r])
            {
                index_size_start[counter].w = start_index_q ; 
                index_size_start[counter].x = start_index_r ; 
                index_size_start[counter].y = buckets_q_size[count_q] ; 
                index_size_start[counter].z = buckets_r_size[count_r] ; 

                // number of points in each bucket 
                //printf("%i == %i bucket r size = %i bucket q size = %i  \n" , buckets_r[count_r], buckets_q[count_q], buckets_r_size[count_r],buckets_q_size[count_q]) ; 

                start_index_q += buckets_q_size[count_q]; 
                start_index_r += buckets_r_size[count_r]; 
                count_r ++ ; 
                count_q ++ ; 
                counter ++ ; 
            }
            else if( buckets_q[count_q] < buckets_r[count_r])
            {
                start_index_q += buckets_q_size[count_q]; 
                count_q ++ ; 
            }
            else
            {
                start_index_r += buckets_r_size[count_r]; 
                count_r ++ ; 
            }
        }
           
        dim3 brute_grid(counter, 1, 1)  ;  
        dim3 brute_bucket(32, 4, 1) ; 
        brute_2nn<<<brute_grid, brute_bucket>>>(sorted, index_r, index_q, index_size_start ,r_points, q_points) ; 
        cudaFree(index_size_start) ; 
   } 
    cudaFree(neighbouring_buckets) ; 
    cudaFree(rand_array) ; 

    cudaFree(index_q) ; 
    cudaFree(index_r) ; 

    cudaFree(code_q) ; 
    cudaFree(code_r) ; 

    cudaFree(dot_res_q) ; 
    cudaFree(dot_res_r) ; 

    cudaFree(buckets_q_size) ; 
    cudaFree(code_by_index_q) ; 
    cudaFree(buckets_r) ; 
    cudaFree(buckets_q) ; 

    cudaFree(buckets_r_size) ; 
    cudaFree(code_by_index_r) ; 
}


