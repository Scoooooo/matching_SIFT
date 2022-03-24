#include <iostream>
#include <thread>        
#include "curand_kernel.h"
#include <string>
#include "cublas_v2.h"
#include "lsh_h.h"
#include "helper.h"
#include <algorithm>
#include "cuda_fp16.h"

// make random data 
void make_vector_h(des_t_h &vec)
{
    int dim = 128 ; 
    float temp[128] ;  
    float t = 0 ;     
    for (size_t i = 0; i < dim; i++)
    {
            temp[i] = ( (((static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) - 0.5) > 0) ? 1 : (-1) ; 
            //temp[i] = ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) - 0.5; 
     //       t +=temp[i] * temp[i] ; 
    }
        // t = sqrtf(t)  ; 
    //    for (size_t i = 0; i < dim; i++)
    //    {
    //        temp[i] /= t ;  
    //    } 
    for (int i = 0; i < dim; i++)
     {
         const float to_half = temp[i]; 
         vec[i] = __float2half(to_half); 
     }
     //  for (int i = 0; i < dim; i++)
     //   {
     //       const float to_half = temp[i]; 
     //       vec[i] = __float2half(to_half); 
     //   }
}


// makes random vectors used to hash 
// atm this is not a very good soulution
void make_vec_h(int nbits, des_t_h2 * vec)
{

    des_t_h *arr = (des_t_h *)vec;
    for (int i = 0; i < nbits; i++)
    {
        
        make_vector_h(arr[i]); 
        /* code */
    }
}

// todo fix equal case 
// uses atomiccas to update the min_2_index array which multiple streams will read/wrtie to at the same time
__device__ inline void atomic_min2_update(min2_index * min_2_index, half2 min2, uint32_t idx, int index, uint32_t * matches)
{
    // use to test 
   min2_index min2_test;
   unsigned long long  * address = &min_2_index[index].ulong; 
   unsigned long long  old = *address;
   do {
        min2_test.ulong = old ;
        min2_index min2_new ; 
        min2_new.ulong = min2_test.ulong ; 
        // do our update acording to the values we have atm 
        
        if(__hgt(min2_test.min_index.min2.x, min2.x))
        {
            min2_new.min_index.min2.x= min2.x ; 
            min2_new.min_index.index= idx ; 

            if(__hgt(min2_test.min_index.min2.y, min2.y))
            {
               min2_new.min_index.min2.y = min2.y; 
            }
        }

        else if(__hgt(min2_test.min_index.min2.y, min2.x))
        {
            min2_new.min_index.min2.y = min2.x; 
        }
        
        old = atomicCAS(address, min2_test.ulong,  min2_new.ulong);
        }
    while (old != min2_test.ulong);
}

// class used to sort 
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

// Q/32 + 1 ,1 ,1 
//32 1, 1

// reads the values from the min2 array checks if it is over the threshold and set the matches array 
// todo 
__global__ void set_matches(min2_index * min2, uint32_t * matches, int size)
{   
    // int idx  = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x ; 
    int idx  = 32 * blockIdx.x + threadIdx.x ;  
    if(idx < size)
    {
        // printf("idx = %i matrch = %i min2 x = %f min2 y = %f  match value = %i \n", idx, min2[32 * blockIdx.x + threadIdx.x].index, __half2float(min2[idx].min2.x ),  __half2float(min2[idx].min2.y ), matches[idx] ) ; 
        //matches[idx] = min2[idx].i[1] ; 

        matches[idx] = min2[idx].min_index.index ; 
    }
}


//sets n bits for each value 
__global__ void set_bit_h_n(int * buckets, int nbits, half2 * dot, int size, int bits_per_vectors)
{
    uint32_t var = 0 ; 
    extern __shared__ half2 dots_shared[];  
    //index of the dot prouduct we need 
    //int dot_idx = blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y * blockDim.x + threadIdx.x ;    
    // x, 1, 1
    // 32, y, 1                       start of block             start of warp  
    int warp_idx  = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x ; 
    //int idx = warp_idx + threadIdx.x ; 

    // the number of threads in the warp which will write to buckets 
    int active_threads_warp = (( warp_idx + 32) < size) ? 32 : (size - warp_idx)  ;     
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
            temp = dots_shared[offset_block + threadIdx.x * (nbits/2) + i] ; 
          //  if(threadIdx.x == 0 && threadIdx.y == 0){

          //    //  printf("1  =  %f  2 = %f\n", __half2float(temp.x),  __half2float(temp.y) ) ; 
          //  }
          // for each value we have to set the bits according to some rule
            if(__hgt(0.0, temp.x))
            {
                var |= 1UL << (i * 2);
            }

            if(__hgt(0.0, temp.y))
            {
                var |= 1UL << (i * 2 + 1) ;
            }
        }
//        printf("%i var \n", var) ; 
        buckets[warp_idx + threadIdx.x] = var ; 
    }

}

//each thread will make its own value, this means we get both coalsed reads and writes 
__global__ void set_bit_h(int * buckets, int nbits, half2 * dot, int size)
{
    uint32_t var = 0 ; 
    extern __shared__ half2 dots_shared[];  
    //index of the dot prouduct we need 
    //int dot_idx = blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y * blockDim.x + threadIdx.x ;    
    // x, 1, 1
    // 32, y, 1                       start of block             start of warp  
    int warp_idx  = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x ; 
    //int idx = warp_idx + threadIdx.x ; 

    // the number of threads in the warp which will write to buckets 
    int active_threads_warp = (( warp_idx + 32) < size) ? 32 : (size - warp_idx)  ;     
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
            temp = dots_shared[offset_block + threadIdx.x * (nbits/2) + i] ; 
          //  if(threadIdx.x == 0 && threadIdx.y == 0){

          //    //  printf("1  =  %f  2 = %f\n", __half2float(temp.x),  __half2float(temp.y) ) ; 
          //  }
            if(__hgt(0.0, temp.x))
            {
                var |= 1UL << (i * 2);
            }
            if(__hgt(0.0, temp.y))
            {
                var |= 1UL << (i * 2 + 1) ;
            }
        }
//        printf("%i var \n", var) ; 
        buckets[warp_idx + threadIdx.x] = var ; 
    }

}

// reduce half 
__device__ inline void reduce_h(half &var)
{
    var = __hadd(__shfl_down_sync( 0xffffffff, var, 16 ), var) ; 
    var = __hadd(__shfl_down_sync( 0xffffffff, var, 8 ), var) ; 
    var = __hadd(__shfl_down_sync( 0xffffffff, var, 4 ), var) ; 
    var = __hadd(__shfl_down_sync( 0xffffffff, var, 2 ), var) ; 
    var = __hadd(__shfl_down_sync( 0xffffffff, var, 1 ), var) ; 
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
// reduction warp
__device__ inline void best_in_warp_h(__half2  &min_2, uint32_t &index)
{
    for (int i = 16; i > 0; i/= 2)
    {          
        half2 temp = __shfl_down_sync( 0xffffffff, min_2, i );
        int index_x  = __shfl_down_sync( 0xffffffff, index, i );

        //half2 val = __hgt2(min_2, temp) ; 
        //if(val.x)
        if(__hgt(min_2.x, temp.x))
        {
            min_2.x = temp.x ; 
            index = index_x ; 
            if(__hgt(min_2.y, temp.y))
            //if(val.y)
            {
                min_2.y = temp.y ; 
            }
        }
        else if(__hgt(min_2.y, temp.x))
        {
            min_2.y = temp.x ; 
        }
    }
}


// takes 2 buckets and find the 2nns for each point from the query bucket 
// called with 
// grid, y, 1, 1 
// block 32, x, 1 
__global__ void brute_2nn_h(min2_index * min_2_index, int shared_size, int * index_r, int * index_q, int4 * start_size, des_t_h2 * Q, des_t_h2 *  R, uint32_t * matches) 
{

    // use a int for now to test 
    int4 start_size_q_r = start_size[blockIdx.x] ; 
    // used to read in the r_points we need 
    extern __shared__ half4 r_points[] ;
    // for each q point 

    half2 best ; 
    uint32_t best_idx ; 
  //  if(threadIdx.x == 0)
  //  {
  //      printf("size q = %i \n ", start_size_q_r.y) ; 
  //      printf("block dim = %i \n ", blockDim.y) ; 
  //      printf("size r = %i \n ",start_size_q_r.z) ; 
  //  }
    for(int i = 0; i < (start_size_q_r.y + blockDim.y); i += blockDim.y)  
    {
        half4 a ; 
        int count = 0 ; 

        best.y = MAXFLOAT ;  
        best.x = MAXFLOAT ; 
       
        // set shared value and read in q point 
        if((i + threadIdx.y) < start_size_q_r.y)
        {
            // read new q point 
            a = ((half4 * )Q[index_q[start_size_q_r.w + (threadIdx.y + i)]])[threadIdx.x]; 
            best.y = MAXFLOAT ;  
            best.x = MAXFLOAT ; 
        } 

        // for every r point find dist to q points we have read in  
        for (int ii = 0; ii < (start_size_q_r.z + shared_size); ii += shared_size)
        {
            // read to shared ? 
            __syncthreads(); 

            if((ii + threadIdx.y) < start_size_q_r.z)
            {
                r_points[32 * threadIdx.y + threadIdx.x] = ((half4 * )R[index_r[start_size_q_r.x + (threadIdx.y + ii)]])[threadIdx.x];  
            }
            __syncthreads() ; 
            if((i + threadIdx.y) < start_size_q_r.y)
            {
                int iii = 0 ; 
                while (((iii + ii) < start_size_q_r.z ) && iii < shared_size)
                {
                    
                    half res = 0.f ; 
                    half4 b = r_points[threadIdx.x + iii * 32]  ; 
                    half4 c ;  
                    c.a = __hsub2(a.a,b.a) ; 
                    c.b = __hsub2(a.b,b.b) ; 
//                    c.x = a.x - b.x ; 
//                    c.y = a.y - b.y ; 
//                    c.z = a.z - b.z ; 
//                    c.w = a.w - b.w ; 
//
                    c.a = __hfma2(c.a, c.a, (__hmul2(c.b,c.b))) ;
                    res = __hadd(c.a.x, c.a.y)  ; 
                   // res =
                   // (c.x )*(c.x ) + (c.y )*(c.y ) +
                   // (c.z )*(c.z ) + (c.w )*(c.w )  ;  
                    reduce_h(res) ;    
                    res = __shfl_sync(0xFFFFFFFF, res, 0 ) ;

                 //   if(threadIdx.x == 0 && threadIdx.y == 0)
                 //   {
                 //     //  printf("%f ghmm \n", __half2float(res)); 
                 //   }
                    // set value 
                    if(threadIdx.x == count)
                    {
                //        printf("%f \n ", __half2float(res)); 
                        best.x = res ;  
                        best_idx = index_r[iii + start_size_q_r.x + ii] ; 
                    } 

                    iii ++ ; 
                    count +=1 ;  

                    // this is a bit usless atm
                    if(count == 32)
                    {
                        best_in_warp_h(best, best_idx) ; 
                        count = 0 ; 
                        if(threadIdx.x == 0)
                        {
                            // could also keep the valus in shared so there is no need to read from sorted more than once 
                            // atomic_min2_update(&(min_2_index[index_q[start_size_q_r.w + (threadIdx.y + i)]]).ulong, best, best_idx, index_q[start_size_q_r.w + (threadIdx.y + i)], matches) ; 
                            atomic_min2_update(min_2_index, best, best_idx, index_q[start_size_q_r.w + (threadIdx.y + i)], matches) ; 
                            // set_sorted_h(min_2_index, index_q[start_size_q_r.w + (threadIdx.y + i)], best, best_idx, matches) ; 
                        }
                        best.y = MAXFLOAT ;  
                        best.x = MAXFLOAT ; 
                    }
                }
            }
            __syncthreads() ; 
        }

        if((i + threadIdx.y) < start_size_q_r.y)
        {

            best_in_warp_h(best, best_idx) ; 
            count = 0 ; 
            if(threadIdx.x == 0)
            {

                // atomic_min2_update(min_2_index, best, best_idx, index_q[start_size_q_r.w + (threadIdx.y + i)], matches) ; 
                // atomic_min2_update(&(min_2_index[index_q[start_size_q_r.w + (threadIdx.y + i)]]).ulong, best, best_idx, index_q[start_size_q_r.w + (threadIdx.y + i)], matches) ; 

                atomic_min2_update(min_2_index, best, best_idx, index_q[start_size_q_r.w + (threadIdx.y + i)], matches) ; 
                // set_sorted_h(min_2_index, index_q[start_size_q_r.w + (threadIdx.y + i)], best, best_idx, matches) ; 
               // atomic_min2_update(&min_2_index[index_q[start_size_q_r.w + (threadIdx.y + i)]].ulong, best, best_idx) ; 
               // set_sorted_h(min_2_index, index_q[start_size_q_r.w + (threadIdx.y + i)], best, best_idx, matches) ; 
            }
            best.y = MAXFLOAT ;  
            best.x = MAXFLOAT ; 
        }
    }   
}

// dots q / r, sorts and reduces               
void lsh_thread_dot_sort_reduce(des_t_h2 * points, uint32_t size, des_t_h2 * rand_array, int rand_array_size, half2 * dot_res, int * code_host, int * code_dev, int * index_host, int * index_dev, int2 * buckets, 
                                cudaStream_t stream, int &buckets_after_reduce) 
{   

   // for block bit 
    int y = 1  ; 

    // we need nbits * 2 * 32 * y bytes of shared memeory for each block 
    int shared_size = rand_array_size * sizeof(half) * 32 * y ; 

     // if (stat != CUBLAS_STATUS_SUCCESS) {
   //     printf ("dot failed, cublas 2nn \n");
   //     cublasDestroy(handle);
   //     exit(EXIT_FAILURE);
   // } 
    dim3 grid_bit((size/(32 * y)) + 1, 1,1) ; 
    dim3 block_bit(32,y,1) ; 

    // one block will read for 32 * y points 
    set_bit_h<<<grid_bit, block_bit, shared_size, stream>>>(code_dev, rand_array_size, dot_res, size) ; 

    cudaStreamSynchronize ( stream ) ; 


    thrust::counting_iterator<int> index_copy(0);

    thrust::copy(thrust::cuda::par.on(stream), index_copy, index_copy + size, index_dev) ;  

    IndexCompare_int code_sort(index_copy, code_dev);

    thrust::device_ptr<int> ptr_index =  thrust::device_pointer_cast(index_dev) ;

    thrust::sort(thrust::cuda::par.on(stream), ptr_index, ptr_index + size, code_sort);

    cudaMemcpyAsync(index_host, index_dev, sizeof(int) * size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(code_host, code_dev, sizeof(int) * size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize ( stream ) ; 
    // reduce 
    int start = code_host[index_host[0]] ; 
    buckets[0].x = index_host[0] ; 
    buckets[0].y = 1 ; 
    buckets_after_reduce = 1 ; 
    int count = 0 ; 
    for(int i = 1; i < size; i++)
    {

        if(code_host[index_host[i]] == start)
        {
         //   printf("%i index i = %i \n ", index_host[i], i ) ; 
            buckets[count].y += 1  ; 
        }
        else
        {
     //       printf("start = %i code sorted = %i\n", start, code_host[index_host[i]] ) ; 
            start = code_host[index_host[i]] ; 
            count ++ ; 
            buckets_after_reduce ++ ; 
            buckets[count].x = index_host[i] ; 
            buckets[count].y = 1 ; 
        }
    }

  //  printf("buckets after reduce = %i \n", buckets_after_reduce) ; 
//    thrust::gather(thrust::cuda::par.on(stream), ptr_index, ptr_index + size, ptr_code, ptr_code_by_index) ; 
//    auto new_end = thrust::reduce_by_key(thrust::cuda::par.on(stream), ptr_code_by_index, ptr_code_by_index + size,
//                        array_of_ones, ptr_buckets, ptr_buckets_size) ; 
//
    // size of the new array 
 //   buckets_after_reduce = new_end.first - (ptr_buckets) ; 
}
// 
//void lsh_thread_dot_sort_reduce(des_t_h2 * points, uint32_t size, des_t_h2 * rand_array, int rand_array_size, half2 * dot_res, int * code, int * index, int2 * buckets, 
//                                cudaStream_t stream, cublasHandle_t handle, int &buckets_after_reduce) 
void lsh_thread(des_t_h2 * R, des_t_h2 * Q, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, int nbits, des_t_h2 * rand_array,  half2 * dot_res_r, half2 * dot_res_q,  
                    int * code_r_dev, int * code_r_host, int * code_q_dev, int * code_q_host, int * index_r_dev, int * index_r_host, int * index_q_dev, int * index_q_host, int2 * buckets_r,
                    int2 * buckets_q, int4 * brute, min2_index * min_2_index, cudaStream_t * stream, int stream_index )
{
    int buckets_size_q ;  
    int buckets_size_r ;  

    std::thread q_thread(lsh_thread_dot_sort_reduce, Q, q_n, rand_array, nbits,
                dot_res_q, code_q_host, code_q_dev, index_q_host, index_q_dev,  buckets_q, stream[stream_index * 2], std::ref(buckets_size_q));   

    std::thread r_thread(lsh_thread_dot_sort_reduce, R, r_n, rand_array, nbits,
                dot_res_r, code_r_host, code_r_dev, index_r_host, index_r_dev,  buckets_r, stream[stream_index * 2 + 1], std::ref(buckets_size_r));   

    q_thread.join() ; 
    r_thread.join() ; 

    int count_q = 0 ; 
    int count_r = 0 ;   
    int brute_count = 0 ; 

    // match buckets 
    
    // printf("count q  %i count_ r  %i \n", buckets_size_q, buckets_size_r) ; 

    while (count_q < buckets_size_q && count_r < buckets_size_r)
    {
        if(code_q_host[buckets_q[count_q].x] ==  code_r_host[buckets_r[count_r].x])
        {
            brute[brute_count].w = buckets_q[count_q].x ; 
            brute[brute_count].y = buckets_q[count_q].y ; 
            brute[brute_count].x = buckets_r[count_r].x ; 
            brute[brute_count].z = buckets_r[count_r].y ; 

            // number of points in each bucket 
     //       printf("%i == %i bucket r size = %i bucket q size = %i  \n" , code_q_host[buckets_q[count_q].x], code_r_host[buckets_r[count_r].x] , brute[brute_count].y ,brute[brute_count].z ) ; 

            count_r ++ ; 
            count_q ++ ; 
            brute_count ++ ; 
        }
        else if( code_q_host[buckets_q[count_q].x] < code_r_host[buckets_r[count_r].x])
        {
            count_q ++ ; 
        }
        else
        {
            count_r ++ ; 
        }
    }

    // for (int i = 0; i < 5000; i++)
    // {
        // printf("val %i and %i \n", index_q_host[i], code_q_host[i]) ; 
    // }
    
    int shared_size = 10 ; 
   // printf("asd %i \n", brute_count) ; 
    // 
    dim3 grid_brute(brute_count,1,1) ; 
    dim3 block_brute(32,shared_size,1) ; 

   int4 * brute_dev; 
   // cudaMallocAsync((void **)&brute_dev, sizeof(int4) * brute_count, stream[stream_index * 2]) ; 
   // cudaMemcpyAsync(brute_dev, brute, sizeof(int4) * brute_count ,cudaMemcpyHostToDevice, stream[stream_index * 2]) ; 
    // pointer for zero copy
   cudaHostGetDevicePointer(&brute_dev, brute, 0);
//    if (cudaStat != cudaSuccess)
//    {
//         printf (" we got em ;) \n");
//
//    }
//
    brute_2nn_h<<<grid_brute, block_brute, shared_size *sizeof( des_t_h2 ) ,stream[stream_index * 2 ] >>>(min_2_index, shared_size, index_r_dev, index_q_dev, brute_dev, Q, R, matches) ; 
    
    cudaStreamSynchronize(stream[stream_index * 2 ]) ;
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
int lsh_gpu(void * q_points, void * r_points, int type, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t handle, int stream_n, int l, int lsh_type, int nbits) 
{
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    des_t_h2 * R;  
    des_t_h2 * Q;


    // if we only run one iteration more streams are ussless streams are usless 
    if(l == 1)
    {
        stream_n = 2 ;
    }

    cudaStream_t stream[stream_n] ; 
    // make streams 
    for (int i = 0; i < stream_n; i++)
    {
      cudaStreamCreate(&stream[i]); 
    }

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

    // cublas want size which is size % 8 = 0 
    // it also makes progrmaing easier 
    // make % 4 or % 2 == 0 insted ? 
    // 2 8 4 cublas wants 8 but at this array size i dont think it is worth 
    int rand_array_size =  (((nbits % 2) == 0) ? nbits : (nbits + 1));   

//    if(nbits <= 8)
//    {
//        rand_array_size = 8 ; 
//    }
//    else if(nbits <= 16)
//    {
//        rand_array_size = 16 ; 
//    }
//    else
//    {
//        rand_array_size = 32 ; 
//    }
    
    des_t_h2 * rand_array_dev[stream_n/2] ;
    des_t_h2 * rand_array_host[stream_n/2] ;
    // dot from random vector to q / r points 
    half2 * dot_res_r[stream_n/2], * dot_res_q[stream_n/2];
    
    // hash codes  
    int *code_r_dev[stream_n/2], *code_q_dev[stream_n/2];
    int *code_r_host[stream_n/2], *code_q_host[stream_n/2];

    // index into bucket array and copy to sort 
    int *index_r_dev[stream_n/2], * index_q_dev[stream_n/2]; 
    int *index_r_host[stream_n/2], * index_q_host[stream_n/2]; 

    // all buckets in use 
    int2 *buckets_r[stream_n/2], *buckets_q[stream_n/2]; 

    //inpuit to brute force 
    int4 * brute_host[stream_n/2]; 

    // size of each of the buckets 
    int * buckets_r_size[stream_n/2], * buckets_q_size[stream_n/2] ;  

    // used to reduce by key 
    int * code_by_index_r[stream_n/2], * code_by_index_q[stream_n/2] ; 

//    // will give us index_copy[0 -> N] = 0 -> N  
//    // used to index into the buckets 
//    thrust::counting_iterator<int> index_copy(0);
//    
//    // will always give us 1 
//    // is used to both find number of elemets in each bucket and make an array of all in use buckets 
//    thrust::constant_iterator<int> array_of_ones(1) ; 


    // used to keep track of the 2 min values and index of min 
    // will be writen to by multiple blocks at once 
    min2_index * min_2_index ; 

    cudaMallocAsync((void **)&min_2_index, sizeof(unsigned long long int) * q_n, stream[0] ) ; 

    // hmmmm
    thrust::fill(thrust::cuda::par.on(stream[0]), (half * )min_2_index,(half *)  (min_2_index + sizeof(min2_index) * q_n), MAXFLOAT) ; 

    // a bucket of all the points each q has to check   
    //int * neighbouring_buckets;

    printf("%i strean n  \n", stream_n) ; 
    //malloc
    for (int i = 0; i < stream_n/2; i++)
    {
        // called either by 0 or by 2  
        //will use manged for now
        //cudaMallocAsync((void **)&rand_array[i], sizeof(des_t_h2) * rand_array_size,stream[ i * 2] ) ; 
 //       cudaMemsetAsync(rand_array[i], 0, sizeof(des_t_h2) * rand_array_size, stream[i * 2]) ;  
        cudaMallocHost((void **)&rand_array_host[i], sizeof(des_t_h2) * rand_array_size) ; 
        cudaMalloc((void **)&rand_array_dev[i], sizeof(des_t_h2) * rand_array_size) ; 

        cudaMallocAsync((void **)&dot_res_q[i], sizeof(half) *q_n * rand_array_size , stream[ i * 2] ) ; 
        cudaMallocAsync((void **)&dot_res_r[i], sizeof(half) *r_n * rand_array_size , stream[ i * 2 + 1] ) ; 

//        cudaMallocAsync((void **)&index_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
//        cudaMallocAsync((void **)&index_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 
//
//        cudaMallocAsync((void **)&code_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
//        cudaMallocAsync((void **)&code_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocAsync((void **)&index_q_dev[i], sizeof(int) *q_n, stream[i * 2]) ; 
        cudaMallocAsync((void **)&index_r_dev[i], sizeof(int) *r_n, stream[i * 2 + 1 ]) ; 

        cudaMallocHost((void **)&index_q_host[i], sizeof(int) *q_n) ; 
        cudaMallocHost((void **)&index_r_host[i], sizeof(int) *r_n) ; 

        cudaMallocAsync((void **)&code_q_dev[i], sizeof(int) *q_n, stream[i * 2]) ; 
        cudaMallocAsync((void **)&code_r_dev[i], sizeof(int) *r_n, stream[i * 2 + 1 ]) ; 

        cudaMallocHost((void **)&code_q_host[i], sizeof(int) *q_n) ; 
        cudaMallocHost((void **)&code_r_host[i], sizeof(int) *r_n) ; 

       // cudaMallocAsync((void **)&buckets_q_size[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
       // cudaMallocAsync((void **)&buckets_r_size[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 

        cudaMallocHost((void **)&buckets_q[i], sizeof(int2) *q_n ) ; 
        cudaMallocHost((void **)&buckets_r[i], sizeof(int2) *r_n ) ; 
        
        cudaMallocHost((void **)&brute_host[i], sizeof(int4) *q_n, cudaHostAllocMapped) ; 

       // cudaMallocAsync((void **)&code_by_index_q[i], sizeof(int) *q_n , stream[ i * 2] ) ; 
       // cudaMallocAsync((void **)&code_by_index_r[i], sizeof(int) *r_n , stream[ i * 2 + 1] ) ; 
    }

    // use to keep track of the streams 
    // for now we assume that the first thread will be done first ..... 
    // to do use mutex / shared varible to fix this  
    // for cublas  

    half a = 1.0f;
    half b = 0.0f;    

    int number_threads = stream_n /2 ; 
    int in_use_threads = 0 ; 
    int active_threads = 0 ; 
    std::thread threads[number_threads] ; 

    //start threads 
    while((in_use_threads < number_threads) && (in_use_threads < l))
    {
            make_vec_h(rand_array_size, rand_array_host[in_use_threads]);
            cudaMemcpyAsync(rand_array_dev[in_use_threads], rand_array_host[in_use_threads],sizeof(des_t_h2) * rand_array_size,cudaMemcpyHostToDevice, stream[in_use_threads * 2]);

            cudaStreamSynchronize(stream[in_use_threads * 2])  ; 
            cublasSetStream(handle, stream[in_use_threads * 2]) ; 
            //  cublasStatus_t stat =
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, q_n, 128, &a, (half *)rand_array_dev[in_use_threads], rand_array_size, (half *)Q, 128, &b, (half *)dot_res_q[in_use_threads], rand_array_size);
           
            cudaStreamSynchronize(stream[in_use_threads * 2 + 1])  ; 
            cublasSetStream(handle, stream[in_use_threads * 2 + 1]) ; 
            // stat =
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, r_n, 128, &a, (half *)rand_array_dev[in_use_threads], rand_array_size, (half *)R, 128, &b, (half *)dot_res_r[in_use_threads], rand_array_size);
            threads[in_use_threads] = std::thread(lsh_thread, R, Q, q_n, r_n, matches, threshold, rand_array_size, (des_t_h2 * )(rand_array_dev[in_use_threads]),
                                    (half2 * )(dot_res_r[in_use_threads]), (half2 *)(dot_res_q[in_use_threads]), (int * )(code_r_dev[in_use_threads]), (int * )(code_r_host[in_use_threads]), 
                                    (int *)(code_q_dev[in_use_threads]), (int *)(code_q_host[in_use_threads]), (int *)(index_r_dev[in_use_threads]), (int *)(index_r_host[in_use_threads]),
                                    (int *)(index_q_dev[in_use_threads]), (int *)(index_q_host[in_use_threads]),(int2 * )(buckets_r[in_use_threads]), (int2 * )(buckets_q[in_use_threads]),(int4 *)(brute_host[in_use_threads]), 
                                    min_2_index, &stream[in_use_threads * 2], 0) ; 
           active_threads ++ ; 
           in_use_threads ++ ; 
    }

    if(active_threads == in_use_threads)
    {
        in_use_threads = 0 ; 
    }

    for (int L = number_threads ; L < l; L++) 
    {

        threads[in_use_threads].join() ; 
        make_vec_h(rand_array_size, rand_array_host[in_use_threads]);
        cudaMemcpyAsync(rand_array_dev[in_use_threads], rand_array_host[in_use_threads],sizeof(des_t_h2) * rand_array_size,cudaMemcpyHostToDevice, stream[in_use_threads * 2]);

        cudaStreamSynchronize(stream[in_use_threads * 2])  ; 
        cublasSetStream(handle, stream[in_use_threads * 2]) ; 
        // cublasStatus_t stat =
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, q_n, 128, &a, (half *)rand_array_dev[in_use_threads], rand_array_size, (half *)Q, 128, &b, (half *)dot_res_q[in_use_threads], rand_array_size);
        cublasSetStream(handle, stream[in_use_threads * 2 + 1]) ; 
        // stat =
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rand_array_size, r_n, 128, &a, (half *)rand_array_dev[in_use_threads], rand_array_size, (half *)R, 128, &b, (half *)dot_res_r[in_use_threads], rand_array_size);

       threads[in_use_threads] = std::thread(lsh_thread, R, Q, q_n, r_n, matches, threshold, rand_array_size, (des_t_h2 * )(rand_array_dev[in_use_threads]),
                                (half2 * )(dot_res_r[in_use_threads]), (half2 *)(dot_res_q[in_use_threads]), (int * )(code_r_dev[in_use_threads]), (int * )(code_r_host[in_use_threads]), 
                                (int *)(code_q_dev[in_use_threads]), (int *)(code_q_host[in_use_threads]), (int *)(index_r_dev[in_use_threads]), (int *)(index_r_host[in_use_threads]),
                                (int *)(index_q_dev[in_use_threads]), (int *)(index_q_host[in_use_threads]),(int2 * )(buckets_r[in_use_threads]), (int2 * )(buckets_q[in_use_threads]),(int4 *)(brute_host[in_use_threads]), 
                                min_2_index, &stream[in_use_threads * 2], 0) ; 
        in_use_threads ++ ; 
        if(in_use_threads == number_threads)
        {
            in_use_threads = 0 ; 
        }
    }

    printf("active %i \n", active_threads) ; 
    //go back to the last started thread 
    
    int last_started = in_use_threads;  
    for (int i = 0; i < (active_threads + 1); i++)
    {
        printf("%i last started \n", last_started); 
        last_started = ((last_started - 1 ) == -1 ) ? (active_threads - 1) :  (last_started - 1) ;  
    }
     
    for (int i = 0; i < (active_threads); i++)
    {
        printf("wait on %i \n", last_started) ; 
        threads[last_started].join() ; 
        last_started = ((last_started + 1 ) == active_threads) ? (0) :  (last_started + 1 ) ;  
    }
    dim3 grid((q_n/32) +1,1,1); 
    dim3 block(32,1,1) ; 
    set_matches<<<grid,block >>>(min_2_index, matches, q_n) ; 
    return 0 ; 
} 
//void lsh_thread(des_t_h2 * R, des_t_h2 * Q, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, int nbits, des_t_h2 * rand_array,  half2 * dot_res_r, half2 * dot_res_q,  
//                    int * code_r, int * code_q, int * index_r, int * index_q, int2 * buckets_r, int2 * buckets_q, int4 * brute, thrust::counting_iterator<int> * index_copy, min2_index * min_2_index, 
//                    cudaStream_t * stream, cublasHandle_t handle, int stream_index )