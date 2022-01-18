#include <iostream>
#include "curand_kernel.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
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
// use to change rand vectors 
int rand_vec_to_zero  = 0 ; 

void make_vec(int dim, des_t_f &vec)
{
    float * vector = vec ; 
    for (size_t i = 0; i < dim; i++)
    {
     //   if(i == rand_vec_to_zero)
     //   {

     //   vector[i] =  0 ;
     //   }
     //   else{

     //   vector[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) -0.5 ;
     //   }
        vector[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) -0.5 ;
    } 

    rand_vec_to_zero += 2   ; 
}

class IndexCompare
{
    thrust::counting_iterator<int> _index_copy;
    int* _code ;

public:
    IndexCompare( thrust::counting_iterator<int> index_copy, int* code)
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
__global__ void set_bit(int *buckets, int nbits, float * dot)
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


// not in use using cublas instead  

//want dot array to be 
// point 0 * rand [0][0 - nbits], .......   point n * rand [n][0 - nbits]
// point n * rand [0][0 - nbits], .......   point n * rand [n][0 - nbits] 
// 
// want rand[n] to only be read by blocks on the same sm. load in shared memory  
//  or want points[n ] to only be read by blocks on the same sm TODO  
__global__ void dot_gpu(des_t_f *  rand, des_t_f * points, float *dot)
{
    // called with 
    // n, nbits, l grid  
    //32 1 1 block 
    // could change to 32, x ,1 block todo test if faster 
    
    float res = 0.f ; 
    float4 a = ((float4 * )points[blockIdx.x])[threadIdx.x ];
    // row major 
    //float4 b = ((float4 * )rand[blockIdx.z * gridDim.y + blockIdx.y])[threadIdx.x]; 
    float4 b ; 

    // colom major  
    b.x =  ((float *) rand) [(threadIdx.x * 4) * gridDim.y + blockIdx.y]; 
    b.y =  ((float *) rand) [((threadIdx.x * 4) + 1) * gridDim.y + blockIdx.y]; 
    b.z =  ((float *) rand) [((threadIdx.x * 4) + 2) * gridDim.y + blockIdx.y]; 
    b.w =  ((float *) rand) [((threadIdx.x * 4) + 3) * gridDim.y + blockIdx.y]; 
    res =
        (a.x )*(b.x ) + (a.y )*(b.y ) +
        (a.z )*(b.z ) + (a.w )*(b.w )  ;  
    reduce(res) ; 
    if(threadIdx.x == 0)
    {
        dot[blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y + blockIdx.y] = res ;     
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
__device__ inline void best_in_warp(float4  &min_2)
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
// takes 2 buckets and find the 2nns 
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
                        best_in_warp(best) ; 
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
            best_in_warp(best) ; 
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



void lsh_test(des_t_f *q_points, des_t_f *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist, cublasHandle_t handle) 
{  
    // see how much memory we have  
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    

    int size_bucket = 0 ;
    if(max_dist == 1)
    {
        size_bucket = nbits ; 
    }
    else{
        size_bucket = ((nbits * (nbits -1 )) / 2) + nbits ;  
    }

    // not accurate atm
    //printf("we need %i mb of space ",(((n_r * 4 * 4)+ (n_q * 4 * 4) + sizeof(int) * n_q * size_bucket) + nbits * 4 * n_q + nbits * 4 * n_r + 4 * 128 * nbits) / 1024 ) ; 

    // arry of vectors 
    des_t_f *rand_array;

    // hash codes  
    int *code_r, *code_q;

    // index into bucket array and copy to sort 
    int *index_r, * index_q; 

    // all buckets in use 
    int *buckets_r, *buckets_q; 

    // size of each of the buckets 
    int * buckets_r_size, * buckets_q_size ;  

    // used to reduce by key 
    int * code_by_index_r, * code_by_index_q ; 
    // will give us index_copy[0 -> N] = 0 -> N  
    // used to index into the buckets 
    thrust::counting_iterator<int> index_copy(0);
    
    // will always give us 1 
    // is used to both find number of elemets in each bucket and make an array of all in use buckets 
    thrust::constant_iterator<int> array_of_ones(1) ; 

    // dot from random vector to q / r points 
    float * dot_res_r, * dot_res_q;
 
    // a bucket of all the points each q has to check   
    int *neighbouring_buckets;
    // number of buckets within hamming distance r given n bits
    cudaMallocManaged((void **)&neighbouring_buckets, sizeof(int) * n_q * size_bucket);

    cudaMallocManaged((void **)&rand_array, sizeof(des_t_f) * nbits);

    cudaMallocManaged((void **)&index_r, sizeof(int) * n_r);  
    cudaMallocManaged((void **)&index_q, sizeof(int) * n_q);  
 
    // need to index with a smaller array 
    // this would in the worst case use 17 gb of memory :(
    // cudaMallocManaged((void **)&bucket_start_r, (2 << nbits) * sizeof(int));

    cudaMallocManaged((void **)&code_r, sizeof(int) * n_r);
    cudaMallocManaged((void **)&code_q, sizeof(int) * n_q);

    cudaMallocManaged((void **)&dot_res_r, nbits * n_r* sizeof(float)); 
    cudaMallocManaged((void **)&dot_res_q, nbits * n_q* sizeof(float));

    cudaMallocManaged((void **)&buckets_r_size, sizeof(int) * n_r);
    cudaMallocManaged((void **)&buckets_r, sizeof(int) * n_r);
    cudaMallocManaged((void **)&code_by_index_r, sizeof(int) * n_r);
    
    cudaMallocManaged((void **)&buckets_q_size, sizeof(int) * n_q);
    cudaMallocManaged((void **)&buckets_q, sizeof(int) * n_q);
    cudaMallocManaged((void **)&code_by_index_q, sizeof(int) * n_q);

    //fill sorted with MAXFLOAT 
    thrust::fill(thrust::device,(float * )sorted,(float*)( sorted + n_q * 4), MAXFLOAT) ; 

    // cublas 
    float a = 1.0f;
    float b = 0.0f;
    
    IndexCompare code_r_sort(index_copy, code_r);
    IndexCompare code_q_sort(index_copy, code_q);
    // thrust pointers for q 
    // todo check if we need thrust pointers 
    thrust::device_ptr<int> ptr_q_index = thrust::device_pointer_cast(index_q);
    thrust::device_ptr<int> ptr_code_by_index_q = thrust::device_pointer_cast(code_by_index_q);
    thrust::device_ptr<int> ptr_code_q = thrust::device_pointer_cast(code_q);
    thrust::device_ptr<int> ptr_buckets_q = thrust::device_pointer_cast(buckets_q);
    thrust::device_ptr<int> ptr_buckets_q_size = thrust::device_pointer_cast(buckets_q_size);
    // thrust pointers for r
    thrust::device_ptr<int> ptr_r_index = thrust::device_pointer_cast(index_r);
    thrust::device_ptr<int> ptr_code_by_index_r = thrust::device_pointer_cast(code_by_index_r);
    thrust::device_ptr<int> ptr_code_r = thrust::device_pointer_cast(code_r);
    thrust::device_ptr<int> ptr_buckets_r = thrust::device_pointer_cast(buckets_r);
    thrust::device_ptr<int> ptr_buckets_r_size = thrust::device_pointer_cast(buckets_r_size);
    for (int L = 0; L < l; L++)
    {
        // memsetstuff
        cudaMemset(neighbouring_buckets, 0, sizeof(int) * n_q * size_bucket);

        // set index arrays  
        thrust::copy(index_copy,index_copy+ n_q,index_q) ;  
        thrust::copy(index_copy,index_copy+ n_r,index_r) ;  

        // to do random vectos gpu curand / thrust 
       // make random vectors
        for (int i = 0; i < nbits; i++)
        {
            make_vec(128, rand_array[i]);
        }
        //using to see how setting values in the rand vec to 0 changes things         
        rand_vec_to_zero = l ; 


        // dot random vectors with n_r
        // using cublas
        //dim3 grid_dot_r(n_r, nbits, 1) ;
        //dim3 block_dot_r(32, 1, 1) ;   
        //dot_gpu<<<grid_dot_r, block_dot_r>>>(rand_array, r_points, dot_res_r); 

        //cublas dot
        // note the rand array is read as colum major not row major 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nbits, n_r, 128, &a, (float *)rand_array, nbits, (float *)r_points, 128, &b, dot_res_r, nbits);

        
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


        // sort and reduce for r buckets  
        thrust::sort(ptr_r_index, ptr_r_index + n_r, code_r_sort );
        thrust::gather(thrust::device, ptr_r_index, ptr_r_index + n_r, ptr_code_r, ptr_code_by_index_r) ; 
        auto new_end_r = thrust::reduce_by_key( ptr_code_by_index_r, ptr_code_by_index_r+ n_r, array_of_ones, ptr_buckets_r, ptr_buckets_r_size) ; 

        // sort and reduce for q buckets 
        thrust::sort(ptr_q_index, ptr_q_index + n_q, code_q_sort );
        thrust::gather(thrust::device, ptr_q_index, ptr_q_index + n_q, ptr_code_q, ptr_code_by_index_q) ; 
        auto new_end_q = thrust::reduce_by_key( ptr_code_by_index_q, ptr_code_by_index_q+ n_q, array_of_ones, ptr_buckets_q, ptr_buckets_q_size) ; 

        int n_r_buckets = new_end_r.first - (ptr_buckets_r) ; 
        int n_q_buckets = new_end_q.first - (ptr_buckets_q) ; 

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
        // can this be done on gpu ? threads ? todo 
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


