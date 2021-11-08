#include <iostream>
#include "curand_kernel.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
#include "knn_brute.h"
#include <memory>
#include "helper.h"
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

#define PREFER_CPU 0

#if PREFER_CPU == 0
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#endif

using namespace std;
int fact(int n){
    return (n==1 || n==0) ? 1: n * fact(n - 1);
}
void host_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
{

    des_t *rand_array;
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits * l);
    // make random vectors
    for (size_t i = 0; i < l * nbits; i++)
    {
        make_vec(128, rand_array[i]);
    }
    // make an array of ints with one int for each r_point
    int *code;
    int * code_test ; 
    int *index;
    // need a copy to sort using sort
    int *index_copy;
    int *bucket_start;

    int * index_test ; 
    int * index_copy_test ; 
    cudaMallocManaged((void **)&index, sizeof(int) * n_r * l);
    cudaMemset(index, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&index_copy, sizeof(int) * n_r * l);
    cudaMemset(index_copy, 0, sizeof(int) * n_r * l);
    
    cudaMallocManaged((void **)&index_test, sizeof(int) * n_r * l);
    cudaMemset(index_test, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&index_copy_test, sizeof(int) * n_r * l);
    cudaMemset(index_copy_test, 0, sizeof(int) * n_r * l);


    cudaMallocManaged((void **)&bucket_start, (2 << nbits) * sizeof(int));

    for (int i = 0; i < (2 << nbits); i++)
    {
        bucket_start[i] = -1;
    }

    cudaMallocManaged((void **)&code, sizeof(int) * n_r * l);
    cudaMemset(code, 0, sizeof(int) * n_r * l);
    cudaMallocManaged((void **)&code_test, sizeof(int) * n_r * l);
    cudaMemset(code_test, 0, sizeof(int) * n_r * l);


    // test if our gpu dot works 
    float * dot_res;
    cudaMallocManaged((void **)&dot_res, l * nbits * n_r* sizeof(float));
    double s = start_timer() ; 
    dim3 grid(n_r, nbits, l) ;
    dim3 block(32, 1, 1) ;   
    dot_gpu<<<grid, block>>>(rand_array, r_points, dot_res); 
    cudaDeviceSynchronize();
    grid.z = 1 ; 
    grid.y = l ; 
    block.x = nbits ;
    set_bit<<<grid, block>>>(code_test, nbits, dot_res) ; 
    cudaDeviceSynchronize();
    print_time(s, "gpu") ; 
    s = start_timer() ; 
    for (int i = 0; i < n_r; i++)
    {
        for (int ii = 0; ii < l; ii++)
        {
            for (int iii = 0; iii < nbits ; iii++)
            {
                float sum = dot(r_points[i] ,rand_array[iii + ii * nbits] );
          //     printf("sum = %f ", sum) ; 
          //     printf("gpu = %f \n", dot_res[i * nbits * l + ii * nbits + iii]) ; 
                if (sum <= 0)
                {
                    code[i * l + ii ] |= 1UL << iii;
                }

            }
                if(code[i * l + ii] != code_test[i * l + ii])
                {
                    printf("cpu = %i ", code[i * l + ii]) ; 
                    printf("gpu = %i \n", code_test[i*l + ii] ) ;                       
 
                }
       }
    }
    print_time(s, "cpu") ; 
     
    
 //   // dot all vectors and add the bit to the coresponding int bit for the r points
 //   for (int i = 0; i < l; i++)
 //   {
 //       for (int ii = 0; ii < n_r; ii++)
 //       {
 //           //            printf(" %i bucket = ", (ii + i * n_r));
 //           for (int iii = 0; iii < nbits; iii++)
 //           {
 //               float sum = dot(r_points[ii], rand_array[iii + i * nbits]);
 //               if (sum <= 0)
 //               {
 //                   code[ii + i * n_r] |= 1UL << iii;
 //               }
 //           }
 //           //           printf(" %i ", code[ii + i * n_r]);
 //           //           printf("\n \n");
 //       }
 //   }

    // make buckets for r points
    grid.x = n_r/32 +1 ; 
    grid.y = l ; 
    block.x = 32 ; 
    block.y = 3 ; 
    set_bucket<<<grid,block>>>(index_test, index_copy_test, n_r) ; 
    cudaDeviceSynchronize();
    for (int i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_r; ii++)
        {
            index[ii + i * n_r] = ii;
            index_copy[ii + i * n_r] = ii;
        }
    }


    std::sort(index, index + n_r * l, [&](const int &i, const int &j) -> bool
              { return (code[index_copy[i]] < code[index_copy[j]]); });
//    std::sort(index_test, index_test + n_r * l, [&](const int &i, const int &j) -> bool
//              { return (code_test[index_copy_test[i]] < code_test[index_copy_test[j]]); });


   // s = start_timer(); 
   // int *helper;
   // cudaMallocManaged((void **)&helper, sizeof(int) * n_r * l);

   // //set bucket start
   // dim3 block_size(32, 1, 1);
   // dim3 grid_size(((l * n_r) / 32) + 1, 1, 1);
   // set_helper<<<grid_size, block_size>>>(helper, code, index);

   // cudaDeviceSynchronize();
   // dim3 blocks(32, 1, 1);
   // dim3 grids(n_r * l, 1, 1);

   // set_bucket_start<<<grids, blocks>>>(helper, bucket_start, l, n_r);
   // cudaDeviceSynchronize();
   // print_time(s,"set start gpu");

    s = start_timer(); 
    for (int i = 0; i < n_r * l; i++)
    {
        if (bucket_start[code[index[i]]] == -1)
        {
            bucket_start[code[index[i]]] = i;
        }
    }
    print_time(s,"set start cpu");

    int *code_q;
    cudaMallocManaged((void **)&code_q, sizeof(int) * n_r * l);
    cudaMemset(code_q, 0, sizeof(int) * n_r * l);
    // dot all q points
    for (int i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_q; ii++)
        {
            for (int iii = 0; iii < nbits; iii++)
            {
                float sum = dot(q_points[ii], rand_array[iii + i * nbits]);
                if (sum <= 0)
                {
                    code_q[ii + i * n_q] |= 1UL << iii;
                }
            }
        }
    }

    int *buckets;
    cudaMallocManaged((void **)&buckets, sizeof(int) * n_q * n_r);
    cudaMemset(buckets, 0, sizeof(int) * n_q * n_r);
    // fill buckets
    for (int i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_q; ii++)
        {
            int bucket = code_q[ii + (i * n_q)];
            int start = bucket_start[bucket];
            int iii = start;
            while ((start != -1) && code[index[iii]] == bucket)
            {
                buckets[ii * n_r + index[iii]] = 1;
                iii++;
            }

            // add from negbouring buckets up to n bits away // n can be given by input but is by defualt 1
            // is there any meaing to adding more than

            // make all buckets with a hamming distance of n

            // 0000
            // 1000 0100 0010 0001
            // 1100 1010 1001
            // 0110 0101
            // 0011

            // 010
            // 011 000 110
            // 001 100
            for (int n = 0; n < max_dist; n++)
            {
                int counters[n + 1];

                for (int q = 0; q < (n + 1); q++)
                {
                    counters[q] = q;
                }

                bool done = false;
                while (!done)
                {
                    int neighbour_bucket = bucket;
                    for (int nn = 0; nn < (n + 1); nn++)
                    {
                        neighbour_bucket ^= 1UL << counters[nn];
                    }
                    // printf("bucket is %i neighbour is %i \n", bucket, neighbour_bucket) ;
                    // we have bucket
                    int start = bucket_start[neighbour_bucket];
                    int iii = start;
                    while ((start != -1) && code[index[iii]] == neighbour_bucket)
                    {
                        buckets[ii * n_r + index[iii]] = 1;
                        iii++;
                    }
                    bool flag = false;
                    int nnn = n;
                    int bits = nbits;
                    while (!flag)
                    {

                        if (((counters[nnn] + 1) >= bits) && nnn == 0)
                        {
                            flag = true;
                            done = true;
                        }
                        else if ((counters[nnn] + 1) < bits)
                        {
                            counters[nnn] += 1;
                            flag = true;
                        }

                        nnn--;
                        bits--;
                    }
                }
            }
            //         printf("\n");
        }
    }

    for (int i = 0; i < n_q; i++)
    {
        sorted[i].w = MAXFLOAT;
        sorted[i].x = MAXFLOAT;
        sorted[i].y = MAXFLOAT;
        sorted[i].z = MAXFLOAT;
        for (int ii = 0; ii < n_r; ii++)
        {
            if (buckets[n_r * i + ii] == 1)
            {
                float dist = host_lenght(r_points[ii], q_points[i]);

                if (dist < sorted[i].x)
                {
                    sorted[i].y = sorted[i].x;
                    sorted[i].x = dist;

                    sorted[i].w = sorted[i].z;
                    sorted[i].z = ii;
                }
                else
                {
                    if (dist < sorted[i].y)
                    {
                        sorted[i].y = dist;
                        sorted[i].w = ii;
                    }
                }
            }
        }
    }
}

void make_vec(int dim, des_t &vec)
{
    for (size_t i = 0; i < dim; i++)
    {
        vec[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

float dot(des_t v1, des_t v2)
{

    float sum = 0.f;
    for (size_t i = 0; i < 128; i++)
    {
        float a = ((float *)v1)[i] -0.5;
        float b = ((float *)v2)[i]- 0.5;
        float c = a * b;
        sum += c;
    }
    return sum;
}
float test_dot(float *v1, des_t v2)
{
    float sum = 0.f;
    for (size_t i = 0; i < 128; i++)
    {
        float a = ((float *)v1)[(4 * 4) * i];
        float b = ((float *)v2)[i];
        float c = a * b;
        sum += c;
    }
    return sum;

}

class IndexCompare
{
    int* _index_copy;
    int* _code ;

public:
    IndexCompare( int* index_copy, int* code)
        : _index_copy( index_copy)
        , _code( code)
    { }

    __host__ __device__
    inline bool operator()( int left, int right ) const
    {
        return (_code[_index_copy[left]] < _code[_index_copy[right]]); 
    }
};

void sort_bucket( )
{

}

//void sort_buckets()

// kernels for finding the start of each bucket in the index array
// not sure if this is really faster than cpu todo TEST
// sets helper values
__global__ void set_helper(int *helper, int *code, int *index)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    helper[idx] = code[index[idx]];
}

__global__ void set_bucket_start(int *helper, int *bucket_start, int l, int n_r)
{
    for (int i = threadIdx.x; i < (l * n_r); i += 32)
    {

        int v = (helper[i] == blockIdx.x);
        unsigned int mask = __ballot_sync(0xffffffff, v);

        if (__popc(mask) == 0)
            continue;
        int x = __ffs(mask);
        if (threadIdx.x == 0)
        {
            int g = i + x - 1;
            bucket_start[blockIdx.x] = g;
            break;
        }
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
    int var = 0 ; 
    // index of the dot prouduct we need 
    
    int dot_idx =  blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y * blockDim.x + threadIdx.x ;    
    // only care if its a relevent thread 
    if((threadIdx.x ) < nbits )
    {
        if(dot[dot_idx] <= 0 ) 
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
        buckets[blockIdx.x * gridDim.y + blockIdx.y ] = var ;     
    }  
}
//want dot array to be 
// point 0 * rand [0][0 - nbits], .......   point n * rand [n][0 - nbits]
// point n * rand [0][0 - nbits], .......   point n * rand [n][0 - nbits] 
// 
// want rand[n] to only be read by blocks on the same sm. load in shared memory  
//  or want points[n ] to only be read by blocks on the same sm TODO  
__global__ void dot_gpu(des_t *  rand, des_t * points, float *dot)
{
    // called with 
    // n, nbits, l grid  
    //32 1 1 block 
    // could change to 32, x ,1 block todo test if faster 0_0 
    
    float res = 0.f ; 
    float4 a = ((float4 * )points[blockIdx.x])[threadIdx.x ];
    float4 b = ((float4 * )rand[blockIdx.z * gridDim.y + blockIdx.y])[threadIdx.x]; 

    res +=
        (a.x -0.5)*(b.x - 0.5) + (a.y -0.5)*(b.y -0.5) +
        (a.z -0.5)*(b.z - 0.5) + (a.w -0.5)*(b.w -0.5)  ;  
    reduce(res) ; 
    if(threadIdx.x == 0)
    {
        dot[blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y + blockIdx.y] = res ;     
    }  
}


// initialize array to a value
__global__ void initialize(int *array, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = value;
}

// may have to make vectors more random ! hmm todo
// makes a random float

__device__ inline float random_float(uint64_t seed, int idx)
{
    curandState s;
    curand_init(seed + idx, 0, 0, &s);
    return curand_uniform(&s);
}
//fills a vector with random floats
__global__ void random_vector(uint64_t seed, des_t *array)
{
    int idx = threadIdx.x + threadIdx.y * 32;
    float * vec = (float *)array[blockIdx.x]  ;
    vec[idx] = random_float(seed, idx);
    
}
__device__ inline void reduce(float &var)
{
    var += __shfl_down_sync( 0xffffffff, var, 16 );
    var += __shfl_down_sync( 0xffffffff, var, 8 ); 
    var += __shfl_down_sync( 0xffffffff, var, 4 ); 
    var += __shfl_down_sync( 0xffffffff, var, 2 );
    var += __shfl_down_sync( 0xffffffff, var, 1 );   
}
// called with 
// only works up to a distance of 3 
//
__global__ void find_all_neigbours(int * neighbouring_buckets, int nbits, int dist, int n_q, int * bucket ) 
{

    // could read more codes to use fewer threads ?
    __shared__ int code[] ;  
    if(threadIdx.x == 0)
    {

        code = bucket[blockIdx.x] ; 
    }
    
    // 0000
    // 1000 0
    if(dist == 1)
    {
        int neigbour = code ;   
        neigbour ^= 1UL << threadIdx.x ; 
        neighbouring_buckets[threadIdx.x + n_q * blockIdx.x] = neigbour ; 

        return ; 
    }  

// 5, 4 
// 5, 2 
// 11000 10100 01100 01010 00110 00101 00011 10010 10001 01001
// set(1, 1+1), set(1, 1+2)
// set(2, 2+1), set(2, 2+2)
// set(3, 3+1), set(3, 3+2)
// set(4, 4+1), set(4, 4+2) over 5 rolls over and back to 0 
// set(5, 5+1), set(5, 5+2)

// 7 * 6 / 2 

// nbits = 7 dist = 2 gives us 7*6/2 = 21 buckets sice nbits is odd we can divide nbits - 1 so 6/2. gives us 7 * 3 threads  
//7,3 
// set(1, 1+1), set(1, 1+2) set(1, 1+3) // 1100000 1010000 1001000 
// set(2, 2+1), set(2, 2+2) set(2, 2+3) // 0110000 0101000 0100100
// set(3, 3+1), set(3, 3+2) set(3, 3+3) // 0011000 0010100 0010010 
// set(4, 4+1), set(4, 4+2) set(4, 4+3) // 0001100 0001010 0001001 
// set(5, 5+1), set(5, 5+2) set(5, 5+3) // 0000110 0000101 1000100 
// set(6, 6+1), set(6, 6+2) set(6, 6+3) // 0000011 1000010 0100010 
// set(7, 7+1), set(7, 7+2) set(7, 7+3) // 1000001 0100001 0010001
// works for odd nbits  hm
    if(dist == 2)
    {
        int neigbour = code ; 
        neigbour  ^= 1UL << threadIdx.x ; 
        neigbour  ^= 1UL << ((threadIdx.x + 1 + threadIdx.y) % nbits)  ; 
    }

// 6, 5
// 3, 5
// 5, 3 ? 
// set(1, 1+1), set(1, 1+2), set(1, 1+3), set(1, 1+4), set(1, 1+5) // 110000 101000 010100 010010 010001 
// set(2, 2+1), set(2, 2+2), set(2, 2+3), set(2, 2+4), set(2, 2+5) // 011000 010100 001010 001001 101000 
// set(3, 3+1), set(3, 3+2), set(3, 3+3), set(3, 3+4), set(3, 3+5) //  

// //110000 101000 100100 
// //011000 010100 010010
// //001100 001010 001001
// //000110 000101 100100
// //000011 100010 010010
// pair 
    if(dist == 2)
    {

    }
}

// CALLED WITH 
// grid n_q, 1, 1 
//block max_dist, 1, 1 
__global__ void hamming(int * neighbouring_buckets, int dist, int size, int * bucket )
{
    int start = bucket[blockIdx.x] ; 
    change_bit(size, threadIdx.x, 1, 0, start) ; 
}

 __device__ void change_bit(int n, int k, int dir, int pos, int start)
{
    for (size_t i = 1; i <= n - k + 1; i++, pos += dir)
    {
        start ^= 1UL << pos ; 
        if (k > 1) 
        {
            change_bit(n - i, k - 1, i % 2 ? dir : -dir, pos + dir * (i % 2 ? 1 : n - i), start);
        }
        else 
        {
            printf("int is %i \n", start) ; 
        }
        start ^= 1UL << pos ; 
    }
}
// called with 
// grid n_r/32 +1 l 1         
// block 32 3 1 
__global__ void set_bucket(int * index, int * index_copy, int n)
{
    int i = blockDim.x * blockDim.y * blockIdx.x +  blockDim.x * threadIdx.y + threadIdx.x ; 
    if(n > i)
    {
        index[i + blockIdx.y * n] = i ; 
        index_copy[i + blockIdx.y * n] = i ; 
    }
}
//    for (int i = 0; i < l; i++)
//    {
//        for (int ii = 0; ii < n_r; ii++)
//        {
//            index[ii + i * n_r] = ii;
//            index_copy[ii + i * n_r] = ii;
//        }
//    }
//given an array of bools n_q * n_r, see if bool true, if true start new kernel which dots the two points 
__global__ void brute_shared(int * bucket)    
{
        //Start kernel |    
}

void lsh_test(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
{  
    // see how much memory we have  

    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    // need at least  

    //double free_db = (double)free_byte ;
    //double total_db = (double)total_byte ;
    //double used_db = total_db - free_db ;

   // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

   //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
 
    // arry of vectors 
    des_t *rand_array;
    // hash codes  
    int *code_r, *code_q;
    // index into bucket array and copy to sort 
    int *index, *index_copy ; 
    // given bucket n gives start index 
    int *bucket_start;
    // dot from random vector to q / r points 
    float * dot_res_r, * dot_res_q;
 
    // a bucket of all the points each q has to check   
    // number of buckets within hamming distance r given n bits = n!/r!(n-r)! where r is 1 -> r  
    int size_bucket = 0 ;
    for (int i = 1; i < max_dist + 1; i++)
    {
        size_bucket += fact(nbits)/(fact(i)*fact(nbits - i)) ;  
    }

    printf("need %i \n", size_bucket) ; 
    int *buckets;

    cudaMallocManaged((void **)&buckets, sizeof(int) * n_q * size_bucket);
    cudaMemset(buckets, 0, sizeof(int) * n_q * size_bucket);
    
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits);
    cudaMallocManaged((void **)&index, sizeof(int) * n_r);  
    cudaMallocManaged((void **)&index_copy, sizeof(int) * n_r);
    cudaMallocManaged((void **)&bucket_start, (2 << nbits) * sizeof(int));

    cudaMallocManaged((void **)&code_r, sizeof(int) * n_r);
    cudaMallocManaged((void **)&code_q, sizeof(int) * n_q);

    cudaMallocManaged((void **)&dot_res_r, nbits * n_r* sizeof(float)); 
    cudaMallocManaged((void **)&dot_res_q, nbits * n_q* sizeof(float));

    for (int L = 0; L < l; L++)
    {
        // memsetstuff
        cudaMemset(index, 0, sizeof(int) * n_r );
        cudaMemset(index_copy, 0, sizeof(int) * n_r );
        cudaMemset(code_r, 0, sizeof(int) * n_r );
        cudaMemset(code_q, 0, sizeof(int) * n_q);
        // todo make kernel to set - 1  
        for (int i = 0; i < (2 << nbits); i++)
        {
            bucket_start[i] = -1;
        }
        // make random vectors
        for (int i = 0; i < nbits; i++)
        {
            make_vec(128, rand_array[i]);
        }
        // dot random vectors with n_r
        dim3 grid_dot_r(n_r, nbits, 1) ;
        dim3 block_dot_r(32, 1, 1) ;   
        dot_gpu<<<grid_dot_r, block_dot_r>>>(rand_array, r_points, dot_res_r); 
        // set bit for code_r 
        dim3 grid_bit_r(n_r,1,1) ; 
        dim3 block_bit_r(nbits,1,1) ; 
        set_bit<<<grid_bit_r, block_bit_r>>>(code_r, nbits, dot_res_r) ; 
        cudaDeviceSynchronize();

        // make buckets for r points
        dim3 grid_set(n_r/32+1, 1, 1) ; 
        dim3 block_set(32,3,1) ;
        set_bucket<<<grid_set, block_set>>>(index, index_copy, n_r) ; 
        cudaDeviceSynchronize();
        //sort bucket by index  
        // gpu or cpu 
        IndexCompare tc(index_copy, code_r);
    #if PREFER_CPU == 0
        thrust::device_ptr<int> ptr = thrust::device_pointer_cast(index);
        thrust::sort( ptr, ptr + n_r, tc );
    #else
        cudaDeviceSynchronize();
        int* ptr = index;
        std::sort( ptr, ptr + n_r, tc );
    #endif
        // set bucket start  
        for (int i = 0; i < n_r; i++)
        {
            if (bucket_start[code_r[index[i]]] == -1)
            {
                bucket_start[code_r[index[i]]] = i;
            }
        }

        // dot random vectors with q
        dim3 grid_dot_q(n_q, nbits, 1) ;
        dim3 block_dot_q(32, 1, 1) ;   
        dot_gpu<<<grid_dot_q, block_dot_q>>>(rand_array, q_points, dot_res_q); 

        //set bit for hash values for code_q 
        dim3 grid_bit_q(n_r,1,1) ; 
        dim3 block_bit_q(nbits,1,1) ; 
        set_bit<<<grid_bit_q, block_bit_q>>>(code_q, nbits, dot_res_q) ; 
        cudaDeviceSynchronize();

        // fill buckets
        // do all the dots at once insted of dividing into L then do  
        // 1 one time for each q use n_r size array so we dont get duplicates  
        // 2 same but  multpile qs at the same time 
        // have a intilzie funtoins which make the data we need  ? 
        // have all the hamming 
        dim3 grid_bucket(n_q, 1, 1) ; 
        dim3 block_bucket(max_dist,1,1) ; 
        hamming<<<grid_bucket, block_bucket>>>(buckets, max_dist, nbits, code_q ) ; 

       cudaDeviceSynchronize();
       for (int ii = 0; ii < n_q; ii++)
       {
            int bucket = code_q[ii];
            int c = 0 ; 
            for (int n = 0; n < max_dist; n++)
            {
                int counters[n + 1];

                for (int q = 0; q < (n + 1); q++)
                {
                    counters[q] = q;
                }

                bool done = false;
                
                while (!done)
                {
                    int neighbour_bucket = bucket;
                    for (int nn = 0; nn < (n + 1); nn++)
                    {
                        neighbour_bucket ^= 1UL << counters[nn];
                    }
                    buckets[ii * size_bucket + c] = neighbour_bucket ; 
                    c ++ ; 
                    bool flag = false;
                    int nnn = n;
                    int bits = nbits;
                    while (!flag)
                    {

                        if (((counters[nnn] + 1) >= bits) && nnn == 0)
                        {
                            flag = true;
                            done = true;
                        }
                        else if ((counters[nnn] + 1) < bits)
                        {
                            counters[nnn] += 1;
                            flag = true;
                        }

                        nnn--;
                        bits--;
                    }
                }
            }
           // printf("int c = %i \n", c) ; 

        }
        for (int i = 0; i < n_q; i++)
        {
            if(L == 0)
            {
                sorted[i].w = MAXFLOAT;
                sorted[i].x = MAXFLOAT;
                sorted[i].y = MAXFLOAT;
                sorted[i].z = MAXFLOAT;
            }
            search_bucket(sorted[i], code_q[i], bucket_start[code_q[i]], code_r, index, r_points, q_points, n_r, i) ; 
            for (int ii = 0; ii < size_bucket; ii++)
            {
                int iii = buckets[max_dist * i + ii]  ; 
                search_bucket(sorted[i], iii, bucket_start[iii], code_r, index, r_points, q_points, n_r, i) ; 
            }
        }
    } 
}

void search_bucket(float4 &min, int bucket, int start, int * code, int * index, des_t * r_p, des_t * q_p, int size_r, int x)
{
    if(start == -1 ) return ; 
    int i = start ; 
    while (i < size_r && code[index[i]] == bucket)
    {
        float dist = host_lenght(r_p[i], q_p[x]);

        if (dist < min.x)
        {
            min.y = min.x;
            min.x = dist;

            min.w = min.z;
            min.z = i;
        }
        else
        {
            if (dist < min.y)
            {
                min.y = dist;
                min.w = i;
            }
        }
        i ++ ; 
    }
}