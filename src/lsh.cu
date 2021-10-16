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

using namespace std;

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
    s = start_timer() ; 
    set_bucket<<<grid,block>>>(index_test, index_copy_test, n_r) ; 
    print_time(s, "gpu"); 
    cudaDeviceSynchronize();
    s = start_timer() ; 
    for (int i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_r; ii++)
        {
            index[ii + i * n_r] = ii;
            index_copy[ii + i * n_r] = ii;
        }
    }

    print_time(s, "cpu") ; 
    std::sort(index, index + n_r * l, [&](const int &i, const int &j) -> bool
              { return (code[index_copy[i]] < code[index_copy[j]]); });

    //  set_bucket_start<<<grid, block>>>(helper, bucket_start, l, n_r);
    //  cudaDeviceSynchronize();
    for (int i = 0; i < n_r * l; i++)
    {
        if (bucket_start[code[index[i]]] == -1)
        {
            bucket_start[code[index[i]]] = i;
        }
    }

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
void gpu_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
{
    // make random vectors

    des_t *rand_array;
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits * l);

    // fill array
    uint64_t seed = clock();
    dim3 grid_size(nbits * l, 1, 1);
    dim3 block_size(128, 1, 1);

    //fill in the rand array
    random_vector<<<grid_size, block_size>>>(seed, rand_array);

    // make an array of ints with one int for each r_point
    int *code;
    int *index;
    // need a copy to sort using sort
    int *index_copy;
    int *bucket_start;

    cudaMallocManaged((void **)&index, sizeof(int) * n_r * l);
    cudaMemset(index, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&index_copy, sizeof(int) * n_r * l);
    cudaMemset(index_copy, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&bucket_start, (2 << nbits) * sizeof(int));

    // dot rand_array and r_points to make buckets  


    // cublas can maybe be used if the query is big enough how big test todo   
    // dot_res layout [x][l * nbits]
    float *dot_res;
    cudaMallocManaged((void **)&dot_res, l * nbits * n_r);

   // float a = 1.0f;
   // float b = 1.0f;
   // cublasHandle_t handle;
   // cublasCreate(&handle);

   // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, l * nbits, n_r, 128, &a, (float *)rand_array, l * nbits, (float *)r_points, 128, &b, dot_res, l * nbits);
   // 
    /**
     * rand_arraay = L*nbitsX128
     * r_points = 128 X n_r
     * dot_res = L*nbits X n_r if we were to use cublas  so n_r * L*nbits  
     * dot_res[] 
     **/
  // test there are some difrencece caused by the way the calculation is done  
  //  for (int i = 0; i < n_r; i++)
  //  {
  //      for (int ii = 0; ii < (l * nbits); ii++)
  //      {
  //          compare_float(test_dot((rand_array + ii), r_points[i]), dot_res[i * (l * nbits) + ii]);
  //      }
  //  }


    dim3 grid(n_r, l, 1) ;
    dim3 block(32, nbits, 1) ;   
    dot_gpu<<<grid, block>>>(rand_array, r_points, dot_res); 

    
    //index
    // code[index[r number ]] -> bucket
    //

    // index[range (bucket start[bucket] ->  [while code [index in r:points]] == bucket)] -> index in r_points of elemtn in bucket
    // r_points[index in r_points ] -> dest vec
//
//    int *helper;
//    cudaMallocManaged((void **)&helper, sizeof(int) * n_r * l);
//
//    //set bucket start
//    dim3 block_size(32, 1, 1);
//    dim3 grid_size(((l * n_r) / 32) + 1, 1, 1);
//    set_helper<<<grid_size, block_size>>>(helper, code, index);
//
//    cudaDeviceSynchronize();
//    dim3 block(32, 1, 1);
//    dim3 grid(n_r * l, 1, 1);
}

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
//  or want points[n ] to only be read by blocks on the same sm 
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
    int idx = threadIdx.x;
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

