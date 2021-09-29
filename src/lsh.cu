#include <iostream>
#include "curand_kernel.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
#include "knn_brute.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

void host_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
{

    des_t *rand_array;
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits * l);
    // make random vectors
    for (size_t i = 0; i < nbits * l; i++)
    {
        make_vec(128, rand_array[i]);
    }
    // make an array of ints with one int for each r_point
    int *code;
    int *index;
    int *bucket_start;
    // need a copy to sort using sort
    int *index_copy;

    cudaMallocManaged((void **)&index, sizeof(int) * n_r * l);
    cudaMemset(index, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&index_copy, sizeof(int) * n_r * l);
    cudaMemset(index_copy, 0, sizeof(int) * n_r * l);

    cudaMallocManaged((void **)&bucket_start, 2 << nbits);

    for (int i = 0; i < (2 << nbits); i++)
    {
        bucket_start[i] = -1;
    }

    cudaMallocManaged((void **)&code, sizeof(int) * n_r * l);
    cudaMemset(code, 0, sizeof(int) * n_r * l);

    float * test ;  
    cudaMallocManaged((void **)&test, sizeof(float) * n_r * l);
    cublasHandle_t handle;
    cublasCreate_v2(&handle) ; 
    
    // dot all vectors and add the bit to the coresponding int bit for the r points
    for (int i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_r; ii++)
        {
            //            printf(" %i bucket = ", (ii + i * n_r));
            for (int iii = 0; iii < nbits; iii++)
            {
                float sum = dot(r_points[ii], rand_array[iii + i * nbits]);
               // cublasSdot(handle, 128, r_points[ii], 1, rand_array[iii + i *nbits], 1, &test[ii + i *n_r]) ; 
                if ( sum<= 0)
                {
                    code[ii + i * n_r] |= 1UL << iii;
                }
            }
            //           printf(" %i ", code[ii + i * n_r]);
            //           printf("\n \n");
        }
    }

    // make buckets for r points

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

    //index
    // code[index[r number ]] -> bucket
    //

    // index[range (bucket start[bucket] ->  [while code [index in r:points]] == bucket)] -> index in r_points of elemtn in bucket
    // r_points[index in r_points ] -> dest vec

    int *helper;
    cudaMallocManaged((void **)&helper, sizeof(int) * n_r * l);

    //set bucket start
    dim3 block_size(32, 1, 1);
    dim3 grid_size(((l * n_r) / 32) + 1, 1, 1);
    set_helper<<<grid_size, block_size>>>(helper, code, index);

    cudaDeviceSynchronize();
    dim3 block(32, 1, 1);
    dim3 grid(n_r * l, 1, 1);

    min_helper<<<grid, block>>>(helper, bucket_start, l, n_r);
    cudaDeviceSynchronize();
 //   for (int i = 0; i < n_r * l; i++)
 //   {
 //       if (bucket_start[code[index[i]]] == -1)
 //       {
 //           bucket_start[code[index[i]]] = i;
 //       }
 //  }
   
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
            // for (int n = 0; n < max_dist; n++)
            // {
            //     int counters[n + 1] ;

            //     for (int q = 0; q < (n + 1); q++)
            //     {
            //         counters[q] = q ;
            //     }
            //
            //     for (int nn = 0; nn < (n+1); n++)
            //     {
            //         for (int nnn = 0; nnn < (nbits - nn) ; nnn++)
            //         {
            //             for (int nnnn = 0; nnnn < ; nnnn++)
            //             {
            //                 /* code */
            //             }
            //
            //
            //         }
            //
            //     }
            //

            //
            //
            // }

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

    // brute for each q point in the right bucket
    // for (int i = 0; i < l; i++)
    // {
    //     for (int ii = 0; ii < n_q; ii++)
    //     {
    //         int bucket = code_q[ii + (i * n_q)] ;
    //         int start = bucket_start[bucket] ;
    //         int iii = start ;
    //         while ((start != -1) && code[index[iii]] == bucket)
    //         {
    //             float dist = host_lenght(r_points[index[iii]], q_points[ii] ) ;
    //             if(i == 0)
    //             {
    //                 sorted[ii].w = MAXFLOAT ;
    //                 sorted[ii].x = MAXFLOAT ;
    //                 sorted[ii].y = MAXFLOAT ;
    //                 sorted[ii].z = MAXFLOAT ;
    //             }
    //             if(dist < sorted[ii].x)
    //             {
    //                 sorted[ii].y = sorted[ii].x ;
    //                 sorted[ii].x = dist;
    //
    //                 sorted[ii].w= sorted[ii].z;
    //                 sorted[ii].z= index[iii];
    //             }
    //             else{
    //                 if(dist < sorted[ii].y)
    //                 {
    //                     sorted[ii].w = index[iii] ;
    //                 }
    //             }
    //             iii ++ ;
    //         }
    //     }
    // }
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
        float a = ((float *)v1)[i] - 0.5;
        float b = ((float *)v2)[i] - 0.5;
        float c = a * b;
        sum += c;
    }
    return sum;
}

void gpu_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l)
{
    // repeat l times !! will lead to comparing the same points multiple times ?

    // make random vectors

    des_t *rand_array;
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits * l);

    // fill array
    uint64_t seed = 9753;
    dim3 grid_size(1, 1, 1);
    dim3 block_size(32, 1, 1);

    //fill in the dist array
    random_vector<<<grid_size, block_size>>>(seed, rand_array);
}

// kernels for finding the start of each bucket in the index array
// not sure if this is really faster than cpu todo TEST 
// sets helper values
__global__ void set_helper(int *helper, int *code, int *index)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    helper[idx] = code[index[idx]];
}

__global__ void min_helper(int *helper, int *bucket_start, int l, int n_r)
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
            bucket_start[blockIdx.x] = g ; 
            break;
        }
    }
}

// dot two arrays of vectors and set bit corresponding to the dot product  

__global__ void dot_set_bit(float * rand, float * points, int * buckets, int size, int nbits, float * test)
{
   // cublasHandle_t handle;
   // float res = 0.f ; 
   // cublasCreate_v2(&handle) ; 
   // cublasSdot(handle, size, (rand + (size * nbits * threadIdx.x)), 1, (points + (blockIdx.x * size)), 1, &res ) ; 
   //   
   // int bucket = 0 ;
   // if(res >= 0)
   // {    
   //     bucket |= 1UL << threadIdx.y ; 
   // } 
   //  // l n_r nbits 
   // bucket += __shfl_down_sync( 0xffffffff, bucket, 16 );
   // bucket += __shfl_down_sync( 0xffffffff, bucket, 8 ); 
   // bucket += __shfl_down_sync( 0xffffffff, bucket, 4 ); 
   // bucket += __shfl_down_sync( 0xffffffff, bucket, 2 );
   // bucket += __shfl_down_sync( 0xffffffff, bucket, 1 );   
   // if(threadIdx.y == 0)
   // {
   //     //test[] = bucket ;  
   // }
   // //code[ii + i * n_r] |= 1UL << iii;
   // // block = n_r, 1 1
   // // thread = l, 32 , 1 
    
}

// initialize array to a value
__global__ void initialize(int *array, int value)
{

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
__global__ void random_vector(uint64_t seed, des_t *vec)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float *vector = (float *)vec;
    vector[idx] = random_float(seed, idx);
}