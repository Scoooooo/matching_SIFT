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


void test_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
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
    int *index;
    // need a copy to sort using sort
    int *index_copy;
    int *bucket_start;

    int * index_test ; 
    int * index_copy_test ; 
    // memsetstuff
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
   
    float * dot_res;
    cudaMallocManaged((void **)&dot_res, l * nbits * n_r* sizeof(float));

    dim3 grid(n_r, nbits, l) ;
    dim3 block(32, 1, 1) ;   
    dot_gpu<<<grid, block>>>(rand_array, r_points, dot_res); 
    cudaDeviceSynchronize();
    grid.z = 1 ; 
    grid.y = l ; 
    block.x = nbits ;
    set_bit<<<grid, block>>>(code, nbits, dot_res) ; 
    cudaDeviceSynchronize();
    
    // make buckets for r points
    grid.x = n_r/32 +1 ; 
    grid.y = l ; 
    block.x = 32 ; 
    block.y = 3 ; 
    set_bucket<<<grid,block>>>(index, index_copy, n_r) ; 
    cudaDeviceSynchronize();

    //free res
    cudaFree(dot_res) ; 

    //sort faster on cpu for now 
    std::sort(index, index + n_r * l, [&](const int &i, const int &j) -> bool
              { return (code[index_copy[i]] < code[index_copy[j]]); });
    //Set bucket start also faster on cpu for now 
    for (int i = 0; i < n_r * l; i++)
    {
        if (bucket_start[code[index[i]]] == -1)
        {
            bucket_start[code[index[i]]] = i;
        }
    }
    
    int *code_q;
    cudaMallocManaged((void **)&code_q, sizeof(int) * n_q * l);
    cudaMemset(code_q, 0, sizeof(int) * n_q * l);
 
    float * dot_res;
    cudaMallocManaged((void **)&dot_res, l * nbits * n_q* sizeof(float));

    dim3 grid(n_q, nbits, l) ;
    dim3 block(32, 1, 1) ;   
    dot_gpu<<<grid, block>>>(rand_array, q_points, dot_res); 
    cudaDeviceSynchronize();
    grid.z = 1 ; 
    grid.y = l ; 
    block.x = nbits ;
    set_bit<<<grid, block>>>(code, nbits, dot_res) ; 
    cudaDeviceSynchronize();

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
void gpu_lsh(des_t *q_points, des_t *r_points, int n_q, int n_r, float4 *sorted, int nbits, int l, int max_dist)
{
    // make random vectors

    des_t *rand_array;
    cudaMallocManaged((void **)&rand_array, sizeof(des_t) * nbits * l);

    // fill array
    uint64_t seed = clock();
    dim3 grid_size(nbits * l, 1, 1);
    dim3 block_size(32, 3, 1);

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
    cudaMallocManaged((void **)&dot_res, nbits * n_r);

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