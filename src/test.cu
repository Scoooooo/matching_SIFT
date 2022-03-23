#include "knn_brute.h"
#include "lsh_h.h"
#include "cuda_profiler_api.h"
#include <iostream>
#include <string>
#include <math.h>
#include <sys/time.h>
#include "helper.h"

#include "cublas_v2.h"

#include <cstring>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
void make_rand_vector(int dim, des_t_f &vec);
void make_rand_vec_array(int dim, int size, des_t_f *array);

void test_float(); 
void test_half_float(); 
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>

int main(int argc, char *argv[])
{
    test_float();
    //test_half_float() ; 
    return 0;
}

void test_float()
{
    int dim = 128;
    int size_q = 10000;
    int size_r = 1000000 ;

    des_t_f *q_points;
    des_t_f *r_points;
    
    des_t_f *gpu_q_points;
    des_t_f *gpu_r_points;

    float4 *sorted_lsh;
    float4 *sorted_2nn;
    //make streams/ handels 
    int stream_n = 2 ; 
    cublasHandle_t handle ;  
    cublasCreate(&handle); 

    uint32_t * matches_lsh ; 
    cudaMallocManaged((void **)&matches_lsh, size_q * sizeof(uint32_t));
    uint32_t * matches_brute ; 
    cudaMallocManaged((void **)&matches_brute, size_q * sizeof(uint32_t));
    
    cudaMallocHost((void **)&q_points, size_q * sizeof(des_t_f));
    cudaMallocHost((void **)&r_points, size_r * sizeof(des_t_f));
 
    cudaMalloc((void **)&gpu_q_points, size_q * sizeof(des_t_f));
    cudaMalloc((void **)&gpu_r_points, size_r * sizeof(des_t_f));

    //output arrays dist and index of dist for 2nn 
  //  cudaMallocManaged((void **)&sorted_lsh, size_q * sizeof(float4));
   // cudaMallocManaged((void **)&sorted_2nn, size_q * sizeof(float4));

    make_rand_vec_array(dim, size_q, q_points);
    make_rand_vec_array(dim, size_r, r_points);

    cudaMemcpy(gpu_q_points, q_points, size_q * sizeof(des_t_f), cudaMemcpyHostToDevice) ; 
    cudaMemcpy(gpu_r_points, r_points, size_r * sizeof(des_t_f), cudaMemcpyHostToDevice) ; 
    //   cudaProfilerStart();
   // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_lsh, 25, 20, 0, handle);
    double s = start_timer() ; 
   //// printf("brute needs to compare %zu points \n", size_q * size_r ) ; 
   //// //host_brute(q_points,r_points,size_q,size_r, sorted_lsh) ;
   // cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;
    cublas_2nn_sift(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_brute, 1, handle, stream_n); 
    cudaDeviceSynchronize() ;
   //device_brute(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn) ;
   print_time(s, "cublas burte") ; 
 
     s = start_timer();

    lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 2, 1, 0, 32); 
    
    cudaDeviceSynchronize() ;
    //    cudaProfilerStop() ;
    // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_2nn, 15 , 10, 0, handle[0]);
   print_time(s, "lsh brute"); 

  int failed = 0 ; 
    s = start_timer() ; 
   for (size_t i = 0; i < size_q; i++)
   {
       if(matches_lsh[i]!= matches_brute[i] )
       {
           failed ++ ; 
            // printf("%i, %i \n", (int)matches_brute[i], (int)matches_lsh[i]); 
       }

       //   printf("%i, %i \n", (int)matches_brute[i], (int)matches_lsh[i]); 
   }
   print_time(s, "for loop") ; 
   printf("failed %i \n", failed); 
   // 
   // return ; 
   // // see how many poins lsh got right 
   // for (size_t i = 0; i < size_q; i++)
   // {
   //    // printf("\n") ;
   //    if(sorted_2nn[i].z !=  sorted_lsh[i].z)
   //     {
   //         printf("failed z %f , %f\n", sorted_2nn[i].z, sorted_lsh[i].z ); 
   //         printf("w = %f , %f\n", sorted_2nn[i].w, sorted_lsh[i].w ); 
   //        failed ++ ; 
// //           printf("z is bad \n") ; 
  ////  printf("lsh 1  %f index %f  lsh 2 %f index %f \n", sorted_host[i].x, sorted_host[i].z, sorted_host[i].y,  sorted_host[i].w) ; 
  ////  printf("cpu 1  %f index %f  cpu 2 %f index %f \n", sorted_dev[i].x, sorted_dev[i].z, sorted_dev[i].y,  sorted_dev[i].w) ;
   //     }
   //     if(sorted_2nn[i].w !=  sorted_lsh[i].w)
   //     {
   //         printf("failed w %f , %f\n", sorted_2nn[i].w, sorted_lsh[i].w ); 
   //         printf("z =  %f , %f\n", sorted_2nn[i].z, sorted_lsh[i].z ); 
   //         failed ++ ; 
 ////           printf("w is not good \n"); 
   //     }
   //     
   // }
    //printf("found %i out of %i nn \n",((size_q * 2)- failed),(size_q *2) ) ; 
    cudaFree(gpu_q_points); 
    cudaFree(gpu_r_points); 
    cudaFree(q_points); 
    cudaFree(r_points); 
    // cudaFree(sorted_2nn); 
    // cudaFree(sorted_lsh); 
}


void test_half_float()
{
    int dim = 128;
    int size_q = 10000;
    int size_r = 100000;

    des_t_h *q_points;
    des_t_h *r_points;
    
    des_t_h2 *gpu_q_points;
    des_t_h2 *gpu_r_points;

    uint32_t * matches ; 
    //make streams/ handels 
    int stream_n = 2 ; 
   
   
    
    cudaMallocHost((void **)&q_points, size_q * sizeof(des_t_h2),  cudaHostAllocMapped);
    cudaMallocHost((void **)&r_points, size_r * sizeof(des_t_h2),  cudaHostAllocMapped);
 
    cudaMalloc((void **)&gpu_q_points, size_q * sizeof(des_t_h2));
    cudaMalloc((void **)&gpu_r_points, size_r * sizeof(des_t_h2));

    //output arrays dist and index of dist for 2nn 
    cudaMallocManaged((void **)&matches, size_q * sizeof(uint32_t));

    make_rand_vec_array_h(dim, size_q, q_points);
    make_rand_vec_array_h(dim, size_r, r_points);

    cudaMemcpy(gpu_q_points, q_points, size_q * sizeof(des_t_h2), cudaMemcpyHostToDevice) ; 
    cudaMemcpy(gpu_r_points, r_points, size_r * sizeof(des_t_h2), cudaMemcpyHostToDevice) ; 
    double s = start_timer();
    //   cudaProfilerStart();
   // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_lsh, 25, 20, 0, handle);
    //device_brute(q_points,r_points,size_q,size_r, sorted_lsh) ;
    cudaDeviceSynchronize() ;
    //    cudaProfilerStop() ;
    //gpu_lsh(q_points, r_points, size_q, size_r, sorted_host, 4, 4, 2);
    print_time(s, "gpu lsh"); 
    s = start_timer() ; 
 //   cublas_2nn_sift(gpu_q_points, gpu_r_points, 2, size_q, size_r, matches, 0.999, handle, stream_n); 

    cudaDeviceSynchronize() ;
    printf("brute needs to compare %zu points \n", size_q * size_r ) ; 
    print_time(s, "gpu brute") ; 
    int failed = 0 ; 
    for (size_t i = 0; i < 10; i++)
    {
        printf("match is %i\n", matches[i]) ; 
    }
    
    return ; 
    // see how many poins lsh got right 
    printf("found %i out of %i nn \n",((size_q * 2)- failed),(size_q *2) ) ; 
    cudaFree(gpu_q_points); 
    cudaFree(gpu_r_points); 
    cudaFree(q_points); 
    cudaFree(r_points); 
}

  