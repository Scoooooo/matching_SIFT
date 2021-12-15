#include "knn_brute.h"
#include "lsh.h"
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
void make_rand_vector(int dim, des_t &vec);
void make_rand_vec_array(int dim, int size, des_t *array);
void test();
#include <iostream>
#include <vector>
#include <functional>

int main_test()
{
    int m = 4;
    int n = 2;
    int k = 3;

    int N = m*k;
    int M = n*k;

    float x0[12] = {0, 2, 2, 
                    3, 4, 5, 
                    6, 7, 8, 
                    9, 10, 11};
                    // 0 8 4 
                    // 2 2 3 
                    // 

    float q0[6]  = {3, 4, 
                    5, 6, 
                    7, 8 };

    float *x, *q, *x_q_multiplication;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&q, M*sizeof(float));
    cudaMallocManaged(&x_q_multiplication, n*m*k);

    std::memcpy(x, x0,  N*sizeof(float));
    std::memcpy(q, q0,  M*sizeof(float));

    float *q_device;
    cudaMallocManaged(&q_device, M*sizeof(float));
    cudaMemcpy(q_device, q, M*sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha, // 1
            q_device, n,
            x, k,
            &beta, // 0
            x_q_multiplication, n);

    cudaDeviceSynchronize();

    for (int i = 0; i < n*m; i++) std::cout << x_q_multiplication[i] << " "; std::cout << std::endl;

    cudaFree(x);
    cudaFree(q);
    cudaFree(x_q_multiplication);
    return 0;
}

int main(int argc, char *argv[])
{
    //main_test() ; 
    test();
    return 0;
}
void test()
{
    int dim = 128;
    int size_q = 1000;
    int size_r = 2000;

    des_t *q_points;
    des_t *r_points;

    float4 *sorted_host;
    float4 *sorted_dev;
    //host
    cudaMallocManaged((void **)&q_points, size_q * sizeof(des_t));
    cudaMallocManaged((void **)&r_points, size_r * sizeof(des_t));

    cudaMallocManaged((void **)&sorted_host, size_q * sizeof(float4));
    cudaMallocManaged((void **)&sorted_dev, size_q * sizeof(float4));

    // 4 2 3 3 
    // 8 2 7 7
    //data
    make_rand_vec_array(dim, size_q, q_points);
    make_rand_vec_array(dim, size_r, r_points);
    double s = start_timer();
    //   cudaProfilerStart();
    lsh_test(q_points, r_points, size_q, size_r, sorted_host, 10, 5 , 0);
    //    cudaProfilerStop() ;
    //gpu_lsh(q_points, r_points, size_q, size_r, sorted_host, 4, 4, 2);
    print_time(s, "gpu lsh"); 
    s = start_timer() ; 
    printf("brute needs to compare %i points \n", size_q * size_r ) ; 
    device_brute(q_points,r_points,size_q,size_r, sorted_dev) ;
    print_time(s, "gpu brute") ; 
    int failed = 0 ; 
    for (size_t i = 0; i < size_q; i++)
    {


       // printf("\n") ;
        if(sorted_dev[i].z !=  sorted_host[i].z)
        {
            failed ++ ; 
//            printf("z is bad \n") ; 

//       printf("lsh 1  %f index %f  lsh 2 %f index %f \n", sorted_host[i].x, sorted_host[i].z, sorted_host[i].y,  sorted_host[i].w) ; 
//    printf("cpu 1  %f index %f  cpu 2 %f index %f \n", sorted_dev[i].x, sorted_dev[i].z, sorted_dev[i].y,  sorted_dev[i].w) ;
        }
        if(sorted_dev[i].w !=  sorted_host[i].w)
        {
            failed ++ ; 
 //           printf("w is not good \n"); 
          
        }
        
    }
    printf("found %i out of %i nn \n",((size_q * 2)- failed),(size_q *2) ) ; 
}



