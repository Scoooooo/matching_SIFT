#include "knn_brute.h"
#include "lsh.h"
#include "cuda_profiler_api.h"
#include <iostream>
#include <string>
#include <math.h>
#include <sys/time.h>
#include "helper.h"

void make_rand_vector(int dim, des_t &vec);
void make_rand_vec_array(int dim, int size, des_t *array);
void test();
#include <iostream>
#include <vector>
#include <functional>

void walkingBits(size_t n, size_t k) {
    std::vector<bool> seq(n, false);
    std::function<void(const size_t, const size_t, const int, size_t)> walk = [&](const size_t n, const size_t k, const int dir, size_t pos){
        for (size_t i = 1; i <= n - k + 1; i++, pos += dir) {
            seq[pos] = true;
            if (k > 1) {
                walk(n - i, k - 1, i % 2 ? dir : -dir, pos + dir * (i % 2 ? 1 : n - i));
            }
            else {
                for (bool v : seq) {
                    std::cout << v;
                }
                std::cout << std::endl;;
            }
            seq[pos] = false;
        }
    };
    walk(n, k, 1, 0);
}


int main(int argc, char *argv[])
{
    test();
    return 0;
}
void test()
{
    int dim = 128;
    int size_q = 200;
    int size_r = 200;
    des_t *q_points;
    des_t *r_points;

    float4 *sorted_host;
    float4 *sorted_dev;
    //host
    cudaMallocManaged((void **)&q_points, size_q * sizeof(des_t));
    cudaMallocManaged((void **)&r_points, size_r * sizeof(des_t));

    cudaMallocManaged((void **)&sorted_host, size_r * sizeof(float4));
    cudaMallocManaged((void **)&sorted_dev, size_r * sizeof(float4));

    // 4 2 3 3 
    // 8 2 7 7
    //data
    make_rand_vec_array(dim, size_q, q_points);
    make_rand_vec_array(dim, size_r, r_points);
    double s = start_timer();
    //   cudaProfilerStart();
    lsh_test(q_points, r_points, size_q, size_r, sorted_host, 4, 1, 1);
    //    cudaProfilerStop() ;
    //gpu_lsh(q_points, r_points, size_q, size_r, sorted_host, 4, 4, 2);
    print_time(s, "gpu lsh"); 
    s = start_timer() ; 
    device_brute(q_points,r_points,size_q,size_r, sorted_dev) ;
    print_time(s, "gpu brute") ; 
    int failed = 0 ; 
    for (size_t i = 0; i < size_q; i++)
    {
       // printf("lsh 1  %f index %f  lsh 2 %f index %f \n", sorted_host[i].x, sorted_host[i].z, sorted_host[i].y,  sorted_host[i].w) ;
       // printf("cpu 1  %f index %f  cpu 2 %f index %f \n", sorted_dev[i].x, sorted_dev[i].z, sorted_dev[i].y,  sorted_dev[i].w) ;
       // printf("\n") ;
        if(sorted_dev[i].z !=  sorted_host[i].z)
        {
            failed ++ ; 
//            printf("z is bad \n") ; 
        }
        if(sorted_dev[i].w !=  sorted_host[i].w)
        {
            failed ++ ; 
 //           printf("w is not good \n"); 
          
        }
        
    }
    printf("found %i out of %i nn \n",((size_q * 2)- failed),(size_q *2) ) ; 

    walkingBits(6,2) ; 
}



