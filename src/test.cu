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
    // test_half_float() ; 
    return 0;
}

void test_float()
{
    int dim = 128;
    //for real data q max == 10000
    // r max == 1000000
    int size_q = 10000;
    int size_r = 1000000;

    // if 1 real data will be read, need to have the files for it to work 
    int real_data = 0 ; 
    int float_char = 1 ; 
    
    des_t_f *q_points;
    des_t_f *r_points;
    
    des_t_f *gpu_q_points;
    des_t_f *gpu_r_points;

    //make streams/ handels 
    cublasHandle_t handle ;  
    cublasCreate(&handle); 

    uint32_t * matches_lsh ; 
    cudaMallocManaged((void **)&matches_lsh, size_q * sizeof(uint32_t));
    // uint32_t * matches_brute ; 
    // cudaMallocManaged((void **)&matches_brute, size_q * sizeof(uint32_t));

    uint32_t * true_values ; 
    int recall = 100  ; 
    cudaMallocHost((void **)&true_values, size_q * sizeof(uint32_t) * recall );

    cudaMallocHost((void **)&q_points, size_q * sizeof(des_t_f), cudaHostAllocMapped);
    cudaMallocHost((void **)&r_points, size_r * sizeof(des_t_f), cudaHostAllocMapped);
 
    cudaMalloc((void **)&gpu_q_points, size_q * sizeof(des_t_f));
    cudaMalloc((void **)&gpu_r_points, size_r * sizeof(des_t_f));

      // random fake data 
    if(real_data == 0)
    {
      make_rand_vec_array(dim, size_q, q_points);
      make_rand_vec_array(dim, size_r, r_points);
    }
    
    if(real_data == 1)
    {
    // read in test data 
      FILE * base_vector = fopen("../../data_set/sift/sift_base.fvecs", "r") ; 
      FILE * query_vector = fopen("../../data_set/sift/sift_query.fvecs", "r") ; 
      FILE * true_val = fopen("../../data_set/sift/sift_groundtruth.ivecs", "r") ; 

      // read base_vectro to r_points 
      int dim_test = 0; 
      float avg_norm  = 0  ; 
      for (int i = 0; i < size_r; i++)
      {
        fread(&dim_test, sizeof(int), 1, base_vector) ; 

        if(dim_test != 128)
        {
          printf(":( bad input ) \n ") ; 
        }
        fread(r_points[i], sizeof(des_t_f), 1, base_vector)  ;

        //make into real float values 
        float norm = 0 ; 
        if(float_char== 1)
        {
          float * vec = (float *)r_points[i] ; 
          for (int ii = 0; ii < 128; ii++)
          {
            vec[ii] = vec[ii]/512; 
            norm += vec[ii] * vec[ii] ; 
          }
          avg_norm += sqrt(norm)/size_r ; 
        }
      }

      // printf("avg norm for input vec is %f \n" ,avg_norm ); 
      avg_norm = 0 ;
      // read query_vector to q_points  
      for (int i = 0; i < size_q; i++)
      {
        fread(&dim_test, sizeof(int), 1, query_vector) ; 
        
        if(dim_test != 128)
        {
          printf(":( bad input ) \n ") ; 
        }
        fread(q_points[i], sizeof(des_t_f), 1, query_vector)  ;

        float norm = 0 ; 
        if(float_char== 1)
        {
          float * vec = (float *)q_points[i] ; 
          for (int ii = 0; ii < 128; ii++)
          {
            vec[ii] = vec[ii]/512; 

            norm += vec[ii] * vec[ii] ; 
          }
          avg_norm += sqrt(norm)/size_q ; 
        }
      }
      // printf("avg norm for q vec is %f \n" ,avg_norm ); 
      int n_check ; 
      // read truth to true_values 
      for (int i = 0; i < size_q + 2; i++)
      {
        fread(&n_check, sizeof(int), 1, true_val) ; 
        if(n_check != 100)
        {
          printf(":( bad input ) \n ") ; 
        }

        fread(&true_values[i* recall], sizeof(uint32_t), recall, true_val) ; 

        fseek(true_val,sizeof(int)* (100 - recall), SEEK_CUR) ; 
      }
      
      fclose(base_vector) ; 
      fclose(query_vector) ; 
      fclose(true_val) ; 
    }
    float4 * sorted_2nn; 
    cudaMallocManaged((void **)&sorted_2nn, size_q * sizeof(float4));
    
    // float4 * sorted_2nn_cublas; 
    // cudaMallocManaged((void **)&sorted_2nn_cublas, size_q * sizeof(float4));


    double s ;  
    cudaMemcpy(gpu_q_points, q_points, size_q * sizeof(des_t_f), cudaMemcpyHostToDevice) ; 
    cudaMemcpy(gpu_r_points, r_points, size_r * sizeof(des_t_f), cudaMemcpyHostToDevice) ; 

    cudaDeviceSynchronize(); 
   
    //  for (int  x = 2; x < 17; x+=2 )
    //   {

    //     s = start_timer() ; 
    //     for (int xx = 0; xx < 100; xx++)
    //     {
    //       /* code */
    //     //  
    //     //  cublas_2nn_sift(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 1.1, handle, 1,(size_r * 2) * (size_t)500, 17); 

    // // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_2nn, 32 , 1, 0, handle);
    //   //  lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 12, 450, 0, 19); 
    //   //  lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 12, 81, 0, 20); 
    //   // device_brute(gpu_q_points, gpu_r_points, size_q,size_r, sorted_2nn); 
    //   //  int count =  lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 8, 450, 0, 19); 
    //   //  cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;
    //    cudaDeviceSynchronize() ;
     
    //     }
    //     print_time_x_y(s,x,100) ; 
    //     /* code */
    //   }
      
    // return ; 

    device_brute(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn) ;
    cudaDeviceSynchronize(); 
    //   cudaProfilerStart();
    // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_lsh, 25, 20, 0, handle);
    // device_brute(gpu_q_points, gpu_r_points, size_q,size_r, sorted_2nn); 
    // host_brute(q_points, r_points, size_q,size_r, sorted_2nn); 
    int temp_val = 1 ; 
   for (int i = 27;  i <  28; i += 1)
   {
     int flag = 0 ; 
       for (int ii = 2; ii <  17; ii += 2)
       {
      
     
       s = start_timer() ; 
        //  
         cublas_2nn_sift(q_points, r_points, 0, size_q, size_r, matches_lsh, 2, handle, 1,(size_r * 2) * (size_t)383, 17); 

      //  lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 16, 450, 0, 19); 
        //  cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;

        // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_2nn, 19 , 450, 0, handle);
       cudaDeviceSynchronize() ;
   
       
       double time = start_timer() - s;
     
      print_time_x_y(s, 0, 1) ; 
      printf("%i %i \n", i, ii) ; 
     
       int recall_count[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ; 
    for (size_t n = 0; n < size_q; n++)
    {
        // 0.8 threshold
        if( sorted_2nn[n].z != matches_lsh[n] && sorted_2nn[n].z != -1)
        {
            recall_count[0] ++ ;  
        }


        int recall_flag[11] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} ; 
        for (int nn = 0; nn < recall; nn++)
        {
          if(true_values[n * recall + nn] == matches_lsh[n] )
          { 
            if(nn == 0)
            {
              memset(recall_flag, 0, sizeof(recall_flag));
              break; 
            }
            else if(nn < 10)
            {
              memset(recall_flag + 1, 0, sizeof(recall_flag)- sizeof(int)* 1);
            }
            else if(nn < 20)
            {
              memset(recall_flag + 2, 0, sizeof(recall_flag)- sizeof(int)* 2);
            }
            else if(nn < 30)
            {
              memset(recall_flag + 3, 0, sizeof(recall_flag)- sizeof(int)* 3);
            }
            else if(nn < 40)
            {
              memset(recall_flag + 4, 0, sizeof(recall_flag)- sizeof(int)* 4);
            }
            else if(nn < 50)
            {
              memset(recall_flag + 5, 0, sizeof(recall_flag)- sizeof(int)* 5);
            }
            else if(nn < 60)
            {
              memset(recall_flag + 6, 0, sizeof(recall_flag)- sizeof(int)* 6);
            }
            else if(nn < 70)
            {
              memset(recall_flag +  7, 0, sizeof(recall_flag)- sizeof(int)* 7);
            }
            else if(nn < 80)
            {
              memset(recall_flag + 8, 0, sizeof(recall_flag)- sizeof(int)* 8);
            }
            else if(nn < 90)
            {
              memset(recall_flag + 9, 0, sizeof(recall_flag)- sizeof(int)* 9);
            }
            else if(nn < 100)
            {
              memset(recall_flag + 10, 0, sizeof(recall_flag)- sizeof(int)* 10);
            }

          }   /* code */
        }
        recall_count[1] += recall_flag[0]  ; 
        recall_count[2] += recall_flag[1]  ; 
        recall_count[3] += recall_flag[2]  ; 
        recall_count[4] += recall_flag[3]  ; 
        recall_count[5] += recall_flag[4]  ; 
        recall_count[6] += recall_flag[5]  ; 
        recall_count[7] += recall_flag[6]  ; 
        recall_count[8] += recall_flag[7]  ; 
        recall_count[9] += recall_flag[8]  ; 
        recall_count[10] += recall_flag[9]  ; 
        recall_count[11] += recall_flag[10]  ; 
       }

        printf("%i %i %i %i %i %i %i %i %i %i %i %i \n", recall_count[0], recall_count[1], recall_count[2], recall_count[3], recall_count[4], recall_count[5],
        recall_count[6], recall_count[7], recall_count[8], recall_count[9], recall_count[10], recall_count[11]) ; 
        if(recall_count[1] < 2000 )
        {
          flag ++ ; 
          ii -- ; 

        }
      if(flag == 3){
        flag = 0 ; 
        printf("new\n"); 
        break;
      } 
     }
      
 
        

 

        
      //  print_time_x_y(s, i, 1) ; 
  }
   return ; 
// 
     
    // device_brute(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn) ;

    // cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;
    // int failed = 0  ; 
    // int no_match = 0 ; 
    // float avg_norm_q_correct = 0 ; 
    // float avg_norm_q_mistaken = 0 ; 
    // float avg_norm_r_correct = 0 ; 
    // float avg_norm_r_mistaken = 0 ; 

   
    // cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;

    // cublas_2nn_sift(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_brute, 1.1, handle, 1,(size_r * 2) * (size_t)500); 
    cudaDeviceSynchronize() ;
  // for (int n = 0; n < size_q; n++)
  //   {
  //       float norm_1 = 0 ; 
  //       float norm_2 = 0 ; 
  //       float * vec_q = (float * )(q_points[n]); 
  //       float * vec_r = (float * )(r_points[true_values[n]]); 


  //       for (int i = 0; i < 128; i++)
  //       {
  //         norm_1 += vec_q[i] * vec_q[i] ; 
  //         norm_2 += vec_r[i] * vec_r[i] ;  
  //       }
  //       if(true_values[n] != sorted_2nn[n].z)
  //       {
  //        printf("%i\n", n)  ; 
  //        failed ++ ;
  //         avg_norm_q_mistaken  += sqrt(norm_1)  ; 
  //         avg_norm_r_mistaken  += sqrt(norm_2)  ; 

  //     }
  //     else{
  //         avg_norm_q_correct += sqrt(norm_1)  ; 
  //         avg_norm_r_correct += sqrt(norm_2)  ; 
        
  //     }
  //     }
    // printf(" no match %i\n", no_match) ; 
    // printf(" failed %i\n", failed) ; 

    // printf("correct %i \n ", (10000 - failed - no_match)) ; 


  //     avg_norm_q_correct = avg_norm_q_correct /(10000 - failed ); 
  //     avg_norm_r_correct = avg_norm_r_correct /(10000 - failed ); 
  //     avg_norm_q_mistaken= avg_norm_q_mistaken /failed ; 
  //     avg_norm_r_mistaken= avg_norm_r_mistaken /failed ; 
  // printf("%i\n norm = q %f r %f \n", failed, avg_norm_q_mistaken, avg_norm_r_mistaken ) ; 
  //   printf("correct  norm = q %f r %f\n", avg_norm_q_correct, avg_norm_r_correct) ; 
 
    // for (int n = 0; n < size_q; n++)
    // {
    //   if(matches_brute[n] != true_values[n] && matches_brute[n] != UINT32_MAX  )
    //   {
    //     failed ++ ;
    //     printf("%i\n", n) ; 
    //   }
    //   if(matches_brute[n] == UINT32_MAX)
    //   {
    //     no_match ++ ; 
    //   }
    // }

    //   return ; 
    // for (int n = 0; n < size_q; n++)
    // {
    //   if(sorted_2nn[n].z != true_values[n] && sorted_2nn[n].z != -1)
    //   {
    //     failed ++ ;
    //   }
    //   if(sorted_2nn[n].z== -1)
    //   {
    //     no_match ++ ; 
    //   }
    // }
 
    // printf("%i\n norm = q %f r %f \n", failed, avg_norm_q_mistaken, avg_norm_r_mistaken ) ; 
    // printf("correct  norm = q %f r %f", avg_norm_q_correct, avg_norm_r_correct) ; 
    // printf(" no match %i\n", no_match) ; 
    // printf(" failed %i\n", failed) ; 
    // printf("correct %i \n ", (10000 - failed - no_match)) ; 

    // return ; 
    
    s = start_timer();

    // lsh_gpu(gpu_q_points, gpu_r_points, 1, size_q, size_r, matches_lsh, 0.8, handle, 16, 100, 0, 22); 
    cudaDeviceSynchronize() ;
    //    cudaProfilerStop() ;
    // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_2nn, 15 , 10, 0, handle[0]);
    print_time(s, "lsh brute"); 

    s = start_timer() ; 

    cublas_2nn_f(gpu_q_points,gpu_r_points,size_q,size_r, sorted_2nn, handle) ;
    print_time(s, "normal brute ") ; 
    int failed_brute = 0 ; 
    int failed_lsh = 0 ; 
    int failed_brute_f = 0 ; 
    for (size_t i = 0; i < size_q; i++)
    {
        if( sorted_2nn[i].z != matches_lsh[i])
        {
            failed_brute ++ ; 
            // printf("%i, %i \n", (int)matches_brute[i], (int)matches_lsh[i]); 
            // printf("%i, %i \n", (int)matches_brute[i], (int)true_values[i]); 

        }

          int test = 1 ; 
        for (int ii = 0; ii < recall; ii++)
        {
         if(true_values[i * recall + ii] == matches_lsh[i] )
        {
            test = 0 ; 
            // printf("%i, %i \n", (int)matches_brute[i], (int)matches_lsh[i]); 
            // printf("%i, %i \n", (int)matches_brute[i], (int)true_values[i]); 
        }   /* code */
        }
        failed_lsh += test ; 
        
        
        //   printf("%i, %i \n", (int)matches_brute[i], (int)matches_lsh[i]); 
       if( - 1 != sorted_2nn[i].z )
        {
            failed_brute_f++ ; 
            // printf("%i, %i \n", (int)sorted_2nn[i].z, (int)true_values[i]); 
        }

        
    }
    printf("failed brute  %i \n", failed_brute); 
    printf("failed lsh %i \n", failed_lsh); 
    // 
    printf("failed brute f %i \n", failed_brute_f); 
    cudaFree(gpu_q_points); 
    cudaFree(sorted_2nn); 
    cudaFree(gpu_r_points); 
    cudaFree(q_points); 
    cudaFree(r_points); 
}


void test_half_float()
{
    int dim = 128;
    int size_q = 1000000;
    int size_r = 10000;

    des_t_h *q_points;
    des_t_h *r_points;
    
    des_t_h2 *gpu_q_points;
    des_t_h2 *gpu_r_points;

    uint32_t * matches ; 
    //make streams/ handels 
    int stream_n = 2 ; 
     cublasHandle_t handle ;  
    cublasCreate(&handle); 

 
   
    
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
    //   cudaProfilerStart();
   // lsh_test(gpu_q_points, gpu_r_points, size_q, size_r, sorted_lsh, 25, 20, 0, handle);
    // device_brute(q_points,r_points,size_q,size_r, sorted_lsh) ;
    double s ;  
    // device_brute(gpu_q_points, gpu_r_points, size_q,size_r, sorted_2nn); 
    // host_brute(q_points, r_points, size_q,size_r, sorted_2nn); 
   for (int i = 1;  i < 17; i++ )
   {
       s = start_timer() ; 
       for (int ii = 0; ii < 10 ; ii++)
       {
         cublas_2nn_sift(gpu_q_points, gpu_r_points, 2, size_q, size_r, matches, 1.1, handle, i,(size_r * 2) * (size_t)3067, 17); 
         cudaDeviceSynchronize() ;
       } 
       print_time_x_y(s, i, 10) ; 
   }
   return ; 
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

  