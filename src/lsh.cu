#include <iostream>
#include "curand_kernel.h"
#include "curand_uniform.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
#include "knn_brute.h"
#include <memory>
#include <vector>
#include<bits/stdc++.h>
#include <map>
using namespace std;

void host_lsh(des_t * q_points, des_t * r_points, int n_q, int n_r, float4  * sorted, int nbits, int l)
{

    des_t * rand_array ; 
    cudaMallocManaged((void **) &rand_array, sizeof(des_t) * nbits * l) ; 

    // make random vectors 
    for (size_t i = 0; i < nbits*l; i++)
    {
        make_vec(128, rand_array[i]) ;
    }   

    // make an array of ints with one int for each r_point
    unsigned int * code ; 
    cudaMallocManaged((void **) &code, sizeof(int ) * n_r * l) ; 
    cudaMemset(code, 0, sizeof(int) * n_r * l);

    // dot all vectors and add the bit to the coresponding int bit for the r points  

    for (size_t i = 0; i < l; i++)
    {
        for (size_t ii = 0; ii < n_r; ii++)
        {
           // printf(" %i bucket = ", ii) ; 
            for (size_t iii = 0; iii < nbits ; iii++)
            {
                float sum = dot(r_points[ii], rand_array[iii + i*nbits]) ; 
                if(sum <= 0)
                {
                    code[ii + i*n_r] |= 1UL << iii;
                }
              //  if(sum >= 0)
              //  {
              //      printf("0") ; 
              //  }
              //  else
              //  {
              //      printf("1") ; 
              //  }
            }
          //  printf(" %u " , code[ii + i*n_r]) ; 
          //  printf("\n \n") ; 
        }
    }
     

    // make buckets for r points      
    map<int, set<int>> buckests ;

    for (size_t i = 0; i < l; i++)
    {
        for (int ii = 0; ii < n_r ; ii++)
        {
            // could also add to neighbouring buckets 
            auto it = buckests.find(code[ii + n_r *i]) ;
            if(it == buckests.end())
            {
                set<int> s = {ii };  
                buckests.insert(make_pair(code[ii + n_r * i], s ));           
            }
            else
            {
                it->second.insert(ii) ; 
            } 
        }
    }

    // test   
    for(const auto& elem : buckests)
    {
        std::cout << elem.first << " " <<  "\n";
        for (auto it = elem.second.begin(); it !=
                            elem.second.end(); ++it)
        cout << ' ' << *it;    
        cout << "\n" ; 
    }

    // for each q point dot with random vectors and find the correct bucket 
    // add all elements for that bucket to the set 
    // do this l times 
    // preform a brute force search on values in the set 
    unsigned int code_q ; 
    set<int> combined_buckets  = {};

    for (size_t i = 0; i < n_q; i++)
    {
        // fill the set 
       for (size_t ii = 0; ii < l; ii++)
       {
            code_q = 0 ;
           
            for (size_t iii = 0; iii < nbits ; iii++)
                {
                float sum = dot(q_points[i], rand_array[iii + ii*nbits]) ; 
                if(sum <= 0)
                {
                    code_q |= 1UL << iii;
                }
            }
            auto it = buckests.find(code_q) ; 

            if(it != buckests.end())
            {
                combined_buckets.insert( it->second.begin(),  it->second.end()) ; 
            }
        }
        //burte force for the one q point 
        cout << "combined bucket for " << i << " is " ; 
        for (auto&  num: combined_buckets)
        {
            std::cout << num << ' ';
        }
        cout << "\n" ;
    }
}

void make_vec(int dim, des_t  &vec)
{
    for (size_t i = 0; i < dim; i++)
    {
        vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }  
}

float dot(des_t v1, des_t v2)
{   

    float sum  = 0.f; 
    for (size_t i = 0; i < 128 ; i++)
    {
        float a = ((float *)v1)[i]; 
        float b = ((float *)v2)[i]; 
        float c = a - b ; 
        sum += c ; 
    }
    return sum ; 
}
// makes a random float 
__device__ inline float random_float(uint64_t seed, int idx, int call_count) 
{
    curandState s ; 
    curand_init(seed + idx + call_count, 0, 0, &s);
    return curand_uniform(&s);
}

//fills in a vector of random floats 
__global__ void random_vector(uint64_t seed, des_t * vec) 
{
    int idx = blockIdx.x * blockDim.x  + threadIdx.x ;
    float * vector = (float *) vec ; 
    for (size_t i = 0; i < 4; i++)
    {
        vector[idx + (32*i)] =  random_float(seed, idx, i);
    }
}


__global__ void dot_make_codes()
{

}


void device_lsh(des_t * q_points, des_t * r_points, int n_q, int n_r, float4  * sorted, int nbits, int l)
{
    // repeat l times !! will lead to comparing the same points multiple times ?


    // do we need random data ? 
    // make random vectors 

    des_t * rand_array ; 
    cudaMallocManaged((void **) &rand_array, sizeof(des_t) * nbits * l) ; 

    // fill array
    uint64_t seed = 132 ;  
    dim3 grid_size(1, 1, 1) ;
    dim3 block_size(32, 1, 1) ;   

    //fill in the dist array
    random_vector<<<grid_size, block_size>>>(seed, rand_array);
 
    cudaDeviceSynchronize();

    for (size_t i = 0; i < 128; i++)
    {
        printf("%i  %f \n  ", i, ((float *)rand_array )[i] ) ; 
    }
     
    // dot vectors with r_points  
    // make hash codes fro doted res  
    


    // make buckets 
    // set ? map ? stuct ?  int array ? vector ?  

    // use pointer vs moving data ? 


    // dot q points 

    // brute in bucket  


}