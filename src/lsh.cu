#include <iostream>
#include <string>
#include "lsh.h"
#include "knn_brute.h"

void host_lsh(des_t * q_points, des_t * r_points, int n_q, int n_r, float2  * sorted, int nbits)
{
   
    des_t * rand_array ; 

    cudaMallocManaged((void **) &rand_array, sizeof(des_t) * nbits) ; 

    // make rand vectors 
    for (size_t i = 0; i < nbits; i++)
    {
        make_vec(128, rand_array[i]) ;
    }   

    // make an array of ints with one int for each r_point
    int * code ; 

    cudaMallocManaged((void **) &code, sizeof(int ) * n_r) ; 

    cudaMemset(code, 0, sizeof(int) * n_r);

    // dot all vectors and add the bit to the coresponding int bit for the r points  
    for (size_t i = 0; i < n_r; i++)
    {
        for (size_t ii = 0; ii < nbits ; ii++)
        {
            float sum = dot(r_points[i],rand_array[ii]) ; 
            if(sum >= 0)
            {
                code[i] |= 1UL << ii;
            }
            if(sum >= 0)
            {
                printf("0") ; 
            }
            else
            {
                printf("1") ; 
            }
        }
        printf(" %u " , code[i]) ; 
        printf("\n \n") ; 
    }
    // make buckets     

    for (size_t i = 0; i < n_r; i++)
    {
        
    }
    
    // for each qury point find bucket  and nearby buckets add all then brute force 
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