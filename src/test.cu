#include "knn_brute.h"

#include <iostream>
#include <string>
#include <math.h>
#include <sys/time.h>


void make_rand_vector(int dim, float * vec);
void make_rand_vec_array(int dim, int size, float * array) ; 
void test() ;
 
int main( int argc, char* argv[])
{
    test() ; 
    return 0 ; 
}

void test()
{
    int dim =  128 ; 
    int size_q = 10 ; 
    int size_r = 10 ; 
    float * q_points ; 
    float * r_points ; 

    float2 * sorted ; 
    //host 
    cudaMallocHost(&q_points, dim*size_q*sizeof(float)) ; 
    cudaMallocHost(&r_points, dim*size_r*sizeof(float)) ;
    
    //data 
    make_rand_vec_array(128, size_q , q_points) ; 
    make_rand_vec_array(128, size_r, r_points) ; 

    //
    for (size_t i = 0; i < dim*size_r; i++)
    {
        printf(" %f \n", r_points[i]) ; 
    }



}

void make_rand_vector(int dim, float * vec)
{
    float * r_vec = (float *)vec ; 
    for (size_t i = 0; i < dim; i++)
    {
        r_vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
   
}

void make_rand_vec_array(int dim, int size, float * array)
{ 
    float * arr = (float *)array ;        
    for (size_t i = 0; (i/dim) < size ;i+=  dim)
    {
        make_rand_vector(dim, &arr[i]) ; 
    }
}
