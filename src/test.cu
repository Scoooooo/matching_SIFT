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
    float * q_points ; 
    float * r_points ; 
    cudaMallocHost(&q_points, 100*sizeof(float)) ; 
    cudaMallocHost(&r_points, 100*sizeof(float)) ;

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
