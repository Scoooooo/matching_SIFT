#include "knn_brute.h"
#include "lsh.h"

#include <iostream>
#include <string>
#include <math.h>
#include <sys/time.h>


void make_rand_vector(int dim, des_t &vec);
void make_rand_vec_array(int dim, int size, des_t * array) ; 
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
    int size_r = 20 ; 
    des_t * q_points ; 
    des_t * r_points ; 

    float2 * sorted_host ; 
    float2 * sorted_dev ; 
    //host 
    cudaMallocManaged((void **)&q_points, size_q*sizeof(des_t)) ; 
    cudaMallocManaged((void **)&r_points, size_r*sizeof(des_t)) ;

    cudaMallocManaged((void **)&sorted_host, size_r * sizeof(float2)) ; 
    cudaMallocManaged((void **)&sorted_dev, size_r * sizeof(float2)) ; 
        
    //data 
    make_rand_vec_array(dim, size_q , q_points) ; 
    make_rand_vec_array(dim, size_r, r_points) ;

    device_brute(q_points, r_points, size_q, size_r, sorted_dev) ; 
    host_brute(q_points,r_points,size_q,size_r, sorted_host) ; 

    host_lsh(q_points,r_points,size_q,size_r, sorted_dev, 32) ; 
    for (size_t i = 0; i < size_q; i++)
    {   
        
        printf("cpu 1  %f  cpu 2 %f \n", sorted_host[i].x, sorted_host[i].y) ; 
        printf("gpu 1  %f  gpu 2 %f \n", sorted_dev[i].x, sorted_dev[i].y) ; 
        printf("\n") ;
    }
    
}

void make_rand_vector(int dim, des_t  &vec)
{
    for (size_t i = 0; i < dim; i++)
    {
        vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }  
}

void make_rand_vec_array(int dim, int size, des_t * array)
{ 
    des_t * arr = (des_t *)array ;        
    for (size_t i = 0; i < size ;i++)
    {
        make_rand_vector(dim,  arr[i]) ; 
    }
}
