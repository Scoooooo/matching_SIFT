#include "knn_brute.h"

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
    int size_r = 10 ; 
    des_t * q_points ; 
    des_t * r_points ; 

    float2 * sorted_host ; 
    float2 * sorted_dev ; 
    //host 
    cudaMallocManaged(&q_points, size_q*sizeof(des_t)) ; 
    cudaMallocManaged(&r_points, size_r*sizeof(des_t)) ;

    cudaMallocManaged(&sorted_host, size_r * sizeof(float2)) ; 
    cudaMallocManaged(&sorted_dev, size_r * sizeof(float2)) ; 
        
    //data 
    make_rand_vec_array(dim, size_q , q_points) ; 
    make_rand_vec_array(dim, size_r, r_points) ;

    for (size_t i = 0; i < (size_q ); i++)
    {
        for (size_t ii = 0; ii < 130; ii++)
        {
            printf("%f \n",q_points[i][ii] )  ;
        }
    }
    host_brute(q_points,r_points,size_q,size_r, sorted_host) ; 
    device_brute(q_points, r_points, size_q, size_r, sorted_dev) ; 

    for (size_t i = 0; i < size_r; i++)
    {
        printf("smallet %f second smallets %f \n", sorted_host[i].x, sorted_host[i].y) ; 
        printf("smallet %f second smallets %f \n", sorted_dev[i].x, sorted_dev[i].y) ; 
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
