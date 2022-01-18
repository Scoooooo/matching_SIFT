#include "helper.h"
#include <sys/time.h>
#include <iostream>
// time stuff 
double start_timer()
{
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void print_time(double start, const char *  s)
{
    double time = start_timer() - start;
    printf("%s took %lf time \n", s, time) ;      
}
// compare stuff 
void compare_float(float x, float y)
{
    if(x != y)
    {
        printf("x = %f != y = %f \n", x, y) ; 
    }
} 
void compare_int(int x, int y) 
{
    if(x != y)
    {
        printf("x = %i != y = %i \n", x, y) ; 
    }
} 

// make random data 
void make_rand_vector(int dim, des_t_f &vec)
{
     for (size_t i = 0; i < dim; i++)
        {
        
            vec[i] =static_cast<float>(rand()); 
    
        }
        float sum = 0 ; 
        for (size_t i = 0; i < dim; i++)
        {
           sum += vec[i]*vec[i]  ; 
        }
        sum = sqrtf(sum)  ; 
        for (size_t i = 0; i < dim; i++)
        {
            vec[i] /= sum ;  
        } 
}

void make_rand_vec_array(int dim, int size, des_t_f *array)
{
    des_t_f *arr = (des_t_f *)array;
    for (size_t i = 0; i < size; i++)
    {
        make_rand_vector(dim, arr[i]);
    }
}
// make random data 
void make_rand_vector_h(int dim, des_t_h &vec)
{
    float temp[dim] ;  
     for (size_t i = 0; i < dim; i++)
        {
            temp[i] = static_cast<float>(rand()); 
        }
        float sum = 0 ; 
        for (size_t i = 0; i < dim; i++)
        {
           sum += temp[i]*temp[i]  ; 
        }
        sum = sqrtf(sum)  ; 
        for (size_t i = 0; i < dim; i++)
        {
            temp[i] /= sum ;  
        } 
        for (int i = 0; i < dim; i++)
        {
            const float to_half = temp[i]; 
            vec[i] = __float2half(to_half); 
        }
}

void make_rand_vec_array_h(int dim, int size, des_t_h *array)
{
    des_t_h *arr = (des_t_h *)array;
    for (size_t i = 0; i < size; i++)
    {
        make_rand_vector_h(dim, arr[i]);
    }
}