
#include <cstdlib>

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