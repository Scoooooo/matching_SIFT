#include "knn_brute.h"
typedef  float des_t_f[128];  
typedef  __half des_t_h[128]; 

double start_timer() ; 
void print_time(double start, const char * s) ; 

void compare_float(float x, float y) ; 
void compare_int(int x, int y) ; 

void make_rand_vec_array(int dim, int size, des_t_f *array) ; 
void make_rand_vector(int dim, des_t_f &vec) ; 

void make_rand_vector_h(int dim, des_t_h &vec) ; 
void make_rand_vec_array_h(int dim, int size, des_t_h *array) ; 