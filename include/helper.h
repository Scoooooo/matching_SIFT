#include "knn_brute.h"

double start_timer() ; 
void print_time(double start, const char * s) ; 

void compare_float(float x, float y) ; 
void compare_int(int x, int y) ; 

void make_rand_vec_array(int dim, int size, des_t *array) ; 
void make_rand_vector(int dim, des_t &vec) ; 