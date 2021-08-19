#include "knn_brute.h"

#include <iostream>
#include <string>
#include <math.h>
#include <sys/time.h>

void make_vector(des_t * vec) ; 
float lenght(des_t x, des_t y) ; 
void sort_host(float * dist, int size, int dim, float2 * sorted) ; 

void test_dist(int q_n, int r_n) ; 
void test2_min(int n, int q) ; 

int main( int argc, char* argv[])

{
    test2_min(1124, 10000) ; 

    return 0;
}
    
// kernel tests 
void test_dist(int q_n, int r_n){
        //malloc host

    des_t * q_points ; 
    des_t * r_points ; 
    
    cudaMallocHost((void **) &q_points, q_n* sizeof(des_t)) ; 
    cudaMallocHost((void **) &r_points, r_n* sizeof(des_t)) ; 

    //random data 
    srand (static_cast <unsigned> (time(0)));
    //query points
    for (size_t i = 0; i < q_n; i++) 
    {
        make_vector(&q_points[i]) ;  
    }
    //reference points
    for (size_t i = 0; i < r_n; i++) 
    {
        make_vector(&r_points[i]) ;
    } 
    //move data to device 
    des_t * dev_q_points ; 
    des_t * dev_r_points ; 
    float * dev_dist ; 

    //cudaMallocManaged()
    
    cudaMalloc((void **) &dev_q_points, q_n* sizeof(des_t)) ; 
    cudaMalloc((void **) &dev_r_points, r_n* sizeof(des_t)) ; 
    // array of dist from each q point to every r point 
    cudaMalloc((void **) &dev_dist, q_n* r_n * sizeof(float)) ; 

    cudaMemcpy(dev_q_points, q_points, q_n* sizeof(des_t), cudaMemcpyHostToDevice) ; 
    cudaMemcpy(dev_r_points, r_points, r_n* sizeof(des_t), cudaMemcpyHostToDevice) ; 

    dim3 block_size(32, 1, 1) ;   
    dim3 grid_size(q_n, r_n, 1) ;

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size, 0>>>(dev_q_points, dev_r_points, dev_dist);
    cudaDeviceSynchronize();
    // free before continuing ? when to free ?  
    cudaFree(q_points) ; 
    cudaFree(r_points) ; 
    cudaFree(dev_q_points) ; 
    cudaFree(dev_r_points) ; 
    cudaFree(dev_dist) ; 

}
void test2_min(int size, int dim)
{
    srand (static_cast <unsigned> (time(0)));
    float * host_dist ; 
    float * dev_dist ; 
    
    float2 * host_sorted ; 
    float2 * dev_sorted ; 

    cudaMallocHost((void **) &host_dist, size*dim*sizeof(float)) ;
    cudaMalloc((void **) &dev_dist, size*dim*sizeof(float)) ;  
    
    cudaMallocHost((void **) &host_sorted, dim * sizeof(float2)) ;
    cudaMalloc((void **) &dev_sorted, dim * sizeof(float2)) ;  
    
    for (int i = 0; i < size*dim; i++)
    {
        host_dist[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
    }

    cudaMemcpy(dev_dist, host_dist, size*dim*sizeof(float), cudaMemcpyHostToDevice) ; 
    
    //dim3 blockSize(1024,1,1) ; 
    //dim3 gridSize(dim/1024 + 1,1,1) ; 
    
    dim3 blockSize(32,3,1) ; 
    dim3 gridSize(dim,1,1) ;
    min_2_4<<<gridSize,blockSize>>>(dev_dist,size,dev_sorted) ; 
    cudaDeviceSynchronize();
    
    printf("cpu ") ; 
    sort_host(host_dist, size, dim, host_sorted) ;
    
    cudaFree(host_dist) ; 

    ////test 
    float2 * test ; 
    cudaMallocHost((void **) &test, dim * sizeof(float2)) ;
    cudaMemcpy(test,dev_sorted,dim * sizeof(float2),cudaMemcpyDeviceToHost) ; 
    for (size_t i = 0; i < dim; i++)
    {
        if(test[i].y != host_sorted[i].y )
        {
        printf("cpu id %d x %f y %f \n",i, host_sorted[i].x,host_sorted[i].y) ;
        printf("gpu id %d x %f y %f \n", i, test[i].x, test[i].y);
        printf("bad y \n") ;  
        }
        if(test[i].x != host_sorted[i].x )
        {
         printf("bad x \n") ;  
        }
        
    }
    

    cudaFree(host_sorted) ; 
    cudaFree(dev_dist) ; 
    cudaFree(dev_sorted) ; 
}


void sort_host(float * dist, int size, int dim, float2 * sorted)
{
    for (int i = 0; i < dim; i++)
    {
        float2 min_2 ;  
        min_2.x = 0xffffff; 
        min_2.y = 0xffffff; 
        int offset = i * size ; 
        for (int ii = 0; ii < size; ii++)
        {
            if(dist[ii + offset] < min_2.x)
            {
                min_2.y = min_2.x ; 
                min_2.x = dist[ii + offset] ;  
            }
            else{
                if(dist[ii + offset] < min_2.y)
                {
                min_2.y = dist[ii + offset] ;  
                }
            }
        }
        sorted[i] = min_2 ;  
    }
}

void make_vector(des_t * vec){
    float * r_vec = (float *)vec ; 
    for (size_t i = 0; i < 128; i++)
    {
        r_vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

float lenght(des_t x, des_t y){
    float * vec1 = (float * )x ; 
    float * vec2 = (float * )y ;
    float dist = 0 ;  
    for (size_t i = 0; i < 128; i++)
    {
        float a = vec1[i] ; 
        float b = vec2[i] ;  
        float c = a - b ; 
        dist += c * c ; 
    }
    return dist ; 
}
