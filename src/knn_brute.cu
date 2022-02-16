#include "curand_kernel.h"
#include "knn_brute.h"
#include <string>
#include "cublas_v2.h"
#include "lsh.h"
#include "knn_brute.h"
#include "helper.h"
#include "cuda_fp16.h"
#include <algorithm>
 
typedef struct{
   half2 x;
   half2 y;
   half2 z;
   half2 w;
} half8;
 

int cublas_2nn_sift(void * q_points, void * r_points, int type ,uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t handle)
{

    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    des_t_h2 * Q;
    des_t_h2 * R;  

    // we have 3 possible inputs chars/floats/half_floats
    // half will be fastest as we do not need to convert 

    // chars  
    if (type == 0)
    {
    
    }
    // floats
    else if (type == 1 )
    {
        // check if we need to pad if we do we need to remake 
        int pad_r_n = (r_n % 8) ;   
        if( pad_r_n != 0)
        {
            cudaMalloc((void **)&R,((8 - pad_r_n) + r_n) * sizeof(des_t_h2));
            cudaMemset(R, 0, ((8 - pad_r_n) + r_n) * sizeof(des_t_h2)); 
            // fill
            float2half<<<r_n, 64>>>((float * )r_points, (half2* )R) ; 
            r_n += (8 - pad_r_n); 
        }
        else
        {
            cudaMalloc((void **)&R, r_n * sizeof(des_t_h2));
            cudaMemset(R, 0, r_n * sizeof(des_t_h2)); 
            // fill
            float2half<<<r_n, 64>>>((float * )r_points, (half2 * )R) ; 
        }
        cudaMalloc((void **)&Q, q_n * sizeof(des_t_h2));
        // fill
        float2half<<<q_n, 64>>>((float * )q_points, (half2 * )Q) ; 
    }
    // halfs 
    else
    {
        // check if we need to pad if we do we need to remake 
        int pad_r_n = (r_n % 8) ;   
        if( pad_r_n != 0)
        {
            cudaMalloc((void **)&R,((8 - pad_r_n) + r_n) * sizeof(des_t_h2));
            cudaMemset(R, 0, ((8 - pad_r_n) + r_n) * sizeof(des_t_h2)); 
            cudaMemcpy(R,r_points, r_n * sizeof(des_t_h2), cudaMemcpyDeviceToDevice ); 
            r_n += (8 - pad_r_n); 
        }
        else{
            R = (des_t_h2 * )r_points ; 
        } 
        Q = (des_t_h2 * )q_points ; 
    }

     cudaError_t cudaStat = cudaDeviceSynchronize();

       if (cudaStat != cudaSuccess)
       {
           printf ("malloc copy failed :( ) \n");
           cublasDestroy(handle);
           exit(EXIT_FAILURE);
       }

    // cublas gemm wants to stasify 
    // m % 8 == 0
    // k % 8 == 0
    // op_B == CUBLAS_OP_N || n%8 == 0
    // intptr_t(A) % 16 == 0
    // intptr_t(B) % 16 == 0
    // intptr_t(C) % 16 == 0
    // intptr_t(A+lda) % 16 == 0
    // intptr_t(B+ldb) % 16 == 0
    // intptr_t(C+ldc) % 16 == 0 
    

    // number of bytes to we want for output array  
    // probly difrent for difrent gpus 

    size_t use = 40000000000 ; 
    // give us how many iterations we need to run  
    uint32_t new_q_n = use /((size_t) r_n * sizeof(half)); 
    printf("%i it \n", new_q_n) ; 
    
    //get number if iterartions +- 1 
    int it = q_n / new_q_n;  
    half2 * dist ; 
    // need to pad because we want the lengh
    cudaMalloc((void **)&dist, new_q_n * r_n * sizeof(half)) ; 
    printf("%i dist size in bytes  \n", (new_q_n * r_n * sizeof(half)) ) ; 

    int i = 0 ;  
    for (i = 0; i < it; i++)
    {
        printf("int %i \n", i); 
        cublas_2nn_sift_batch(Q + (i * new_q_n), R, new_q_n, r_n, dist, matches + (i * new_q_n), threshold, handle); 
    }
    if((q_n % new_q_n ) > 0 )
    {
        int left =  q_n % new_q_n; 
        printf("left %i \n", left) ; 
        // need to meemset as it will not fill the whole dist array         
        cudaMemset(dist, 0, left * r_n * sizeof(half));
        cublas_2nn_sift_batch(Q + (i * new_q_n), R, left, r_n, dist, matches + (i * new_q_n), threshold, handle); 
    }
    cudaFree(dist); 
    return 0  ; 
}

// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie or just manged 
int cublas_2nn_sift_batch(des_t_h2 * Q, des_t_h2 * R, uint32_t q_n, uint32_t r_n, half2 * dist, uint32_t * matches, float threshold, cublasHandle_t handle)
{
    // d^t = r * q^t is what cublas dose if see from cloum major
    // which is d = r^t * q   
   // float a = -1.f;
   // float b = 0.f;
    // singel for more accuracy but a bit slower 
    // cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (half *)Q, CUDA_R_16F, 128, 
                            // (half *)R, CUDA_R_16F, 128, &b, (half * )dist,CUDA_R_16F, r_n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
 
    //cublasStatus_t stat = cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (half *)Q, CUDA_R_16F, 128, (half *)R, CUDA_R_16F, 128, &b, (half *)dist,CUDA_R_16F, r_n);

    half a = -2.f;
    half b = 0.f; 
    cublasStatus_t stat = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (half *)R, 128, (half *)Q, 128, &b, (half *)dist, r_n);
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("dot failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    } 
    // want to find min value for each dist array 
    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,4,1) ; 
    find_matches<<<gridSize,blockSize>>>(dist, r_n / 2, matches, threshold) ; 
    cudaError_t cudaStat = cudaDeviceSynchronize();

    if (cudaStat != cudaSuccess)
    {
        printf ("min dist failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
    return 0 ; 
 
}

//__device__ inline void min_half8(__half2  &min_2, half8 temp, int2 &index, int2 temp_index)
//{
//
//    half2 val1 = __hgt2(temp.x, temp.y) ; 
//
//    half2 val2 = __hgt2(temp.x, temp.z) ; 
//    half2 val3 = __hgt2(temp.x, temp.w) ; 
//
//    half2 val4 = __hgt2(temp.y, temp.z) ; 
//    half2 val5 = __hgt2(temp.y, temp.w) ; 
//
//    half2 val6 = __hgt2(temp.z, temp.w) ; 
//
//    int a, b ; 
//    if(val1.x && val2.x && val3.x) ; 
//
//}

__global__ void float2half(float * points, half2 * output)
{
    float2 f2 =  ((float2 * )points)[threadIdx.x + blockDim.x * blockIdx.x] ; 
    output[threadIdx.x + blockDim.x * blockIdx.x] = __float22half2_rn(f2) ;
}

__global__ void find_matches(half2 *  dist, int size , uint32_t * matches, float threshold)
{
    //  finds the dist array         x dim       y dim pos                      
    int offset = (blockIdx.x * size) + threadIdx.y * blockDim.x ;

    half2 min_2 ;  
    int2 index ; 
   // half8 temp ; 
    half2 temp ; 
    //  our values will be negative so setting to 0 is fine  
    min_2.x = 0.0f; 
    min_2.y = 0.0f; 
   
    //  maybe better to read values first, however compiler may just be doing it for me

    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+= (blockDim.x * blockDim.y) )
    //for (int i = 0; (i + threadIdx.x) < size ; i+= (blockDim.x * blockDim.y) )
    {   
    
        // todo if time
        // need to read as uint4 to get 128 bits at once, half8 not suported ? 
        // why do we read 4 ?
        // performance is better idk why 0_0
        //temp = ((half8 * )dist)[i + offset + threadIdx.x]  ;     
        temp = dist[i + offset + threadIdx.x]  ;     

        int2 temp_index  ;  

        temp_index.x = (i + threadIdx.x + threadIdx.y * blockDim.x) * 2  ; 
        temp_index.y = temp_index.x + 1;  
        min_half(min_2, temp, index, temp_index); 

     //  min_half(min_2, temp.x, index, temp_index); 
     //  temp_index.x ++;   
     //  temp_index.y ++;  
     //  
     //  min_half(min_2, temp.y, index, temp_index); 
     //  temp_index.x ++;   
     //  temp_index.y ++;  
     //  
     //  min_half(min_2, temp.z, index, temp_index); 
     //  temp_index.x ++;   
     //  temp_index.y ++;  
     //  
     //  min_half(min_2, temp.w, index, temp_index); 

    }

    best_in_warp(min_2, index) ;   
    // hmm we do not alwasy need yhis much 
    __shared__ __half2 best_val[32] ; 
    __shared__ int2 best_index[32] ; 

    if(threadIdx.y == 0 )
    {
       best_val[threadIdx.x].x = 0.0f ;
       best_val[threadIdx.x].y = 0.0f  ;
    }
   __syncthreads() ; 
   if(threadIdx.x == 0)
   {
       best_val[threadIdx.y] = min_2 ;
       best_index[threadIdx.y] = index ;  
   }
   __syncthreads() ; 
   if(threadIdx.y == 0 ){
       min_2 = best_val[threadIdx.x] ; 
       index = best_index[threadIdx.x] ; 
       best_in_warp(min_2, index) ;   
       if(threadIdx.x == 0)
       {    
           
           half2 val ; 
           val.x = 2 ; val.y = 2 ; 
           val = __hadd2 ( val, min_2 ) ;  
           val = h2sqrt(val) ; 
           half t = __hdiv(val.x,val.y) ; 
           if(__half2float(t) <  threshold)  
           {

               matches[blockIdx.x] = index.x ; 
           }
           else
           {
               
               matches[blockIdx.x] = UINT32_MAX  ; 
           }
           
       }
    }
}

__device__ inline void min_half(__half2  &min_2, __half2 temp, int2 &index, int2 temp_index)
{

    // sort temp 
    if(__hgt(temp.x, temp.y))
    {
            half val = temp.x ; 
            temp.x = temp.y ; 
            temp.y = val ;  
            temp_index.x ++ ; 
            temp_index.y -- ;
    }

    //half2 val = __hgt2(min_2, temp) ; 
    //if(val.x)
    if(__hgt(min_2.x, temp.x))
    {
        min_2.x = temp.x ; 
        index.x = temp_index.x ; 
    }
    else if(__hgt(min_2.y, temp.x))
    {
        min_2.y = temp.x ; 
        index.y = temp.y ; 
    }

    if(__hgt(min_2.y, temp.y))
    //if(val.y)
    {
        min_2.y = temp.y ; 
        index.y = temp_index.y ; 
    }

}


// makes sure that we have enough memory  
void cublas_2nn_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle)
{
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    int i = 1 ; 
    size_t need_byte = (size_t)q_n *  (size_t)r_n * 4 ;  

    int temp = q_n ; 
    size_t dont_use = 6000000000 ; 
    while ( need_byte > (free_byte - dont_use ))
    {
       // printf("%i, %zu, %zu is \n", i, need_byte, free_byte - dont_use) ; 
        i ++ ; 
        temp = q_n / i ; 
        need_byte = (size_t)temp* (size_t) r_n * 4 ;  
    }
   // printf("%i, %zu, %zu is \n", i, need_byte, free_byte - dont_use) ; 
    float * dist ; 
    cudaMalloc((void **)&dist, (size_t)temp*  (size_t)r_n * 4) ; 
    int ii; 
    for (ii = 0; ii < i; ii++)
    {
        cublas_2nn_brute_f(q_points + (ii * temp), r_points, temp, r_n, sorted + (ii * temp), dist, handle); 
        printf("%i \n", ii) ; 
    }
    if((q_n % temp ) > 0 )
    {
        int left =  q_n % temp ; 
        printf("left %i \n", left) ; 
        cublas_2nn_brute_f(q_points + (ii * temp), r_points, left, r_n, sorted + (ii * temp), dist, handle); 
    }
   cudaFree(dist); 
}
// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie or just manged 
void cublas_2nn_brute_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, float * dist,cublasHandle_t handle)
{
   size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);


// steps d(x,y)^2 = ||x||^2 + ||y||^2 - 2x*y^T
// for sift we only need to solve d(x,y)^2 = 2 + -2x*y^t
// or -x*y^t
// notes cublas works in colum major,, c++ is in row major :(  
// meaning our input q_points and r_poins are already transpoed 
    // add minus 
    float a = -1.f;
    float b = 0.f;
 
   // we are in row major so we want our output from cublas to be in row major as well 
    // d^t = r * q^t is what cublas dose if see from cloum major
    // which is d = r^t * q   
   cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (float *)r_points, CUDA_R_32F, 128, (float *)q_points, CUDA_R_32F, 128, &b, dist,CUDA_R_32F, r_n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    //  cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, 
    // (signed char *)r_points, CUDA_R_8I, 128, (signed char *)q_points, CUDA_R_8I, 128, &b, dist,CUDA_R_32F, r_n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    
    //cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (float *)r_points, 128, (float *)q_points, 128, &b, dist, r_n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("dot failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    } 
    // want to find min value for each dist array 
    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,4,1) ; 
    min_dist_f<<<gridSize,blockSize>>>(dist, r_n, sorted) ; 
    cudaError_t cudaStat = cudaDeviceSynchronize();

    if (cudaStat != cudaSuccess) {
        printf ("min dist failed, cublas 2nn \n");
        cudaFree (dist);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }
 
}

void device_brute(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted)
{
    // array of the distances between all q and r points 
    float * dev_dist ; 
    //array of dist from each q point to every r point 
    cudaMalloc((void **)&dev_dist, q_n* r_n * sizeof(float)) ; 

    dim3 grid_size(q_n, r_n, 1) ;
    dim3 block_size(32, 1, 1) ;   

    //fill in the dist array
    sqrEuclidianDist<<<grid_size, block_size>>>(q_points,r_points, dev_dist);
    cudaDeviceSynchronize();

    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,1,1) ; 

    min_dist_f<<<gridSize,blockSize>>>(dev_dist, r_n , sorted) ; 
    cudaDeviceSynchronize();

    cudaFree(dev_dist) ; 
}

//kernels
//finds the sqr euclidan distance between two 128 vector arrays
__global__ void sqrEuclidianDist(des_t_f * q_points, des_t_f * r_points, float * dist_array)   
{   
    float dist = 0.0f ;
    // find dist 
    for (size_t i = 0; i < 4; i++)
    {
        float a = ((float *)q_points[blockIdx.x])[threadIdx.x+(i*32)]; 
        float b = ((float *)r_points[blockIdx.y])[threadIdx.x+(i*32)]; 
        float c = a - b ; 
        dist += c * c ; 
    }

    dist += __shfl_down_sync( 0xffffffff, dist, 16 );
    dist += __shfl_down_sync( 0xffffffff, dist, 8 ); 
    dist += __shfl_down_sync( 0xffffffff, dist, 4 ); 
    dist += __shfl_down_sync( 0xffffffff, dist, 2 );
    dist += __shfl_down_sync( 0xffffffff, dist, 1 );   
    if(threadIdx.x == 0)
    {
        dist_array[blockIdx.x * gridDim.y + blockIdx.y] = dist; 
    }
}
//find smallest vlaue in the warp and index  
__device__ inline void best_in_warp_float(float4  &min_2)
{    
    for (int i = 16; i > 0; i/= 2)
    {          
        float x_dist = __shfl_down_sync( 0xffffffff, min_2.x, i );
        float y_dist = __shfl_down_sync( 0xffffffff, min_2.y, i );
        float w_value = __shfl_down_sync( 0xffffffff, min_2.w, i );
        float z_value = __shfl_down_sync( 0xffffffff, min_2.z, i );
        if(x_dist < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = x_dist ;  
                
            min_2.w = min_2.z ; 
            min_2.z = z_value;  
        }
        else{
            if(x_dist < min_2.y)
            {
                min_2.y = x_dist ; 
                min_2.w = z_value;  
                continue ; 
            }
        } 
        if(y_dist < min_2.y)
        {
            min_2.y = y_dist ; 
            min_2.w = w_value;  
        }
    }
}

__device__ inline void best_in_warp(__half2  &min_2, int2 &index)
{
    for (int i = 16; i > 0; i/= 2)
    {          
        half2 temp = __shfl_down_sync( 0xffffffff, min_2, i );
        int index_x  = __shfl_down_sync( 0xffffffff, index.x, i );
        int index_y  = __shfl_down_sync( 0xffffffff, index.y, i );

        //half2 val = __hgt2(min_2, temp) ; 
        //if(val.x)
        if(__hgt(min_2.x, temp.x))
        {
            min_2.x = temp.x ; 
            index.x = index_x ; 
        }
        else if(__hgt(min_2.y, temp.x))
        {
            min_2.y = temp.x ; 
            index.y = index_x ; 
        }

        if(__hgt(min_2.y, temp.y))
        //if(val.y)
        {
            min_2.y = temp.y ; 
            index.y = index_y ; 
        }
    }
}
// x warps per dist 
__global__ void min_dist_f(float *  dist, int size ,float4 * sorted)
{
    //            finds the dist array         x dim       y dim pos                      
    int offset = (blockIdx.x * size)+ threadIdx.y * blockDim.x ;
   
    float4 min_2 ;  
    min_2.x = MAXFLOAT; 
    min_2.y = MAXFLOAT; 
    //maybe better to read values first, however compiler may just be doing it for me
    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+=(blockDim.x * blockDim.y) )
    {

        float temp = dist[i + offset + threadIdx.x]; 
        if(temp < min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = temp ;  
            min_2.w = min_2.z ;  
            min_2.z = i + threadIdx.x + threadIdx.y * blockDim.x  ; 
        }
        else{
            if(temp< min_2.y)
            {
                min_2.y = temp;  
                min_2.w = i + threadIdx.x + threadIdx.y * blockDim.x   ;  
            }
        }
    }
    best_in_warp_float(min_2) ;   

    __shared__ float4 best[32] ; 
    
    if(threadIdx.y == 0)
    {
        best[threadIdx.x].x = MAXFLOAT ;
        best[threadIdx.x].y = MAXFLOAT ; 
    }
    __syncthreads() ; 
    if(threadIdx.x == 0)
    {
        best[threadIdx.y] = min_2 ;
    }
    
    __syncthreads() ; 
    if(threadIdx.y == 0){
        min_2 = best[threadIdx.x] ; 
        best_in_warp_float(min_2) ; 
        if(threadIdx.x == 0)
        { 
            sorted[blockIdx.x ] = min_2 ; 
        }
    }
}

//host brute

void host_brute(des_t_f * q_points, des_t_f * r_points, int q_points_size, int r_points_size, float4  * sorted)
{
    
    float * lenght;
    cudaMallocHost((void **)&lenght, r_points_size * q_points_size * sizeof(float)) ; 
    for (size_t i = 0; i < q_points_size; i++)
    {
        for (size_t ii = 0; ii < r_points_size; ii++)
        {
            lenght[(i * r_points_size) + ii ] = host_lenght(q_points[i], r_points[ii]) ;  
        }
    }
    
    host_sort(lenght,r_points_size, q_points_size, sorted) ; 
    cudaFree(lenght) ; 
}

//given an array of arrays of lenghts it sorts the array and returns a sroted array 
//containing the 2 shortests lenghts from each array 
void host_sort(float * dist, int size, int array_size, float4 * sorted)
{   
    for (int i = 0; i < array_size ; i++)
    {
        float4 min_2 ;  
        min_2.x = MAXFLOAT ; 
        min_2.y = MAXFLOAT ;
        min_2.z = MAXFLOAT ;  
        min_2.w = MAXFLOAT ;  
        int offset = i * size ; 
        for (int ii = 0; ii < size; ii++)
        {
            if(dist[ii + offset] < min_2.x)
            {
                min_2.y = min_2.x ; 
                min_2.x = dist[ii + offset] ;  
                
                min_2.w = min_2.z ; 
                min_2.z = ii ;  
                
            }
            else{
                if(dist[ii + offset] < min_2.y)
                {
                    min_2.y = dist[ii + offset] ; 
                    min_2.w = ii ;  
                }
            }
        }
        sorted[i] = min_2 ;  
    }
}

// given 2 points find the euclidina lenght before taking the root
float host_lenght(des_t_f x, des_t_f y){
    float * vec1 = (float * )x ; 
    float * vec2 = (float * )y ;
    float dist =  0.0f  ;  
    for (size_t i = 0; i < 128; i++)
    {
        float a = vec1[i] ; 
        float b = vec2[i] ;  
        float c = a - b ; 
        dist += c * c ; 
    }
    return dist ; 
}



