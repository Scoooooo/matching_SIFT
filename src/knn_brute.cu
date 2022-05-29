#include "knn_brute.h"
#include <string>
#include "cublas_v2.h"
#include "knn_brute.h"
#include "helper.h"
#include "cuda_fp16.h"
 
typedef struct{
   half2 x;
   half2 y;
   half2 z;
   half2 w;
} half8;

typedef enum {
    FLOAT_HOST=0, 
    FLOAT_DEVICE=1,
    HALF_DEVICE=2,
} type_mem  ;  
 
// full half 2nn for sift 
// make sure that gpu has memeory use * number of streams 
// warps range 1 - 32, stream_n 1 - 16 
int cublas_2nn_sift(void * q_points, void * r_points, int type, uint32_t q_n, uint32_t r_n, uint32_t * matches, float threshold, cublasHandle_t handle, int stream_n, size_t use, int warps)
{
   // make streams 
    cudaStream_t stream[stream_n] ; 
    
    for (int i = 0; i < stream_n; i++)
    {
      cudaStreamCreate(&stream[i]); 
    }
  
    des_t_h2 * R;  
    // flag to see if we need to clear R at the end 
    int flag = 0 ; 
    // we have 2 possible inputs floats/half_floats if half it has to be in device memory if floats it can be device / host 
    // half will be fastest as we do not need to convert 
    // R needs to be in device memory and of type half2 since we use the whole array for every cublas call
    // for R size cublas wants r_n % 16 == 0 

    // floats in host memory  
    // will just use zero copy for now assumese that memory is mapped and pinned 
    // note for some reason this is faster than float_device sometimes ?    
    if (type == FLOAT_HOST)
    {
        // float * q_copy ; 
        // r_copy = (float * )r_points ;
        // cudaHostGetDevicePointer(&q_copy, q_points, 0);
 
        // des_t_h2 * q ;
        // cudaMalloc((void **)&q, q_n * sizeof(des_t_h2));
        // float2half<<<q_n, 64>>>((float * )q_copy, (half2 * )q) ;
        // q_points = q ; 
       
        float * r_copy ; 
        // r_copy = (float * )r_points ;
        cudaHostGetDevicePointer(&r_copy, r_points, 0);
        // check if we need to pad if we do we need to remake 
        // cudaMalloc((void **)&r_copy, r_n * sizeof(des_t_f));
        // cudaMemcpy(r_copy, r_points, r_n* sizeof(des_t_f), cudaMemcpyHostToDevice) ; 

        int pad_r_n = (r_n % 16) ;   
        if( pad_r_n !=  0)
        {
            cudaMalloc((void **)&R,((16 - pad_r_n) + r_n) * sizeof(des_t_h2));
            // cudaMemcpy(test, r_points, r_n* sizeof(des_t_f), cudaMemcpyHostToDevice) ; 
            cudaMemset(R, 0, ((16 - pad_r_n) + r_n) * sizeof(des_t_h2)); 
            // covert to halfs
            float2half<<<r_n, 64>>>((float * )r_copy, (half2* )R) ; 
            r_n += (16 - pad_r_n); 
            
        }
        else
        {
            cudaMalloc((void **)&R, r_n * sizeof(des_t_h2));
            // cudaMemset(R, 0, r_n * sizeof(des_t_h2)); 
            // convert to halfs
            float2half<<<r_n, 64>>>((float * )r_copy, (half2 * )R) ; 
        }
    }
    // floats in device memory 
    else if (type == FLOAT_DEVICE )
    {
        // des_t_h2 * q ;
        // cudaMalloc((void **)&q, q_n * sizeof(des_t_h2));
        // float2half<<<q_n, 64>>>((float * )q_points, (half2 * )q) ;
        // q_points = q ; 
        // check if we need to pad if we do we need to remake 
        int pad_r_n = (r_n % 16) ;   
        if( pad_r_n != 0)
        {
            cudaMalloc((void **)&R,((16 - pad_r_n) + r_n) * sizeof(des_t_h2));
            cudaMemset(R, 0, ((16 - pad_r_n) + r_n) * sizeof(des_t_h2)); 
            // covert to halfs
            float2half<<<r_n, 64>>>((float * )r_points, (half2* )R) ; 
            r_n += (16 - pad_r_n); 
        }
        else
        {
            cudaMalloc((void **)&R, r_n * sizeof(des_t_h2));
            // cudaMemset(R, 0, r_n * sizeof(des_t_h2)); 
            // convert to halfs
            float2half<<<r_n, 64>>>((float * )r_points, (half2 * )R) ; 
        }
         
    }
    // halfs in device memory  
    else if(type == HALF_DEVICE)
    {
        // check if we need to pad if we do we need to remake 
        int pad_r_n = (r_n % 16) ;   
        if( pad_r_n != 0)
        {
            cudaMalloc((void **)&R,((16 - pad_r_n) + r_n) * sizeof(des_t_h2));
            cudaMemset(R, 0, ((16 - pad_r_n) + r_n) * sizeof(des_t_h2)); 
            cudaMemcpy(R,r_points, r_n * sizeof(des_t_h2), cudaMemcpyDeviceToDevice ); 
            flag = 1 ; 
            r_n += (16 - pad_r_n); 
        }
        else{
            R = (des_t_h2 * )r_points ; 
        } 
    }

    // number of bytes to we want for output array  
    // probly difrent for difrent gpus 

    // give us how many iterations we need to run with the number of bytes we want
    uint32_t new_q_n = use /((size_t) r_n * sizeof(half)); 
    
    // if the whole dist array fits in memeory we just do it in one batch
    if(new_q_n >= q_n)
    {
        new_q_n = q_n ; 
        stream_n = 1 ; 
    }
    
    //get number of iterartions 
    int it = q_n / new_q_n;  
    // malloc for each stream 
    half2 * dist[stream_n] ;
    des_t_h2 * Q[stream_n] ; 
    for (int i = 0; i < stream_n; i++)
    {
        cudaMallocAsync((void **)&dist[i], new_q_n * r_n * sizeof(half), stream[i]) ; 
        // do not need to malloc if half2 is in memory 
        if(type != HALF_DEVICE)
        {
            cudaMallocAsync((void **)&Q[i], new_q_n * sizeof(des_t_h2), stream[i]); 
        }
    }


    int i = 0 ;  
    int stream_id = 0 ; 
    // run batches 
    for (i = 0; i < it; i++)
    {
       // printf("int %i \n", i); 
        if(type != HALF_DEVICE)
        {
            cublas_2nn_sift_batch((des_t_f * )q_points + (i * new_q_n), Q[stream_id],  type , R, new_q_n, r_n, dist[stream_id], matches + (i * new_q_n), threshold, handle, stream[stream_id], warps); 
        } 
        else
        {
            cublas_2nn_sift_batch((des_t_h2 * )q_points + (i * new_q_n), Q[stream_id],  type , R, new_q_n, r_n, dist[stream_id], matches + (i * new_q_n), threshold, handle, stream[stream_id], warps); 
        }
        stream_id ++ ; 
        if(stream_id == stream_n)
        {
            stream_id = 0 ; 
        }
    }

    // check if we got all or if there are any q points left 
    if((q_n % new_q_n ) > 0 )
    {
        int left =  q_n % new_q_n; 
        //printf("left %i \n", left) ; 
        if(type != HALF_DEVICE)
        {
            cublas_2nn_sift_batch((des_t_f * )q_points + (i * new_q_n), Q[stream_id],  type , R, left, r_n, dist[stream_id], matches + (i * new_q_n), threshold, handle, stream[stream_id], warps); 
        }
        else
        {
            cublas_2nn_sift_batch((des_t_h2 * )q_points + (i * new_q_n), Q[stream_id],  type , R, left, r_n, dist[stream_id], matches + (i * new_q_n), threshold, handle, stream[stream_id], warps); 
        }
    }

    // free
    for (int i = 0; i < stream_n; i++)
    {
        cudaFreeAsync(dist[i], stream[i]);
        cudaFreeAsync(Q[i], stream[i]);
    }
    for (int i = 0; i < stream_n; i++)
    {
        cudaStreamDestroy(stream[i]) ; 
    }

    if(type != HALF_DEVICE || flag == 1)
    {
        cudaFree(R);
    }
    
    // cudaFree(q_points) ;
    return 0  ; 
}

// gpu brute force 2nn 
// takes pointer with data on device as input, sorted output should also be on devcie or just manged 
int cublas_2nn_sift_batch(void * q_points, des_t_h2 * Q, int type, des_t_h2 * R, uint32_t q_n, uint32_t r_n, half2 * dist, uint32_t * matches, float threshold, cublasHandle_t handle, cudaStream_t stream, int warps)
{
    //Q can be either in device or host memory and of half / float type  

    // in Host memory type float 
    if(type == FLOAT_HOST)
    {
        float * q_copy ; 
        // cudaMallocAsync((void **)&q_copy, r_n * sizeof(des_t_f), stream);
        // cudaMemcpyAsync(q_copy, q_points, r_n* sizeof(des_t_f), cudaMemcpyHostToDevice, stream) ; 
        cudaHostGetDevicePointer(&q_copy, q_points, 0);
        
        float2half<<<q_n, 64,0, stream>>>((float * )q_copy, (half2 * )Q) ;
    }
    // // in device memory but of type float
    else if(type == FLOAT_DEVICE)
    {
        // fill
        float2half<<<q_n, 64,0, stream>>>((float * )q_points, (half2 * )Q) ;
    }
    // in device memory and half
    else if(type == HALF_DEVICE)
    {
        Q = (des_t_h2 *)q_points ; 
    }

// 
   half a = -2.f;
   half b = 0.f; 
   cublasSetStream(handle, stream) ; 
//    cublasStatus_t stat = 
   cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (half *)R, 128, (half *)Q, 128, &b, (half *)dist, r_n);

    //singel, bit slower but more accuracy 
//    float a = -2.f;
//    float b = 0.f;
   // cublasSetStream(handle, stream) ; 
//    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (half *)R, CUDA_R_16F, 128, 
                        //    (half *)Q, CUDA_R_16F, 128, &b, (half * )dist, CUDA_R_16F, r_n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// 
   
    // error 
    //if (stat != CUBLAS_STATUS_SUCCESS) {
    //    printf ("dot failed, cublas 2nn \n");
    //    cudaFree (dist);
    //    cublasDestroy(handle);
    //    exit(EXIT_FAILURE);
    //} 
    // want to find min value for each dist array 
    dim3 gridSize(q_n,1,1) ;
    // // y dim can be changed 
    dim3 blockSize(32,warps,1) ; 
    find_matches<<<gridSize,blockSize, 0, stream>>>(dist, r_n / 8, matches, threshold) ; 
    // //error 
   // cudaError_t cudaStat = cudaDeviceSynchronize();

   // if (cudaStat != cudaSuccess)
   // {
   //     printf ("min dist failed, cublas 2nn \n");
   //     cudaFree (dist);
   //     cublasDestroy(handle);
   //     exit(EXIT_FAILURE);
   // }

    return 0 ;  
}

// the reduction of the half8 can probly be done better todo if time  
//__device__ inline void min_half_helper(__half2  &min_2, half8 temp, int2 &index, int2 temp_index)
//{
//
//}
////// todo if i have time 
//__device__ inline void min_half8(__half2  &min_2, half8 temp, int2 &index, int2 temp_index)
//{
//    // given 10 halfs find the 2min and their index
//    // reduce 2 finding the 
//    if(__hgt(temp.x.x, temp.x.y))
//    {
//        half val = temp.x.x ; 
//        temp.x.x = temp.x.y ; 
//        temp.x.y = val ;  
//        temp_index.x ++ ; 
//        temp_index.y -- ;
//    }
//
//    if(__hgt(temp.x.x, temp.y.x))
//    {
//        half val = temp.x.x ; 
//        temp.x.x = temp.x.y ; 
//        temp.x.y = val ;  
//        temp_index.x ++ ; 
//        temp_index.y -- ;
//    }
//
//
//
//    int a, b ; 
//    if(val1.x && val2.x && val3.x) ; 
//}

// converts from floats to halfs  
__global__ void float2half(float * points, half2 * output)
{
    float2 f2 =  ((float2 * )points)[threadIdx.x + blockDim.x * blockIdx.x] ; 
    half2 temp ;
    // need to divide if values are in unsiged char form
    // f2.x = f2.x/512; 
    // f2.y = f2.y/512; 
    // rn
    temp  = __float22half2_rn(f2) ;

    // rd 
    // temp.x = __float2half_rd(f2.x) ; 
    // temp.y = __float2half_rd(f2.y) ; 

    //ru
    // temp.x = __float2half_ru(f2.x) ; 
    // temp.y = __float2half_ru(f2.y) ; 

    //rz 
    // temp.x = __float2half_rz(f2.x) ; 
    // temp.y = __float2half_rz(f2.y) ; 

    output[threadIdx.x + blockDim.x * blockIdx.x] = temp ; 
}

__global__ void find_matches(half2 *  dist, int size, uint32_t * matches, float threshold)
{

    //  finds the dist array         x dim       y dim pos                      
    int  offset = (blockIdx.x * size) + threadIdx.y * blockDim.x ;

    half2 min_2 ;  
    int2 index ; 

    // half2 temp ; 
    //  our values will be negative so setting to 0 is fine  
    min_2.x = 0.0f; 
    min_2.y = 0.0f; 
   
    for (int i = 0; (i + threadIdx.x +  threadIdx.y * blockDim.x  ) < size ; i+= (blockDim.x * blockDim.y) )
    //for (int i = 0; (i + threadIdx.x) < size ; i+= (blockDim.x * blockDim.y) )
    {   

        // performance is better when we read half8 
        half8 temp = ((half8 * )dist)[i + offset + threadIdx.x]  ;     
        // temp = dist[(i + offset + threadIdx.x)]  ;     

        int2 temp_index  ;  

        temp_index.x = (i + threadIdx.x + threadIdx.y * blockDim.x) * 8  ; 
        temp_index.y = temp_index.x + 1;  
        //min_half(min_2, temp, index, temp_index); 


        min_half(min_2, temp.x, index, temp_index); 
        temp_index.x = (i + threadIdx.x + threadIdx.y * blockDim.x) * 8 + 2  ; 
        temp_index.y = temp_index.x + 1 ;  
        
        min_half(min_2, temp.y, index, temp_index); 
        temp_index.x = (i + threadIdx.x + threadIdx.y * blockDim.x) * 8 + 4  ; 
        temp_index.y = temp_index.x + 1 ;  
        
        min_half(min_2, temp.z, index, temp_index); 
        
        temp_index.x = (i + threadIdx.x + threadIdx.y * blockDim.x) * 8 + 6  ; 
        temp_index.y = temp_index.x + 1 ;  
        
        min_half(min_2, temp.w, index, temp_index); 

    }

    best_in_warp(min_2, index) ;   
    // block wide reduction 
    __shared__ __half2 best_val[32] ; 
    __shared__ int2 best_index[32] ; 

    if(threadIdx.y == 0 )
    {
       best_val[threadIdx.x].x = 0.0f ;
       best_val[threadIdx.x].y = 0.0f  ;
    }
   __syncthreads() ; 
   if(threadIdx.x == 0 )
   {
       best_val[threadIdx.y] = min_2 ;
       best_index[threadIdx.y] = index ;  
   }
   __syncthreads() ; 
   if(threadIdx.y == 0){
       min_2 = best_val[threadIdx.x] ; 
       index = best_index[threadIdx.x] ; 
       best_in_warp(min_2, index) ;   
       if(threadIdx.x == 0)
       {    
        //    todo half threshold check         
        //    half2 val ; 
        //    val.x = 2 ; val.y = 2 ; 
        //    val = __hadd2( val, min_2 ) ;  
        //    val = h2sqrt(val) ; 
        //    values are amlost identical for the random values i use to test -> for big q and r no matches pass   
        //    printf("val x %f, val y %f \n", __half2float( val1.x),  __half2float( val1.y)) ; 
// 
        //    val.x = __hmul(val.x, __float2half_rn(threshold)) ; 
// 
        //    printf("val x %f, val y %f \n", __half2float( val.x),  __half2float( val.y)) ; 
        //    if(__hgt(val.y, val.x))
        //    {
            //    matches[blockIdx.x] = index.x ; 
        //    }
        //    else
        //    {
            //    matches[blockIdx.x] = UINT32_MAX  ; 
        //    }
          
           // float thershold test 
           float2 val = __half22float2(min_2) ; 
           val.x = val.x + 2 ; 
           val.y = val.y + 2 ; 
           val.x  = sqrt(val.x) ; 
           val.y  = sqrt(val.y) ; 
           if( (val.x / val.y) < threshold)
           {
               matches[blockIdx.x] = index.x ; 
           }
           else
           {
                matches[blockIdx.x] = UINT32_MAX  ; 
           }
            
            //remove this line for threhold to be of any use 
            //  matches[blockIdx.x] = index.x ; 
       }
    }
}

// reduction 2 half2s
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

   // half2 val = __hgt2(min_2, temp) ; 
    if(__hgt(min_2.x, temp.x))
    //if(val.x)
    {
        min_2.x = temp.x ; 
        index.x = temp_index.x ; 
        if(__hgt(min_2.y, temp.y))
        //if(val.y)
        {
            min_2.y = temp.y ; 
            index.y = temp_index.y ; 
        }
    }
    else if(__hgt(min_2.y, temp.x))
    {
        min_2.y = temp.x ; 
        index.y = temp_index.y ; 
    }
}

// reduction warp
__device__ inline void best_in_warp(__half2  &min_2, int2 &index)
{
    for (int i = 16; i > 0; i/= 2)
    {          
        half2 temp = __shfl_down_sync( 0xffffffff, min_2, i );
        int index_x  = __shfl_down_sync( 0xffffffff, index.x, i );
        // dont really need index_y hmm
        int index_y  = __shfl_down_sync( 0xffffffff, index.y, i );

        //half2 val = __hgt2(min_2, temp) ; 
        //if(val.x)
        if(__hgt(min_2.x, temp.x))
        {
            min_2.x = temp.x ; 
            index.x = index_x ; 
            if(__hgt(min_2.y, temp.y))
            //if(val.y)
            {
                min_2.y = temp.y ; 
                index.y = index_y ; 
            }
        }
        else if(__hgt(min_2.y, temp.x))
        {
            min_2.y = temp.x ; 
            index.y = index_x ; 
        }
    }
}






// this is for float and in general just worse than half but gives more accuracy i guess 
// cublas brute for floats with 16 bit dot 
// makes sure that we have enough memory  
void cublas_2nn_f(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted, cublasHandle_t handle)
{
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;


    int i = 1 ; 
    size_t need_byte = (size_t)q_n *  (size_t)r_n * 4 ;  

    int temp = q_n ; 
    size_t dont_use = 7000000000 ; 
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
        // printf("%i \n", ii) ; 
    }
    if((q_n % temp ) > 0 )
    {
        int left =  q_n % temp ; 
        // printf("left %i \n", left) ; 
        cublas_2nn_brute_f(q_points + (ii * temp), r_points, left, r_n, sorted + (ii * temp), dist, handle); 
    }
    cudaDeviceSynchronize() ; 
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



// steps d(x,y)^2 = ||x||^2 + ||y||^2 - 2x*y^T
// for sift we only need to solve d(x,y)^2 = 2 + -2x*y^t
// or -x*y^t
// notes cublas works in colum major,, c++ is in row major :(  
// meaning our input q_points and r_poins are already transpoed 
    // add minus 
    float a = -2.f;
    float b = 0.f;
 
   // we are in row major so we want our output from cublas to be in row major as well 
    // d^t = r * q^t is what cublas dose if see from cloum major
    // which is d = r^t * q   

   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (float *)r_points, CUDA_R_32F, 128, (float *)q_points, CUDA_R_32F, 128, &b, dist,CUDA_R_32F, r_n, CUBLAS_COMPUTE_32F , CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //  cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, 
    // (signed char *)r_points, CUDA_R_8I, 128, (signed char *)q_points, CUDA_R_8I, 128, &b, dist,CUDA_R_32F, r_n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, r_n, q_n, 128, &a, (float *)r_points, 128, (float *)q_points, 128, &b, dist, r_n);
    // if (stat != CUBLAS_STATUS_SUCCESS) {
    //     printf ("dot failed, cublas 2nn \n");
    //     cudaFree (dist);
    //     cublasDestroy(handle);
    //     exit(EXIT_FAILURE);
    // } 
    // // want to find min value for each dist array 
    dim3 gridSize(q_n,1,1) ;
    dim3 blockSize(32,4,1) ; 
    min_dist_f<<<gridSize,blockSize>>>(dist, r_n, sorted) ; 
    // cudaError_t cudaStat = cudaDeviceSynchronize();

    // if (cudaStat != cudaSuccess) {
    //     printf ("min dist failed, cublas 2nn \n");
    //     cudaFree (dist);
    //     cublasDestroy(handle);
    //     exit(EXIT_FAILURE);
    // }
 
}

// normal gpu brute, extremly slow 
void device_brute(des_t_f * q_points, des_t_f * r_points, int q_n, int r_n, float4  * sorted)
{
    // how much memroy to use
    size_t use = 800000000 ; 

    // give us how many iterations we need to run with the number of bytes we want
    uint32_t new_q_n = use /((size_t) r_n * sizeof(float)); 
    
    // if the whole dist array fits in memeory we just do it in one batch
   
    //get number of iterartions 
    int it = q_n / new_q_n;  
    // array of the distances between all q and r points 
    float * dev_dist ; 
    //array of dist from each q point to every r point 
    cudaMalloc((void **)&dev_dist, new_q_n * r_n * sizeof(float)) ; 

    int i = 0 ; 
    //batching 
   for (i = 0; i < it; i++)
    {

        dim3 grid_size(r_n, new_q_n, 1) ;
        dim3 block_size(32, 1, 1) ;   
        //fill in the dist array
        sqrEuclidianDist<<<grid_size, block_size>>>((des_t_f  * ) q_points + (i * new_q_n),r_points, dev_dist);
        cudaDeviceSynchronize();
        dim3 gridSize(new_q_n,1,1) ;
        dim3 blockSize(32,1,1) ; 
        min_dist_f<<<gridSize,blockSize>>>(dev_dist, r_n ,(float4 * )sorted + (i * new_q_n)) ; 
    }
    
    if((q_n % new_q_n ) > 0 )
    {
        int left =  q_n % new_q_n; 
        dim3 grid_size(r_n, left, 1) ;
        dim3 block_size(32, 1, 1) ;   
        //fill in the dist array
        sqrEuclidianDist<<<grid_size, block_size>>>((des_t_f  * ) q_points + (i * new_q_n),r_points, dev_dist);
        cudaDeviceSynchronize();
        dim3 gridSize(left,1,1) ;
        dim3 blockSize(32,1,1) ; 
        min_dist_f<<<gridSize,blockSize>>>(dev_dist, r_n ,(float4 * )sorted + (i * new_q_n)) ; 
 
    }
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
        float a = ((float *)q_points[blockIdx.y])[threadIdx.x+(i*32)]; 
        float b = ((float *)r_points[blockIdx.x])[threadIdx.x+(i*32)]; 
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
        dist_array[blockIdx.y * gridDim.x + blockIdx.x] = dist; 
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
        if(temp <= min_2.x)
        {
            min_2.y = min_2.x ; 
            min_2.x = temp ;  
            min_2.w = min_2.z ;  
            min_2.z = i + threadIdx.x + threadIdx.y * blockDim.x  ; 
        }
        else{
            if(temp < min_2.y)
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
            float2 val ; 
            val.x = min_2.x ; 
            val.y = min_2.y ; 
            val.x = val.x + 2 ; 
            val.y = val.y + 2 ; 
            val.x  = sqrt(val.x) ; 
            val.y  = sqrt(val.y) ; 
           if( (val.x / val.y) < 0.8)
           {
                sorted[blockIdx.x] = min_2 ; 
           }
           else
           {
                sorted[blockIdx.x].z =  - 1 ; 
           }

            sorted[blockIdx.x] = min_2 ; 
         
                
        }
    }
}


// normal cpu brute 2nn 
//host brute
void host_brute(des_t_f * q_points, des_t_f * r_points, int q_points_size, int r_points_size, float4  * sorted)
{
    for (int i = 0; i < q_points_size; i++)
    {

        float temp = 0 ; 
        float4 min_2 ;  
        min_2.x = MAXFLOAT ; 
        min_2.y = MAXFLOAT ;
        min_2.z = MAXFLOAT ;  
        min_2.w = MAXFLOAT ;  
        for (int ii = 0; ii < r_points_size; ii++)
        {

            temp = host_lenght(q_points[i], r_points[ii]) ; 
            if(temp < min_2.x)
            {
                min_2.y = min_2.x ; 
                min_2.x = temp ;  
                
                min_2.w = min_2.z ; 
                min_2.z = ii ;  
                
            }
            else{
                if(temp < min_2.y)
                {
                    min_2.y = temp ;  
                    min_2.w = ii ;  
                }
            }
        }
        sorted[i] = min_2 ;  
    }
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



