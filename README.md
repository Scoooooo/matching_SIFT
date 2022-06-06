# matching SIFT vectors on GPU 

knn_brute contains a 2nn matching algorithm for SIFT vectors which uses cuBLAS called cublas_2nn_sift, also it contaisn a navie cuda brute force. 

lsh_h.cu contatins a GPU implementation of basic LSH for SIFT vectors   

make only works for compute capability of 86 or up

![comparions](https://user-images.githubusercontent.com/42589827/172199644-5e313723-7b61-4875-8799-547bcc0b55b4.png)

Shows time used for LSH on GPU, brute-force using cuBLAS and Naive CUDA brute force on the ANN_SIFT1M data set. Time is in
seconds and is represented by the y value, the lower the better. The brute-force approaches are represented as straight lines, while LSH on GPU
shows the best results at different number of bits used for hash value, at different levels of precision (recall is around 0.8). The brute-force with cuBLAS bit stands for the level of precision used for the cuBLAS GEMM call, 16 or 32 bit
