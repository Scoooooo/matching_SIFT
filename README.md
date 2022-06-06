# matching SIFT vectors on GPU 

knn_brute contains a 2nn matching algorithm for SIFT vectors which uses cuBLAS called cublas_2nn_sift, also it contaisn a navie cuda brute force. 

lsh_h.cu contatins a GPU implementation of basic LSH for SIFT vectors   

make only works for compute capability of 86 or up

