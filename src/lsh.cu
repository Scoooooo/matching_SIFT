#include <iostream>
#include <string>
#include "lsh.h"
#include "knn_brute.h"
#include <memory>
#include <vector>
#include<bits/stdc++.h>
#include <map>
using namespace std;

void host_lsh(des_t * q_points, des_t * r_points, int n_q, int n_r, float4  * sorted, int nbits)
{
    // number of codes for each point  
    int l = 1 ;  

    des_t * rand_array ; 

    cudaMallocManaged((void **) &rand_array, sizeof(des_t) * nbits) ; 

    // make random vectors 
    for (size_t i = 0; i < nbits; i++)
    {
        make_vec(128, rand_array[i]) ;
    }   

    // make an array of ints with one int for each r_point
    unsigned int * code ; 

    cudaMallocManaged((void **) &code, sizeof(int ) * n_r * l) ; 

    cudaMemset(code, 0, sizeof(int) * n_r);

    // dot all vectors and add the bit to the coresponding int bit for the r points  
    for (size_t i = 0; i < n_r; i++)
    {
        printf(" %i bucket = ", i) ; 
        for (size_t ii = 0; ii < nbits ; ii++)
        {
            float sum = dot(r_points[i],rand_array[ii]) ; 
            if(sum <= 0)
            {
                code[i] |= 1UL << ii;
            }
            if(sum >= 0)
            {
                printf("0") ; 
            }
            else
            {
                printf("1") ; 
            }
        }
        printf(" %u " , code[i]) ; 
        printf("\n \n") ; 
    }
    
    // make buckets     
    
    map<int, set<int>> buckests ;

    for (size_t i = 0; i < n_r; i++)
    {
        // could also add to neighbouring buckets 
        map<int, set<int>>::iterator it = buckests.find(code[i]) ;
        if(it == buckests.end())
        {
            set<int> s = {1};  
            buckests.insert(make_pair(code[i], s ));           
        }
        else
        {
            it->second.insert(i) ; 
        } 
    }
    
    for(const auto& elem : buckests)
    {
        std::cout << elem.first << " " <<  "\n";
        for (auto it = elem.second.begin(); it !=
                            elem.second.end(); ++it)
        cout << ' ' << *it;    
        cout << "\n" ; 
    }

    // for each q point dot with random vectors and find the correct bucket 
    // preform a brute force search on values in the buckets 

}

void make_vec(int dim, des_t  &vec)
{
    for (size_t i = 0; i < dim; i++)
    {
        vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }  
}

float dot(des_t v1, des_t v2)
{   

    float sum  = 0.f; 
    for (size_t i = 0; i < 128 ; i++)
    {
        float a = ((float *)v1)[i]; 
        float b = ((float *)v2)[i]; 
        float c = a - b ; 
        sum += c ; 
    }
    return sum ; 
}