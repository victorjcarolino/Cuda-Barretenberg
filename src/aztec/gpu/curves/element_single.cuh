#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

using namespace std;

namespace gpu_barretenberg_single {
namespace gpu_group_elements_single { 
    
/* -------------------------- Jacobian Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Jacobian coordinates
 */
template < typename fq_single, typename fr_single> 
class element_single {
    public:    
        fq_single x, y, z;

        __device__ __forceinline__ element_single() noexcept {}
        
        __device__ __forceinline__ element_single(const fq_single &a, const fq_single &b, const fq_single &c) noexcept;
        
        __device__ __forceinline__ element_single(const element_single& other) noexcept;
};

/* -------------------------- Affine Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Affine coordinates
 */
template < typename fq_single, typename fr_single> 
class affine_element_single {
    public:    
        fq_single x, y;

        __device__ __forceinline__ affine_element_single() noexcept {}

        __device__ __forceinline__ affine_element_single(const fq_single &a, const fq_single &b) noexcept;

        __device__ __forceinline__ affine_element_single(const affine_element_single &other) noexcept;        
};

/* -------------------------- Projective Coordinate Representation ---------------------------------------------- */

/**
 * Implements elliptic curve group arithmetic using Projective coordinates
 */
template < typename fq_single, typename fr_single> 
class projective_element_single {
    public:    
        fq_single x, y, z;

        __device__ __forceinline__ projective_element_single() noexcept {}

        __device__ __forceinline__ projective_element_single(const fq_single &a, const fq_single &b, const fq_single &c) noexcept;

        __device__ __forceinline__ projective_element_single(const projective_element_single &other) noexcept;  
};

}
}