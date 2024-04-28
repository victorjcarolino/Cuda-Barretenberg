#include "field.cu"
#include "element.cu"
#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

using namespace std;

namespace gpu_barretenberg { // does this namespace need to be chnaged

/* -------------------------- G1 Elliptic Curve Operations ---------------------------------------------- */

/**
 * Group class that represents an elliptic curve group element
 */
template < typename fq_single, typename fr_single > 
class group_gpu_single {
    public:    
        typedef gpu_group_elements_single::element_single<fq_single, fr_single> element_single;
        typedef gpu_group_elements_single::affine_element_single<fq_single, fr_single> affine_element_single;
        typedef gpu_group_elements_single::projective_element_single<fq_single, fr_single> projective_element_single;

        /* -------------------------- Affine and Jacobian Coordinate Operations ---------------------------------------------- */
        
        __device__ __forceinline__ void load_affine(affine_element_single &X, affine_element_single &result);

        __device__ __forceinline__ static void load_jacobian(element_single &X, element_single &result);

        __device__ __forceinline__ static void add(fq_single X1, fq_single Y1, fq_single Z1, fq_single X2, fq_single Y2, fq_single Z2, fq_single &res_x, fq_single &res_y, fq_single &res_z);

        __device__ __forceinline__ static void mixed_add(fq_single X1, fq_single Y1, fq_single Z1, fq_single X2, fq_single Y2, fq_single &res_x, fq_single &res_y, fq_single &res_z);

        __device__ __forceinline__ static void doubling(fq_single X, fq_single Y, fq_single Z, fq_single &res_x, fq_single &res_y, fq_single &res_z);

        /* -------------------------- Projective Coordinate Operations ---------------------------------------------- */

        __device__ __forceinline__ static void load_projective(projective_element_single &X, projective_element_single &result);
        
        projective_element_single from_affine(const affine_element_single &other);
        
        __device__ __forceinline__ static void add_projective(fq_single X1, fq_single Y1, fq_single Z1, fq_single X2, fq_single Y2, fq_single Z2, fq_single &res_x, fq_single &res_y, fq_single &res_z);
};
typedef group_gpu_single<fq_single, fr_single> g1_single;

}