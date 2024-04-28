#include "element_single.cuh"

using namespace std;
// using namespace gpu_barretenberg;
// using namespace gpu_group_elements;
using namespace gpu_barretenberg_single;
using namespace gpu_group_elements_single;

/* -------------------------- Jacobian Elements ---------------------------------------------- */

template <class fq_single, class fr_single> 
__device__ __forceinline__ element_single<fq_single, fr_single>::element_single(const fq_single &a, const fq_single &b, const fq_single &c) noexcept
    : x{a}, y{b}, z{c} {};

template <class fq_single, class fr_single> 
__device__ __forceinline__ element_single<fq_single, fr_single>::element_single(const element_single &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};

/* -------------------------- Affine Elements ---------------------------------------------- */

template <class fq_single, class fr_single> 
__device__ __forceinline__ affine_element_single<fq_single, fr_single>::affine_element_single(const fq_single &a, const fq_single &b) noexcept 
    : x(a), y(b) {};

template <class fq_single, class fr_single> 
__device__ __forceinline__ affine_element_single<fq_single, fr_single>::affine_element_single(const affine_element_single &other) noexcept 
    : x(other.x), y(other.y) {};

/* -------------------------- Projective Elements ---------------------------------------------- */

template <class fq_single, class fr_single> 
__device__ __forceinline__ projective_element_single<fq_single, fr_single>::projective_element_single(const fq_single &a, const fq_single &b, const fq_single &c) noexcept
    : x{a}, y{b}, z{c} {};

template <class fq_single, class fr_single> 
__device__ __forceinline__ projective_element_single<fq_single, fr_single>::projective_element_single(const projective_element_single &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};