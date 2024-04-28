#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>
#include <iostream>
#include <type_traits>
#include <fixnum.cu>
#include "field_common.cuh"

// #include "field.cuh"

// using namespace gpu_barretenberg;
using namespace std;

namespace gpu_barretenberg_single {

using namespace gpu_barretenberg_common;

typedef std::uint64_t var;
// static constexpr size_t BYTES_PER_ELEM = LIMBS * sizeof(var);

struct uint254 {
    std::uint64_t limbs[4]; 

    __host__ __device__ __forceinline__
    uint254() {
        limbs[0] = 0;
        limbs[1] = 0;
        limbs[2] = 0;
        limbs[3] = 0;
    }

    __host__ __device__ __forceinline__
    uint254(uint64_t a, uint64_t b, uint64_t c, uint64_t d) : limbs{a, b, c, d} {}

    template<
        typename T,
        typename = std::enable_if_t<std::is_array_v<T>>,
        typename = std::enable_if_t<std::is_same_v<std::remove_all_extents<T>, uint64_t>>
    >
    __host__ __device__ __forceinline__
    uint254(const T data) : limbs{data[0], data[1], data[2], data[3]} {}

    __host__ __device__ __forceinline__
    uint254(const uint64_t x) : limbs{x, x, x, x} {}
};


// /* -------------------------- Finite Field Arithmetic for G1 ---------------------------------------------- */

template < typename params > 
class field_single {
    public:  
        uint254 data;

        __device__ __host__ field_single(uint254 val) : data{val} {}
        __device__ __host__ field_single() : data{} {}
        
        __device__ __forceinline__ field_single(const var a, const var b, const var c, const var d) noexcept;

        __device__ __forceinline__ static void add(const uint254 a, const uint254 b, uint254 &res);

        __device__ __forceinline__ static void sub(const uint254 a, const uint254 b, uint254 &res);

        __device__ __forceinline__ static void sub_inplace(uint254 &af, const var *b);

        __device__ __forceinline__ static void square_inplace(uint254 &x);
        
        __device__ __forceinline__ static void square(const uint254 x, uint254 &res);

        __device__ __forceinline__ static void neg(uint254 x, uint254 &res);

        // TODO
        // __device__ __forceinline__ field_single() noexcept {}

        // TODO
        __device__ __forceinline__ static field_single zero();
        
        // TODO
        __device__ __forceinline__ static field_single one();
        
        // TODO
        __device__ __forceinline__ static bool is_zero(const uint254 &x);

        __device__ __forceinline__ static bool equal(const uint254 x, const uint254 y);

        __device__ __forceinline__ static uint254 load(uint254 x, uint254 &res);

        __device__ __forceinline__ static void store(uint254 *mem, const uint254 &x);  

        __device__ __forceinline__ static void mul(const uint254 a, const uint254 b, uint254 &res);

        __device__ __forceinline__ static void to_monty(uint254 x, uint254 &resf);
        
        __device__ __forceinline__ static void from_monty(uint254 x, uint254 &resf);
};
typedef field_single<gpu_barretenberg_single::BN254_MOD_BASE> fq_single;
typedef field_single<gpu_barretenberg_single::BN254_MOD_SCALAR> fr_single;

}
