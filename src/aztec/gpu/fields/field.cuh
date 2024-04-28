#ifndef FIELD_CUH
#define FIELD_CUH

#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>
#include <iostream>
#include <fixnum.cu>
#include "field_common.cuh"

using namespace std;

namespace gpu_barretenberg
{

    using namespace gpu_barretenberg_common;

    /* -------------------------- Finite Field Arithmetic for G1 ---------------------------------------------- */

    template <typename params>
    class field_gpu
    {
    public:
        var data[4];

        __device__ __forceinline__ field_gpu() noexcept {}

        __device__ __forceinline__ field_gpu(const var a, const var b, const var c, const var d) noexcept;

        __device__ __forceinline__ static field_gpu zero();

        __device__ __forceinline__ static field_gpu one();

        __device__ __forceinline__ static bool is_zero(const var &x);

        __device__ __forceinline__ static var equal(const var x, const var y);

        __device__ __forceinline__ static var load(var x, var &res);

        __device__ __forceinline__ static void store(var *mem, const var &x);

        __device__ __forceinline__ static var add(const var a, const var b, var &res);

        __device__ __forceinline__ static var sub(const var x, const var y, var &z);

        __device__ __forceinline__ static var square(var x, var &y);

        __device__ __forceinline__ static var mul(const var a, const var b, var &res);

        __device__ __forceinline__ static var to_monty(var x, var &res);

        __device__ __forceinline__ static var from_monty(var x, var &res);

        __device__ __forceinline__ static var neg(var &x, var &res);
    };
    typedef field_gpu<BN254_MOD_BASE> fq_gpu;
    typedef field_gpu<BN254_MOD_SCALAR> fr_gpu;
}

#endif /* FIELD_CUH */