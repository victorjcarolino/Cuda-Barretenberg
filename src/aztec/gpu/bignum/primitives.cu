#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

/*
 * Perform operations on fixed-precision integers using
 * Nvidia's inline PTX Assembly Language
 */
namespace internal {
    typedef std::uint32_t u32;
    typedef std::uint64_t u64;
    typedef std::int32_t i32;
    typedef std::int64_t i64;

    /* struct uint254 {
        u64 limbs[4];

        __device__ __forceinline__
        uint254() {
            limbs[0] = 0;
            limbs[1] = 0;
            limbs[2] = 0;
            limbs[3] = 0;
        }
    }; */

    // Add two 32-bit signed integers and set carry flag on overflow ('s' > INT_MAX)
    void addc(i32 &s, i32 a, i32 b) {
        asm ("addc.s32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }
    // struct addc_functor {
    //     __device__ __forceinline__ 
    //     int operator() (const thrust::tuple<int, int>& input) {
    //         int s;
    //         asm ("addc.s32 %0, %1, %2;"
    //              : "=r"(s)
    //              : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)));
    //         return s;
    //     }
    // };
    // void addc_thrust(thrust::device_vector<int>& d_input1, thrust::device_vector<int>& d_input2, thrust::device_vector<int>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end())),
    //                       d_output.begin(),
    //                       addc_functor());
    // }

    // Add two 32-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void addc(u32 &s, u32 a, u32 b) {
        asm ("addc.u32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }
    // struct addc_functor {
    //     __device__ __forceinline__ 
    //     u32 operator() (const thrust::tuple<u32, u32>& input) {
    //         u32 s;
    //         asm ("addc.u32 %0, %1, %2;"
    //              : "=r"(s)
    //              : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)));
    //         return s;
    //     }
    // };
    // void addc_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end())),
    //                       d_output.begin(),
    //                       addc_u32_functor());
    // }

    // // add two 64-bit SIGNED integers and set carry flag on overflow ('s' > INT_MAX)
    // __device__ __forceinline__
    // void addc(i64)

    // Add two 64-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void addc(u64 &s, u64 a, u64 b) {
        asm ("addc.u64 %0, %1, %2;"
             : "=l"(s)
             : "l"(a), "l" (b));
    }

    // struct addc_functor {
    //     __device__ __forceinline__ 
    //     u64 operator() (const thrust::tuple<u64, u64>& input) {
    //         u64 s;
    //         asm ("addc.u64 %0, %1, %2;"
    //              : "=l"(s)
    //              : "l"(thrust::get<0>(input)), "l" (thrust::get<1>(input)));
    //         return s;
    //     }
    // };

    // void addc_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end())),
    //                       d_output.begin(),
    //                       addc_u64_functor());
    // }

   


    // add two 254-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    /* __device__ __forceinline__
    void addc_functor(uint254) */

    /*
     * hi * 2^n + lo = a * b
     */
    __device__ __forceinline__
    void mul_wide(u32 &hi, u32 &lo, u32 a, u32 b) {
        asm ("{\n\t"
             " .reg .u64 tmp;\n\t"
             " mul.wide.u32 tmp, %2, %3;\n\t"
             " mov.b64 { %1, %0 }, tmp;\n\t"
             "}"
             : "=r"(hi), "=r"(lo)
             : "r"(a), "r"(b));
    }

    struct mul_wide_functor {
        __device__ __forceinline__ 
        thrust::tuple<u32, u32> operator() (const thrust::tuple<u32, u32>& input) {
            u32 hi, lo;
            asm ("{\n\t"
                 " .reg .u64 tmp;\n\t"
                 " mul.wide.u32 tmp, %2, %3;\n\t"
                 " mov.b64 { %1, %0 }, tmp;\n\t"
                 "}"
                 : "=r"(hi), "=r"(lo)
                 : "r"(thrust::get<0>(input)), "r"(thrust::get<1>(input)));
            return thrust::make_tuple(hi, lo);
        }
    }

    void mul_wide_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_output1, thrust::device_vector<u32>& d_output2) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_output1.begin(), d_output2.begin())),
                          mul_wide_functor());
    }

    __device__ __forceinline__
    void mul_wide(u64 &hi, u64 &lo, u64 a, u64 b) {
        asm ("mul.hi.u64 %0, %2, %3;\n\t"
             "mul.lo.u64 %1, %2, %3;"
             : "=l"(hi), "=l"(lo)
             : "l"(a), "l"(b));
    }

    struct mul_wide_functor {
        __device__ __forceinline__ 
        thrust::tuple<u64, u64> operator() (const thrust::tuple<u64, u64>& input) {
            u64 hi, lo;
            asm ("mul.hi.u64 %0, %2, %3;\n\t"
                 "mul.lo.u64 %1, %2, %3;"
                 : "=l"(hi), "=l"(lo)
                 : "l"(thrust::get<0>(input)), "l"(thrust::get<1>(input)));
            return thrust::make_tuple(hi, lo);
        }
    }

    void mul_wide_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_output1, thrust::device_vector<u64>& d_output2) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_output1.begin(), d_output2.begin())),
                          mul_wide_functor());
    }

    // lo = a * b + c (mod 2^n)
    __device__ __forceinline__
    void mad_lo(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    struct mad_lo_functor {
        __device__ __forceinline__ 
        u32 operator() (const thrust::tuple<u32, u32, u32>& input) {
            u32 lo;
            asm ("mad.lo.u32 %0, %1, %2, %3;"
                 : "=r"(lo)
                 : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)), "r" (thrust::get<2>(input)));
            return lo;
        }
    }

    void mad_lo_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_input3, thrust::device_vector<u32>& d_output) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
                          d_output.begin(),
                          mad_lo_functor());
    }

    __device__ __forceinline__
    void mad_lo(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    struct mad_lo_functor {
        __device__ __forceinline__ 
        u64 operator() (const thrust::tuple<u64, u64, u64>& input) {
            u64 lo;
            asm ("mad.lo.u64 %0, %1, %2, %3;"
                 : "=l"(lo)
                 : "l"(thrust::get<0>(input)), "l" (thrust::get<1>(input)), "l" (thrust::get<2>(input)));
            return lo;
        }
    }

    void mad_lo_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_input3, thrust::device_vector<u64>& d_output) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
                          d_output.begin(),
                          mad_lo_functor());
    }

    // as above but with carry in cy
    __device__ __forceinline__
    void mad_lo_cc(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.cc.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    // struct mad_lo_cc_functor {
    //     __device__ __forceinline__ 
    //     u32 operator() (const thrust::tuple<u32, u32, u32>& input) {
    //         u32 lo;
    //         asm ("mad.lo.cc.u32 %0, %1, %2, %3;"
    //              : "=r"(lo)
    //              : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)), "r" (thrust::get<2>(input)));
    //         return lo;
    //     }
    // }

    // void mad_lo_cc_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_input3, thrust::device_vector<u32>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
    //                       d_output.begin(),
    //                       mad_lo_cc_functor());
    // }

    __device__ __forceinline__
    void mad_lo_cc(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.cc.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    // struct mad_lo_cc_functor {
    //     __device__ __forceinline__ 
    //     u64 operator() (const thrust::tuple<u64, u64, u64>& input) {
    //         u64 lo;
    //         asm ("mad.lo.cc.u64 %0, %1, %2, %3;"
    //              : "=l"(lo)
    //              : "l"(thrust::get<0>(input)), "l" (thrust::get<1>(input)), "l" (thrust::get<2>(input)));
    //         return lo;
    //     }
    // }

    // void mad_lo_cc_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_input3, thrust::device_vector<u64>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
    //                       d_output.begin(),
    //                       mad_lo_cc_functor());
    // }

    __device__ __forceinline__
    void mad_hi(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    struct mad_hi_functor {
        __device__ __forceinline__ 
        u32 operator() (const thrust::tuple<u32, u32, u32>& input) {
            u32 hi;
            asm ("mad.hi.u32 %0, %1, %2, %3;"
                 : "=r"(hi)
                 : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)), "r" (thrust::get<2>(input)));
            return hi;
        }
    }

    void mad_hi_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_input3, thrust::device_vector<u32>& d_output) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
                          d_output.begin(),
                          mad_hi_functor());
    }

    __device__ __forceinline__
    void mad_hi(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    struct mad_hi_functor {
        __device__ __forceinline__ 
        u64 operator() (const thrust::tuple<u64, u64, u64>& input) {
            u64 hi;
            asm ("mad.hi.u64 %0, %1, %2, %3;"
                 : "=l"(hi)
                 : "l"(thrust::get<0>(input)), "l" (thrust::get<1>(input)), "l" (thrust::get<2>(input)));
            return hi;
        }
    }

    void mad_hi_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_input3, thrust::device_vector<u64>& d_output) {
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
                          d_output.begin(),
                          mad_hi_functor());
    }

    __device__ __forceinline__
    void mad_hi_cc(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.cc.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    // struct mad_hi_cc_functor {
    //     __device__ __forceinline__ 
    //     u32 operator() (const thrust::tuple<u32, u32, u32>& input) {
    //         u32 hi;
    //         asm ("mad.hi.cc.u32 %0, %1, %2, %3;"
    //              : "=r"(hi)
    //              : "r"(thrust::get<0>(input)), "r" (thrust::get<1>(input)), "r" (thrust::get<2>(input)));
    //         return hi;
    //     }
    // }

    // void mad_hi_cc_thrust(thrust::device_vector<u32>& d_input1, thrust::device_vector<u32>& d_input2, thrust::device_vector<u32>& d_input3, thrust::device_vector<u32>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
    //                       d_output.begin(),
    //                       mad_hi_cc_functor());
    // }

    __device__ __forceinline__
    void mad_hi_cc(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.cc.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    // struct mad_hi_cc_functor {
    //     __device__ __forceinline__ 
    //     u64 operator() (const thrust::tuple<u64, u64, u64>& input) {
    //         u64 hi;
    //         asm ("mad.hi.cc.u64 %0, %1, %2, %3;"
    //              : "=l"(hi)
    //              : "l"(thrust::get<0>(input)), "l" (thrust::get<1>(input)), "l" (thrust::get<2>(input)));
    //         return hi;
    //     }
    // }

    // void mad_hi_cc_thrust(thrust::device_vector<u64>& d_input1, thrust::device_vector<u64>& d_input2, thrust::device_vector<u64>& d_input3, thrust::device_vector<u64>& d_output) {
    //     thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input1.begin(), d_input2.begin(), d_input3.begin())),
    //                       thrust::make_zip_iterator(thrust::make_tuple(d_input1.end(), d_input2.end(), d_input3.end())),
    //                       d_output.begin(),
    //                       mad_hi_cc_functor());
    // }
} 