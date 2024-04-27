/* Inspired by: https://github.com/data61/cuda-fixnum/blob/5a422c70fcdb2571387270edabaf828adbc69fc7/src/fixnum/internal/primitives.cu*/

#pragma once

#include <cooperative_groups.h>
#include "primitives.cu"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// 4 * 64 = 256 bits, but we'll only use 254 of them since we are using 
// BN254 with order 254 and scalars/field elements are 254 bits
// struct uint254 {
//     std::uint64_t limbs[4]; 

//     __device__ __forceinline__
//     uint254() {
//         limbs[0] = 0;
//         limbs[1] = 0;
//         limbs[2] = 0;
//         limbs[3] = 0;
//     }

//     // __device__ __forceinline__
//     // uint254 operator+(const uint254& other) const {
//     //     uint254 result;
//     //     std::uint64_t carry = 0;
//     //     for (int i = 0; i < 4; i++) {
//     //         result.limbs[i] = limbs[i] + other.limbs[i] + carry;
//     //         carry = (result.limbs[i] < limbs[i] || result.limbs[i] < other.limbs[i] ? 1 : 0);
//     //     }
//     //     return result;
//     // }
// };

/*
 * var is the basic register type that we deal with. The interpretation of 
 * such registers is determined by the struct used, e.g. digit and fixnum
 */
typedef std::uint64_t var;

struct digit {
    // Add the values of two variables 'a' and 'b' and stores the result in 's'
    __device__ __forceinline__
    static void add(var &s, var a, var b) {
        s = a + b;
    }

    // struct add_functor {
    //     __device__ __forceinline__
    //     var operator()(const var &a, const var &b) const {
    //         var s;
    //         digit::add(s, a, b);
    //         return s;
    //     }
    // }

    // void add_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &s) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
    //         s.begin(),
    //         add_functor()
    //     );
    // }

    // Add the values of two variables 'a' and b and stores the result in 's' 
    // Store the carry bit in variable 'cy'
    __device__ __forceinline__
    static void add_cy(var &s, int &cy, var a, var b) {
        s = a + b;
        cy = s < a;
    }

    // struct add_cy_functor {
    //     __device__ __forceinline__
    //     thrust::tuple<var, int> operator()(const var &a, const var &b) const {
    //         var s;
    //         int cy;
    //         digit::add_cy(s, cy, a, b);
    //         return thrust::make_tuple(s, cy);
    //     }
    // }

    // void add_cy_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &s, thrust::device_vector<int> &cy) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
    //         thrust::make_zip_iterator(thrust::make_tuple(s.begin(), cy.begin())),
    //         add_cy_functor()
    //     );
    // }

    // Subtract the value of one variable 'b' from 'a' and stores the result in 'd'
    __device__ __forceinline__
    static void sub(var &d, var a, var b) {
        d = a - b;
    }

    // struct sub_functor {
    //     __device__ __forceinline__
    //     var operator()(const var &a, const var &b) const {
    //         var d;
    //         digit::sub(d, a, b);
    //         return d;
    //     }
    // }   

    // void sub_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &d) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
    //         d.begin(),
    //         sub_functor()
    //     );
    // }

    // Subtract the value of variable 'b' from 'a' and stores the result in 'd'
    // Store the borrow bit in variable 'br'
    __device__ __forceinline__
    static void sub_br(var &d, int &br, var a, var b) {
        d = a - b;
        br = d > a;
    }

    // struct sub_br_functor {
    //     __device__ __forceinline__
    //     thrust::tuple<var, int> operator()(const var &a, const var &b) const {
    //         var d;
    //         int br;
    //         digit::sub_br(d, br, a, b);
    //         return thrust::make_tuple(d, br);
    //     }
    // }

    // void sub_br_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &d, thrust::device_vector<int> &br) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
    //         thrust::make_zip_iterator(thrust::make_tuple(d.begin(), br.begin())),
    //         sub_br_functor()
    //     );
    // }

    // Return zero value of the var type
    __device__ __forceinline__
    static var zero() { return 0ULL; }

    // struct zero_functor {
    //     __device__ __forceinline__
    //     var operator()() const {
    //         return digit::zero();
    //     }
    // }

    // void zero_thrust(thrust::device_vector<var> &z) {
    //     thrust::fill(z.begin(), z.end(), 0ULL);
    // }

    // Return true if variable 'a' is equal to the maximum value of the var type, and false otherwise
    __device__ __forceinline__
    static int is_max(var a) { return a == ~0ULL; }

    // struct is_max_functor {
    //     __device__ __forceinline__
    //     int operator()(const var &a) const {
    //         return digit::is_max(a);
    //     }
    // }

    // void is_max_thrust(thrust::device_vector<var> &a, thrust::device_vector<int> &res) {
    //     thrust::transform(a.begin(), a.end(), res.begin(), is_max_functor());
    // }

    // Return true if variable 'a' is equal to the minimum value of the var type, and false otherwise
    __device__ __forceinline__
    static int is_min(var a) { return a == 0ULL; }

    // struct is_min_functor {
    //     __device__ __forceinline__
    //     int operator()(const var &a) const {
    //         return digit::is_min(a);
    //     }
    // }

    // void is_min_thrust(thrust::device_vector<var> &a, thrust::device_vector<int> &res) {
    //     thrust::transform(a.begin(), a.end(), res.begin(), is_min_functor());
    // }

    // Return true if variable 'a' is equal to zero, and false otherwise
    __device__ __forceinline__
    static int is_zero(var a) { return a == zero(); }

    // struct is_zero_functor {
    //     __device__ __forceinline__
    //     int operator()(const var &a) const {
    //         return digit::is_zero(a);
    //     }
    // }

    // void is_zero_thrust(thrust::device_vector<var> &a, thrust::device_vector<int> &res) {
    //     thrust::transform(a.begin(), a.end(), res.begin(), is_zero_functor());
    // }

    // Multiply two variables 'a' and 'b' and stores the lower 64 bits of the result in 'lo'
    __device__ __forceinline__
    static void mul_lo(var &lo, var a, var b) {
        lo = a * b;
    }

    // struct mul_lo_functor {
    //     __device__ __forceinline__
    //     var operator()(const var &a, const var &b) const {
    //         var lo;
    //         digit::mul_lo(lo, a, b);
    //         return lo;
    //     }
    // }

    // void mul_lo_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &lo) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
    //         lo.begin(),
    //         mul_lo_functor()
    //     );
    // }

    // Compute the result of the operation a * b + c and stores the lower 64 bits in 'lo'
    // lo = a * b + c (mod 2^64)
    __device__ __forceinline__
    static void mad_lo(var &lo, var a, var b, var c) {
        internal::mad_lo(lo, a, b, c);
    }

    // struct mad_lo_functor {
    //     __device__ __forceinline__
    //     var operator()(const var &a, const var &b, const var &c) const {
    //         var lo;
    //         digit::mad_lo(lo, a, b, c);
    //         return lo;
    //     }
    // }

    // void mad_lo_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &c, thrust::device_vector<var> &lo) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end())),
    //         lo.begin(),
    //         mad_lo_functor()
    //     );
    // }

    // Compute the result of the operation a * b + c and stores the lower 64 bits in 'lo'
    // Increment the value of 'cy' by the mad carry
    __device__ __forceinline__
    static void mad_lo_cy(var &lo, int &cy, var a, var b, var c) {
        internal::mad_lo_cc(lo, a, b, c);
        internal::addc(cy, cy, 0);
    }

    // struct mad_lo_cy_functor {
    //     __device__ __forceinline__
    //     thrust::tuple<var, int> operator()(const var &a, const var &b, const var &c) const {
    //         var lo;
    //         int cy;
    //         digit::mad_lo_cy(lo, cy, a, b, c);
    //         return thrust::make_tuple(lo, cy);
    //     }
    // }

    // void mad_lo_cy_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &c, thrust::device_vector<var> &lo, thrust::device_vector<int> &cy) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end())),
    //         thrust::make_zip_iterator(thrust::make_tuple(lo.begin(), cy.begin())),
    //         mad_lo_cy_functor()
    //     );
    // }

    // Compute the result of the operation a * b + c and stores the upper 64 bits in 'hi'
    __device__ __forceinline__
    static void mad_hi(var &hi, var a, var b, var c) {
        internal::mad_hi(hi, a, b, c);
    }

    // struct mad_hi_functor {
    //     __device__ __forceinline__
    //     var operator()(const var &a, const var &b, const var &c) const {
    //         var hi;
    //         digit::mad_hi(hi, a, b, c);
    //         return hi;
    //     }
    // }

    // void mad_hi_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &c, thrust::device_vector<var> &hi) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end())),
    //         hi.begin(),
    //         mad_hi_functor()
    //     );
    // }

    // Compute the result of the operation a * b + c and stores the upper 64 bits in 'hi'
    // Increment the value of 'cy' by the mad carry
    __device__ __forceinline__
    static void mad_hi_cy(var &hi, int &cy, var a, var b, var c) {
        internal::mad_hi_cc(hi, a, b, c);
        internal::addc(cy, cy, 0);
    }

    // struct mad_hi_cy_functor {
    //     __device__ __forceinline__
    //     thrust::tuple<var, int> operator()(const var &a, const var &b, const var &c) const {
    //         var hi;
    //         int cy;
    //         digit::mad_hi_cy(hi, cy, a, b, c);
    //         return thrust::make_tuple(hi, cy);
    //     }
    // }

    // void mad_hi_cy_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &c, thrust::device_vector<var> &hi, thrust::device_vector<int> &cy) {
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end())),
    //         thrust::make_zip_iterator(thrust::make_tuple(hi.begin(), cy.begin())),
    //         mad_hi_cy_functor()
    //     );
    // }
};

struct fixnum {

    // Return the layout of the current thread block as a thread_block_tile object with WIDTH threads
    // This shouldn't be relevant when using Thrust instead of CUDA
    static constexpr unsigned WIDTH = 4;
    __device__ __forceinline__
    static cooperative_groups::thread_block_tile<WIDTH> layout() {
        return cooperative_groups::tiled_partition<WIDTH>(cooperative_groups::this_thread_block());
    }

    // Return zero value of var type
    __device__ __forceinline__
    static var zero() { 
        return digit::zero(); 
    }

    // void zero_thrust(thrust::device_vector<var> &z) {
    //     thrust::fill(z.begin(), z.end(), digit::zero());
    // }

    // Return one value of var type
    __device__ __forceinline__
    static var one() {
        auto t = layout().thread_rank();
        return (var)(t == 0);
    }

    // Add the values of two variables 'a' and 'b' and stores the result in 'r'
    // Store the carry bit in the variable 'cy_hi'. If the result of the addition overflows, 
    // it is propagated to the 'cy_hi' variable
    __device__ __forceinline__
    static void add_cy(var &r, int &cy_hi, const var &a, const var &b) {
        int cy;
        digit::add_cy(r, cy, a, b);
        // r propagates carries iff r = FIXNUM_MAX
        var r_cy = effective_carries(cy_hi, digit::is_max(r), cy);
        digit::add(r, r, r_cy);
    }

    // struct add_cy_functor {
    //     __device__ __forceinline__
    //     void operator()(thrust::tuple<var&, int&, const var&, const var&> t) {
    //         var &r = thrust::get<0>(t);
    //         int &cy_hi = thrust::get<1>(t);
    //         const var &a = thrust::get<2>(t);
    //         const var &b = thrust::get<3>(t);
    //         digit::add_cy(r, cy_hi, a, b);
    //     }
    // }

    // void add_cy_thrust(thrust::device_vector<var> &a, thrust::device_vector<var> &b, thrust::device_vector<var> &r, thrust::device_vector<int> &cy_hi) {
    //     thrust::for_each(
    //         thrust::make_zip_iterator(thrust::make_tuple(r.begin(), cy_hi.begin(), a.begin(), b.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(r.end(), cy_hi.end(), a.end(), b.end())),
    //         add_cy_functor()
    //     );
    // }

    // Add the values of two variables 'a' and 'b' and stores the result in a third variable 'r' 
    // If the result of the addition overflows, it is propagated to the next higher digit
    __device__ __forceinline__
    static void add(var &r, const var &a, const var &b) {
        int cy_hi;
        add_cy(r, cy_hi, a, b);
    }

    // Ssubtract the value of one variable 'b' from 'a' and stores the result in 'r'
    // Store the borrow bit in the variable 'br_lo'. If the result of the subtraction underflows, 
    // it is propagated to the 'br_lo' variable
    __device__ __forceinline__
    static void sub_br(var &r, int &br_lo, const var &a, const var &b) {
        int br;
        digit::sub_br(r, br, a, b);
        // r propagates borrows iff r = FIXNUM_MIN
        var r_br = effective_carries(br_lo, digit::is_min(r), br);
        digit::sub(r, r, r_br);
    }


    // Subtract the value of one variable 'b' from 'a' and stores the result in 'r'. If the result
    // of the subtraction underflows, it is propagated to the next higher digit
    __device__ __forceinline__
    static void sub(var &r, const var &a, const var &b) {
        int br_lo;
        sub_br(r, br_lo, a, b);
    }

    __device__ __forceinline__ 
    static uint32_t nonzero_mask(var r) {
        return fixnum::layout().ballot( ! digit::is_zero(r));
    }

    __device__ __forceinline__
    static int is_zero(var r) {
        return nonzero_mask(r) == 0U;
    }

    // Compare equality of two var arrays
    __device__ __forceinline__
    static int cmp(var x, var y) {
        var r;
        int br;
        sub_br(r, br, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    // Helper function 
    __device__ __forceinline__
    static var effective_carries(int &cy_hi, int propagate, int cy) {
        uint32_t allcarries, p, g;
        auto grp = fixnum::layout();

        g = grp.ballot(cy);                       // carry generate
        p = grp.ballot(propagate);                // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        cy_hi = (allcarries >> grp.size()) & 1;   // detect hi overflow
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        return (allcarries >> grp.thread_rank()) & 1;
    }
};