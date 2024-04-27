#include <cstdint>
#include <stdio.h>
#include "field.cuh"

using namespace std;
using namespace gpu_barretenberg;

/**
 * Function declerations are templated with base and scalar fields represented as 'params'
 */
template<class params> 
__device__ __forceinline__ field_gpu<params>::field_gpu(var a, var b, var c, var d) noexcept
    : data{ a, b, c, d } {};

template<class params> 
__device__ __forceinline__ field_gpu<params> field_gpu<params>::zero() {
    return field_gpu(0x0, 0x0, 0x0, 0x0); 
}

// Montgomery form of 1
template<class params> 
__device__ __forceinline__ field_gpu<params> field_gpu<params>::one() {
    return field_gpu(0xd35d438dc58f0d9d, 0xa78eb28f5c70b3d, 0x666ea36f7879462c, 0xe0a77c19a07df2f); 
}

template<class params> 
__device__ __forceinline__ bool field_gpu<params>::is_zero(const var &x) {
    return fixnum::is_zero(x); 
}

// Note: converted the return type from var to bool
template<class params> 
__device__ __forceinline__ var field_gpu<params>::equal(const var x, const var y) { 
    return fixnum::cmp(x, y) == 0; 
}

/**
 * Load operation copies data from main memory into a register
 */
template<class params> 
__device__ __forceinline__ var field_gpu<params>::load(var x, var &res) {
    int id = params::lane();
    res = x;
    return res;
}

// Not used in msm implementation
/**
 * Store operation copies data from a register into main memory
 */
template<class params> 
__device__ __forceinline__ void field_gpu<params>::store(var *mem, const var &x) {
    int id = params::lane();
    if (id < LIMBS) {
        mem[id] = x;
    }
}

template<class params> 
__device__ __forceinline__ var field_gpu<params>::add(const var a, const var b, var &res) {
    int br;
    var x = a, y = b, z, r;
    var mod = params::mod();
    // z = a+b
    fixnum::add(z, x, y);
    // r = z-mod
    fixnum::sub_br(r, br, z, mod);
    // res = (z < mod) ? z : (z-mod)
    res = br ? z : r;
    return res;
}

template<class params> 
__device__ __forceinline__ var field_gpu<params>::sub(const var x, const var y, var &res) {
    int br;
    var r, mod = params::mod();
    // r = x-y
    fixnum::sub_br(r, br, x, y);
    // if (x<y) r += mod
    if (br)
        fixnum::add(r, r, mod);
    res = r;
    return r;
}

/**
 * Square operation (special casing may yield 1.5 - 2x speed improvement)
 */
template<class params> 
__device__ __forceinline__ var field_gpu<params>::square(var x, var &y) {
    field_gpu::mul(x, x, y);
    return y;
}

/**
 * Convert to montgomery representation 
 */
template<class params> 
__device__ __forceinline__ var field_gpu<params>::to_monty(var x, var &res) {
    var r_sqr_mod = params::monty();
    field_gpu::mul(x, r_sqr_mod, res);
    return res;
}

/**
 * Convert from montgomery representation 
 */
template<class params> 
__device__ __forceinline__ var field_gpu<params>::from_monty(var x, var &res) {
    var mont;
    mont = fixnum::one();
    mul(x, mont, res);
    return res;
}

template<class params> 
__device__ __forceinline__ var field_gpu<params>::neg(var &x, var &res) {
    var mod = params::mod();
    fixnum::sub(res, mod, x);
    return res;
}

/**
 * Montgomery multiplication (CIOS algorithm) for modular multiplications
 */
template<class params> 
__device__ __forceinline__ var field_gpu<params>::mul(const var a, const var b, var &res) {
    auto grp = fixnum::layout();
    int L = grp.thread_rank();  // L indicates the limb this thread is associated with
    var mod = params::mod();

    var x = a, y = b, z = digit::zero();
    var tmp;

    // r_inv[L] = (-Q^{-1} (mod 2^256))[L]
    // tmp[L] = lower 64 bits of (x[L] * r_inv[L])
    if (is_same<params, BN254_MOD_BASE>::value) {
        digit::mul_lo(tmp, x, gpu_barretenberg::r_inv_base);    
    } else {
        digit::mul_lo(tmp, x, gpu_barretenberg::r_inv_scalar);    
    }

    // tmp[L] = lower 64 bits of (tmp[L] * y.limb[0])
    digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
    int cy = 0;

    for (int i = 0; i < LIMBS; ++i) {
        var u;
        var xi = grp.shfl(x, i);
        var z0 = grp.shfl(z, 0);
        var tmpi = grp.shfl(tmp, i);

        // u[L] = lower 64 bits of (z[0]*r_inv[L] + tmp[i])
        if (is_same<params, BN254_MOD_BASE>::value) {
            digit::mad_lo(u, z0, gpu_barretenberg::r_inv_base, tmpi);
        } else {
            digit::mad_lo(u, z0, gpu_barretenberg::r_inv_scalar, tmpi);
        }

        // z[L] = lower 64 bits of (mod[L]*u[L] + z[L])
        digit::mad_lo_cy(z, cy, mod, u, z);
        // z[L] = lower 64 bits of (y[L]*x[i] + z[L])
        digit::mad_lo_cy(z, cy, y, xi, z);

        // if (L == 0) assert(z == 0);
        assert(L || z == 0);
        // z >>= 64 [as a whole group]
        z = grp.shfl_down(z, 1); 
        z = (L >= LIMBS - 1) ? 0 : z;

        // z[L] += cy[L]
        digit::add_cy(z, cy, z, cy);
        // z[L] = higher 64 bits of (mod[L]*u[L] + z[L])
        digit::mad_hi_cy(z, cy, mod, u, z);
        // z[L] = higher 64 bits of (y[L]*x[i] + z[L])
        digit::mad_hi_cy(z, cy, y, xi, z);

        printf("TAL %lx %lx %lx %lx\n", grp.shfl(z, 0), grp.shfl(z, 1), grp.shfl(z, 2), grp.shfl(z, 3));
    }
    printf("\n");
    
    // Resolve carries
    // msb[L] = cy[3] // msb set if most significant limb overflows
    int msb = grp.shfl(cy, LIMBS - 1);
    // cy <<= 64  [as a whole group]  // add value carried from lower limb
    cy = grp.shfl_up(cy, 1);
    cy = (L == 0) ? 0 : cy;

    // FIXNUM: z[L] += cy[L]
    fixnum::add_cy(z, cy, z, cy);
    // msb[L] += cy[L]
    msb += cy;
    // assert(msb[L] <= 1)     // If most significant limb already overflowed, carrying the +1 should not cause another overflow
    assert(msb == !!msb);

    // the entire following section more or less equates to:
    // return (z < mod) ? z : z-mod;
    var r;
    int br;
    // r = z - mod
    fixnum::sub_br(r, br, z, mod);
    // if (msb > 0 || z > mod)
    if (msb || br == 0) {
        // if (msb > 0) assert(z[L] > mod[L]);
        assert(!msb || msb == br);
        z = r;
    }   
    res = z;
    return res;
}


// /**
//  * Montgomery multiplication (CIOS algorithm) for modular multiplications
//  */
// template<class params> 
// __device__ __forceinline__ var field_gpu<params>::mul_thrust(const var a, const var b, var &res) {
//     auto grp = fixnum::layout();
//     int L = grp.thread_rank();
//     var mod = params::mod();

//     var x = a, y = b, z = digit::zero();
//     var tmp;

//     if (is_same<params, BN254_MOD_BASE>::value) {
//         digit::mul_lo(tmp, x, gpu_barretenberg::r_inv_base);    
//     }
//     else {
//          digit::mul_lo(tmp, x, gpu_barretenberg::r_inv_scalar);    
//     }

//     digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
//     int cy = 0;

//     for (int i = 0; i < LIMBS; ++i) {
//         var u;
//         var xi = grp.shfl(x, i);
//         var z0 = grp.shfl(z, 0);
//         var tmpi = grp.shfl(tmp, i);

//         if (is_same<params, BN254_MOD_BASE>::value) {
//             digit::mad_lo(u, z0, gpu_barretenberg::r_inv_base, tmpi);
//         }
//         else {
//             digit::mad_lo(u, z0, gpu_barretenberg::r_inv_scalar, tmpi);
//         }

//         digit::mad_lo_cy(z, cy, mod, u, z);
//         digit::mad_lo_cy(z, cy, y, xi, z);

//         assert(L || z == 0);  
//         z = grp.shfl_down(z, 1); 
//         z = (L >= LIMBS - 1) ? 0 : z;

//         digit::add_cy(z, cy, z, cy);
//         digit::mad_hi_cy(z, cy, mod, u, z);
//         digit::mad_hi_cy(z, cy, y, xi, z);
//     }
    
//     // Resolve carries
//     int msb = grp.shfl(cy, LIMBS - 1);
//     cy = grp.shfl_up(cy, 1); 
//     cy = (L == 0) ? 0 : cy;

//     fixnum::add_cy(z, cy, z, cy);
//     msb += cy;
//     assert(msb == !!msb);

//     var r;
//     int br;
//     fixnum::sub_br(r, br, z, mod);
//     if (msb || br == 0) {
//         assert(!msb || msb == br);
//         z = r;
//     }
//     res = z;
//     return res;
// }