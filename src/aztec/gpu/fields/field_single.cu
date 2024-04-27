#pragma once

#include <cooperative_groups.h>
#include "primitives.cu"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <type_traits>
#include "field.cuh"
#include "field_single.cuh"

using namespace gpu_barretenberg;

template<class params>
__device__ __forceinline__ void field_single<params>::add(const uint254 a, const uint254 b, uint254 &res) {
    const var *p;
    if (std::is_same_v<params, BN254_MOD_BASE>) {
        p = gpu_barretenberg::MOD_Q_BASE;
    } else {
        p = gpu_barretenberg::MOD_Q_SCALAR;
    }
    asm(
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.u64 %3, %7, %11;"
        : "=l"(res.limbs[0]), "=l"(res.limbs[1]), "=l"(res.limbs[2]), "=l"(res.limbs[3])
        : "l"(a.limbs[0]), "l"(a.limbs[1]), "l"(a.limbs[2]), "l"(a.limbs[3]),
          "l"(b.limbs[0]), "l"(b.limbs[1]), "l"(b.limbs[2]), "l"(b.limbs[3])
    );
    bool res_ge_p =
        (res.limbs[3] > p[3]) ? true :
        (res.limbs[3] < p[3]) ? false :
        (res.limbs[2] > p[2]) ? true :
        (res.limbs[2] < p[2]) ? false :
        (res.limbs[1] > p[1]) ? true :
        (res.limbs[1] < p[1]) ? false :
        (res.limbs[0] >= p[0]);
    if (!res_ge_p) 
        
    printf("    xx %lu %lu %lu\n", a.limbs[0], b.limbs[0], res.limbs[0]);
} 

template<class params>
__device__ __forceinline__ void field_single<params>::sub(const uint254 a, const uint254 b, uint254 &res) {
    /** // sub and sub_inplace are each more efficient for thees    res = a;
    sub_inplace(res, b);
     */
    
    const var *p;
    if (std::is_same_v<params, BN254_MOD_BASE>) {
        p = gpu_barretenberg::MOD_Q_BASE;
    } else {
        p = gpu_barretenberg::MOD_Q_SCALAR;
    }

    asm(
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;"
        : "=l"(res.limbs[0]), "=l"(res.limbs[1]), "=l"(res.limbs[2]), "=l"(res.limbs[3])
        : "l"(a.limbs[0]), "l"(a.limbs[1]), "l"(a.limbs[2]), "l"(a.limbs[3]),
          "l"(b.limbs[0]), "l"(b.limbs[1]), "l"(b.limbs[2]), "l"(b.limbs[3])
    );

    bool a_ge_b =
        (a.limbs[3] > b.limbs[3]) ? true :
        (a.limbs[3] < b.limbs[3]) ? false :
        (a.limbs[2] > b.limbs[2]) ? true :
        (a.limbs[2] < b.limbs[2]) ? false :
        (a.limbs[1] > b.limbs[1]) ? true :
        (a.limbs[1] < b.limbs[1]) ? false :
        (a.limbs[0] >= b.limbs[0]);
    if (!a_ge_b)
        asm(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, %5;\n\t"
            "addc.cc.u64 %2, %2, %6;\n\t"
            "addc.u64 %3, %3, %7;"
            : "+l"(res.limbs[0]), "+l"(res.limbs[1]), "+l"(res.limbs[2]), "+l"(res.limbs[3])
            : "l"(p[0]), "l"(p[1]), "l"(p[2]), "l"(p[3])
        );
}

template<class params>
__device__ __forceinline__ void field_single<params>::sub_inplace(uint254 &af, const var *b) {
    var *a = af.limbs;

    const var *p;
    if (std::is_same_v<params, BN254_MOD_BASE>) {
        p = gpu_barretenberg::MOD_Q_BASE;
    } else {
        p = gpu_barretenberg::MOD_Q_SCALAR;
    }

    bool a_ge_b =
        (a[3] > b[3]) ? true :
        (a[3] < b[3]) ? false :
        (a[2] > b[2]) ? true :
        (a[2] < b[2]) ? false :
        (a[1] > b[1]) ? true :
        (a[1] < b[1]) ? false :
        (a[0] >= b[0]);
    if (!a_ge_b)
        asm(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, %5;\n\t"
            "addc.cc.u64 %2, %2, %6;\n\t"
            "addc.u64 %3, %3, %7;"
            : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3])
            : "l"(p[0]), "l"(p[1]), "l"(p[2]), "l"(p[3])
        );
    asm(
        "sub.cc.u64 %0, %0, %4;\n\t"
        "subc.cc.u64 %1, %1, %5;\n\t"
        "subc.cc.u64 %2, %2, %6;\n\t"
        "subc.u64 %3, %3, %7;"
        : "+l"(a[0]), "+l"(a[1]), "+l"(a[2]), "+l"(a[3])
        : "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
}

template<class params>
__device__ __forceinline__ void field_single<params>::mul(const uint254 af, const uint254 bf, uint254 &resf) {
    const var *a = af.limbs;
    const var *b = bf.limbs;
    var T[6]{0, 0, 0, 0, 0, 0};
    var r_inv;
    const var *p;
    if (std::is_same_v<params, BN254_MOD_BASE>) {
        r_inv = gpu_barretenberg::r_inv_base;
        p = gpu_barretenberg::MOD_Q_BASE;
    } else {
        r_inv = gpu_barretenberg::r_inv_scalar;
        p = gpu_barretenberg::MOD_Q_SCALAR;
    }

    for (int i = 0; i < 4; i++) {
        asm(
            "mad.lo.cc.u64 %0, %11, %12, %6;\n\t"   // T[0] = a[i]*b[0] + T[0];
            "madc.lo.cc.u64 %1, %11, %13, %7;\n\t"  // T[1] = a[i]*b[1] + T[1];
            "madc.lo.cc.u64 %2, %11, %14, %8;\n\t"  // T[2] = a[i]*b[2] + T[2];
            "madc.lo.cc.u64 %3, %11, %15, %9;\n\t"  // T[3] = a[i]*b[3] + T[3];
            "addc.cc.u64 %4, %10, 0;\n\t"           //
            "addc.u64 %5, 0, 0;\n\t"                //
            "mad.hi.cc.u64 %1, %11, %12, %1;\n\t"   //
            "madc.hi.cc.u64 %2, %11, %13, %2;\n\t"  //
            "madc.hi.cc.u64 %3, %11, %14, %3;\n\t"  //
            "madc.hi.cc.u64 %4, %11, %15, %4;\n\t"  //
            "addc.u64 %5, 0, 0;"
            : "=l"(T[0]), "=l"(T[1]), "=l"(T[2]), "=l"(T[3]), "=l"(T[4]), "=l"(T[5])
              // 6
            : "l"(T[0]), "l"(T[1]), "l"(T[2]), "l"(T[3]), "l"(T[4]),
              // 11
              "l"(a[i]),
              // 12
              "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
        );

        var m = T[0] * r_inv;
        asm(
            "{\n\t"
            " .reg .u64 tmp;"
            " mad.lo.cc.u64 tmp, %11, %12, %5;\n\t"  // tmp = {m*p[0]}.lo + T[0]
            " madc.lo.cc.u64 %0, %11, %13, %6;\n\t"  // T[0] = {m*p[1]}.lo + T[1] + cf
            " madc.lo.cc.u64 %1, %11, %14, %7;\n\t"  // T[1] = {m*p[2]}.lo + T[2] + cf
            " madc.lo.cc.u64 %2, %11, %15, %8;\n\t"  // T[2] = {m*p[3]}.lo + T[3] + cf
            " addc.cc.u64 %3, %9, 0;\n\t"            // T[3] = T[4] + cf
            " addc.u64 %4, %10, 0;\n\t"              // T[4] = T[5] + cf
            " mad.hi.cc.u64 %0, %11, %12, %0;\n\t"   // T[0] += {m*p[0]}.hi
            " madc.hi.cc.u64 %1, %11, %13, %1;\n\t"  // T[1] += {m*p[1]}.hi + cf
            " madc.hi.cc.u64 %2, %11, %14, %2;\n\t"  // T[2] += {m*p[2]}.hi + cf
            " madc.hi.cc.u64 %3, %11, %15, %3;\n\t"  // T[3] += {m*p[3]}.hi + cf
            " addc.u64 %4, %4, 0;\n\t"               // T[4] += cf
            "}"
            : "=l"(T[0]), "=l"(T[1]), "=l"(T[2]), "=l"(T[3]), "=l"(T[4])
              // 5
            : "l"(T[0]), "l"(T[1]), "l"(T[2]), "l"(T[3]), "l"(T[4]), "l"(T[5])
              // 11
              "l"(m),
              // 12
              "l"(p[0]), "l"(p[1]), "l"(p[2]), "l"(p[3])
        );
    }
    printf("\n");

    // uint254 resf {T[0], T[1], T[2], T[3]};
    resf.limbs[0] = T[0];
    resf.limbs[1] = T[1];
    resf.limbs[2] = T[2];
    resf.limbs[3] = T[3];

    bool t_ge_p =
        (T[4] > 0) ? true :
        (T[3] > p[3]) ? true :
        (T[3] < p[3]) ? false :
        (T[2] > p[2]) ? true :
        (T[2] < p[2]) ? false :
        (T[1] > p[1]) ? true :
        (T[1] < p[1]) ? false :
        (T[0] >= p[0]);

    if (t_ge_p)
        sub_inplace(resf, p);
}

template<class params>
__device__ __forceinline__ void field_single<params>::square_inplace(uint254 &resf) {
    uint254 x{resf.limbs[0], resf.limbs[1], resf.limbs[2], resf.limbs[3]};
    mul(x, x, resf);
}

template<class params>
__device__ __forceinline__ void field_single<params>::square(const uint254 x, uint254 &res) {
    res = x;
    square_inplace(res);
}

template<class params>
__device__ __forceinline__ void field_single<params>::neg(uint254 &resf) {
    uint254 x{resf.limbs[0], resf.limbs[1], resf.limbs[2], resf.limbs[3]};

    if (std::is_same_v<params, BN254_MOD_BASE>) {
        uint254 p {gpu_barretenberg::MOD_Q_BASE[0], gpu_barretenberg::MOD_Q_BASE[1], gpu_barretenberg::MOD_Q_BASE[2], gpu_barretenberg::MOD_Q_BASE[3]};
        sub(p, x, resf);
    } else {
        uint254 p {gpu_barretenberg::MOD_Q_SCALAR[0], gpu_barretenberg::MOD_Q_SCALAR[1], gpu_barretenberg::MOD_Q_SCALAR[2], gpu_barretenberg::MOD_Q_SCALAR[3]};
        sub(p, x, resf);
    }
}
