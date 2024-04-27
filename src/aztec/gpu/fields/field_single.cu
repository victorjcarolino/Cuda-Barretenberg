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
    // printf("??? %lu %lu %lu %lu\n", af.limbs[0], af.limbs[1], af.limbs[2], af.limbs[3]);
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
        var C = 0;
        for (int j = 0; j < 4; j++) {
            var newC;
            asm(
                "mad.lo.cc.u64 %1, %2, %3, %1;\n\t"
                "madc.hi.u64 %0, %2, %3, 0;\n\t"
                "add.cc.u64 %1, %1, %4;\n\t"
                "addc.u64 %0, %0, 0;"
                : "=l"(newC), "+l"(T[j])
                : "l"(a[i]), "l"(b[j]), "l"(C)
            );
            // printf("  %d: ai %lu bj %lu\n", i, a[i], b[j]);
            C = newC;
        }
        asm(
            "add.cc.u64 %1, %1, %0;\n\t"
            "addc.u64 %0, 0, 0;"
            : "+l"(C), "+l"(T[4])
        );
        T[5] = C;

        // printf("%d: %lu %lu %lu %lu\n", i, T[0], T[1], T[2], T[3]);

        var m = T[0] * r_inv;
        var S;
        asm(
            "mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
            "madc.hi.u64 %0, %2, %3, 0;"
            : "=l"(C), "=l"(S)
            : "l"(m), "l"(p[0]), "l"(T[0])
        );
        for (int j = 1; j < 4; j++) {
            asm(
                "mad.lo.cc.u64 %1, %2, %3, %0;\n\t"
                "madc.hi.u64 %0, %2, %3, 0;\n\t"
                "add.cc.u64 %1, %1, %4;\n\t"
                "addc.u64 %0, %0, 0;"
                : "+l"(C), "=l"(S)
                : "l"(m), "l"(p[j]), "l"(T[j])
            );
            T[j-1] = S;
        }
        asm(
            "add.cc.u64 %1, %0, %2;\n\t"
            "addc.u64 %2, %3, 0;"
            : "+l"(C), "=l"(T[3]), "+l"(T[4])
            : "l"(T[5])
        );
    }

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
