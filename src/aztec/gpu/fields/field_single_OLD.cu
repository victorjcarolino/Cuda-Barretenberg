// typedef std::uint64_t var;

// BN254 with order 254 and scalars/field elements are 254 bits

// __device__ __forceinline__ mul_add4(const uint254 a, const uint254 b, uint254 &out) {
//     //       7654
//     //     * ba98
//     //     ------
//     //     8*7654
//     //   + 9*654
//     //   + a*54
//     //   + b*4
//     asm(
//         // %8 * whatever
//         "mad.lo.cc.u64 %0, %4, %8, %0;\n\t"
//         "madc.lo.cc.u64 %1, %5, %8, %1;\n\t"
//         "madc.lo.cc.u64 %2, %6, %8, %2;\n\t"
//         "madc.lo.u64 %3, %7, %8, %3;\n\t"
//         "mad.hi.cc.u64 %1, %4, %8, %1;\n\t"
//         "madc.hi.cc.u64 %2, %5, %8, %2;\n\t"
//         "madc.hi.u64 %3, %6, %8, %3;\n\t"

//         // %9 * whatever
//         "mad.lo.cc.u64 %1, %4, %9, %1;\n\t"
//         "madc.lo.cc.u64 %2, %5, %9, %2;\n\t"
//         "madc.lo.u64 %3, %6, %9, %3;\n\t"
//         "mad.hi.cc.u64 %2, %4, %9, %2;\n\t"
//         "madc.hi.u64 %3, %5, %9, %3;\n\t"

//         // %10 * whatever
//         "mad.lo.cc.u64 %2, %4, %10, %2;\n\t"
//         "madc.lo.u64 %3, %5, %10, %3;\n\t"
//         "mad.hi.u64 %3, %4, %10, %3;\n\t"

//         // %11 * whatever
//         "mad.lo.u64 %3, %4, %11, %3;"
//         : "+l"(out[0]), "+l"(out[1]), "+l"(out[2]), "+l"(out[3])
//         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
//           "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
//     )
// }

// __device__ __forceinline__ mul_add4_repeat(const uint254 a, const var b, uint254 &out) {
//     asm(
//         "mad.lo.cc.u64 %0, %4, %8, %0;\n\t"
//         "madc.lo.cc.u64 %1, %5, %8, %1;\n\t"
//         "madc.lo.cc.u64 %2, %6, %8, %2;\n\t"
//         "madc.lo.u64 %3, %7, %8, %3;\n\t"
//         "mad.hi.cc.u64 %1, %4, %8, %1;\n\t"
//         "madc.hi.cc.u64 %2, %5, %8, %2;\n\t"
//         "madc.hi.u64 %3, %6, %8, %3;"
//         : "+l"(out[0]), "+l"(out[1]), "+l"(out[2]), "+l"(out[3])
//         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b)
//     )
// }

// __device__ __forceinline__ mul4(const uint254 a, const uint254 b, uint254 &out) {
//     //       7654
//     //     * ba98
//     //     ------
//     //     8*7654
//     //   + 9*654
//     //   + a*54
//     //   + b*4
//     asm(
//         // %8 * whatever
//         "mul.lo.u64 %0, %4, %8;\n\t"
//         "mul.lo.u64 %1, %5, %8;\n\t"
//         "mul.lo.u64 %2, %6, %8;\n\t"
//         "mul.lo.u64 %3, %7, %8;\n\t"
//         "mad.hi.cc.u64 %1, %4, %8, %1;\n\t"
//         "madc.hi.cc.u64 %2, %5, %8, %2;\n\t"
//         "madc.hi.u64 %3, %6, %8, %3;\n\t"

//         // %9 * whatever
//         "mad.lo.cc.u64 %1, %4, %9, %1;\n\t"
//         "madc.lo.cc.u64 %2, %5, %9, %2;\n\t"
//         "madc.lo.u64 %3, %6, %9, %3;\n\t"
//         "mad.hi.cc.u64 %2, %4, %9, %2;\n\t"
//         "madc.hi.u64 %3, %5, %9, %3;\n\t"

//         // %10 * whatever
//         "mad.lo.cc.u64 %2, %4, %10, %2;\n\t"
//         "madc.lo.u64 %3, %5, %10, %3;\n\t"
//         "mad.hi.u64 %3, %4, %10, %3;\n\t"

//         // %11 * whatever
//         "mad.lo.u64 %3, %4, %11, %3;"
//         : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
//         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
//           "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
//     )
// }

// __device__ __forceinline__ mul4_repeat(const uint254 a, const var b, uint254 &out) {
//     asm(
//         "mul.lo.u64 %0, %4, %8;\n\t"
//         "mul.lo.u64 %1, %5, %8;\n\t"
//         "mul.lo.u64 %2, %6, %8;\n\t"
//         "mul.lo.u64 %3, %7, %8;\n\t"
//         "mad.hi.cc.u64 %1, %4, %8, 0;\n\t"
//         "madc.hi.cc.u64 %2, %5, %8, %2;\n\t"
//         "madc.hi.u64 %3, %6, %8, %3;"
//         : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
//         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b)
//     )
// }

// // __device__ __forceinline__ add4(const uint254 a, const uint254 b, uint254 &out) {
// //     asm(
// //         "add.cc.u64 "
// //         : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
// //         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
// //           "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
// //     )
// // }

// /// returns a MASK representing the output for each thread
// __device__ __forceinline__ var effective_carries(int &cy_hi, const int propagate[4], const int cy[4]) {
//     int g = cy[0] | (cy[1]<<1) | (cy[2]<<2) | (cy[3]<<3);
//     int p = propagate[0] | (propagate[1]<<1) | (propagate[2]<<2) | (propagate[3]<<3);
//     int allcarries = (p | g) + g;
//     cy_hi = (allcarries >> 4) & 1;
//     allcarries = (allcarries ^ p) | (g << 1);
//     return allcarries;
// }

// __device__ __forceinline__ static field_gpu zero();
    
//     __device__ __forceinline__ static field_gpu one();
    
//     __device__ __forceinline__ static bool is_zero(const var &x);

//     __device__ __forceinline__ static var equal(const var x, const var y);

// Where is store used in filed_gpu?
// /**
//  * Store operation copies data from a register into main memory
//  */
// template<class params> 
// __device__ __forceinline__ void field_gpu<params>::store(var *mem, const var &x) {
//     if (id < LIMBS) {
//         mem[id] = x;
//     }
// }

// template<class params> 
// __device__ __forceinline__ void field_gpu<params>::store(var *mem, const var &x) {

// __device__ __forceinline__ uint254 mul(const uint254 a, const uint254 b, uint254 &res) {
//     var T{0, 0, 0, 0};
//     uint254 p;

//     if (is_same<params, BN254_MOD_BASE>::value) {
//         p = gpu_barretenberg::MOD_Q_BASE;
//     } else {
//         p = gpu_barretenberg::MOD_Q_SCALAR;
//     }

//     for (int i = 0; i < 4; i++) {
//         var C = 0;
//         for (int j = 0; j < 4; j++) {
//             var newC;
//             asm(
//                 "mad.lo.cc.u64 %1, %2, %3, %1;\n\t"
//                 "madc.hi.u64 %0, %2, %3, 0;\n\t"
//                 "add.cc.u64 %1, %1, %4;\n\t"
//                 "addc.u64 %0, %0, 0;"
//                 : "=l"(newC), "+l"(T[j])
//                 : "l"(a[i]), "l"(b[j]), "l"(C)
//             )
//             C = newC;
//         }
//         T[4] = C;

//         var m = T[0];
//         C = T[0];
//         for (int j = 1; j < 4; j++) {
//             var S;
//             var newC;
//             asm(
//                 "mad.lo.cc.u64 %1, %3, %4, %2;\n\t"
//                 "madc.hi.u64 %0, %3, %4, 0;\n\t"
//                 "add.cc.u64 %1, %1, %5;\n\t"
//                 "addc.u64 %0, %0, 0"
//                 : "=l"(newC), "=l"(S)
//                 : "l"(T[j]), "l"(m), "l"(p[j]), "l"(C)
//             )
//             C = newC;
//             T[j-1] = S;
//         }
//         T[3] = T[4] + C;
//     }

//     return T; // 5 limbs?!
// }

// __device__ __forceinline__ var mul(const uint254 a, const uint254 b, uint254 &res) {
//     uint254 x = a, y = b, z = digit::zero();

//     uint254 r_inv;
//     if (is_same<params, BN254_MOD_BASE>::value) {
//         r_inv = gpu_barretenberg::r_inv_base;
//     } else {
//         r_inv = gpu_barretenberg::r_inv_scalar;
//     }

//     uint254 carry {0, 0, 0, 0};
//     for (int i = 0; i < LIMBS; i++) {
//         uint254 u;
//         var tmpi = x[i] * r_inv[i] * y[0];

//         digit::mad_lo(u[0], z[0], r_inv[0], tmpi);
//         digit::mad_lo_cy(z[0], cy[0], mod[0], u[0], z[0]);

//         digit::mad_lo(u[1], z[0], r_inv[1], tmpi);
//         digit::mad_lo_cy(z[1], cy[1], mod[1], u[1], z[1]);

//         digit::mad_lo(u[2], z[0], r_inv[2], tmpi);
//         digit::mad_lo_cy(z[2], cy[2], mod[2], u[2], z[2]);

//         digit::mad_lo(u[3], z[0], r_inv[3], tmpi);
//         digit::mad_lo_cy(z[3], cy[3], mod[3], u[3], z[3]);

//         assert(z[0] == 0);

//         z[0] = z[1];
//         z[1] = z[2];
//         z[2] = z[3];
//         z[3] = z[0];

//         digit::add_cy(z[0], cy[0], z[0], cy[0]);
//         digit::add_cy(z[1], cy[1], z[1], cy[1]);
//         digit::add_cy(z[2], cy[2], z[2], cy[2]);
//         digit::add_cy(z[3], cy[3], z[3], cy[3]);

//         digit::mad_hi_cy(z[0], cy[0], mod[0], u[0], z[0]);
//         digit::mad_hi_cy(z[1], cy[1], mod[1], u[1], z[1]);
//         digit::mad_hi_cy(z[2], cy[2], mod[2], u[2], z[2]);
//         digit::mad_hi_cy(z[3], cy[3], mod[3], u[3], z[3]);
        
//         digit::mad_hi_cy(z[0], cy[0], y[0], x[i], z[0]);
//         digit::mad_hi_cy(z[1], cy[1], y[1], x[i], z[1]);
//         digit::mad_hi_cy(z[2], cy[2], y[2], x[i], z[2]);
//         digit::mad_hi_cy(z[3], cy[3], y[3], x[i], z[3]);
//     }

//     int msb[4] {cy[3], cy[3], cy[3], cy[3]};
//     cy[3] = cy[2];
//     cy[2] = cy[1];
//     cy[1] = cy[0];
//     cy[0] = 0;

//     // TODO: fixnum: add_cy

//     msb[0] += cy[0];
//     msb[1] += cy[1];
//     msb[2] += cy[2];
//     msb[3] += cy[3];

//     assert(msb[0] <= 1);
//     assert(msb[1] <= 1);
//     assert(msb[2] <= 1);
//     assert(msb[3] <= 1);

//     uint254 r;
//     int br[4];
// }

//     __device__ __forceinline__ static var to_monty(var x, var &res);
    
//     __device__ __forceinline__ static var from_monty(var x, var &res);

//     __device__ __forceinline__ static var neg(var &x, var &res);