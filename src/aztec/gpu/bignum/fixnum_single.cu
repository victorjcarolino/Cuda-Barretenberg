struct fixnum_single {
    __device__ __forceinline__ mul_add4(const uint254 a, const uint254 b, uint254 &out) {
        //       7654
        //     * ba98
        //     ------
        //     8*7654
        //   + 9*654
        //   + a*54
        //   + b*4
        asm(
            // %8 * whatever
            "mad.lo.cc.u64 %0, %4, %8, %0"
            "madc.lo.cc.u64 %1, %5, %8, %1"
            "madc.lo.cc.u64 %2, %6, %8, %2"
            "madc.lo.u64 %3, %7, %8, %3"
            "mad.hi.cc.u64 %1, %4, %8, %1"
            "madc.hi.cc.u64 %2, %5, %8, %2"+
            "madc.hi.u64 %3, %6, %8, %3"

            // %9 * whatever
            "mad.lo.cc.u64 %1, %4, %9, %1"
            "madc.lo.cc.u64 %2, %5, %9, %2"
            "madc.lo.u64 %3, %6, %9, %3"
            "mad.hi.cc.u64 %2, %4, %9, %2"
            "madc.hi.u64 %3, %5, %9, %3"

            // %10 * whatever
            "mad.lo.cc.u64 %2, %4, %10, %2"
            "madc.lo.u64 %3, %5, %10, %3"
            "mad.hi.u64 %3, %4, %10, %3"

            // %11 * whatever
            "mad.lo.u64 %3, %4, %11, %3"
            : "+l"(out[0]), "+l"(out[1]), "+l"(out[2]), "+l"(out[3])
            : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
              "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
        )
    }

    __device__ __forceinline__ mul_add4_repeat(const uint254 a, const var b, uint254 &out) {
        asm(
            "mad.lo.cc.u64 %0, %4, %8, %0"
            "madc.lo.cc.u64 %1, %5, %8, %1"
            "madc.lo.cc.u64 %2, %6, %8, %2"
            "madc.lo.u64 %3, %7, %8, %3"
            "mad.hi.cc.u64 %1, %4, %8, %1"
            "madc.hi.cc.u64 %2, %5, %8, %2"+
            "madc.hi.u64 %3, %6, %8, %3"
            : "+l"(out[0]), "+l"(out[1]), "+l"(out[2]), "+l"(out[3])
            : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b)
        )
    }

    __device__ __forceinline__ mul4(const uint254 a, const uint254 b, uint254 &out) {
        //       7654
        //     * ba98
        //     ------
        //     8*7654
        //   + 9*654
        //   + a*54
        //   + b*4
        asm(
            // %8 * whatever
            "mul.lo.u64 %0, %4, %8"
            "mul.lo.u64 %1, %5, %8"
            "mul.lo.u64 %2, %6, %8"
            "mul.lo.u64 %3, %7, %8"
            "mad.hi.cc.u64 %1, %4, %8, %1"
            "madc.hi.cc.u64 %2, %5, %8, %2"+
            "madc.hi.u64 %3, %6, %8, %3"

            // %9 * whatever
            "mad.lo.cc.u64 %1, %4, %9, %1"
            "madc.lo.cc.u64 %2, %5, %9, %2"
            "madc.lo.u64 %3, %6, %9, %3"
            "mad.hi.cc.u64 %2, %4, %9, %2"
            "madc.hi.u64 %3, %5, %9, %3"

            // %10 * whatever
            "mad.lo.cc.u64 %2, %4, %10, %2"
            "madc.lo.u64 %3, %5, %10, %3"
            "mad.hi.u64 %3, %4, %10, %3"

            // %11 * whatever
            "mad.lo.u64 %3, %4, %11, %3"
            : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
            : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
              "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
        )
    }

    __device__ __forceinline__ mul4_repeat(const uint254 a, const var b, uint254 &out) {
        asm(
            "mul.lo.u64 %0, %4, %8"
            "mul.lo.u64 %1, %5, %8"
            "mul.lo.u64 %2, %6, %8"
            "mul.lo.u64 %3, %7, %8"
            "mad.hi.cc.u64 %1, %4, %8, 0"
            "madc.hi.cc.u64 %2, %5, %8, %2"+
            "madc.hi.u64 %3, %6, %8, %3"
            : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
            : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(b)
        )
    }

    // __device__ __forceinline__ add4(const uint254 a, const uint254 b, uint254 &out) {
    //     asm(
    //         "add.cc.u64 "
    //         : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
    //         : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
    //           "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    //     )
    // }

    /// returns a MASK representing the output for each thread
    __device__ __forceinline__ var effective_carries(int &cy_hi, const int propagate[4], const int cy[4]) {
        int g = cy[0] | (cy[1]<<1) | (cy[2]<<2) | (cy[3]<<3);
        int p = propagate[0] | (propagate[1]<<1) | (propagate[2]<<2) | (propagate[3]<<3);
        int allcarries = (p | g) + g;
        cy_hi = (allcarries >> 4) & 1;
        allcarries = (allcarries ^ p) | (g << 1);
        return allcarries;
    }

    __device__ __forceinline__ var add_cy()

    __device__ __forceinline__ var mul(const uint254 a, const uint254 b, uint254 &res) {
        uint254 x = a, y = b, z = digit::zero();

        uint254 r_inv;
        if (is_same<params, BN254_MOD_BASE>::value) {
            r_inv = gpu_barretenberg::r_inv_base;
        } else {
            r_inv = gpu_barretenberg::r_inv_scalar;
        }

        uint254 carry {0, 0, 0, 0};
        for (int i = 0; i < LIMBS; i++) {
            uint254 u;
            var tmpi = x[i] * r_inv[i] * y[0];

            digit::mad_lo(u[0], z[0], r_inv[0], tmpi);
            digit::mad_lo(u[1], z[0], r_inv[1], tmpi);
            digit::mad_lo(u[2], z[0], r_inv[2], tmpi);
            digit::mad_lo(u[3], z[0], r_inv[3], tmpi);

            digit::mad_lo_cy(z[0], cy[0], mod[0], u[0], z[0]);
            digit::mad_lo_cy(z[1], cy[1], mod[1], u[1], z[1]);
            digit::mad_lo_cy(z[2], cy[2], mod[2], u[2], z[2]);
            digit::mad_lo_cy(z[3], cy[3], mod[3], u[3], z[3]);

            assert(z[0] == 0);

            z[0] = z[1];
            z[1] = z[2];
            z[2] = z[3];
            z[3] = z[0];

            digit::add_cy(z[0], cy[0], z[0], cy[0]);
            digit::add_cy(z[1], cy[1], z[1], cy[1]);
            digit::add_cy(z[2], cy[2], z[2], cy[2]);
            digit::add_cy(z[3], cy[3], z[3], cy[3]);

            digit::mad_hi_cy(z[0], cy[0], mod[0], u[0], z[0]);
            digit::mad_hi_cy(z[1], cy[1], mod[1], u[1], z[1]);
            digit::mad_hi_cy(z[2], cy[2], mod[2], u[2], z[2]);
            digit::mad_hi_cy(z[3], cy[3], mod[3], u[3], z[3]);
            
            digit::mad_hi_cy(z[0], cy[0], y[0], x[i], z[0]);
            digit::mad_hi_cy(z[1], cy[1], y[1], x[i], z[1]);
            digit::mad_hi_cy(z[2], cy[2], y[2], x[i], z[2]);
            digit::mad_hi_cy(z[3], cy[3], y[3], x[i], z[3]);
        }

        int msb[4] {cy[3], cy[3], cy[3], cy[3]};
        cy[3] = cy[2];
        cy[2] = cy[1];
        cy[1] = cy[0];
        cy[0] = 0;

        // TODO: fixnum:
add_cy

        msb[0] += cy[0];
        msb[1] += cy[1];
        msb[2] += cy[2];
        msb[3] += cy[3];

        assert(msb[0] <= 1);
        assert(msb[1] <= 1);
        assert(msb[2] <= 1);
        assert(msb[3] <= 1);

        uint254 r;
        int br[4];    }

    __device__ __forceinline__var tommy_mul(const uint254 a, const uint254 b, uint254 &res) {
        uint254 mod = params::mod();
        uint254 x = a, y = b, z = digit::zero();
        uint254 tmp;
        uint254 r_inv = gpu_barreten;

        uint64 tmp0 = x[0]*r_inv[0];
        for(int l = 0; l < LIMBS; l++) {
            tmp[l] = x[l]*r_inv[l];
            tmp[l] *= tmp[0];
        }
                
        int cy[LIMBS];
        for(int l = 0; l < LIMBS; l++){
            cy[l] = 0;
        }

        for (int i = 0; i < LIMBS; i++) {
            for(int l = 0; l < LIMBS; l++) {
                u[l] = z[0]*r_inv[l] + tmp[i];
                mad_lo_cy(z[l], cy[l], mod[l], u[l], z[l]);
                mad_lo_cy(z[l], cy[l], m[l], x[i], z[l]);
            }

            assert(z[0] == 0);
            for(int l = 0; l < LIMBS-1; l ++){
                z[l] = z[l+1];
            }
            z[LIMBS-1] = 0;

            for(int l = 0; l<LIMBS; l++){
                add_cy(z[l], cy[l], z[l], cy[l]);
                // z[l] += mod[l]*u[l];
                mad_hi_cy(z[l], cy[l], mod[l], u[l], z[l]);
                mad_hi_cy(z[l], cy[l], y[l], x[i], z[l]);
            }
        } 
        
        // Resolve carries

        uint254 msb;
        for(int l = 0; l < LIMBS; l ++){
            msb[l] = cy[LIMBS-1];
        }
        for(int l = LIMBS-1; l > 0; l++){
            cy[l] = c[l-1]
        }
        cy[0] = 0;

        for(int l = 0; l < LIMBS; l ++){
            // MUST REWRITE add_cy functionality
            // fixnum::add_cy(z[l], cy[l], z[l], cy[l]);
            z[l] += cy[l];
            if((l+1 < LIMBS) && (z[l] < cy[l])) {
                cy[l+1] += 1; // fix this - need to propagate all the existing carries
            }
            
            msb[l] += cy[l];
        }

        uint254 r;
        int br[];
        for(int l = 0; l < LIMBS; l ++){
            // MUST REWRITE sub_br functionality
            fixnum::sub_br(r[l], br[l], z[l], mod[l]);
            
            if(msb[l] || br[l] == 0){
                assert(!msb[l] || msb[l] == br[l]);
                z[l] = r[l];
            }
        }
        return z;
    }

};