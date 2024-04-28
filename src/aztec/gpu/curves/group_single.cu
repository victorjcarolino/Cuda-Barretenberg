#include "group_single.cuh"

using namespace std;
// using namespace gpu_barretenberg;
using namespace gpu_barretenberg_single;

/* -------------------------- Affine and Jacobian Coordinate Operations ---------------------------------------------- */

template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::load_affine(affine_element &X, affine_element &result) {
    fq_single::load(X.x, result.x);        
        
    fq_single::load(X.y, result.y);       
}

template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::load_jacobian(element &X, element &result) {
    fq_single::load(X.x, result.x);          
        
    fq_single::load(X.y, result.y);      

    fq_single::load(X.z, result.z);      
}

/**
 * Elliptic curve algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-zadd-2007-m
 */
template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::mixed_add(uint254 X, uint254 Y, uint254 Z, uint254 A, uint254 B, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    uint254 z1z1, u2, s2, h, hh, i, j, r, v, t0, t1;

     // X Element
    fq_single::square(Z, z1z1);   
    fq_single::mul(A, z1z1, u2); 
    fq_single::mul(B, Z, s2);
    fq_single::mul(s2, z1z1, s2);

    fq_single::sub(u2, X, h);   
    fq_single::square(h, hh);    
    fq_single::add(hh, hh, i);     
    fq_single::add(i, i, i);      
    fq_single::mul(i, h, j);      
    fq_single::sub(s2, Y, r);      
    fq_single::add(r, r, r);      
    fq_single::mul(X, i, v);      
    fq_single::square(r, t0);     
    fq_single::add(v, v, t1);
    fq_single::sub(t0, j, t0);    
    fq_single::sub(t0, t1, res_x); 

    // Y Element
    fq_single::sub(v, res_x, t0);  
    fq_single::mul(Y, j, t1);     
    fq_single::add(t1, t1, t1);
    fq_single::mul(t0, r, t0);     
    fq_single::sub(t0, t1, res_y);

    // Z Element
    fq_single::add(Z, h, t0);      
    fq_single::square(t0, t0);     
    fq_single::sub(t0, z1z1, t0);  
    fq_single::sub(t0, hh, res_z); 
}

template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::doubling(uint254 X, uint254 Y, uint254 Z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    uint254 T0, T1, T2, T3;

    // Check P == 0
    if (fq_single::is_zero(Z)) {
        fq_single::zero();
    }

    // X Element
    fq_single::square(X, T0);              // T0 = x*x
    fq_single::square(Y, T1);              // T1 = y*y
    fq_single::square(T1, T2);             // T2 = T1*T1 = y*y*y*y
    fq_single::add(T1, X, T1);             // T1 = T1+x = x + y*y
    fq_single::square(T1, T1);             // T1 = T1*T1
    fq_single::add(T0, T2, T3);            // T3 = T0 +T2 = x*x + y*y*y*y
    fq_single::sub(T1, T3, T1);            // T1 = T1-T3 = x*x + y*y*y*y + 2*x*x*y*y*y*y - x*x - y*y*y*y = 2*x*x*y*y*y*y = 2*S
    fq_single::add(T1, T1, T1);            // T1 = 2T1 = 4*S
    fq_single::add(T0, T0, T3);            // T3 = 2T0
    fq_single::add(T3, T0, T3);            // T3 = T3+T0 = 3T0
    fq_single::add(T1, T1, T0);            // T0 = 2T1
    fq_single::square(T3, X);              // X = T3*T3
    fq_single::sub(X, T0, X);              // X = X-T0 = X-2T1
    fq_single::load(X, res_x);             // X = X-T0 = X-2T1

    // Z Element
    fq_single::add(Z, Z, Z);               // Z2 = 2Z
    fq_single::mul(Z, Y, res_z);           // Z2 = Z*Y = 2*Y*Z

    // Y Element
    fq_single::add(T2, T2, T2);            // T2 = T2+T2 = 2T2
    fq_single::add(T2, T2, T2);            // T2 = T2+T2 = 4T2
    fq_single::add(T2, T2, T2);            // T2 = T2+T2 = 8T2
    fq_single::sub(T1, X, Y);              // Y = T1-X
    fq_single::mul(Y, T3, Y);              // Y = Y*T3
    fq_single::sub(Y, T2, res_y);          // Y = Y - T2
}

/**
 * Jacobian addition has cost 16T multiplications
 */
template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::add(uint254 X1, uint254 Y1, uint254 Z1, uint254 X2, uint254 Y2, uint254 Z2, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    uint254 Z1Z1, Z2Z2, U1, U2, S1, S2, F, H, I, J;

    // Check P == 0 or Q == 0
    if (fq_single::is_zero(Z1)) {
        res_x = X2;
        res_y = Y2;
        res_z = Z2;
        return;
    } else if (fq_single::is_zero(Z2)) {
        res_x = X1;
        res_y = Y1;
        res_z = Z1;
        return;
    }

    // X Element
    fq_single::square(Z1, Z1Z1);            // Z1Z1 = Z1^2
    fq_single::square(Z2, Z2Z2);            // Z1Z1 = Z2^2
    fq_single::mul(Z1Z1, Z1, S2);           // S2 = Z1Z1 * Z1 
    fq_single::mul(Z1Z1, X2, U2);           // U2 = Z1Z1 * X2
    fq_single::mul(S2, Y2, S2);             // S2 = S2 * Y2
    fq_single::mul(Z2Z2, X1, U1);           // U1 = Z2Z2 * X1
    fq_single::mul(Z2Z2, Z2, S1);           // S1 = Z2Z2 * Z2
    fq_single::mul(S1, Y1, S1);             // S1 = S1 * Y1
    fq_single::sub(S2, S1, F);              // F = S2 - S1
    fq_single::sub(U2, U1, H);              // H = U2 - U1
    fq_single::add(F, F, F);                // F = F + F
    fq_single::add(H, H, I);                // I = H + H
    fq_single::square(I, I);                // I = I * H
    fq_single::mul(H, I, J);                // J = H * H
    fq_single::mul(U1, I, U1);              // U1 = U1 * I
    fq_single::add(U1, U1, U2);             // U2 = U1 + U1
    fq_single::add(U2, J, U2);              // U2 = U2 * J
    fq_single::square(F, X1);               // X1 = F^2
    fq_single::sub(X1, U2, X1);             // X1 = X1 - U2
    fq_single::load(X1, res_x);             // res_x = X1

    // Y Element
    fq_single::mul(J, S1, J);               // J = J * S1
    fq_single::add(J, J, J);                // J = J + J
    fq_single::sub(U1, X1, Y1);             // Y1 = U1 - X1
    fq_single::mul(Y1, F, Y1);              // Y1 = Y1 + F
    fq_single::sub(Y1, J, Y1);              // Y1 = Y1 - J
    fq_single::load(Y1, res_y);             // res_y = Y1

    // Z Element
    fq_single::add(Z1, Z2, Z1);             // Z1 = Z1 + Z2
    fq_single::add(Z1Z1, Z2Z2, Z1Z1);       // Z1Z1 = Z2Z2 + Z1Z1
    fq_single::square(Z1, Z1);              // Z1 = Z1^2
    fq_single::sub(Z1, Z1Z1, Z1);           // Z1 = Z1 - Z1Z1
    fq_single::mul(Z1, H, Z1);              // Z1 = Z1 * H
    fq_single::load(Z1, res_z);             // res_z = Z1;

    
    // Tommy: include the Point1 == Point2 check inside the add() function
    if (fq_single::is_zero(res_x) && fq_single::is_zero(res_y) && fq_single::is_zero(res_z)) {
        g1_single::doubling(
            X1, Y1, Z1, 
            res_x, res_y, res_z
        );
    }
}

/* -------------------------- Projective Coordinate Operations ---------------------------------------------- */

template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::load_projective(projective_element &X, projective_element &result) {
    fq_single::load(X.x, result.x);       
        
    fq_single::load(X.x, result.y);      

    fq_single::load(X.z, result.z);      
}

template <class fq_single, class fr_single> 
projective_element_single<fq_single, fr_single> group_gpu_single<fq_single, fr_single>::from_affine(const affine_element &other) {
    projective_element projective;
    projective.x = other.x;
    projective.y = other.y;
    return { projective.x, projective.y, fq_single::one() };
}

/**
 * Projective addition has cost 14T multiplications
 */
template <class fq_single, class fr_single> 
__device__ __forceinline__ void group_gpu_single<fq_single, fr_single>::add_projective(
uint254 X1, uint254 Y1, uint254 Z1, uint254 X2, uint254 Y2, uint254 Z2, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    uint254 t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10;
    uint254 t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21;
    uint254 t22, t23, t24, t25, t26, t27, t28, t29, t30, t31;
    uint254 X3, Y3, Z3;

    fq_single::mul(X1, X2, t00);                                       // t00 ← X1 · X2
    fq_single::mul(Y1, Y2, t01);                                       // t01 ← Y1 · Y2
    fq_single::mul(Z1, Z2, t02);                                       // t02 ← Z1 · Z2
    fq_single::add(X1, Y1, t03);                                       // t03 ← X1 + Y1
    fq_single::add(X2, Y2, t04);                                       // t04 ← X2 + Y2
    fq_single::mul(t03, t04, t03);                                     // t03 ← t03 + t04
    fq_single::add(t00, t01, t04);                                     // t04 ← t00 + t01
    fq_single::sub(t03, t04, t03);                                     // t03 ← t03 - t04
    fq_single::add(X1, Z1, t04);                                       // t04 ← X1 + Z1
    fq_single::add(X2, Z2, t05);                                       // t05 ← X2 + Z2
    fq_single::mul(t04, t05, t04);                                     // t04 ← t04 * t05
    fq_single::add(t00, t02, t05);                                     // t05 ← t00 + t02
    fq_single::sub(t04, t05, t04);                                     // t04 ← t04 - t05
    fq_single::add(Y1, Z1, t05);                                       // t05 ← Y1 + Z1
    fq_single::add(Y2, Z2, X3);                                        // X3 ← Y2 + Z23
    fq_single::mul(t05, X3, t05);                                      // t05 ← t05 * X3
    fq_single::add(t01, t02, X3);                                      // X3 ← t01 + t02
    fq_single::sub(t05, X3, t05);                                      // t05 ← t05 - X3
    fq_single::mul(0, t04, Z3);                                        // Z3 ← a * t04
    fq_single::mul({3 * gpu_barretenberg_single::b, 3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b}, t02, X3);                  // X3 ← b3 * t02 
    fq_single::add(X3, Z3, Z3);                                        // Z3 ← X3 + Z3
    fq_single::sub(t01, Z3, X3);                                       // X3 ← t01 - Z3
    fq_single::add(t01, Z3, Z3);                                       // Z3 ← t01 + Z3
    fq_single::mul(X3, Z3, Y3);                                        // Y3 ← X3 * Z3
    fq_single::add(t00, t00, t01);                                     // t01 ← t00 + t00
    fq_single::add(t01, t00, t01);                                     // t01 ← t01 + t00
    fq_single::mul(0, t02, t02);                                       // t02 ← a * t02
    fq_single::mul({3 * gpu_barretenberg_single::b, 3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b,3 * gpu_barretenberg_single::b}, t04, t04);                 // t04 ← b3 * t04 
    fq_single::add(t01, t02, t01);                                     // t01 ← t01 + t02
    fq_single::sub(t00, t02, t02);                                     // t02 ← t00 - t02
    fq_single::mul(0, t02, t02);                                       // t02 ← a * t02
    fq_single::add(t04, t02, t04);                                     // t04 ← t04 + t02
    fq_single::mul(t01, t04, t00);                                     // t00 ← t01 * t04
    fq_single::add(Y3, t00, Y3);                                       // Y3 ← Y3 + t00
    fq_single::mul(t05, t04, t00);                                     // t00 ← t05 * t04
    fq_single::mul(t03, X3, X3);                                       // X3 ← t03 * X3
    fq_single::sub(X3, t00, X3);                                       // X3 ← X3 - t00
    fq_single::mul(t03, t01, t00);                                     // t00 ← t03 * t01
    fq_single::mul(t05, Z3, Z3);                                       // Z3 ← t05 * Z3
    fq_single::add(Z3, t00, Z3);                                       // Z3 ← Z3 + t00

    fq_single::load(X3, res_x);  
    fq_single::load(Y3, res_y);  
    fq_single::load(Z3, res_z);  
}