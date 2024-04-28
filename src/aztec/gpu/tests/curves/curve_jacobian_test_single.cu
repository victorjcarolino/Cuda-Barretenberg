#include "group_single.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg_single;


static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Mixed Addition Test ---------------------------------------------- */

__global__ void initialize_mixed_add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z) {
    a = { 0x92716caa6cac6d26, 0x1e6e234136736544, 0x1bb04588cde00af0, 0x9a2ac922d97e6f5 };
    b = { 0x9e693aeb52d79d2d, 0xf0c1895a61e5e975, 0x18cd7f5310ced70f, 0xac67920a22939ad };
    c = { 0xfef593c9ce1df132, 0xe0486f801303c27d, 0x9bbd01ab881dc08e, 0x2a589badf38ec0f9 };
    x = { 0xa1ec5d1398660db8, 0x6be3e1f6fd5d8ab1, 0x69173397dd272e11, 0x12575bbfe1198886 };
    y = { 0xcfbfd4441138823e, 0xb5f817e28a1ef904, 0xefb7c5629dcc1c42, 0x1a9ed3d6f846230e };
    expected_x = { 0x2a9d0201fccca20, 0x36f969b294f31776, 0xee5534422a6f646, 0x911dbc6b02310b6 };
    expected_y = { 0x14c30aaeb4f135ef, 0x9c27c128ea2017a1, 0xf9b7d80c8315eabf, 0x35e628df8add760 };
    expected_z = { 0xa43fe96673d10eb3, 0x88fbe6351753d410, 0x45c21cc9d99cb7d, 0x3018020aa6e9ede5 };
}

#define DBG_LIMBS(msg, val) printf(msg " %lx %lx %lx %lx\n", val.limbs[0], val.limbs[1], val.limbs[2], val.limbs[3]);

__global__ void mixed_add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element lhs;
    g1_single::affine_element rhs;
    g1_single::element result;
    g1_single::element expected;
    
    // Calculate global thread ID, and boundry check
    fq_single::to_monty(a, res_x);
    lhs.x = res_x;
    fq_single::to_monty(b, res_y);
    lhs.y = res_y;
    fq_single::to_monty(c, res_z);
    lhs.z = res_z;
    fq_single::to_monty(x, res_x);
    rhs.x = res_x;
    fq_single::to_monty(y, res_y);
    rhs.y = res_y;

    DBG_LIMBS("mixed_add 0 lhs.x", lhs.x.data);
    DBG_LIMBS("mixed_add 0 lhs.y", lhs.y.data);
    DBG_LIMBS("mixed_add 0 lhs.z", lhs.z.data);
    DBG_LIMBS("mixed_add 0 rhs.x", rhs.x.data);
    DBG_LIMBS("mixed_add 0 rhs.y", rhs.y.data);

    // lhs + rhs (affine element + jacobian element)
    g1_single::mixed_add(
        lhs.x.data, lhs.y.data, lhs.z.data, 
        rhs.x.data, rhs.y.data, 
        res_x, res_y, res_z
    );

    DBG_LIMBS("mixed_add 1 x", res_x);
    DBG_LIMBS("mixed_add 1 y", res_y);
    DBG_LIMBS("mixed_add 1 z", res_z);

    // Return results from montgomery form 
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);

    DBG_LIMBS("mixed_add 2 x", res_x);
    DBG_LIMBS("mixed_add 2 y", res_y);
    DBG_LIMBS("mixed_add 2 z", res_z);
}

/* -------------------------- Doubling Test ---------------------------------------------- */

__global__ void initialize_dbl_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z) {
    a = { 0x8d1703aa518d827f, 0xd19cc40779f54f63, 0xabc11ce30d02728c, 0x10938940de3cbeec };
    b = { 0xcf1798994f1258b4, 0x36307a354ad90a25, 0xcd84adb348c63007, 0x6266b85241aff3f };
    c = { 0xe213e18fd2df7044, 0xb2f42355982c5bc8, 0xf65cf5150a3a9da1, 0xc43bde08b03aca2 };
    expected_x = { 0xd5c6473044b2e67c, 0x89b185ea20951f3a, 0x4ac597219cf47467, 0x2d00482f63b12c86 };
    expected_y = { 0x4e7e6c06a87e4314, 0x906a877a71735161, 0xaa7b9893cc370d39, 0x62f206bef795a05 };
    expected_z = { 0x8813bdca7b0b115a, 0x929104dffdfabd22, 0x3fff575136879112, 0x18a299c1f683bdca };
}

__global__ void dbl_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element lhs;
    g1_single::element result;
    g1_single::element expected;

    fq_single::to_monty(a, res_x);
    lhs.x = res_x;
    fq_single::to_monty(b, res_y);
    lhs.y = res_y;
    fq_single::to_monty(c, res_z);
    lhs.z = res_z;

    // lhs.doubling
    g1_single::doubling(
        lhs.x.data, lhs.y.data, lhs.z.data, 
        res_x, res_y, res_z
    );

    // (lhs.doubling).doubling
    g1_single::doubling(
        res_x, res_y, res_z, 
        res_x, res_y, res_z
    );

    // ((lhs.doubling).doubling).doubling
    g1_single::doubling(
        res_x, res_y, res_z, 
        res_x, res_y, res_z
    );

    // Return results from montgomery form
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);
}

// /* -------------------------- Addition Test ---------------------------------------------- */

__global__ void initialize_add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z) {
    a = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    b = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    c = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    x = { 0xafdb8a15c98bf74c, 0xac54df622a8d991a, 0xc6e5ae1f3dad4ec8, 0x1bd3fb4a59e19b52 };
    y = { 0x21b3bb529bec20c0, 0xaabd496406ffb8c1, 0xcd3526c26ac5bdcb, 0x187ada6b8693c184 };
    z = { 0xffcd440a228ed652, 0x8a795c8f234145f1, 0xd5279cdbabb05b95, 0xbdf19ba16fc607a };
    expected_x = { 0x18764da36aa4cd81, 0xd15388d1fea9f3d3, 0xeb7c437de4bbd748, 0x2f09b712adf6f18f };
    expected_y = { 0x50c5f3cab191498c, 0xe50aa3ce802ea3b5, 0xd9d6125b82ebeff8, 0x27e91ba0686e54fe };
    expected_z = { 0xe4b81ef75fedf95, 0xf608edef14913c75, 0xfd9e178143224c96, 0xa8ae44990c8accd };
}

__global__ void add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element lhs;
    g1_single::element rhs;
    g1_single::element result;
    g1_single::element expected;

    fq_single::to_monty(a, res_x);
    lhs.x = res_x;
    fq_single::to_monty(b, res_y);
    lhs.y = res_y;
    fq_single::to_monty(c, res_z);
    lhs.z = res_z;
    fq_single::to_monty(x, res_x);
    rhs.x = res_x;
    fq_single::to_monty(y, res_y);
    rhs.y = res_y;
    fq_single::to_monty(z, res_z);
    rhs.z = res_z;      

    // lhs + rhs (affine element + affine element)
    g1_single::add(
         lhs.x.data, lhs.y.data, lhs.z.data,
         rhs.x.data, rhs.y.data, rhs.z.data,
         res_x, res_y, res_z
    );
        
    // Return results from montgomery form
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);
}

/* -------------------------- Add Exception Test ---------------------------------------------- */

__global__ void initialize_add_exception_test_dbl
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z) {
    a = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    b = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    c = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    x = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    y = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    z = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
}

__global__ void add_exception_test_dbl
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element lhs;
    g1_single::element rhs;
    g1_single::element result;
    g1_single::element expected;

    lhs.x = fq_single::load(a, expected_x);
    lhs.y = fq_single::load(b, expected_x);
    lhs.z = fq_single::load(c, expected_x);
    rhs.x = fq_single::load(x, expected_x);
    rhs.y = fq_single::load(y, expected_x);
    rhs.z = fq_single::load(z, expected_x);

    // lhs + rhs
    g1_single::add(
        lhs.x.data, lhs.y.data, lhs.z.data,
        rhs.x.data, rhs.y.data, rhs.z.data,
        res_x, res_y, res_z
    );

    // Tal comment: Temporarily handle case where P = Q -- NEED TO MOVE TO 'group.cu' file //Tommy - this should be fixed in group_single.cu
    if (fq_single::is_zero(res_x) && fq_single::is_zero(res_y) && fq_single::is_zero(res_z)) {
        g1_single::doubling(
            lhs.x.data, lhs.y.data, lhs.z.data, 
            res_x, res_y, res_z
        );
    }

    // lhs.doubling
    g1_single::doubling(
        lhs.x.data, lhs.y.data, lhs.z.data, 
        expected_x, expected_y, expected_z
    );

    // Transform results from montgomery form 
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);

    // Transform results from montgomery form 
    fq_single::from_monty(expected_x, expected_x);
    fq_single::from_monty(expected_y, expected_y);
    fq_single::from_monty(expected_z, expected_z);

    // EXPECT(lsh + rhs == lhs.doubling);
}

/* -------------------------- Add Double Consistency Test ---------------------------------------------- */

__global__ void initialize_add_dbl_consistency
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z) {
    a = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    b = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    c = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    x = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    y = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    z = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
}

__global__ void add_dbl_consistency
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element a_element;
    g1_single::element b_element;
    g1_single::element c_element;
    g1_single::element d_element;
    g1_single::element add_result;
    g1_single::element dbl_result;

    

    // Calculate global thread ID, and boundry check
    //int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    //if (tid < LIMBS) {
        a_element.x = fq_single::load(a, res_x);
        a_element.y = fq_single::load(b, res_x);
        a_element.z = fq_single::load(c, res_x);
        b_element.x = fq_single::load(x, res_x);
        b_element.y = fq_single::load(y, res_x);
        b_element.z = fq_single::load(z, res_x);

        // c = a + b
        g1_single::add(
            a_element.x.data, a_element.y.data, a_element.z.data, 
            b_element.x.data, b_element.y.data, b_element.z.data, 
            c_element.x.data, c_element.y.data, c_element.z.data
        ); 
        
        // b = -b
        fq_single::neg(b_element.y.data, b_element.y.data);                                                                                                                                                      
        
        // d = a + b
        g1_single::add(
            b_element.x.data, b_element.y.data, b_element.z.data, 
            a_element.x.data, a_element.y.data, a_element.z.data, 
            d_element.x.data, d_element.y.data, d_element.z.data
        );
       
        // result = c + d
        g1_single::add(
            c_element.x.data, c_element.y.data, c_element.z.data, 
            d_element.x.data, d_element.y.data, d_element.z.data, 
            res_x, res_y, res_z
        );

         //   Tommy - this is no longer needed
        //Temporarily handle case where P = Q -- NEED TO MOVE TO 'group.cu' file
        //if (fq_single::is_zero(res_x) && fq_single::is_zero(res_y) && fq_single::is_zero(res_z)) {
        //    g1_single::doubling(
        //        a_element.x.data, a_element.y.data, a_element.z.data, 
        //        res_x, res_y, res_z
        //    );
        //}

        // a.doubling
        g1_single::doubling(
            a_element.x.data, a_element.y.data, a_element.z.data, 
            expected_x, expected_y, expected_z
        );
         
        // Transform results from montgomery form 
        fq_single::from_monty(res_x, res_x);
        fq_single::from_monty(res_y, res_y);
        fq_single::from_monty(res_z, res_z);
        
        // Transform results from montgomery form 
        fq_single::from_monty(expected_x, expected_x);
        fq_single::from_monty(expected_y, expected_y);
        fq_single::from_monty(expected_z, expected_z);

        // EXPECT (c + d == a.doubling);
    //}
}

/* -------------------------- Add Double Consistency Repeated Test ---------------------------------------------- */

__global__ void initialize_add_dbl_consistency_repeated
(uint254 &a, uint254 &b, uint254 &c) {
    a = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    b = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    c = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };

}

__global__ void add_dbl_consistency_repeated
(uint254 a, uint254 b, uint254 c, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    g1_single::element a_element;
    g1_single::element b_element;
    g1_single::element c_element;
    g1_single::element d_element;
    g1_single::element e_element;
    g1_single::element result;
    g1_single::element expected;

    // Calculate global thread ID, and boundry check
    //int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    //if (tid < LIMBS) {
        a_element.x = fq_single::load(a, res_x);
        a_element.y = fq_single::load(b, res_x);
        a_element.z = fq_single::load(c, res_x);

        // b = 2a
        g1_single::doubling(
            a_element.x.data, a_element.y.data, a_element.z.data, 
            b_element.x.data, b_element.y.data, b_element.z.data
        );

        // c = 4a
        g1_single::doubling(
            b_element.x.data, b_element.y.data, b_element.z.data, 
            c_element.x.data, c_element.y.data, c_element.z.data
        );
         
        // d = 3a
        g1_single::add(
            a_element.x.data, a_element.y.data, a_element.z.data, 
            b_element.x.data, b_element.y.data, b_element.z.data, 
            d_element.x.data, d_element.y.data, d_element.z.data
        ); 

        // e = 5a
        g1_single::add(
            a_element.x.data, a_element.y.data, a_element.z.data, 
            c_element.x.data, c_element.y.data, c_element.z.data, 
            e_element.x.data, e_element.y.data, e_element.z.data
        ); 
  
        // result = 8a
        g1_single::add(
            d_element.x.data, d_element.y.data, d_element.z.data, 
            e_element.x.data, e_element.y.data, e_element.z.data, 
            res_x, res_y, res_z
        );

        // c.doubling
        g1_single::doubling(
            c_element.x.data, c_element.y.data, c_element.z.data, 
            expected_x, expected_y, expected_z
        );

        // Transform results from montgomery form 
        fq_single::from_monty(res_x, res_x);
        fq_single::from_monty(res_y, res_y);
        fq_single::from_monty(res_z, res_z);

        // Transform results from montgomery form 
        fq_single::from_monty(expected_x, expected_x);
        fq_single::from_monty(expected_y, expected_y);
        fq_single::from_monty(expected_z, expected_z);

        // EXPECT (d + e == c.doubling)
    //}
}

/* -------------------------- Group Exponentiation Check Against Constants Test ---------------------------------------------- */

__global__ void initialize_group_exponentiation
(uint254 &a, uint254 &b, uint254 &c, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z) {
    a = { 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    b = { 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    c = { 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    expected_x = { 0xC22BA855EE138794, 0xA61591A7E7FD82BF, 0xE156E7E491B4B7E2, 0x2F4620C8373C106A };
    expected_y = { 0xFAFBA721679C418, 0xE5491810D637BB55, 0x64B6FAD0A59D97B2, 0x111DA26AEEE41706 };
    expected_z = { 0x59F11DAE3A07BF31, 0xDB2756DB66333FB, 0x34F2D97DAD44161, 0xD1A485A89C277DA };
}

__global__ void group_exponentiation(uint254 &a, uint254 &b, uint254 &c, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //g1_gpu::element one; 
    g1_single::element one;
    //g1_gpu::element R;
    g1_single::element R;
    //g1_gpu::element Q;
    g1_single::element Q;

     //Fix this constructor syntax
    //fr_gpu exponent{ 0xb67299b792199cf0, 0xc1da7df1e7e12768, 0x692e427911532edf, 0x13dd85e87dc89978 };
    fr_single exponent(uint254{0xb67299b792199cf0, 0xc1da7df1e7e12768, 0x692e427911532edf, 0x13dd85e87dc89978});

     //fq_gpu::load(gpu_barretenberg::one_x_bn_254[tid], one.x.data[tid]);
     //fq_gpu::load(gpu_barretenberg::one_y_bn_254[tid], one.y.data[tid]);
     //fq_gpu::load(fq_gpu::one().data[tid], one.z.data[tid]);
     // Tommy - THIS NEEDS A FIX. one_x_bn_254 is type var[], one.x.data is type uint254. Same problem for y coordinate
    fq_single::load(uint254{gpu_barretenberg_single::one_x_bn_254[0],gpu_barretenberg_single::one_x_bn_254[1],gpu_barretenberg_single::one_x_bn_254[2],gpu_barretenberg_single::one_x_bn_254[3]}, one.x.data);
    fq_single::load(uint254{gpu_barretenberg_single::one_y_bn_254[0],gpu_barretenberg_single::one_y_bn_254[1],gpu_barretenberg_single::one_y_bn_254[2],gpu_barretenberg_single::one_y_bn_254[3]}, one.y.data);
    fq_single::load(fq_single::one().data, one.z.data);

    //if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_single::load(uint254{0,0,0,0}, R.x.data); 
        fq_single::load(uint254{0,0,0,0}, R.y.data); 
        fq_single::load(uint254{0,0,0,0}, R.z.data); 

        fq_single::load(one.x.data, Q.x.data);
        fq_single::load(one.y.data, Q.y.data);
        fq_single::load(one.z.data, Q.z.data);

        // Loop for each limb starting with the last limb
        for (int j = 3; j >= 0; j--) {
            // Loop for each bit of scalar
            for (int i = 64; i >= 0; i--) {
                // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                // and extracting the i-th bit of scalar in limb.
                if (((exponent.data.limbs[j] >> i) & 1) ? 1 : 0)
                    g1_single::add(
                        R.x.data, R.y.data, R.z.data, 
                        Q.x.data, Q.y.data, Q.z.data, 
                        R.x.data, R.y.data, R.z.data
                    );
                printf("\nEXPONENT i %d j %d PART 1\n", i, j);
                DBG_LIMBS("  group exp R.x", R.x.data);
                DBG_LIMBS("  group exp R.y", R.y.data);
                DBG_LIMBS("  group exp R.z", R.z.data);
                DBG_LIMBS("  group exp Q.x", Q.x.data);
                DBG_LIMBS("  group exp Q.y", Q.y.data);
                DBG_LIMBS("  group exp Q.z", Q.z.data);
                if (i != 0) 
                    g1_single::doubling(
                        R.x.data, R.y.data, R.z.data, 
                        R.x.data, R.y.data, R.z.data
                    );
                printf("\nEXPONENT i %d j %d PART 2\n", i, j);
                DBG_LIMBS("  group exp R.x", R.x.data);
                DBG_LIMBS("  group exp R.y", R.y.data);
                DBG_LIMBS("  group exp R.z", R.z.data);
                DBG_LIMBS("  group exp Q.x", Q.x.data);
                DBG_LIMBS("  group exp Q.y", Q.y.data);
                DBG_LIMBS("  group exp Q.z", Q.z.data);
            }
        }
    //}

    // Store the final value of R into the result array for this limb
    fq_single::load(R.x.data, res_x);
    fq_single::load(R.y.data, res_y);
    fq_single::load(R.z.data, res_z);

    // Convert back from montgomery form
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);
}

// /* -------------------------- Operator Ordering Test ---------------------------------------------- */

// __global__ void initialize_operator_ordering(uint254 &a, uint254 &b, uint254 &c, uint254 &d) {
//          // Fix this conversion (for a & b) - assigning uint254 to be a var[]
//         a = gpu_barretenberg_single::one_x_bn_254;
//         b = gpu_barretenberg_single::one_y_bn_254;
//         c = fq_single::one();
//         d = { 0xcfbfd4441138823e, 0xb5f817e28a1ef904, 0xefb7c5629dcc1c42, 0x1a9ed3d6f846230e };
// }

// __global__ void operator_ordering(uint254 a, uint254 b, uint254 c, uint254 d, uint254 &res_x, uint254 &res_y, uint254 &res_z) {
//     //int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     g1_single::element b_new;
//     g1_single::element c_new;
//     g1_single::element d_new;

//     // Copy a into b
//     fq_single::load(a, b_new.x.data);
//     fq_single::load(b, b_new.y.data);
//     fq_single::load(c, b_new.z.data);

//     // c = a + b
//     g1_single::add(
//         a, b, c, 
//         b_new.x.data, b_new.y.data, b_new.z.data, 
//         c_new.x.data, c_new.y.data, c_new.z.data
//     );

//      // Tommy - this is_zero() check should be redundant; already is included in group_single.cu
//     // Double is c == 0
//     //if (fq_single::is_zero(c_new.x.data) && fq_single::is_zero(c_new.y.data) && fq_single::is_zero(c_new.z.data)) {
//     //    g1_single::doubling(
//     //        a, b, c, 
//     //        c_new.x.data, c_new.y.data, c_new.z.data
//     //    );
//     //}

//     // d = b + a
//     g1_single::add(
//         b_new.x.data, b_new.y.data, b_new.z.data, 
//         a, b, c, 
//         d_new.x.data, d_new.y.data, d_new.z.data
//     );

//      // Tommy - this is_zero() check should be redundant; already is included in group_single.cu
//     // Double is d == 0
//     //if (fq_single::is_zero(d_new.x.data) && fq_single::is_zero(d_new.y.data) && fq_single::is_zero(d_new.z.data)) {
//     //    g1_single::doubling(
//     //        b_new.x.data, b_new.y.data, b_new.z.data, 
//     //        d_new.x.data, d_new.y.data, d_new.z.data
//     //    );
//     //}

//     // Return final result. Expect c == d
//     fq_single::load(d_new.x.data, res_x);
//     fq_single::load(d_new.y.data, res_y);
//     fq_single::load(d_new.z.data, res_z);

//     fq_single::from_monty(res_x, res_x);
//     fq_single::from_monty(res_y, res_y);
//     fq_single::from_monty(res_z, res_z);
// }

// /* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(uint254 *expected, uint254 *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print statements
    printf("expected[0] is: %lx\n", expected->limbs[0]);
    printf("expected[1] is: %lx\n", expected->limbs[1]);
    printf("expected[2] is: %lx\n", expected->limbs[2]);
    printf("expected[3] is: %lx\n", expected->limbs[3]);
    printf("result[0] is: %lx\n", result->limbs[0]);
    printf("result[1] is: %lx\n", result->limbs[1]);
    printf("result[2] is: %lx\n", result->limbs[2]);
    printf("result[3] is: %lx\n", result->limbs[3]);

    // Assert clause
    assert(expected->limbs[0] == result->limbs[0]);
    assert(expected->limbs[1] == result->limbs[1]);
    assert(expected->limbs[2] == result->limbs[2]);
    assert(expected->limbs[3] == result->limbs[3]);
}

void execute_kernels
(uint254 *a, uint254 *b, uint254 *c, uint254 *x, uint254 *y, uint254 *z, uint254 *expected_x, uint254 *expected_y, uint254 *expected_z, uint254 *res_x, uint254 *res_y, uint254 *res_z) {
    // Mixed Addition Test
    initialize_mixed_add_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z);
    cudaDeviceSynchronize();
    mixed_add_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *res_x, *res_y, *res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Doubling Test
    initialize_dbl_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z);
    dbl_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *res_x, *res_y, *res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Addition Test
    initialize_add_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z);
    add_check_against_constants<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *res_x, *res_y, *res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Add Exception Test
    initialize_add_exception_test_dbl<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z);
    add_exception_test_dbl<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // // Add Double Consistency Test
    // initialize_add_dbl_consistency<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z);
    // add_dbl_consistency<<<BLOCKS, 1>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z);
    // assert_checks(expected_x, res_x);
    // assert_checks(expected_y, res_y);
    // assert_checks(expected_z, res_z);

    // Add Double Consistency Repeated Test
    initialize_add_dbl_consistency_repeated<<<BLOCKS, 1>>>(*a, *b, *c);
    add_dbl_consistency_repeated<<<BLOCKS, 1>>>(*a, *b, *c, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z);

    // Group Exponentiation Consistency Test
    initialize_group_exponentiation<<<BLOCKS, 1>>>(*a, *b, *c, *expected_x, *expected_y, *expected_z);
    group_exponentiation<<<BLOCKS, 1>>>(*a, *b, *c, *res_x, *res_y, *res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // // Operator Ordering Test
    // initialize_operator_ordering<<<BLOCKS, 1>>>(*a, *b, *c, *x);
    // operator_ordering<<<BLOCKS, 1>>>(*a, *b, *c, *res_x, *res_y, *res_z);
    // assert_checks(expected_x, res_x);
    // assert_checks(expected_y, res_y);
    // assert_checks(expected_z, res_z);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to 'uint64_t' type
    uint254 *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, res_x, res_y, res_z);

    // Successfull execution of unit tests
    cout << "******* All 'g1_gpu BN-254 Curve' unit tests passed! **********" << endl;

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(expected_x);
    cudaFree(expected_y);
    cudaFree(expected_z);
    cudaFree(res_x);
    cudaFree(res_y);
    cudaFree(res_z);

    cout << "Completed sucessfully!" << endl;

    return 0;
}
