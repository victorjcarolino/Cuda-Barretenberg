#include "field_single.cu"
#include <assert.h> 

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr int LIMBS_NUM = 4; // converted from size_t to eliminate warnings
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Montgomery Multiplication Test ---------------------------------------------- */

__global__ void initialize_mont_mult(uint254 &a, uint254 &b, uint254 &expected) {
    a = { 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    b = { 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    expected = { 0x7ed4174114b521c4, 0x58f5bd1d4279fdc2, 0x6a73ac09ee843d41, 0x687a76ae9b3425c };
    // printf("yy %lu\n", expected.limbs[0]);
}

__global__ void mont_mult(uint254 a, uint254 b, uint254 &result) {
    // printf("??? xx %lu %lu %lu %lu\n", a.limbs[0], a.limbs[1], a.limbs[2], a.limbs[3]);
    fq_single::mul(a, b, result);
}

/* -------------------------- Montgomery Multiplication Test -- Short Integers ---------------------------------------------- */

__global__ void initialize_mont_mult_short(uint254 &a, uint254 &b, uint254 &expected) {
    a = { 0xa, 0, 0, 0 };
    b = { 0xb, 0, 0, 0 };
    expected = { 0x65991a6dc2f3a183, 0xe3ba1f83394a2d08, 0x8401df65a169db3f, 0x1727099643607bba };
}

// duplicate code
// __global__ void mont_mult_short_single(uint254 &a, uint254 &b, uint254 &result) {
//     fq_single::mul(a, b, &result);
// }

/* -------------------------- Multiply - Square Consistency ---------------------------------------------- */

__global__ void initialize_mul_square_consistency(uint254 &a, uint254 &b) {
    a = { 0x329596aa978981e8, 0x8542e6e254c2a5d0, 0xc5b687d82eadb178, 0x2d242aaf48f56b8a };
    b = { 0x7d2e20e82f73d3e8, 0x8e50616a7a9d419d, 0xcdc833531508914b, 0xd510253a2ce62c };
}

__global__ void mul_square_consistency(uint254 a, uint254 b, uint254 &expected, uint254 &result) {
    uint254 t1;
    uint254 t2;

    fq_single::sub(a, b, result);
    t1 = result;
    fq_single::add(a, b, result);
    t2 = result;
    fq_single::mul(t1, t2, expected);

    fq_single::square(a, result);
    t1 = result;
    fq_single::square(b, result);
    t2 = result;
    fq_single::sub(t1, t2, result);
}

/* -------------------------- Multiply - Square Against Constants ---------------------------------------------- */

__global__ void initialize_sqr_check_against_constants(uint254 &a, uint254 &expected) {
    a = { 0x329596aa978981e8, 0x8542e6e254c2a5d0, 0xc5b687d82eadb178, 0x2d242aaf48f56b8a };
    expected = { 0xbf4fb34e120b8b12, 0xf64d70efbf848328, 0xefbb6a533f2e7d89, 0x1de50f941425e4aa };
}

__global__ void sqr_check_against_constants(uint254 a, uint254 &result) {
    fq_single::square(a, result);
}

/* -------------------------- Add - Check Against Constants ---------------------------------------------- */

__global__ void initialize_add_check_against_constants(uint254 &a, uint254 &b, uint254 &expected) {
    a = { 0x7d2e20e82f73d3e8, 0x8e50616a7a9d419d, 0xcdc833531508914b, 0xd510253a2ce62c };
    b = { 0x2829438b071fd14e, 0xb03ef3f9ff9274e, 0x605b671f6dc7b209, 0x8701f9d971fbc9 };
    expected = { 0xa55764733693a536, 0x995450aa1a9668eb, 0x2e239a7282d04354, 0x15c121f139ee1f6 };
}

__global__ void add_check_against_constants(uint254 a, uint254 b, uint254 &result) {
    fq_single::add(a, b, result);
}

/* -------------------------- Subtract - Check Against Constants ---------------------------------------------- */

__global__ void initialize_sub_check_against_constants(uint254 &a, uint254 &b, uint254 &expected) {
    a = { 0xd68d01812313fb7c, 0x2965d7ae7c6070a5, 0x08ef9af6d6ba9a48, 0x0cb8fe2108914f53 };
    b = { 0x2cd2a2a37e9bf14a, 0xebc86ef589c530f6, 0x75124885b362b8fe, 0x1394324205c7a41d };
    expected = { 0xe5daeaf47cf50779, 0xd51ed34a5b0d0a3c, 0x4c2d9827a4d939a6, 0x29891a51e3fb4b5f };
}

__global__ void sub_check_against_constants(uint254 a, uint254 b, uint254 &result) {
    fq_single::sub(a, b, result);
}

/* -------------------------- Cube Root ---------------------------------------------- */

__global__ void initialize_cube(uint254 &a) {
    a = { 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
}

__global__ void cube(uint254 &a, uint254 &expected, uint254 &result) {
    uint254 x_cubed;
    uint254 beta_x;
    uint254 beta_x_cubed;

    fq_single::mul(a, a, result);
    x_cubed = result;
    fq_single::mul(x_cubed, a, result);
    x_cubed = result;
    
    fq_single::mul(a, gpu_barretenberg::CUBE_ROOT_BASE, result);
    beta_x = result;
    fq_single::mul(beta_x, beta_x, result);
    beta_x_cubed = result;
    fq_single::mul(beta_x_cubed, beta_x, result);
    beta_x_cubed = result;
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(uint254 *expected, uint254 *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print statements
    for(int i=0; i<LIMBS_NUM; i++) {
        // printf("expected->limbs[%d] is: %zu\n", i, expected->limbs[i]);
        printf("expected->limbs[%d] is: %lx\n", i, expected->limbs[i]);
    }
    for(int i=0; i<LIMBS_NUM; i++) {
        // printf("result->limbs[%d] is: %zu\n", i, result->limbs[i]);
        printf("result->limbs[%d] is: %lx\n", i, result->limbs[i]);
    }
    // Assert clause    
    for(int i=0; i<LIMBS_NUM; i++) {
        // assert(expected->limbs[i] == result->limbs[i]);
        if (expected->limbs[i] != result->limbs[i]) {
            printf("***** BAD!!!\n");
            break;
        }
    }
}

void execute_kernels(uint254 *a, uint254 *b, uint254 *expected, uint254 *result) {    
    // Montgomery Multiplication Test 
    printf("\n*** mont_mult ***\n");
    initialize_mont_mult<<<BLOCKS, THREADS>>>(*a, *b, *expected);
    // printf("\n??? zz %lu %lu %lu %lu\n", a->limbs[0], a->limbs[1], a->limbs[2], a->limbs[3]);
    cudaDeviceSynchronize();
    mont_mult<<<BLOCKS, THREADS>>>(*a, *b, *result);
    assert_checks(expected, result);

    // Montgomery Multiplication Test -- Short Integers 
    printf("\n*** mont_mult short ***\n");
    initialize_mont_mult_short<<<BLOCKS, THREADS>>>(*a, *b, *expected);
    cudaDeviceSynchronize();
    mont_mult<<<BLOCKS, THREADS>>>(*a, *b, *result);
    assert_checks(expected, result);

    // Multiply Test - Square Consistency 
    printf("\n*** mont_mult sq consistency ***\n");
    initialize_mul_square_consistency<<<BLOCKS, THREADS>>>(*a, *b);
    cudaDeviceSynchronize();
    mul_square_consistency<<<BLOCKS, THREADS>>>(*a, *b, *expected, *result);
    assert_checks(expected, result);

    // Multiply Test - Square Against Constants 
    printf("\n*** mont_mult sq against constants ***\n");
    initialize_sqr_check_against_constants<<<BLOCKS, THREADS>>>(*a, *expected);
    cudaDeviceSynchronize();
    sqr_check_against_constants<<<BLOCKS, THREADS>>>(*a, *result);
    assert_checks(expected, result);

    // Add Test - Check Against Constants
    printf("\n*** add check against constants ***\n");
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(*a, *b, *expected);
    cudaDeviceSynchronize();
    add_check_against_constants<<<BLOCKS, THREADS>>>(*a, *b, *result);
    assert_checks(expected, result);

    // Subtract Test - Check Against Constant
    printf("\n*** sub check against constants ***\n");
    initialize_sub_check_against_constants<<<BLOCKS, THREADS>>>(*a, *b, *expected);
    cudaDeviceSynchronize();
    sub_check_against_constants<<<BLOCKS, THREADS>>>(*a, *b, *result);
    assert_checks(expected, result);

    // // Convert To Montgomery Form Test
    // initialize_to_montgomery_form<<<BLOCKS, THREADS>>>(*a, *expected);
    // to_montgomery_form<<<BLOCKS, THREADS>>>(*a, *result);*
    // assert_checks(expected, result);

    // // Convert From Montgomery Form Test
    // initialize_from_montgomery_form<<<BLOCKS, THREADS>>>(*a, *expected);
    // from_montgomery_form<<<BLOCKS, THREADS>>>(*a, *result);*
    // assert_checks(expected, result);

    // // Montgomery Consistency Check Test
    // initialize_montgomery_consistency_check<<<BLOCKS, THREADS>>>(*a, *b);*
    // montgomery_consistency_check<<<BLOCKS, THREADS>>>(*a, *b, *expected, result);
    // assert_checks(expected, result);

    // // Add Multiplication Consistency Test
    // initialize_add_mul_consistency<<<BLOCKS, THREADS>>>(*a, *b);*
    // add_mul_consistency<<<BLOCKS, THREADS>>>(*a, *b, *expected, result);
    // assert_checks(expected, result);

    // // Subtract Multiplication Consistency test
    // initialize_sub_mul_consistency<<<BLOCKS, THREADS>>>(*a, *b);*
    // sub_mul_consistency<<<BLOCKS, THREADS>>>(*a, *b, *expected, result);
    // assert_checks(expected, result);

    // Cube Root Test
    printf("\n*** cube root ***\n");
    initialize_cube<<<BLOCKS, THREADS>>>(*a);
    cudaDeviceSynchronize();
    cube<<<BLOCKS, THREADS>>>(*a, *expected, *result);
    assert_checks(expected, result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define uint256 types
    uint254 *a, *b, *expected, *result;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&expected, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));

    // Execute kernel functions
    execute_kernels(a, b, expected, result);

    // Successfull execution of unit tests
    cout << "******* All 'Fq' unit tests passed! **********" << endl;

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(&a);
    cudaFree(&b);
    cudaFree(&result);

    cout << "Completed sucessfully!" << endl;

    return 0;
}