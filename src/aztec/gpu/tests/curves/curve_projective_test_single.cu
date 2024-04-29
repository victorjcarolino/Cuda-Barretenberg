#include "group_single.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg_single;

// Kernel launch parameters
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;
static constexpr size_t POINTS = 1 << 10;

/* -------------------------- Addition Test ---------------------------------------------- */

__global__ void initialize_add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &expected_x, uint254 &expected_y, uint254 &expected_z, g1_single::element &t1, g1_single::element &t2, g1_single::element &expected) {
    fq_single::load(0x0, a);
    fq_single::load(field_single<fq_single>::one().data, b);
    fq_single::load(0x0, c);

    fq_single::load(a, t1.x.data); 
    fq_single::load(b, t1.y.data); 
    fq_single::load(c, t1.z.data); 
}

__global__ void initialize_affine(g1_single::affine_element &aff) {
    fq_single a_x{ 0x01, 0x0, 0x0, 0x0 };
    fq_single a_y{ 0x02, 0x0, 0x0, 0x0 };

    fq_single::load(a_x.data, aff.x.data); 
    fq_single::load(a_y.data, aff.y.data); 
}

__global__ void initialize_jacobian(g1_single::element &jac) {
    fq_single a_x{ 0x1, 0x0, 0x0, 0x0 };
    fq_single a_y{ 0x2, 0x0, 0x0, 0x0 };
    fq_single a_z{ 0x0, 0x0, 0x0, 0x0 };

    fq_single::load(a_x.data, jac.x.data); 
    fq_single::load(a_y.data, jac.y.data); 
    fq_single::load(a_z.data, jac.z.data); 
}

__global__ void initialize_projective(g1_single::projective_element &proj) {
    fq_single a_x{ 0x1, 0x0, 0x0, 0x0 };
    fq_single a_y{ 0x2, 0x0, 0x0, 0x0 };
    fq_single a_z{ 0x0, 0x0, 0x0, 0x0 };

    fq_single::load(a_x.data, proj.x.data); 
    fq_single::load(a_y.data, proj.y.data); 
    fq_single::load(a_z.data, proj.z.data); 
}


__global__ void add_check_against_constants
(uint254 &a, uint254 &b, uint254 &c, uint254 &x, uint254 &y, uint254 &z, uint254 &res_x, uint254 &res_y, uint254 &res_z, g1_single::element &t1, g1_single::projective_element &t2, g1_single::projective_element &t3) {
    g1_single::projective_element lhs;
    g1_single::projective_element rhs;

    // Calculate global thread ID, and boundry check
    fq_single::to_monty(t1.x.data, lhs.x.data);
    fq_single::to_monty(t1.y.data, lhs.y.data);
    fq_single::to_monty(t1.z.data, lhs.z.data);
    fq_single::to_monty(t2.x.data, rhs.x.data);
    fq_single::to_monty(t2.y.data, rhs.y.data);
    fq_single::to_monty(t2.z.data, rhs.z.data);

    // lhs + rhs (projective element + projective element)
    g1_single::add_projective(
        lhs.x.data, lhs.y.data, lhs.z.data, 
        rhs.x.data, rhs.y.data, rhs.z.data, 
        res_x, res_y, res_z
    );

    // Transform results from montgomery form 
    fq_single::from_monty(res_x, res_x);
    fq_single::from_monty(res_y, res_y);
    fq_single::from_monty(res_z, res_z);

    fq_single::load(res_x, t3.x.data); 
    fq_single::load(res_y, t3.y.data); 
    fq_single::load(res_z, t3.z.data); 
}

// Compare two elliptic curve elements
__global__ void comparator_kernel(g1_single::element &point, g1_single::projective_element &point_2, uint64_t &result) {     
    fq_single lhs_zz;
    fq_single lhs_zzz;
    fq_single rhs_zz;
    fq_single rhs_zzz;
    fq_single lhs_x;
    fq_single lhs_y;
    fq_single rhs_x;
    fq_single rhs_y;
    
    fq_single::square(point.z.data, lhs_zz.data);
    fq_single::mul(lhs_zz.data, point.z.data, lhs_zzz.data);
    fq_single::square(point_2.z.data, rhs_zz.data);
    fq_single::mul(rhs_zz.data, point_2.z.data, rhs_zzz.data);
    fq_single::mul(point.x.data, rhs_zz.data, lhs_x.data);
    fq_single::mul(point.y.data, rhs_zzz.data, lhs_y.data);
    fq_single::mul(point_2.x.data, lhs_zz.data, rhs_x.data);
    fq_single::mul(point_2.y.data, lhs_zzz.data, rhs_y.data);
    result = ((lhs_x.data == rhs_x.data) && (lhs_y.data == rhs_y.data));
}

__global__ void affine_to_projective(g1_single::affine_element &point, g1_single::projective_element &point_2) {     
    fq_single::load(point.x.data, point_2.x.data);
    fq_single::load(point.y.data, point_2.y.data);
    fq_single::load(field_single<fq_single>::one().data, point_2.z.data);
}

__global__ void jacobian_to_projective(g1_single::element &point, g1_single::projective_element &point_2) {     
    fq_single t1; 

    fq_single::mul(point.x.data, point.z.data, point_2.x.data);
    fq_single::load(point.y.data, point_2.y.data);
    fq_single::square(point.z.data, t1.data);
    fq_single::mul(t1.data, point.z.data, point_2.z.data);
}

__global__ void projective_to_jacobian(g1_single::projective_element &point, g1_single::element &point_2) {     
    fq_single t1; 

    fq_single::mul(point.x.data, point.z.data, point_2.x.data);
    fq_single::square(point.z.data, t1.data);
    fq_single::mul(point.y.data, t1.data, point_2.y.data);
    fq_single::load(point.z.data, point_2.z.data);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(var *expected, var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();
    
    // Assert clause
    // assert(expected[0] == result[0]);
    // assert(expected[1] == result[1]);
    // assert(expected[2] == result[2]);
    // assert(expected[3] == result[3]);

    // Print statements
    printf("expected[0] is: %zu\n", expected[0]);
    printf("expected[1] is: %zu\n", expected[1]);
    printf("expected[2] is: %zu\n", expected[2]);
    printf("expected[3] is: %zu\n", expected[3]);

    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n\n", result[3]);
}

void print_affine(g1_single::affine_element *aff) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", aff->x.data.limbs[0]);
    printf("expected[0] is: %zu\n", aff->x.data.limbs[1]);
    printf("expected[0] is: %zu\n", aff->x.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", aff->x.data.limbs[3]);

    printf("expected[0] is: %zu\n", aff->y.data.limbs[0]);
    printf("expected[0] is: %zu\n", aff->y.data.limbs[1]);
    printf("expected[0] is: %zu\n", aff->y.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", aff->y.data.limbs[3]);
}

void print_projective(g1_single::projective_element *proj) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", proj->x.data.limbs[0]);
    printf("expected[0] is: %zu\n", proj->x.data.limbs[1]);
    printf("expected[0] is: %zu\n", proj->x.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", proj->x.data.limbs[3]);

    printf("expected[0] is: %zu\n", proj->y.data.limbs[0]);
    printf("expected[0] is: %zu\n", proj->y.data.limbs[1]);
    printf("expected[0] is: %zu\n", proj->y.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", proj->y.data.limbs[3]);

    printf("expected[0] is: %zu\n", proj->z.data.limbs[0]);
    printf("expected[0] is: %zu\n", proj->z.data.limbs[1]);
    printf("expected[0] is: %zu\n", proj->z.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", proj->z.data.limbs[3]);
}

void print_jacobian(g1_single::element *jac) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", jac->x.data.limbs[0]);
    printf("expected[0] is: %zu\n", jac->x.data.limbs[1]);
    printf("expected[0] is: %zu\n", jac->x.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", jac->x.data.limbs[3]);

    printf("expected[0] is: %zu\n", jac->y.data.limbs[0]);
    printf("expected[0] is: %zu\n", jac->y.data.limbs[1]);
    printf("expected[0] is: %zu\n", jac->y.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", jac->y.data.limbs[3]);

    printf("expected[0] is: %zu\n", jac->z.data.limbs[0]);
    printf("expected[0] is: %zu\n", jac->z.data.limbs[1]);
    printf("expected[0] is: %zu\n", jac->z.data.limbs[2]);
    printf("expected[0] is: %zu\n\n", jac->z.data.limbs[3]);
}

void print_result(var result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result is: %zu\n", result);
}

void execute_kernels
(uint254 *a, uint254 *b, uint254 *c, uint254 *x, uint254 *y, uint254 *z, uint254 *expected_x, uint254 *expected_y, uint254 *expected_z, uint64_t *result, uint254 *res_x, uint254 *res_y, uint254 *res_z) {
    // Allocate unified memory accessible by host and device
    g1_single::element *t1;
    g1_single::element *t2;
    g1_single::projective_element *t3;
    g1_single::element *jac;
    g1_single::affine_element *aff;
    g1_single::projective_element *proj;
    g1_single::element *expected;

    cudaMallocManaged(&t1, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&t2, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&t3, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&jac, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&aff, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&proj, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected, 3 * 2 * LIMBS * sizeof(uint64_t));

    // Initialize points
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(*a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *t1, *t2, *expected);
    initialize_affine<<<BLOCKS, THREADS>>>(*aff);

    // Affine conversion
    affine_to_projective<<<1, 1>>>(*aff, *proj);
    cudaDeviceSynchronize();

    // Execute projective addition
    add_check_against_constants<<<1, 1>>>(*a, *b, *c, *x, *y, *z, *res_x, *res_y, *res_z, *t1, *proj, *t3);
    print_projective(t3);

    // Compare results
    cudaDeviceSynchronize();
    comparator_kernel<<<1, 1>>>(*expected, *t3, *result);
    print_result(*result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to 'uint64_t' type
    uint254 *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z;    
    uint64_t *result;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, sizeof(uint64_t));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, result, res_x, res_y, res_z);

    // Successfull execution of unit tests
    cout << "******* All 'g1_single BN-254 Curve' unit tests passed! **********" << endl;

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