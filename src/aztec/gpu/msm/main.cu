#include "pippenger.cu"
#include <chrono>
#include <common/assert.hpp>
#include <cstdlib>
#include <ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp>
#include <plonk/reference_string/file_reference_string.hpp>
#include <polynomials/polynomial_arithmetic.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace pippenger_common;
using namespace waffle;
using namespace barretenberg;

void usage() {
    printf("Usage: pippenger_cu <num_points> <num_acc_bucket_threads>\n");
}

int main(int argc, char *argv[]) {
    // if (argc != 2) {
    //     usage();
    //     return 1;
    // }
    // if (argc != 3) {
    //     usage();
    //     return 1;
    // }
    long long n = atoll(argv[1]);
    if (n == 0 || n > SIZE_MAX) {
        usage();
        return 1;
    }
    // long long acc_buck_threads = atoll(argv[2]);
    long long acc_buck_threads = 7;
    NUM_POINTS = n;

    // Initialize dynamic 'msm_t' object 
    msm_t<point_t, scalar_t> *msm = new msm_t<point_t, scalar_t>();
    
    // Construct elliptic curve points from SRS
    auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db/ignition");
    g1::affine_element* points = reference_string->get_monomials();

    // Construct random scalars 
    std::vector<fr> scalars;
    scalars.reserve(NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        scalars.emplace_back(fr::random_element());
    }

    // Number of streams
    int num_streams = 1;

    // Initialize dynamic pippenger 'context' object
    Context<point_t, scalar_t> *context = msm->pippenger_initialize(points, &scalars[0], num_streams, NUM_POINTS);
    // // Execute "Double-And-Add" reference kernel
    // cout << "start double and add..." << endl;
    // g1_gpu::element *result_1 = msm->msm_double_and_add(context, NUM_POINTS, points, &scalars[0]);
    // cout << "initialize pippenger context" << endl;

    // Execute "Pippenger's Bucket Method" kernel
    cout << "start pippenger..." << endl;
    g1_gpu::element **result_2 = msm->msm_bucket_method(context, points, &scalars[0], num_streams, acc_buck_threads);


    // // Print results 
    // context->pipp.print_result(result_1, result_2);

    // // Verify the final results are equal
    // context->pipp.verify_result(result_1, result_2);

    return 0;
}