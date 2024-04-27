# 375 Notes for Implementation TODOs
- [ ] Implement addc_functor and addc_thurst for uint254 in fixnum.cu instead of primitives.cu
- Remove one declaration of uint254 (from primitives.cu)
- Write the correct + operator overloading for uint254 in fixnum.cu using the correct addc_functor and addc_thurst in primitives.cu
- Combine digit and fixnum namespace functions into a single namespace thurst_fixnum using the required add functions in primitives.cu
- [ ] Fix arithmetic in fixnum to not compute over 4 threads to avoid need of sync
    - [ ] Fix fixnum to handle carry bits over all operations in limb indicies over the same thread on the uint254 type rather than propogating over another limb
    - [ ] Fix field.cu to use updated functions from fixnum.cu
    - [ ] Fix group.cu to use updated functions froom field.cu
    - [ ] Fix kernels to pass entire coordinate to math function, instead of passing only one limb with "[tid % 4]"
- [ ] Look into replacing functionality from cuda unbound to thrust api calls
    - thrust::unique_by_key to get bucket offsets and indicies from sorted scalar-point pairs
    - thrust sort can do the radix sort and unique_by_key can handle the gathering of single_bucket_indicies and bucket_offsets and bucket_sizes 
- [ ] Modify code to not rely on cooperative groups
    - How do we make sure that threads are computing over the correct portion of any single msm?
- Check for performance on different parts of Pippenger algortihm and Compare with Tal's implementation
    - Benchmark on Tal's implementation is already taken and we have numbers
- [ ] Convert kernel.cu split_scalars_kernel to Thrust
- [ ] Convert kernel.cu decompose_scalars to Thrust functor


# Things we did
1. Changed the field arithmetic and elliptic curve operations on spearate limbs of a 256 bit integer to a single variable with 4 limbs and using thrust libarary to handle the parallel operations between the limbs. Because the operation between the limbs contains dependencies due to the carries and thus requires lot of synchronization to perfrom in parallel.

2. Implemented assembly for single threaded 254 bit multiplication and multiplication+add to replace the field arithmetic from the existing 4 threaded execution for the 4 limbs to avoid synchronization between the 4 threads currently being done with cuda cooperative groups






    
# Implementation Plan (Tal)
- [x] Set up a cloud infrastructure (NVIDIA A10 (Ampere) w/ 24 GB VRAM) on Oracle Cloud
- [x] Initialize Barretenberg repo
- [x] Integrate cuda-fixnum external library and add cuda / nvcc support
- [x] Set up Nvsight Compute profiling software
- [x] Implement finite field (FF) logic on the GPU
    - [x] Understand the algorithm, and the difference between montogomery multiplication and montogomery reduction schemes
    - [x] Extract the montogomery multiplication (CIOS) implementation to a seperate benchmarking file and get it compiling
    - [x] Implement addition/subtraction operations on GPU
    - [x] Understand Barretenberg's C++ implementation, and benchmark for correctness and performance
    - [x] Implement unit tests for fq / fr (base and scalar fields)
- [x] Implement BN254 / Grumpkin elliptic curve (EC) logic on the GPU   
    - [x] Understand the differences between BN-254 and Grumpkin ECs
    - [x] Implement unit tests for G1 BN-254
- [x] Benchmark FF and ECC implementations on CPU
    - [x] Benchmark finite field arithemtic to establish CPU baseline (bench suite already created)
    - [x] Benchmark elliptic curve operations to establish CPU baseline
- [x] Benchmark Aztec's MSM, FFT, and Plonk algorithms on CPU
    - [x] Benchmark Pippenger's bucket algorithm
    - [x] Benchmark Fast Fourier Transform algorithm
    - [x] Benchmark vanilla Plonk prover
    - [x] Benchmark TurboPlonk and UltraPlonk provers
- [x] Multi-scalar multiplication (MSM) for Pippenger's Bucket Method
    - [x] Understand Barretenberg’s multi-exponentiation CPU implementation 
    - [x] Benchmark Zprize MSM implementation on GPU 
    - [x] Understand Supranational’s SPPARK MSM GPU kernel
    - [x] Adapt Supranational’s and Ingonyama's Icicle MSM scheme
        - [x] Add comments to understand code
        - [x] Remove rust bindings
        - [x] Read in curve points and scalars
        - [x] Port MSM implemenation over BN-254 curve
            - [x] Add naive double and add kernel (reference)
            - [x] Add sum reduction kernel 
            - [x] Add bucket method
        - [x] Implement simple double-and-add algorithm for correctness
            - [x] Implement for single finite field elements
            - [x] Implement for single elliptic curve points
            - [x] Implement for vector of finite field elements
            - [x] Implement for vector of elliptic curve points
        - [x] MVP KZG-10 commitment scheme 
- [x] Test MSM kernel implementation on GPU
    - [x] Create unit test suite
- [ ] Benchmark FF and ECC arithmetic and MSM on GPU
    - [ ] Create finite field bench suite 
    - [ ] Create elliptic curve bench suite 
    - [ ] Create MSM bench suite
- [x] Implement remaining G1 arthmetic
    - [x] Finish adding missing elliptic curve functions
- [ ] Implement Projective coordinates G2 arithmetic
    - [ ] Add extension fields
    - [ ] Add zero and infinity checks 
    - [ ] Create unit test suite for extension fields
    - [x] Implement Projective coordinates and testing bench
- [x] NTT port
    - [x] Generate twiddle factors (roots of unity) over BN-254
    - [x] Store twiddle factors in table
- [ ] Implement Plonk pipeline wrapper
    - [x] General wrapper that overrides prover functions
    - [ ] Connect wrapper to MSM GPU kernel
- [ ] Enhance MSM with performance optimizations
- [ ] Design memory pipeline for data used in MSM GPU kernel
- [ ] Set proper licenses for repository