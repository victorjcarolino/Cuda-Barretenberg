#include "common.cuh"
#include <cooperative_groups.h>
#include <cuda.h>

using namespace cooperative_groups;

namespace pippenger_common
{

#define MAX_THREADS_PER_BLOCK 128

    /* ----------------------------------------- Sum Reduction Kernels ---------------------------------------------- */

    /**
     * Naive multiplication kernel
     */
    __global__ void multiplication_kernel(g1_gpu::element *point, fr_gpu *scalar, g1_gpu::element *result_vec, size_t npoints)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        // 3 * N field multiplications
        fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
                    scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4],
                    result_vec[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
        fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
                    scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4],
                    result_vec[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
        fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4],
                    scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4],
                    result_vec[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
    }

    /**
     * Sum reduction with shared memory
     */
    __global__ void sum_reduction_kernel(g1_gpu::element *points, g1_gpu::element *result)
    {
        // Global thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Perform reduction in shared memory
        __shared__ g1_gpu::element partial_sum[128];

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].x.data[tid % 4],
                     partial_sum[subgroup * 2].x.data[tid % 4]);
        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].y.data[tid % 4],
                     partial_sum[subgroup * 2].y.data[tid % 4]);
        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].z.data[tid % 4],
                     partial_sum[subgroup * 2].z.data[tid % 4]);

        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].x.data[tid % 4],
                     partial_sum[(subgroup * 2) + 1].x.data[tid % 4]);
        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].y.data[tid % 4],
                     partial_sum[(subgroup * 2) + 1].y.data[tid % 4]);
        fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].z.data[tid % 4],
                     partial_sum[(subgroup * 2) + 1].z.data[tid % 4]);

        // Local sync barrier for load operations
        __syncthreads();

        /*
            For 2^10 (1024) points, the unrolled loop iteration is as follows:
                * First pass -- 1024 threads --> 512 points to 256 points
                * Second pass -- 512 threads --> 256 points to 128 points
                * Third pass -- 256 threads --> 128 points to 64 points
                * Fourth pass -- 128 threads --> 64 points to 32 points
                * Fifth pass -- 64 threads --> 32 points to 16 points
                * Sixth pass -- 32 threads --> 16 points to 8 points
                * Seventh pass -- 16 threads --> 8 points to 4 points
                * Eighth pass -- 8 threads --> 4 points to 2 points
                * Ninth pass -- 4 threads --> 2 points to 1 point
        */

        int t = blockDim.x;
        for (int s = 0; s < log2f(blockDim.x) - 1; s++)
        {
            if (threadIdx.x < t)
            {
                g1_gpu::add(
                    // This indexing is not correct!
                    partial_sum[subgroup * 2].x.data[tid % 4],
                    partial_sum[subgroup * 2].y.data[tid % 4],
                    partial_sum[subgroup * 2].z.data[tid % 4],
                    partial_sum[(subgroup * 2) + 1].x.data[tid % 4],
                    partial_sum[(subgroup * 2) + 1].y.data[tid % 4],
                    partial_sum[(subgroup * 2) + 1].z.data[tid % 4],
                    partial_sum[subgroup].x.data[tid % 4],
                    partial_sum[subgroup].y.data[tid % 4],
                    partial_sum[subgroup].z.data[tid % 4]);
            }
            __syncthreads(); // this is what we are trying to avoid
            t -= t / 2;
        }

        // Global synchronization directive for entire grid
        grp.sync();

        // Load data from shared memory to global memory
        if (threadIdx.x < 4)
        {
            fq_gpu::load(partial_sum[subgroup].x.data[tid % 4], result[blockIdx.x].x.data[tid % 4]);
            fq_gpu::load(partial_sum[subgroup].y.data[tid % 4], result[blockIdx.x].y.data[tid % 4]);
            fq_gpu::load(partial_sum[subgroup].z.data[tid % 4], result[blockIdx.x].z.data[tid % 4]);
        }
    }

    /* ----------------------------------------- Naive Double-and-Add Kernel ---------------------------------------------- */

    /**
     * Double and add implementation for multiple points and scalars using bit-decomposition with time complexity: O(k)
     */
    __global__ void double_and_add_kernel(fr_gpu *test_scalars, g1_gpu::element *test_points, g1_gpu::element *final_result, size_t num_points)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        g1_gpu::element R;
        g1_gpu::element Q;

        if (tid < LIMBS)
        {
            // Initialize result as 0
            fq_gpu::load(0, final_result[0].x.data[tid % 4]);
            fq_gpu::load(0, final_result[0].y.data[tid % 4]);
            fq_gpu::load(0, final_result[0].z.data[tid % 4]);
            // Loop for each bucket module
            for (unsigned z = 0; z < num_points; z++)
            {
                // Initialize 'R' to the identity element, Q to the curve point
                fq_gpu::load(0, R.x.data[tid % 4]);
                fq_gpu::load(0, R.y.data[tid % 4]);
                fq_gpu::load(0, R.z.data[tid % 4]);

                // Load partial sums
                fq_gpu::load(test_points[z].x.data[tid % 4], Q.x.data[tid % 4]);
                fq_gpu::load(test_points[z].y.data[tid % 4], Q.y.data[tid % 4]);
                fq_gpu::load(test_points[z].z.data[tid % 4], Q.z.data[tid % 4]);

                // Sync loads
                __syncthreads();

                // Loop for each limb starting with the last limb
                for (int j = 3; j >= 0; j--)
                {
                    // Loop for each bit of scalar
                    for (int i = 64; i >= 0; i--)
                    {
                        // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                        // and extracting the i-th bit of scalar in limb.
                        if (((test_scalars[z].data[j] >> i) & 1) ? 1 : 0)
                            g1_gpu::add(
                                Q.x.data[tid % 4], Q.y.data[tid % 4], Q.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]);
                        if (i != 0)
                            g1_gpu::doubling(
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]);
                    }
                }
                g1_gpu::add(
                    R.x.data[tid % 4],
                    R.y.data[tid % 4],
                    R.z.data[tid % 4],
                    final_result[0].x.data[tid % 4],
                    final_result[0].y.data[tid % 4],
                    final_result[0].z.data[tid % 4],
                    final_result[0].x.data[tid % 4],
                    final_result[0].y.data[tid % 4],
                    final_result[0].z.data[tid % 4]);
            }
        }
    }

    /* ----------------------------------------- Pippenger's "Bucket Method" MSM Kernels ---------------------------------------------- */

    /**
     * Initialize buckets kernel for large MSM
     */
    // This is just initializing the buckets in GPU to zero. I guess no need to convert
    __global__ void initialize_buckets_kernel(g1_gpu::element *bucket)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        // Load zero value into the three coordinates of the each element in the bucket
        fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].x.data[tid % 4]);
        fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].y.data[tid % 4]);
        fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].z.data[tid % 4]);
    }

    // __global__ void initialize_buckets_kernel_thrust(g1_gpu::element *bucket) {
    //     int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //     // set bucket[i].x.data
    //     fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].x.data[tid % 4]);
    //     fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].y.data[tid % 4]);
    //     fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].z.data[tid % 4]);
    // }

    /**
     * Scalar digit decomposition
     * Convert this to a thrust transform functor
     */
    __device__ __forceinline__ uint64_t decompose_scalar_digit(fr_gpu scalar, unsigned num, unsigned width)
    {
        // Determine which 64-bit limb to access
        const uint64_t limb_lsb_idx = (num * width) / 64;
        const uint64_t shift_bits = (num * width) % 64;

        // Shift limb to right to extract scalar digit
        uint64_t rv = scalar.data[limb_lsb_idx] >> shift_bits;

        // Check if scalar digit crosses boundry of current limb
        if ((shift_bits + width > 64) && (limb_lsb_idx + 1 < 4))
        {
            rv += scalar.data[limb_lsb_idx + 1] << (64 - shift_bits);
        }
        // Bit mask to extract LSB of size width
        rv &= ((1 << width) - 1);

        return rv;
    }

    /**
     * Decompose b-bit scalar into c-bit scalar, where c <= b
     * Convert to thrust
     */
    __global__ void split_scalars_kernel(unsigned *bucket_indices, unsigned *point_indices, fr_gpu *scalars, unsigned npoints, unsigned num_bucket_modules, unsigned c)
    {
        unsigned bucket_index;
        unsigned current_index;
        fr_gpu scalar;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // "num_bucket_modules" = #windows
        for (int i = 0; i < num_bucket_modules; i++)
        {
            bucket_index = decompose_scalar_digit(scalars[tid], i, c);
            current_index = i * npoints + tid;

            // Bitwise performs addition here -- packing information about bucket module and specific bucket index
            bucket_indices[current_index] = (i << c) | bucket_index;
            point_indices[current_index] = tid;
        }
    }

    // Instead of accessing scalars[tid], we'll use thrust to iterate of the scalars vector

    /**
     * Accumulation kernel adds up points in each bucket -- this can be swapped out for efficient sum reduction kernel (tree reduction method)
     */
    __global__ void accumulate_buckets_kernel
    (g1_gpu::element *buckets, unsigned *bucket_offsets, unsigned *bucket_sizes, unsigned *single_bucket_indices,
    unsigned *point_indices, g1_gpu::element *points, unsigned num_buckets) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        // Stores the indices, sizes, and offsets of the buckets and points
        unsigned bucket_index = single_bucket_indices[(subgroup + (subgroup_size * blockIdx.x))];
        unsigned bucket_size = bucket_sizes[(subgroup + (subgroup_size * blockIdx.x))];
        unsigned bucket_offset = bucket_offsets[(subgroup + (subgroup_size * blockIdx.x))];

        // printf("bucket size is: %d", bucket_size);

        // Sync loads
        grp.sync();

        // Return empty bucket
        if (bucket_size == 0) {
            return;
        }

        for (unsigned i = 0; i < bucket_size; i++) {
            g1_gpu::add(
                buckets[bucket_index].x.data[tid % 4],
                buckets[bucket_index].y.data[tid % 4],
                buckets[bucket_index].z.data[tid % 4],
                points[point_indices[bucket_offset + i]].x.data[tid % 4],
                points[point_indices[bucket_offset + i]].y.data[tid % 4],
                points[point_indices[bucket_offset + i]].z.data[tid % 4],
                buckets[bucket_index].x.data[tid % 4],
                buckets[bucket_index].y.data[tid % 4],
                buckets[bucket_index].z.data[tid % 4]
            );

            if (fq_gpu::is_zero(buckets[bucket_index].x.data[tid % 4]) &&
                fq_gpu::is_zero(buckets[bucket_index].y.data[tid % 4]) &&
                fq_gpu::is_zero(buckets[bucket_index].z.data[tid % 4])) {
                    g1_gpu::doubling(
                        points[point_indices[bucket_offset + i]].x.data[tid % 4],
                        points[point_indices[bucket_offset + i]].y.data[tid % 4],
                        points[point_indices[bucket_offset + i]].z.data[tid % 4],
                        buckets[bucket_index].x.data[tid % 4],
                        buckets[bucket_index].y.data[tid % 4],
                        buckets[bucket_index].z.data[tid % 4]
                    );
            }
        }
    }

    // struct accumulate_buckets_functor
    // {
    // protected:
    //     g1_single::element *buckets, points;
    //     unsigned *point_indices;

    // public:
    //     accumulate_buckets_functor(g1_single::element *buckets, g1_single::element *points, unsigned *point_indices) : buckets(buckets), points(points), point_indices(point_indices) {}

    //     __device__ 
    // };

    // /**
    //  * Accumulation kernel adds up points in each bucket -- this can be swapped out for efficient sum reduction kernel (tree reduction method)
    //  */
    // __global__ void accumulate_buckets_kernel(g1_single::element *buckets, unsigned *bucket_offsets, unsigned *bucket_sizes, unsigned *single_bucket_indices,
    //                                           unsigned *point_indices, g1_single::element *points, unsigned num_buckets)
    // {
    //     // Stores the indices, sizes, and offsets of the buckets and points
    //     unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    //     if (tid >= num_buckets)
    //         return;

    //     unsigned bucket_index = single_bucket_indices[tid];
    //     unsigned bucket_size = bucket_sizes[tid];
    //     unsigned bucket_offset = bucket_offsets[tid];

    //     // printf("bucket size is: %d", bucket_size);

    //     // Sync loads
    //     // grp.sync();
    //     // __syncthreads();

    //     // Return empty bucket
    //     if (bucket_size == 0)
    //     {
    //         return;
    //     }

    //     for (unsigned i = 0; i < bucket_size; i++)
    //     {
    //         g1_single::add(
    //             buckets[bucket_index].x.data,
    //             buckets[bucket_index].y.data,
    //             buckets[bucket_index].z.data,
    //             points[point_indices[bucket_offset + i]].x.data,
    //             points[point_indices[bucket_offset + i]].y.data,
    //             points[point_indices[bucket_offset + i]].z.data,
    //             buckets[bucket_index].x.data,
    //             buckets[bucket_index].y.data,
    //             buckets[bucket_index].z.data);
    //         // 0G_1 0G_2 0 1 1 2 2 3 3G_8
    //         // [0G_1-G_3 3G_4-G_5 5 7]

    //         // if (fq_single::is_zero(buckets[bucket_index].x.data) &&
    //         //     fq_single::is_zero(buckets[bucket_index].y.data) &&
    //         //     fq_single::is_zero(buckets[bucket_index].z.data)) {
    //         //         g1_single::doubling(
    //         //             points[point_indices[bucket_offset + i]].x.data,
    //         //             points[point_indices[bucket_offset + i]].y.data,
    //         //             points[point_indices[bucket_offset + i]].z.data,
    //         //             buckets[bucket_index].x.data,
    //         //             buckets[bucket_index].y.data,
    //         //             buckets[bucket_index].z.data
    //         //         );
    //         // }
    //     }
    // }

    struct point_add_functor
    {
        __device__ g1_single::element operator()(g1_single::element a, g1_single::element b) const
        {

            g1_single::add(a.x.data,
                           a.y.data,
                           a.z.data,
                           b.x.data,
                           b.y.data,
                           b.z.data,
                           a.x.data,
                           a.y.data,
                           a.z.data);
            return a;
        }
    };

    struct gather_functor
    {
        const g1_single::element* points;
        const unsigned* point_indices;

        gather_functor(const g1_single::element* _points, const unsigned* _point_indices)
            : points(_points), point_indices(_point_indices) {}

        __device__ g1_single::element operator()(const unsigned& idx) const {
            return points[point_indices[idx]];
        }
    };

    // void accumulate_buckets_thrust(thrust::device_vector<g1_single::element> buckets, thrust::device_vector<unsigned> bucket_indices,
    //                                thrust::device_vector<unsigned> point_indices, thrust::device_vector<g1_single::element> points)
    // {
        
    //    //thrust::permutation_iterator<

    //     thrust::device_vector<g1_single::element> sorted_points(points.size());
    //     thrust::device_vector<unsigned> dummy_key_output(buckets.size());

    //     thrust::counting_iterator<unsigned> indices_begin(0);
    //     thrust::counting_iterator<unsigned> indices_end(points.size());

    //     gather_functor gather_op(thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(point_indices.data()));

    //     thrust::transform(thrust::device, indices_begin, indices_end, sorted_points.begin(), gather_op);

    //     // assuming some bucket_indices = [1,1,2,3,5], and points is sorted as desribed by point_indices
    //     point_add_functor add_op;
    //     thrust::equal_to<unsigned> binary_pred;
    //     thrust::reduce_by_key(thrust::device, bucket_indices.begin(), bucket_indices.end(), sorted_points.begin(),
    //                           dummy_key_output.begin(), buckets.begin(), binary_pred, add_op);

    //     // thrust::inclusive_scan();
    // }

    /**
     * Running sum kernel that accumulates partial bucket sums using running sum method
     */
    __global__ void bucket_running_sum_kernel(g1_gpu::element *buckets, g1_gpu::element *final_sum, uint64_t c)
    {
        // Global thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        g1_gpu::element line_sum;

        // Load intitial pointsinitialize_buckets_kernel
        fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].x.data[tid % 4], line_sum.x.data[tid % 4]);
        fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].y.data[tid % 4], line_sum.y.data[tid % 4]);
        fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].z.data[tid % 4], line_sum.z.data[tid % 4]);

        fq_gpu::load(line_sum.x.data[tid % 4], final_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
        fq_gpu::load(line_sum.y.data[tid % 4], final_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
        fq_gpu::load(line_sum.z.data[tid % 4], final_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);

        // Sync loads
        __syncthreads();

        // Running sum method
        for (unsigned i = (1 << c) - 2; i > 0; i--)
        {
            g1_gpu::add(
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].x.data[tid % 4],
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].y.data[tid % 4],
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].z.data[tid % 4],
                line_sum.x.data[tid % 4],
                line_sum.y.data[tid % 4],
                line_sum.z.data[tid % 4],
                line_sum.x.data[tid % 4],
                line_sum.y.data[tid % 4],
                line_sum.z.data[tid % 4]);

            g1_gpu::add(
                line_sum.x.data[tid % 4],
                line_sum.y.data[tid % 4],
                line_sum.z.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
                final_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);

            if (fq_gpu::is_zero(final_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]) &&
                fq_gpu::is_zero(final_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]) &&
                fq_gpu::is_zero(final_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]))
            {
                g1_gpu::doubling(
                    line_sum.x.data[tid % 4],
                    line_sum.y.data[tid % 4],
                    line_sum.z.data[tid % 4],
                    final_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
                    final_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
                    final_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
            }
        }
    }

    /**
     * Running sum kernel that accumulates partial bucket sums
     * References PipeMSM (Algorithm 2) -- https://eprint.iacr.org/2022/999.pdf
     */
    __global__ void bucket_running_sum_kernel_2(g1_gpu::element *buckets, g1_gpu::element *S_, g1_gpu::element *G_, unsigned M, unsigned U)
    {
        // Global thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        // Define variables
        g1_gpu::element G;
        g1_gpu::element S;

        // Initialize G and S with 0
        fq_gpu::load(0x0, G.x.data[tid % 4]);
        fq_gpu::load(0x0, G.y.data[tid % 4]);
        fq_gpu::load(0x0, G.z.data[tid % 4]);

        fq_gpu::load(0x0, S.x.data[tid % 4]);
        fq_gpu::load(0x0, S.y.data[tid % 4]);
        fq_gpu::load(0x0, S.z.data[tid % 4]);

        // Sync loads
        __syncthreads();

        // Each of the M segment sums of size U can be computed seperately
        for (unsigned u = U - 1; u < U; u--)
        {
            g1_gpu::add(
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4],
                G.x.data[tid % 4],
                G.y.data[tid % 4],
                G.z.data[tid % 4],
                G.x.data[tid % 4],
                G.y.data[tid % 4],
                G.z.data[tid % 4]);

            if (fq_gpu::is_zero(G.x.data[tid % 4]) && fq_gpu::is_zero(G.y.data[tid % 4]) && fq_gpu::is_zero(G.z.data[tid % 4]))
            {
                g1_gpu::doubling(
                    S.x.data[tid % 4],
                    S.y.data[tid % 4],
                    S.z.data[tid % 4],
                    G.x.data[tid % 4],
                    G.y.data[tid % 4],
                    G.z.data[tid % 4]);
            }

            g1_gpu::add(
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4],
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << (M - 1)) + u].x.data[tid % 4],
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << (M - 1)) + u].y.data[tid % 4],
                buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << (M - 1)) + u].z.data[tid % 4],
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4]);
        }

        fq_gpu::load(S.x.data[tid % 4], S_[blockIdx.x].x.data[tid % 4]);
        fq_gpu::load(S.y.data[tid % 4], S_[blockIdx.x].y.data[tid % 4]);
        fq_gpu::load(S.z.data[tid % 4], S_[blockIdx.x].z.data[tid % 4]);

        fq_gpu::load(G.x.data[tid % 4], G_[blockIdx.x].x.data[tid % 4]);
        fq_gpu::load(G.y.data[tid % 4], G_[blockIdx.x].y.data[tid % 4]);
        fq_gpu::load(G.z.data[tid % 4], G_[blockIdx.x].z.data[tid % 4]);
    }

    __global__ void bucket_running_sum_kernel_3(g1_gpu::element *result, g1_gpu::element *S_, g1_gpu::element *G_, unsigned M, unsigned U)
    {
        // Global thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        // Define variables
        g1_gpu::element S;
        g1_gpu::element S_k;

        // Initialize S_k and S with 0
        fq_gpu::load(0x0, S.x.data[tid % 4]);
        fq_gpu::load(0x0, S.y.data[tid % 4]);
        fq_gpu::load(0x0, S.z.data[tid % 4]);

        fq_gpu::load(0x0, S_k.x.data[tid % 4]);
        fq_gpu::load(0x0, S_k.y.data[tid % 4]);
        fq_gpu::load(0x0, S_k.z.data[tid % 4]);

        // Sync loads
        __syncthreads();

        // Add up each segement M for each window K
        for (unsigned m = 0; m < M - 1; m++)
        {
            g1_gpu::add(
                S_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].x.data[tid % 4],
                S_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].y.data[tid % 4],
                S_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].z.data[tid % 4],
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4],
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4]);

            g1_gpu::add(
                S_k.x.data[tid % 4],
                S_k.y.data[tid % 4],
                S_k.z.data[tid % 4],
                S.x.data[tid % 4],
                S.y.data[tid % 4],
                S.z.data[tid % 4],
                S_k.x.data[tid % 4],
                S_k.y.data[tid % 4],
                S_k.z.data[tid % 4]);

            if (fq_gpu::is_zero(S_k.x.data[tid % 4]) && fq_gpu::is_zero(S_k.y.data[tid % 4]) && fq_gpu::is_zero(S_k.z.data[tid % 4]))
            {
                g1_gpu::doubling(
                    S.x.data[tid % 4],
                    S.y.data[tid % 4],
                    S.z.data[tid % 4],
                    S_k.x.data[tid % 4],
                    S_k.y.data[tid % 4],
                    S_k.z.data[tid % 4]);
            }
        }

        __syncthreads();

        // 2.2
        unsigned v = log2f(U);
        for (unsigned m = 0; m < v; m++)
        {
            g1_gpu::doubling(
                S_k.x.data[tid % 4],
                S_k.y.data[tid % 4],
                S_k.z.data[tid % 4],
                S_k.x.data[tid % 4],
                S_k.y.data[tid % 4],
                S_k.z.data[tid % 4]);
        }

        __syncthreads();

        g1_gpu::element G_k;

        // Initialize G and S with 0
        fq_gpu::load(0x0, G_k.x.data[tid % 4]);
        fq_gpu::load(0x0, G_k.y.data[tid % 4]);
        fq_gpu::load(0x0, G_k.z.data[tid % 4]);

        // 2.3
        for (unsigned m = 0; m < M; m++)
        {
            g1_gpu::add(
                G_k.x.data[tid % 4],
                G_k.y.data[tid % 4],
                G_k.z.data[tid % 4],
                G_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].x.data[tid % 4],
                G_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].y.data[tid % 4],
                G_[(subgroup + (subgroup_size * blockIdx.x)) * (M - 1) + m].z.data[tid % 4],
                G_k.x.data[tid % 4],
                G_k.y.data[tid % 4],
                G_k.z.data[tid % 4]);
        }

        __syncthreads();

        // 2.4
        g1_gpu::add(
            S_k.x.data[tid % 4],
            S_k.y.data[tid % 4],
            S_k.z.data[tid % 4],
            G_k.x.data[tid % 4],
            G_k.y.data[tid % 4],
            G_k.z.data[tid % 4],
            G_k.x.data[tid % 4],
            G_k.y.data[tid % 4],
            G_k.z.data[tid % 4]);

        __syncthreads();

        // load result
        fq_gpu::load(S_k.x.data[tid % 4], result[blockIdx.x].x.data[tid % 4]);
        fq_gpu::load(S_k.y.data[tid % 4], result[blockIdx.x].y.data[tid % 4]);
        fq_gpu::load(S_k.z.data[tid % 4], result[blockIdx.x].z.data[tid % 4]);
    }

    /**
     * Final bucket accumulation to produce single group element
     */
    __global__ void final_accumulation_kernel(g1_gpu::element *final_sum, g1_gpu::element *final_result, size_t num_bucket_modules, unsigned c)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        g1_gpu::element R;
        g1_gpu::element Q;

        fr_gpu exponent{1024, 0, 0, 0};

        if (tid < LIMBS)
        {
            // Initialize result as 0
            fq_gpu::load(0, final_result[0].x.data[tid % 4]);
            fq_gpu::load(0, final_result[0].y.data[tid % 4]);
            fq_gpu::load(0, final_result[0].z.data[tid % 4]);
            // Loop for each bucket module
            for (unsigned z = 26; z > 0; z--)
            {
                // Initialize 'R' to the identity element, Q to the curve point
                fq_gpu::load(0, R.x.data[tid % 4]);
                fq_gpu::load(0, R.y.data[tid % 4]);
                fq_gpu::load(0, R.z.data[tid % 4]);

                // Load partial sums
                fq_gpu::load(final_result[0].x.data[tid % 4], Q.x.data[tid % 4]);
                fq_gpu::load(final_result[0].y.data[tid % 4], Q.y.data[tid % 4]);
                fq_gpu::load(final_result[0].z.data[tid % 4], Q.z.data[tid % 4]);

                // Sync loads
                __syncthreads();

                // Loop for each limb starting with the last limb
                for (int j = 3; j >= 0; j--)
                {
                    // Loop for each bit of scalar
                    for (int i = 64; i >= 0; i--)
                    {
                        // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB,
                        // extracting the i-th bit of scalar in limb.
                        if (((exponent.data[j] >> i) & 1) ? 1 : 0)
                            g1_gpu::add(
                                Q.x.data[tid % 4], Q.y.data[tid % 4], Q.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]);
                        if (i != 0)
                            g1_gpu::doubling(
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4],
                                R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]);
                    }
                }
                g1_gpu::add(
                    R.x.data[tid % 4],
                    R.y.data[tid % 4],
                    R.z.data[tid % 4],
                    final_sum[z - 1].x.data[tid % 4],
                    final_sum[z - 1].y.data[tid % 4],
                    final_sum[z - 1].z.data[tid % 4],
                    final_result[0].x.data[tid % 4],
                    final_result[0].y.data[tid % 4],
                    final_result[0].z.data[tid % 4]);

                if (fq_gpu::is_zero(final_result[0].x.data[tid % 4]) && fq_gpu::is_zero(final_result[0].y.data[tid % 4]) && fq_gpu::is_zero(final_result[0].z.data[tid % 4]))
                {
                    g1_gpu::doubling(
                        R.x.data[tid % 4],
                        R.y.data[tid % 4],
                        R.z.data[tid % 4],
                        final_result[0].x.data[tid % 4],
                        final_result[0].y.data[tid % 4],
                        final_result[0].z.data[tid % 4]);
                }
            }
        }
    }

    // /**
    //  * Final bucket accumulation to produce single group element
    //  */
    // __global__ void final_accumulation_kernel(g1_single::element *final_sum, g1_single::element *final_result, size_t num_bucket_modules, unsigned c) {
    //     g1_single::element R;
    //     g1_single::element Q;

    //     // 1024 = 2^10 = 2^C
    //     fr_gpu exponent{ 1024, 0, 0, 0 };

    //     // Initialize result as 0
    //     fq_single::load(0, final_result[0].x.data);
    //     fq_single::load(0, final_result[0].y.data);
    //     fq_single::load(0, final_result[0].z.data);
    //     // Loop for each bucket module
    //     // 26 = B/C by default
    //     for (unsigned z = 26; z > 0; z--) {
    //         // Initialize 'R' to the identity element, Q to the curve point
    //         fq_single::load(0, R.x.data);
    //         fq_single::load(0, R.y.data);
    //         fq_single::load(0, R.z.data);

    //         // Load partial sums
    //         fq_single::load(final_result[0].x.data, Q.x.data);
    //         fq_single::load(final_result[0].y.data, Q.y.data);
    //         fq_single::load(final_result[0].z.data, Q.z.data);

    //         // Sync loads
    //         __syncthreads();

    //         // Loop for each limb starting with the last limb
    //         // R = Q^1024
    //         for (int j = 3; j >= 0; j--) {
    //             // Loop for each bit of scalar
    //             for (int i = 64; i >= 0; i--) {
    //                 // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB,
    //                 // extracting the i-th bit of scalar in limb.
    //                 if (((exponent.data[j] >> i) & 1) ? 1 : 0)
    //                     g1_single::add(
    //                         Q.x.data, Q.y.data, Q.z.data,
    //                         R.x.data, R.y.data, R.z.data,
    //                         R.x.data, R.y.data, R.z.data
    //                     );
    //                 if (i != 0)
    //                     g1_single::doubling(
    //                         R.x.data, R.y.data, R.z.data,
    //                         R.x.data, R.y.data, R.z.data
    //                     );
    //             }
    //         }
    //         // final_result = final_sum[z-1] + R
    //         g1_single::add(
    //             R.x.data,
    //             R.y.data,
    //             R.z.data,
    //             final_sum[z - 1].x.data,
    //             final_sum[z - 1].y.data,
    //             final_sum[z - 1].z.data,
    //             final_result[0].x.data,
    //             final_result[0].y.data,
    //             final_result[0].z.data
    //         );

    //         // if (final_result == 0) { final_result = 2*R; }
    //         if (fq_single::is_zero(final_result[0].x.data)
    //             && fq_single::is_zero(final_result[0].y.data)
    //             && fq_single::is_zero(final_result[0].z.data)) {
    //             g1_single::doubling(
    //                 R.x.data,
    //                 R.y.data,
    //                 R.z.data,
    //                 final_result[0].x.data,
    //                 final_result[0].y.data,
    //                 final_result[0].z.data
    //             );
    //         }
    //     }
    // }

    /* ----------------------------------------- Helper Kernels ---------------------------------------------- */

    /**
     * Convert affine to jacobian or projective coordinates
     */
    __global__ void affine_to_jacobian(g1_gpu::affine_element *a_point, g1_gpu::element *j_point, size_t npoints)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Parameters for coperative groups
        auto grp = fixnum::layout();
        int subgroup = grp.meta_group_rank();
        int subgroup_size = grp.meta_group_size();

        fq_gpu::load(
            a_point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
            j_point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
        fq_gpu::load(
            a_point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
            j_point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
        fq_gpu::load(field_gpu<fq_gpu>::one().data[tid % 4], j_point[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
    }

    // __global__ void affine_to_jacobian_thrust(g1_gpu::affine_element *a_point, g1_gpu::element *j_point, size_t npoints) {
    //     int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //      // Parameters for coperative groups
    //     auto grp = fixnum::layout();
    //     int subgroup = grp.meta_group_rank();
    //     int subgroup_size = grp.meta_group_size();

    //     fq_gpu::load(
    //         a_point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4],
    //         j_point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]
    //     );
    //     fq_gpu::load(
    //         a_point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4],
    //         j_point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]
    //     );
    //     fq_gpu::load(field_gpu<fq_gpu>::one().data[tid % 4], j_point[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
    // }

    /**
     * Compare group elements kernel
     */
    __global__ void comparator_kernel(g1_gpu::element *point, g1_gpu::element *point_2, uint64_t *result)
    {
        fq_gpu lhs_zz;
        fq_gpu lhs_zzz;
        fq_gpu rhs_zz;
        fq_gpu rhs_zzz;
        fq_gpu lhs_x;
        fq_gpu lhs_y;
        fq_gpu rhs_x;
        fq_gpu rhs_y;

        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        lhs_zz.data[tid] = fq_gpu::square(point[0].z.data[tid], lhs_zz.data[tid]);
        lhs_zzz.data[tid] = fq_gpu::mul(lhs_zz.data[tid], point[0].z.data[tid], lhs_zzz.data[tid]);
        rhs_zz.data[tid] = fq_gpu::square(point_2[0].z.data[tid], rhs_zz.data[tid]);
        rhs_zzz.data[tid] = fq_gpu::mul(rhs_zz.data[tid], point_2[0].z.data[tid], rhs_zzz.data[tid]);
        lhs_x.data[tid] = fq_gpu::mul(point[0].x.data[tid], rhs_zz.data[tid], lhs_x.data[tid]);
        lhs_y.data[tid] = fq_gpu::mul(point[0].y.data[tid], rhs_zzz.data[tid], lhs_y.data[tid]);
        rhs_x.data[tid] = fq_gpu::mul(point_2[0].x.data[tid], lhs_zz.data[tid], rhs_x.data[tid]);
        rhs_y.data[tid] = fq_gpu::mul(point_2[0].y.data[tid], lhs_zzz.data[tid], rhs_y.data[tid]);
        printf("x[%d]: LHS %lx RHS %lx\n", tid, lhs_x.data[tid], rhs_x.data[tid]);
        printf("y[%d]: LHS %lx RHS %lx\n", tid, lhs_y.data[tid], rhs_y.data[tid]);
        result[tid] = ((lhs_x.data[tid] == rhs_x.data[tid]) && (lhs_y.data[tid] == rhs_y.data[tid]));
    }

}