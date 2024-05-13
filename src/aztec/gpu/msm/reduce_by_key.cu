// Adapted from reduce_by_key.inl in Thrust

#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

template <typename ValueType, typename TailFlagType, typename AssociativeOperator>
struct reduce_by_key_functor
{
  AssociativeOperator binary_op;

  typedef typename thrust::tuple<ValueType, TailFlagType> result_type;

  __host__ __device__ reduce_by_key_functor(AssociativeOperator _binary_op)
      : binary_op(_binary_op)
  {}

  __host__ __device__ result_type operator()(result_type a, result_type b)
  {
    return result_type(thrust::get<1>(b) ? thrust::get<0>(b) : binary_op(thrust::get<0>(a), thrust::get<0>(b)),
                       thrust::get<1>(a) | thrust::get<1>(b));
  }
};

typedef unsigned int FlagType;

template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator1,
          typename BinaryPredicate,
          typename BinaryFunction>
__host__ void reduce_by_key_into_map(
  const ExecutionPolicy &exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  InputIterator3 scatter_map,
  OutputIterator1 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op,
  thrust::device_vector<FlagType> &head_flags,
  thrust::device_vector<FlagType> &tail_flags,
  thrust::device_vector<FlagType> &scanned_tail_flags,
  thrust::device_vector<typename thrust::iterator_value<InputIterator2>::type> &scanned_values
  )
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;

  // Use the input iterator's value type per https://wg21.link/P0571
  using ValueType = typename thrust::iterator_value<InputIterator2>::type;

  if (keys_first == keys_last)
  {
    return;
  }

  // input size
  difference_type n = keys_last - keys_first;

  InputIterator2 values_last = values_first + n;

  // compute head flags
  // thrust::device_vector<FlagType> head_flags(n);
  printf("Here\n");
  fflush(stdout);
  thrust::transform(
    exec, keys_first, keys_last - 1, keys_first + 1, head_flags.begin() + 1, thrust::detail::not2(binary_pred));
  head_flags[0] = 1;

  // compute tail flags
  // thrust::device_vector<FlagType> tail_flags(n); // COPY INSTEAD OF TRANSFORM
  printf("Here1\n");
  fflush(stdout);
  thrust::transform(
    exec, keys_first, keys_last - 1, keys_first + 1, tail_flags.begin(), thrust::detail::not2(binary_pred));
  tail_flags[n - 1] = 1;

  // scan the values by flag
  // thrust::device_vector<ValueType> scanned_values(n);
  // thrust::device_vector<FlagType> scanned_tail_flags(n);

  thrust::inclusive_scan(
    exec,
    thrust::make_zip_iterator(thrust::make_tuple(values_first, head_flags.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(values_last, head_flags.end())),
    thrust::make_zip_iterator(thrust::make_tuple(scanned_values.begin(), scanned_tail_flags.begin())),
    reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));
  
  thrust::scatter_if(
    exec,
    scanned_values.begin(),
    scanned_values.end(),
    scatter_map,
    tail_flags.begin(),
    values_output
  );

  // thrust::exclusive_scan(
  //   exec, tail_flags.begin(), tail_flags.end(), scanned_tail_flags.begin(), FlagType(0), thrust::plus<FlagType>());

  // // number of unique keys
  // FlagType N = scanned_tail_flags[n - 1] + 1;

  // // scatter the keys and accumulated values
  // thrust::scatter_if(exec, keys_first, keys_last, scanned_tail_flags.begin(), head_flags.begin(), keys_output);
  // thrust::scatter_if(
  //   exec, scanned_values.begin(), scanned_values.end(), scanned_tail_flags.begin(), tail_flags.begin(), values_output);

  // return thrust::make_pair(keys_output + N, values_output + N);
} // end reduce_by_key()