/**
 * This file contains OpenCL definitions for the macros and functions needed for the
 * common compute framework.
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
__attribute__((overloadable)) unsigned long atom_add(volatile __global unsigned long* p, unsigned long val) {
    device atomic_uint* word = (device atomic_uint*) p;
    unsigned int lower = as_type<uint2>(val)[0];
    unsigned int upper = as_type<uint2>(val)[1];
    unsigned int previous = atomic_fetch_add_explicit(word + 0, lower, memory_order_relaxed);
    int carry = (lower + previous < lower) ? 1 : 0;
    upper += carry;
    if (upper != 0)
        atomic_fetch_add_explicit(word + 1, upper, memory_order_relaxed);
    return 0;
}
#endif

#define KERNEL kernel
#define DEVICE
#define LOCAL threadgroup
#define LOCAL_ARG threadgroup
#define GLOBAL device
#define RESTRICT
#define LOCAL_ID thread_position_in_threadgroup
#define LOCAL_SIZE threads_per_threadgroup
#define SIMD_LANE_ID thread_index_in_simdgroup
#define SIMD_ID simdgroup_index_in_threadgroup
#define SIMD_SIZE threads_per_simdgroup
#define GLOBAL_ID thread_position_in_grid
#define GLOBAL_SIZE threads_per_grid
#define GROUP_ID threadgroup_position_in_grid
#define NUM_GROUPS threadgroups_per_grid
#define SYNC_THREADS threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
#define MEM_FENCE threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
#define ATOMIC_ADD(dest, value) atom_add(dest, value)

#define DISPATCH_ARGUMENTS \
ushort thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
ushort threads_per_threadgroup [[threads_per_threadgroup]], \
ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
ushort threads_per_simdgroup [[threads_per_simdgroup]], \
uint thread_position_in_grid [[thread_position_in_grid]], \
uint threads_per_grid [[threads_per_grid]], \
uint threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
uint threadgroups_per_grid [[threadgroups_per_grid]]

typedef long mm_long;
typedef unsigned long mm_ulong;

#define make_short2(x...) ((short2) (x))
#define make_short3(x...) ((short3) (x))
#define make_short4(x...) ((short4) (x))
#define make_int2(x...) ((int2) (x))
#define make_int3(x...) ((int3) (x))
#define make_int4(x...) ((int4) (x))
#define make_float2(x...) ((float2) (x))
#define make_float3(x...) ((float3) (x))
#define make_float4(x...) ((float4) (x))
#define make_double2(x...) ((double2) (x))
#define make_double3(x...) ((double3) (x))
#define make_double4(x...) ((double4) (x))

#define trimTo3(v) (v).xyz

// OpenCL has overloaded versions of standard math functions for single and double
// precision arguments.  CUDA has separate functions.  To allow them to be called
// consistently, we define the "single precision" functions to just be synonyms
// for the standard ones.

#define sqrtf(x) sqrt(x)
#define rsqrtf(x) rsqrt(x)
#define expf(x) exp(x)
#define logf(x) log(x)
#define powf(x) pow(x)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define tanf(x) tan(x)
#define acosf(x) acos(x)
#define asinf(x) asin(x)
#define atanf(x) atan(x)
#define atan2f(x, y) atan2(x, y)

float __openmm_erf(float x);
float __openmm_erfc(float x);

// Call this before summing energy.
// Source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
__attribute__((__always_inline__))
void accumulateEnergy(float input, thread float& sum, thread float& c) {
    volatile float t = sum + input;
    if (abs(sum) >= abs(input)) {
        c += (sum - t) + input;
    } else {
        c += (input - t) + sum;
    }
    sum = t;
}

// Call this before storing energy.
// Source: https://en.wikipedia.org/wiki/2Sum
__attribute__((__always_inline__))
void redistributeEnergy(thread float& sum, thread float& c) {
    volatile float new_sum = sum + c;
    volatile float c_virtual = new_sum - sum;
    sum = new_sum;
    c = c - c_virtual;
}

inline long realToFixedPoint(real x) {
    return (long) (x * 0x100000000);
}
