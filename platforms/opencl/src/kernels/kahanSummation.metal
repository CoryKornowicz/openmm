/**
 * Sum the energy buffer.
 */

// Compile this without fast math.


__attribute__((__always_inline__))
void kahanSum(float input, thread float &sum, thread float &c, bool sumLarger) {
    float t = sum + input;
    if (sumLarger) {
        c += (sum - t) + input;
    } else {
        c += (input - t) + sum;
    }
    sum = t;
};

__attribute__((__always_inline__))
void twoSum(thread float &sum, thread float &c) {
    float new_sum = sum + c;
    float c_virtual = new_sum - sum;
    sum = new_sum;
    c = c - c_virtual;
}

// Each pass divides the buffer by ~1000. Repeat this multiple times, until
// the size reaches 1. For the final result, you don't need to add the
// compensation. The larger part contains the closest FP32 number to the energy.
//
// Aim to just overload each workgroup by a factor of ~2-4. This makes the bulk
// summation stage comparable to the reduction stage, while decreasing the
// number of utilized GPU cores. This kernel is flexible; you could easily
// assign all work to a single threadgroup (and observe the performance delta).
__kernel void reduceEnergyPass(GLOBAL const float2* energyBuffer, GLOBAL float2* result, int bufferSize, int elementsPerWorkGroup, DISPATCH_ARGUMENTS) {
    float sum = 0;
    float c = 0;
    int groupStart = GROUP_ID * elementsPerWorkGroup;
    int groupEnd = min(groupStart + elementsPerWorkGroup, bufferSize);
    for (int index = groupStart + LOCAL_ID; index < groupEnd; ++index) {
        float2 value = energyBuffer[index];
        float other_sum = value[0];
        float other_c = value[1];
        kahanSum(other_sum, sum, c, abs(sum) >= abs(other_sum))
        kahanSum(other_c, sum, c, /*sumLarger=*/true);
    }
    
    // Although threadgroup bandwidth might not be a significant bottleneck,
    // threadgroup latency will be. We do SIMD shuffling instead, even though
    // it increases executable size.
    for (uint offset=SIMD_SIZE/2; offset>0; offset/=2) {
        float other_sum = simd_shuffle_down(sum, offset);
        float other_c = simd_shuffle_down(c, offset);
        kahanSum(other_sum, sum, c, abs(sum) >= abs(other_sum))
        kahanSum(other_c, sum, c, /*sumLarger=*/true);
    }
    
    threadgroup float sum_buffer[32 + 1];
    threadgroup float c_buffer[32 + 1];
    if (simd_is_first()) {
        sum_buffer[SIMD_ID] = sum;
        c_buffer[SIMD_ID] = c;
    }
    if (LOCAL_ID >= LOCAL_SIZE/SIMD_SIZE) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = sum_buffer[SIMD_LANE_ID];
    c = c_buffer[SIMD_LANE_ID];
    
    for (uint offset=SIMD_SIZE/2; offset>0; offset/=2) {
        float other_sum = simd_shuffle_down(sum, offset);
        float other_c = simd_shuffle_down(c, offset);
        kahanSum(other_sum, sum, c, abs(sum) >= abs(other_sum))
        kahanSum(other_c, sum, c, /*sumLarger=*/true);
    }
    twoSum(sum, c);
    if (simd_is_first()) {
        result[GROUP_ID] = float2(sum, c);
    }
}

