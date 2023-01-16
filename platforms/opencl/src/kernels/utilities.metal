/**
 * Encode a command to fill multiple buffers with zero.
 */

struct AddressWrapper1 {
    ulong address;
}
struct AddressWrapper2 {
    device void *buffer;
}
 
// Just pass each resource's address to the shader. This should bypass bugs in
// Metal Frame Capture. The command buffer inherits the compute pipeline from
// the parent encoder.
kernel void prepareClearBuffers(device ulong *addresses [[buffer(0)]],
                                constant int *sizes [[buffer(1)]],
                                constant ulong icb_id [[buffer(2)]],
                                uint tid [[thread_position_in_grid]])
{
    auto icb = as_type<command_buffer>(icb_id);
    compute_command command(icb, tid);
    command.reset();
    
    AddressWrapper1 wrapper(addresses[tid]);
    auto *buffer = reinterpret_cast<thread AddressWrapper2&>(wrapper).buffer;
    command.set_kernel_buffer(buffer, 0);
    
    uint grid_size = sizes[tid];
    uint tg_size;
    if (VENDOR_AMD == 1) {
        command.set_kernel_buffer(sizes + tid, 1);
        grid_size = (grid_size + 3) / 4;
        tg_size = 128;
    } else {
        tg_size = 256;
    }
    command.concurrent_dispatch_threads(
        uint3(grid_size, 1, 1), uint3(tg_size, 1, 1));
}

/**
 * Fill a buffer with 0.
 */
kernel void clearBufferApple(device int *buffer, DISPATCH_ARGUMENTS) {
    buffer[GLOBAL_ID] = 0;
}

/**
 * Fill a buffer with 0.
 */
kernel void clearBufferAMD(device int *buffer, constant int &size, DISPATCH_ARGUMENTS) {
    auto buffer4 = (device int4*)buffer;
    buffer4[GLOBAL_ID] = int4(0);
    
    if (GLOBAL_ID == 0)
        for (int i = size & ~int(3)); i < size; i++)
            buffer[i] = 0;
}

/**
 * Sum a collection of buffers into the first one.
 * Also, write the result into a 64-bit fixed point buffer (overwriting its contents).
 */

__kernel void reduceReal4Buffer(__global real4* restrict buffer, __global long* restrict longBuffer, int bufferSize, int numBuffers) {
    int index = get_global_id(0);
    int totalSize = bufferSize*numBuffers;
    while (index < bufferSize) {
        real4 sum = buffer[index];
        for (int i = index+bufferSize; i < totalSize; i += bufferSize)
            sum += buffer[i];
        buffer[index] = sum;
        longBuffer[index] = (long) (sum.x*0x100000000);
        longBuffer[index+bufferSize] = (long) (sum.y*0x100000000);
        longBuffer[index+2*bufferSize] = (long) (sum.z*0x100000000);
        index += get_global_size(0);
    }
}

/**
 * Sum the various buffers containing forces.
 */
__kernel void reduceForces(__global long* restrict longBuffer, __global real4* restrict buffer, int bufferSize, int numBuffers) {
    int totalSize = bufferSize*numBuffers;
    real scale = 1/(real) 0x100000000;
    for (int index = get_global_id(0); index < bufferSize; index += get_global_size(0)) {
        real4 sum = (real4) (scale*longBuffer[index], scale*longBuffer[index+bufferSize], scale*longBuffer[index+2*bufferSize], 0);
        for (int i = index; i < totalSize; i += bufferSize)
            sum += buffer[i];
        buffer[index] = sum;
        longBuffer[index] = realToFixedPoint(sum.x);
        longBuffer[index+bufferSize] = realToFixedPoint(sum.y);
        longBuffer[index+2*bufferSize] = realToFixedPoint(sum.z);
    }
}

/**
 * This is called to determine the accuracy of various native functions.
 */

KERNEL void determineNativeAccuracy(GLOBAL vec<float, 8>* restrict values, int numValues, DISPATCH_ARGUMENTS) {
    for (int i = GLOBAL_ID; i < numValues; i += GLOBAL_SIZE) {
        float v = values[i][0];
        values[i] = vec<float, 8> (v, fast::sqrt(v), fast::rsqrt(v), fast::divide(1.0f, v), fast::exp(v), fast::log(v), 0.0f, 0.0f);
    }
}

/**
 * Record the atomic charges into the posq array.
 */
__kernel void setCharges(__global real* restrict charges, __global real4* restrict posq, __global int* restrict atomOrder, int numAtoms) {
    for (int i = get_global_id(0); i < numAtoms; i += get_global_size(0))
        posq[i].w = charges[atomOrder[i]];
}
