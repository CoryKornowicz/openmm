/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include <cmath>
#include "OpenCLContext.h"
#include "OpenCLArray.h"
#include "OpenCLBondedUtilities.h"
#include "OpenCLEvent.h"
#include "OpenCLForceInfo.h"
#include "OpenCLIntegrationUtilities.h"
#include "OpenCLKernelSources.h"
#include "OpenCLNonbondedUtilities.h"
#include "OpenCLProgram.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <typeinfo>

using namespace OpenMM;
using namespace std;

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
  #define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
  #define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif

const int OpenCLContext::ThreadBlockSize = 64;
const int OpenCLContext::TileSize = 32;

OpenCLContext::OpenCLContext(const System& system, int platformIndex, int deviceIndex, const string& precision, OpenCLPlatform::PlatformData& platformData, OpenCLContext* originalContext) :
        ComputeContext(system), platformData(platformData), numForceBuffers(0), hasAssignedPosqCharges(false),
        integration(NULL), expression(NULL), bonded(NULL), nonbonded(NULL), pinnedBuffer(NULL) {
    // Right now, force-enable Metal API validation to catch bugs.
    setenv("METAL_DEVICE_WRAPPER_TYPE", "1", 1);
    setenv("METAL_ERROR_MODE", "5", 1);
    setenv("METAL_DEBUG_ERROR_MODE", "5", 1);
    
    // Catch all objects autoreleased during initialization.
    commandPool = NS::AutoreleasePool::alloc()->init();
    
    if (precision == "single") {
        useDoublePrecision = false;
        useMixedPrecision = false;
    }
    else
        throw OpenMMException("Illegal value for Precision: "+precision);
    try {
        NS::Array* devices = MTL::CopyAllDevices();

        // We are simultaneously initializing Metal and OpenCL. First, search
        // through the Metal devices. `OpenCLDeviceIndex` translates to an index
        // in the array from `MTLCopyAllDevices()`, whose order seems stable.
        // Then, we scour all OpenCL devices over all platforms. Find one with
        // the same name, or fail otherwise.
        int bestDevice = -1;
        for (int j = 0; j < devices->count(); j++) {
            // Metal backend currently doesn't support Intel GPUs. The CPU has
            // more FLOPS than the GPU for these devices.
            if (device->isLowPower())
                continue;

            if (deviceIndex < -1 || deviceIndex >= (int) devices->count())
                throw OpenMMException("Illegal value for DeviceIndex: "+intToString(deviceIndex));

            // If they supplied a valid deviceIndex, we only look through that one
            if (i != deviceIndex && deviceIndex != -1)
                continue;
            
            // We're not going to query OpenCL parameters because that tactic is
            // broken on macOS. If you have two AMD GPUs, choose the one you
            // want through `OpenCLDeviceIndex`.
            bestDevice = i;
        }
        if (bestDevice == -1)
            throw OpenMMException("No compatible OpenCL device is available");

        this->device = NS::TransferPtr(devices->object(bestDevice));
        std::string device_description(
            device->name()->cString(NS::UTF8StringEncoding));
        
        std::vector<cl::Platform> clPlatforms;
        cl::Platform::get(&platforms);
        bool foundInfoDevice = false;
        for (auto clPlatform : clPlatforms) {
            std::vector<cl::Device> clDevices;
            clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto clDevice : clDevices) {
                auto name = clDevice.getInfo<CL_DEVICE_NAME>();
                if (name == device_description) {
                    this->infoDevice = clDevice;
                    break;
                }
            }
        }
        if (!foundInfoDevice) {
            throw OpenMMException("Could not find OpenCL device matching: " + device_description);
        }
        
        // Assert that we've enabled Metal API validation.
        std::string description(
            device->description()->cString(NS::UTF8StringEncoding));
        if (description.find("MTLDebugDevice") == std::string::npos)
            throw OpenMMException("Metal API validation not active.")
        
        this->deviceIndex = bestDevice;
        commandQueue = NS::TransferPtr(device->newCommandQueue());
        commandBuffer = commandQueue->commandBuffer();
        
        compilationDefines["WORK_GROUP_SIZE"] = intToString(ThreadBlockSize);
        defaultOptimizationOptions = NS::TransferPtr(
            MTL::CompileOptions::alloc()->init());
        if (useDoublePrecision || useMixedPrecision)
            throw OpenMMException("This device does not support double precision");
        
        // Query SIMD width through a compute pipeline.
        NS::Error* error;
        auto testLibrary = NS::TransferPtr(
            device->newLibrary("kernel void getSIMDWidth() {}", NULL, &error));
        if (error) throw OpenMMException("Could not create test library.");
        auto testFunction = NS::TransferPtr(
            library->newFunction("getSIMDWidth", &error));
        if (error) throw OpenMMException("Could not create test function.");
        auto testPipeline = NS::TransferPtr(
            device->newComputePipelineState(testFunction.get(), &error));
        if (error) throw OpenMMException("Could not create test pipeline.");
        simdWidth = testPipeline->threadExecutionWidth();
        if (simdWidth != 32 && simdWidth != 64)
            throw OpenMMException("Unsupported SIMD width: " + std::to_string(simdWidth));
        
        int numThreadBlocksPerComputeUnit;
        #if defined(__aarch64__)
        // We target 2048 of 3072 threads on Apple, until we know more about
        // performance. This is also the only size we can enforce by checking a
        // pipeline's max threadgroup size.
        numThreadBlocksPerComputeUnit = 2048 / ThreadBlockSize;
        #else
        // For AMD, it seems that 12/CU works best on RDNA, but 16/CU on GCN. We
        // cannot trust Metal to report the true SIMD width; it probably treats
        // both architectures as 64-wide.
        std::vector<std::string> rdnaGPUs = {
            "5300", "5500", "5600", "5700", "5800",
            "6300", "6400", "6500", "6600", "6700", "6800", "6900",
            "6350", "6450", "6550", "6650", "6750", "6850", "6950",
            "7300", "7400", "7500", "7600", "7700", "7800", "7900",
            "7350", "7450", "7550", "7650", "7750", "7850", "7950",
        };
        bool isRDNA = false;
        for (auto candidate : rdnaGPUs) {
            if (device_description.find(candidate) != std::string::npos) {
                isRDNA = true;
                break;
            }
        }
        if (isRDNA) {
            // 2 x 32-wide simds
            numThreadBlocksPerComputeUnit = 6 * 2;
        } else {
            // 4 x 16-wide simds
            numThreadBlocksPerComputeUnit = 4 * 4;
        }
        #endif
        
        compilationDefines["SYNC_WARPS"] =
            "threadgroup_barrier(mem_flags::mem_threadgroup);";
        numAtoms = system.getNumParticles();
        paddedNumAtoms = TileSize*((numAtoms+TileSize-1)/TileSize);
        numAtomBlocks = (paddedNumAtoms+(TileSize-1))/TileSize;
        numThreadBlocks = numThreadBlocksPerComputeUnit *
            infoDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        {
            posq.initialize<mm_float4>(*this, paddedNumAtoms, "posq");
            velm.initialize<mm_float4>(*this, paddedNumAtoms, "velm");
            compilationDefines["convert_real4"] = "float4";
            compilationDefines["make_real2"] = "float2";
            compilationDefines["make_real3"] = "float3";
            compilationDefines["make_real4"] = "float4";
            compilationDefines["convert_mixed4"] = "float4";
            compilationDefines["make_mixed2"] = "float2";
            compilationDefines["make_mixed3"] = "float3";
            compilationDefines["make_mixed4"] = "float4";
        }
        longForceBuffer.initialize<cl_long>(*this, 3*paddedNumAtoms, "longForceBuffer");
        posCellOffsets.resize(paddedNumAtoms, mm_int4(0, 0, 0, 0));
        atomIndexDevice.initialize<cl_int>(*this, paddedNumAtoms, "atomIndexDevice");
        atomIndex.resize(paddedNumAtoms);
        for (int i = 0; i < paddedNumAtoms; ++i)
            atomIndex[i] = i;
        atomIndexDevice.upload(atomIndex);
    }
    catch (cl::Error err) {
        std::stringstream str;
        str<<"Error initializing context: "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }

    // Create utility kernels that are used in multiple places.

    // TODO: Speed up copying through indirect command buffers. Have the GPU
    // encode commands and execute them concurrently. You can make just one
    // kernel that encodes everything in parallel, instead of multiple kernels
    // for different amounts of buffers.
    OpenCLProgram utilities(
        *this, createProgram(OpenCLKernelSources::utilities));
    reduceReal4Kernel = utilities.createKernel("reduceReal4Buffer");
    reduceForcesKernel = utilities.createKernel("reduceForces");
    reduceEnergyKernel = utilities.createKernel("reduceEnergy");
    setChargesKernel = utilities.createKernel("setCharges");
    
    // Configure dynamic functions library.
    auto dynamicFunctionsOptions = NS::TransferPtr(
        MTL::CompileOptions::alloc()->init());
    dynamicFunctionsOptions->setFastMathEnabled(false);
    dynamicFunctionsOptions->setLibraryType(MTL::LibraryTypeDynamic);
    dynamicFunctionsOptions->setInstallName(
        "@loader_path/libOpenMM_dynamicFunctions.metallib");
    dynamicFunctionsLibrary = NS::TransferPtr(
        createProgram(OpenCLKernelSources::erf, erfCompileOptions));
    auto dynamicFunctionsLibraries = NS::TransferPtr(
        NS::Array::alloc()->init(erfLibrary.get()));
    defaultOptimizationOptions->setLibraries(dynamicFunctionsLibraries.get());

    // Decide whether native_sqrt(), native_rsqrt(), and native_recip() are sufficiently accurate to use.

    auto accuracyKernel = utilities.createKernel("determineNativeAccuracy");
    OpenCLArray valuesArray(*this, 20, sizeof(mm_float8), "values");
    vector<mm_float8> values(valuesArray.getSize());
    float nextValue = 1e-4f;
    for (auto& val : values) {
        val.s0 = nextValue;
        nextValue *= (float) M_PI;
    }
    valuesArray.upload(values);
    accuracyKernel.setArg(0, valuesArray);
    accuracyKernel.setArg<cl_int>(1, values.size());
    executeKernel(accuracyKernel, values.size());
    valuesArray.download(values);
    double maxSqrtError = 0.0, maxRsqrtError = 0.0, maxRecipError = 0.0, maxExpError = 0.0, maxLogError = 0.0;
    for (auto& val : values) {
        double v = val.s0;
        double correctSqrt = sqrt(v);
        maxSqrtError = max(maxSqrtError, fabs(correctSqrt-val.s1)/correctSqrt);
        maxRsqrtError = max(maxRsqrtError, fabs(1.0/correctSqrt-val.s2)*correctSqrt);
        maxRecipError = max(maxRecipError, fabs(1.0/v-val.s3)/val.s3);
        maxExpError = max(maxExpError, fabs(exp(v)-val.s4)/val.s4);
        maxLogError = max(maxLogError, fabs(log(v)-val.s5)/val.s5);
    }
    compilationDefines["SQRT"] = (maxSqrtError < 1e-6) ? "fast::sqrt" : "precise::sqrt";
    compilationDefines["RSQRT"] = (maxRsqrtError < 1e-6) ? "fast::rsqrt" : "precise::rsqrt";
    compilationDefines["RECIP(x)"] = (maxRecipError < 1e-6) ? "fast::divide(1.0f, x)" : "precise::divide(1.0f, x);";
    compilationDefines["EXP"] = (maxExpError < 1e-6) ? "fast::exp" : "precise::exp";
    compilationDefines["LOG"] = (maxLogError < 1e-6) ? "fast::log" : "precise::log";
    // Pause: I'm curious what the results are.
    throw OpenMMException(
        "Accuracy results:" + compilationDefines["SQRT"] +
         compilationDefines["RSQRT"] + compilationDefines["RECIP(x)"] +
         compilationDefines["EXP"] + compilationDefines["LOG"]);
    
    compilationDefines["POW"] = "precise::pow";
    compilationDefines["COS"] = "precise::cos";
    compilationDefines["SIN"] = "precise::sin";
    compilationDefines["TAN"] = "precise::tan";
    compilationDefines["ACOS"] = "precise::acos";
    compilationDefines["ASIN"] = "precise::asin";
    compilationDefines["ATAN"] = "precise::atan";
    compilationDefines["ERF"] = "__openmm_erf";
    compilationDefines["ERFC"] = "__openmm_erfc";

    // Set defines for applying periodic boundary conditions.

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    boxIsTriclinic = (boxVectors[0][1] != 0.0 || boxVectors[0][2] != 0.0 ||
                      boxVectors[1][0] != 0.0 || boxVectors[1][2] != 0.0 ||
                      boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0.0);
    if (boxIsTriclinic) {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "{"
            "real scale3 = floor(delta.z*invPeriodicBoxSize.z+0.5f); \\\n"
            "delta.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
            "real scale2 = floor(delta.y*invPeriodicBoxSize.y+0.5f); \\\n"
            "delta.xy -= scale2*periodicBoxVecY.xy; \\\n"
            "real scale1 = floor(delta.x*invPeriodicBoxSize.x+0.5f); \\\n"
            "delta.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "{"
            "real scale3 = floor(pos.z*invPeriodicBoxSize.z); \\\n"
            "pos.xyz -= scale3*periodicBoxVecZ.xyz; \\\n"
            "real scale2 = floor(pos.y*invPeriodicBoxSize.y); \\\n"
            "pos.xy -= scale2*periodicBoxVecY.xy; \\\n"
            "real scale1 = floor(pos.x*invPeriodicBoxSize.x); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "real scale3 = floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f); \\\n"
            "pos.x -= scale3*periodicBoxVecZ.x; \\\n"
            "pos.y -= scale3*periodicBoxVecZ.y; \\\n"
            "pos.z -= scale3*periodicBoxVecZ.z; \\\n"
            "real scale2 = floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f); \\\n"
            "pos.x -= scale2*periodicBoxVecY.x; \\\n"
            "pos.y -= scale2*periodicBoxVecY.y; \\\n"
            "real scale1 = floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f); \\\n"
            "pos.x -= scale1*periodicBoxVecX.x;}";
    }
    else {
        compilationDefines["APPLY_PERIODIC_TO_DELTA(delta)"] =
            "delta.xyz -= floor(delta.xyz*invPeriodicBoxSize.xyz+0.5f)*periodicBoxSize.xyz;";
        compilationDefines["APPLY_PERIODIC_TO_POS(pos)"] =
            "pos.xyz -= floor(pos.xyz*invPeriodicBoxSize.xyz)*periodicBoxSize.xyz;";
        compilationDefines["APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)"] =
            "{"
            "pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x; \\\n"
            "pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y; \\\n"
            "pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;}";
    }

    // Create utilities objects.

    bonded = new OpenCLBondedUtilities(*this);
    nonbonded = new OpenCLNonbondedUtilities(*this);
    integration = new OpenCLIntegrationUtilities(*this, system);
    expression = new OpenCLExpressionUtilities(*this);
    
    // Garbage-collect autoreleased objects.
    commandPool->drain();
    commandPool = NS::AutoreleasePool::alloc()->init();
}

OpenCLContext::~OpenCLContext() {
    for (auto force : forces)
        delete force;
    for (auto listener : reorderListeners)
        delete listener;
    for (auto computation : preComputations)
        delete computation;
    for (auto computation : postComputations)
        delete computation;
    if (pinnedBuffer != NULL)
        delete pinnedBuffer;
    if (integration != NULL)
        delete integration;
    if (expression != NULL)
        delete expression;
    if (bonded != NULL)
        delete bonded;
    if (nonbonded != NULL)
        delete nonbonded;
}

void OpenCLContext::initialize() {
    bonded->initialize(system);
    numForceBuffers = std::max(numForceBuffers, (int) platformData.contexts.size());
    int energyBufferSize = max(numThreadBlocks*ThreadBlockSize, nonbonded->getNumEnergyBuffers());
    if (useDoublePrecision) {
        forceBuffers.initialize<mm_double4>(*this, paddedNumAtoms*numForceBuffers, "forceBuffers");
        force.initialize<mm_double4>(*this, &forceBuffers.getDeviceBuffer(), paddedNumAtoms, "force");
        energyBuffer.initialize<cl_double>(*this, energyBufferSize, "energyBuffer");
        energySum.initialize<cl_double>(*this, 1, "energySum");
    }
    else if (useMixedPrecision) {
        forceBuffers.initialize<mm_float4>(*this, paddedNumAtoms*numForceBuffers, "forceBuffers");
        force.initialize<mm_float4>(*this, &forceBuffers.getDeviceBuffer(), paddedNumAtoms, "force");
        energyBuffer.initialize<cl_double>(*this, energyBufferSize, "energyBuffer");
        energySum.initialize<cl_double>(*this, 1, "energySum");
    }
    else {
        forceBuffers.initialize<mm_float4>(*this, paddedNumAtoms*numForceBuffers, "forceBuffers");
        force.initialize<mm_float4>(*this, &forceBuffers.getDeviceBuffer(), paddedNumAtoms, "force");
        energyBuffer.initialize<cl_float>(*this, energyBufferSize, "energyBuffer");
        energySum.initialize<cl_float>(*this, 1, "energySum");
    }
    reduceForcesKernel.setArg<MTL::Buffer>(0, longForceBuffer.getDeviceBuffer());
    reduceForcesKernel.setArg<MTL::Buffer>(1, forceBuffers.getDeviceBuffer());
    reduceForcesKernel.setArg<cl_int>(2, paddedNumAtoms);
    reduceForcesKernel.setArg<cl_int>(3, numForceBuffers);
    addAutoclearBuffer(longForceBuffer);
    addAutoclearBuffer(forceBuffers);
    addAutoclearBuffer(energyBuffer);
    int numEnergyParamDerivs = energyParamDerivNames.size();
    if (numEnergyParamDerivs > 0) {
        if (useDoublePrecision || useMixedPrecision)
            energyParamDerivBuffer.initialize<cl_double>(*this, numEnergyParamDerivs*energyBufferSize, "energyParamDerivBuffer");
        else
            energyParamDerivBuffer.initialize<cl_float>(*this, numEnergyParamDerivs*energyBufferSize, "energyParamDerivBuffer");
        addAutoclearBuffer(energyParamDerivBuffer);
    }
    int bufferBytes = max(max((int) velm.getSize()*velm.getElementSize(),
            energyBufferSize*energyBuffer.getElementSize()),
            (int) longForceBuffer.getSize()*longForceBuffer.getElementSize());
    pinnedBuffer = NS::TransferPtr(MTL::Buffer((context, CL_MEM_ALLOC_HOST_PTR, bufferBytes));
    pinnedMemory = currentQueue.enqueueMapBuffer(*pinnedBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bufferBytes);
    for (int i = 0; i < numAtoms; i++) {
        double mass = system.getParticleMass(i);
        if (useDoublePrecision || useMixedPrecision)
            ((mm_double4*) pinnedMemory)[i] = mm_double4(0.0, 0.0, 0.0, mass == 0.0 ? 0.0 : 1.0/mass);
        else
            ((mm_float4*) pinnedMemory)[i] = mm_float4(0.0f, 0.0f, 0.0f, mass == 0.0 ? 0.0f : (cl_float) (1.0/mass));
    }
    velm.upload(pinnedMemory);
    findMoleculeGroups();
    nonbonded->initialize(system);
    
    // Garbage-collect autoreleased objects.
    commandPool->drain();
    commandPool = NS::AutoreleasePool::alloc()->init();
}

void OpenCLContext::initializeContexts() {
    getPlatformData().initializeContexts(system);
}

void OpenCLContext::addForce(ComputeForceInfo* force) {
    ComputeContext::addForce(force);
    OpenCLForceInfo* clinfo = dynamic_cast<OpenCLForceInfo*>(force);
    if (clinfo != NULL)
        requestForceBuffers(clinfo->getRequiredForceBuffers());
}

void OpenCLContext::requestForceBuffers(int minBuffers) {
    numForceBuffers = std::max(numForceBuffers, minBuffers);
}

void OpenCLContext::maybeFlushCommands(bool forceFlush, bool waitOnFlush) {
    // forceFlush works even if no commands are buffered.
    if (forceFlush || (numBufferedCommands >= maxBufferedCommands)) {
        if (computeEncoder) {
            computeEncoder->endEncoding();
            computeEncoder = nullptr;
            if (blitEncoder)
                throw OpenMMException("Unexpected blit encoder.");
        } else if (blitEncoder) {
            blitEncoder->endEncoding();
            blitEncoder = nullptr;
            if (computeEncoder)
                throw OpenMMException("Unexpected compute encoder.");
        }
        commandBuffer->commit();
        if (waitOnFlush) {
            commandBuffer->waitUntilCompleted();
        }
        commandPool->drain();
        numBufferedCommands = 0;
        commandPool = NS::AutoreleasePool::alloc()->init();
        commandBuffer = commandQueue->commandBuffer();
    }
}

MTL::ComputeCommandEncoder* OpenCLContext::nextComputeCommand() {
    maybeFlushCommands();
    if (blitEncoder) {
        blitEncoder->endEncoding();
        blitEncoder = nullptr;
        if (computeEncoder)
            throw OpenMMException("Unexpected compute encoder.");
    }
    if (!computeEncoder) {
        computeEncoder = commandBuffer->computeCommandEncoder();
    }
    numBufferedCommands += 1;
    return computeEncoder;
}

MTL::ComputeCommandEncoder* OpenCLContext::nextBlitCommand() {
    maybeFlushCommands();
    if (computeEncoder) {
        computeEncoder->endEncoding();
        computeEncoder = nullptr;
        if (blitEncoder)
            throw OpenMMException("Unexpected blit encoder.");
    }
    if (!blitEncoder) {
        blitEncoder = commandBuffer->blitCommandEncoder();
    }
    numBufferedCommands += 1;
    return blitEncoder;
}

dispatch_semaphore_t OpenCLContext::createSemaphoreAndFlush() {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    commandBuffer->addCompletedHandler(^void(MTL::CommandBuffer* commandBuffer) {
        dispatch_semaphore_signal(semaphore);
    });
    maybeFlushCommands(true);
    return semaphore;
}

MTL::Buffer* OpenCLContext::newTemporaryBuffer(void *ptr, int64_t size, int64_t* offset) {
    auto address = (uintptr_t)ptr;
    auto rounded_address = address & uintptr_t(vm_page_size - 1);
    *offset = address - rounded_address;
    auto rounded_size = (address - rounded_address) + size;
    auto rounded_ptr = (void*)rounded_address;
    return device->newBuffer(
        rounded_ptr, rounded_size, MTL::ResourceStorageModeShared, NULL);
}

MTL::Library* OpenCLContext::createProgram(const string source, MTL::CompileOptions* optimizationFlags) {
    return createProgram(source, map<string, string>(), optimizationFlags);
}

MTL::Library* OpenCLContext::createProgram(const string source, const map<string, string>& defines, MTL::CompileOptions* optimizationFlags) {
    MTL::CompileOptions options = (optimizationFlags == NULL)
        ? defaultOptimizationOptions : optimizationFlags;
    stringstream src;
    src << "// Compilation Options: ";
    if (options->fastMathEnabled())
        src << "-ffast-math" << " ";
    if (options->preserveInvariance())
        src << "-fpreserve-invariance" << " ";
    MTL::LanguageVersion version = options->languageVersion();
    NSUInteger versionMajor = version >> 16;
    NSUInteger versionMinor = version - (versionMajor << 16);
    src << "-std=metal" << std::to_string(versionMajor) << "." << std::to_string(versionMinor) << " ";
    
    NS::Dictionary* macros = options->preprocessorMacros();
    NS::Enumerator* macros_enumerator = macros->keyEnumerator();
    while (true) {
        NS::String* key = macros_enumerator->nextObject();
        if (!key) {
            break;
        }
        NS::String* value = macros->object(key);
        const char* key_description =
            key->description()->cString(NS::UTF8StringEncoding);
        const char* value_description =
            object->description()->cString(NS::UTF8StringEncoding);
        src << "-D" << std::string(key_description);
        src << "=" << std::string(value_description) << " ";
    }
    switch (options->optimizationLevel()) {
        case MTL::LibraryOptimizationLevelDefault: {
            src << "-O2" << " ";
            break;
        }
        case MTL::LibraryOptimizationLevelSize: {
            src << "-Os" << " ";
            break;
        }
    }
    
    NS::Array* libraries = options->libraries();
    for (NS::UInteger i = 0; i < libraries->count(); ++i) {
        MTL::DynamicLibrary* value = libraries->object(i);
        NS::String* name = value->installName();
        const char* name_description =
            name->description()->cString(NS::UTF8StringEncoding);
        src << "-l" << std::string(name_description) << " ";
    }
    if (options->libraryType() == MTL::LibraryTypeDynamic) {
        src << "-dynamiclib" << " ";
        NS::String* name = options->installName();
        const char* name_description =
            name->description()->cString(NS::UTF8StringEncoding);
        src << "-install_name=" << std::string(name_description) << " ";
    }
    
    src << endl << endl;
    for (auto& pair : compilationDefines) {
        // Query defines to avoid duplicate variables
        if (defines.find(pair.first) == defines.end()) {
            src << "#define " << pair.first;
            if (!pair.second.empty())
                src << " " << pair.second;
            src << endl;
        }
    }
    if (!compilationDefines.empty())
        src << endl;
    if (supportsDoublePrecision)
        src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    {
        src << "typedef float real;\n";
        src << "typedef float2 real2;\n";
        src << "typedef float3 real3;\n";
        src << "typedef float4 real4;\n";
    }
    {
        src << "typedef float mixed;\n";
        src << "typedef float2 mixed2;\n";
        src << "typedef float3 mixed3;\n";
        src << "typedef float4 mixed4;\n";
    }
    src << OpenCLKernelSources::common << endl;
    for (auto& pair : defines) {
        src << "#define " << pair.first;
        if (!pair.second.empty())
            src << " " << pair.second;
        src << endl;
    }
    if (!defines.empty())
        src << endl;
    src << source << endl;
    
    // We need an encapsulating autoreleasepool because the string and error are
    // autoreleased. The caller should ensure an autoreleasepool is active.
    NS::Error* error;
    auto sources = NS::String::string(sources.c_str(), NS::UTF8StringEncoding);
    MTL::Library* program = device->newLibrary(sources, options, &error);
    if (error) {
        const char* error_description =
            error->localizedDescription()->cString(NS::UTF8StringEncoding);
        throw OpenMMException(
            "Error compiling kernel: " + std::string(error_description));
    }
    return program;
}

OpenCLArray* OpenCLContext::createArray() {
    return new OpenCLArray();
}

ComputeEvent OpenCLContext::createEvent() {
    return shared_ptr<ComputeEventImpl>(new OpenCLEvent(*this));
}

ComputeProgram OpenCLContext::compileProgram(const std::string source, const std::map<std::string, std::string>& defines) {
    MTL::Library* program = createProgram(source, defines);
    return shared_ptr<ComputeProgramImpl>(new OpenCLProgram(*this, program));
}

OpenCLArray& OpenCLContext::unwrap(ArrayInterface& array) const {
    OpenCLArray* clarray;
    ComputeArray* wrapper = dynamic_cast<ComputeArray*>(&array);
    if (wrapper != NULL)
        clarray = dynamic_cast<OpenCLArray*>(&wrapper->getArray());
    else
        clarray = dynamic_cast<OpenCLArray*>(&array);
    if (clarray == NULL)
        throw OpenMMException("Array argument is not an OpenCLArray");
    return *clarray;
}

void OpenCLContext::executeKernel(OpenCLKernel kernel, int workUnits, int blockSize) {
    if (blockSize == -1)
        blockSize = ThreadBlockSize;
    int size = std::min((workUnits+blockSize-1)/blockSize, numThreadBlocks)*blockSize;
    try {
        kernel.execute();
        currentQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NDRange(blockSize));
    }
    catch (cl::Error err) {
        stringstream str;
        str<<"Error invoking kernel "<<kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()<<": "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }
}

int OpenCLContext::computeThreadBlockSize(double memory) const {
#if defined(__aarch64__)
    // Many assignable sizes divide into 60 KB/core, instead of 64 KB/core.
    int maxShared = 64 * 1024;
    int groupsPerCore = maxShared / int(memory);
    bool groupsPower2 = std::popcount(uint32_t(groupsPerCore)) == 1;
    bool sharedIsSmall = memory < 2 * 1024; // roundUp(65536/(3072/64))
    if ((!groupsPower2 || sharedIsSmall) &&
        (groupsPerCore * memory > 60 * 1024)) {
        groupsPerCore = 60 * 1024 / int(memory);
    }
    
    // The actual maximum is 3072, but register pressure can invisibly throttle
    // this to 2048.
    int threadsPerCore = 2048;
    int threadsPerThreadgroup = threadsPerCore / groupsPerCore;
    
    // TODO: Determine whether we should strongly favor 256 on M1.
    threadsPerThreadgroup = threadsPerThreadgroup & ~int(64 - 1);
    threadsPerThreadgroup = max(32, threadsPerThreadgroup);
    threadsPerThreadgroup = min(1024, threadsPerThreadgroup);
    return threadsPerThreadgroup;
#else
    int maxShared = device->maxThreadgroupMemoryLength();
    // On some implementations, more local memory gets used than we calculate by
    // adding up the sizes of the fields.  To be safe, include a factor of 0.5.
    int max = (int) (0.5*maxShared/memory);
    if (max < 64)
        return 32;
    int threads = 64;
    while (threads+64 < max)
        threads += 64;
    return threads;
#endif
}

void OpenCLContext::clearBuffer(ArrayInterface& array) {
    clearBuffer(unwrap(array).getDeviceBuffer(), array.getSize()*array.getElementSize());
}

void OpenCLContext::clearBuffer(MTL::Buffer* memory, int size) {
    int words = size/4;
    clearBufferKernel.setArg<MTL::Buffer>(0, memory);
    clearBufferKernel.setArg<cl_int>(1, words);
    executeKernel(clearBufferKernel, words, 128);
}

void OpenCLContext::addAutoclearBuffer(ArrayInterface& array) {
    addAutoclearBuffer(unwrap(array).getDeviceBuffer(), array.getSize()*array.getElementSize());
}

void OpenCLContext::addAutoclearBuffer(MTL::Buffer* memory, int size) {
    autoclearBuffers.push_back(&memory);
    autoclearBufferSizes.push_back(size/4);
}

void OpenCLContext::clearAutoclearBuffers() {
    int base = 0;
    int total = autoclearBufferSizes.size();
    while (total-base >= 6) {
        clearSixBuffersKernel.setArg<MTL::Buffer>(0, *autoclearBuffers[base]);
        clearSixBuffersKernel.setArg<cl_int>(1, autoclearBufferSizes[base]);
        clearSixBuffersKernel.setArg<MTL::Buffer>(2, *autoclearBuffers[base+1]);
        clearSixBuffersKernel.setArg<cl_int>(3, autoclearBufferSizes[base+1]);
        clearSixBuffersKernel.setArg<MTL::Buffer>(4, *autoclearBuffers[base+2]);
        clearSixBuffersKernel.setArg<cl_int>(5, autoclearBufferSizes[base+2]);
        clearSixBuffersKernel.setArg<MTL::Buffer>(6, *autoclearBuffers[base+3]);
        clearSixBuffersKernel.setArg<cl_int>(7, autoclearBufferSizes[base+3]);
        clearSixBuffersKernel.setArg<MTL::Buffer>(8, *autoclearBuffers[base+4]);
        clearSixBuffersKernel.setArg<cl_int>(9, autoclearBufferSizes[base+4]);
        clearSixBuffersKernel.setArg<MTL::Buffer>(10, *autoclearBuffers[base+5]);
        clearSixBuffersKernel.setArg<cl_int>(11, autoclearBufferSizes[base+5]);
        executeKernel(clearSixBuffersKernel, max(max(max(max(max(autoclearBufferSizes[base], autoclearBufferSizes[base+1]), autoclearBufferSizes[base+2]), autoclearBufferSizes[base+3]), autoclearBufferSizes[base+4]), autoclearBufferSizes[base+5]), 128);
        base += 6;
    }
    if (total-base == 5) {
        clearFiveBuffersKernel.setArg<MTL::Buffer>(0, *autoclearBuffers[base]);
        clearFiveBuffersKernel.setArg<cl_int>(1, autoclearBufferSizes[base]);
        clearFiveBuffersKernel.setArg<MTL::Buffer>(2, *autoclearBuffers[base+1]);
        clearFiveBuffersKernel.setArg<cl_int>(3, autoclearBufferSizes[base+1]);
        clearFiveBuffersKernel.setArg<MTL::Buffer>(4, *autoclearBuffers[base+2]);
        clearFiveBuffersKernel.setArg<cl_int>(5, autoclearBufferSizes[base+2]);
        clearFiveBuffersKernel.setArg<MTL::Buffer>(6, *autoclearBuffers[base+3]);
        clearFiveBuffersKernel.setArg<cl_int>(7, autoclearBufferSizes[base+3]);
        clearFiveBuffersKernel.setArg<MTL::Buffer>(8, *autoclearBuffers[base+4]);
        clearFiveBuffersKernel.setArg<cl_int>(9, autoclearBufferSizes[base+4]);
        executeKernel(clearFiveBuffersKernel, max(max(max(max(autoclearBufferSizes[base], autoclearBufferSizes[base+1]), autoclearBufferSizes[base+2]), autoclearBufferSizes[base+3]), autoclearBufferSizes[base+4]), 128);
    }
    else if (total-base == 4) {
        clearFourBuffersKernel.setArg<MTL::Buffer>(0, *autoclearBuffers[base]);
        clearFourBuffersKernel.setArg<cl_int>(1, autoclearBufferSizes[base]);
        clearFourBuffersKernel.setArg<MTL::Buffer>(2, *autoclearBuffers[base+1]);
        clearFourBuffersKernel.setArg<cl_int>(3, autoclearBufferSizes[base+1]);
        clearFourBuffersKernel.setArg<MTL::Buffer>(4, *autoclearBuffers[base+2]);
        clearFourBuffersKernel.setArg<cl_int>(5, autoclearBufferSizes[base+2]);
        clearFourBuffersKernel.setArg<MTL::Buffer>(6, *autoclearBuffers[base+3]);
        clearFourBuffersKernel.setArg<cl_int>(7, autoclearBufferSizes[base+3]);
        executeKernel(clearFourBuffersKernel, max(max(max(autoclearBufferSizes[base], autoclearBufferSizes[base+1]), autoclearBufferSizes[base+2]), autoclearBufferSizes[base+3]), 128);
    }
    else if (total-base == 3) {
        clearThreeBuffersKernel.setArg<MTL::Buffer>(0, *autoclearBuffers[base]);
        clearThreeBuffersKernel.setArg<cl_int>(1, autoclearBufferSizes[base]);
        clearThreeBuffersKernel.setArg<MTL::Buffer>(2, *autoclearBuffers[base+1]);
        clearThreeBuffersKernel.setArg<cl_int>(3, autoclearBufferSizes[base+1]);
        clearThreeBuffersKernel.setArg<MTL::Buffer>(4, *autoclearBuffers[base+2]);
        clearThreeBuffersKernel.setArg<cl_int>(5, autoclearBufferSizes[base+2]);
        executeKernel(clearThreeBuffersKernel, max(max(autoclearBufferSizes[base], autoclearBufferSizes[base+1]), autoclearBufferSizes[base+2]), 128);
    }
    else if (total-base == 2) {
        clearTwoBuffersKernel.setArg<MTL::Buffer>(0, *autoclearBuffers[base]);
        clearTwoBuffersKernel.setArg<cl_int>(1, autoclearBufferSizes[base]);
        clearTwoBuffersKernel.setArg<MTL::Buffer>(2, *autoclearBuffers[base+1]);
        clearTwoBuffersKernel.setArg<cl_int>(3, autoclearBufferSizes[base+1]);
        executeKernel(clearTwoBuffersKernel, max(autoclearBufferSizes[base], autoclearBufferSizes[base+1]), 128);
    }
    else if (total-base == 1) {
        clearBuffer(*autoclearBuffers[base], autoclearBufferSizes[base]*4);
    }
}

void OpenCLContext::reduceForces() {
    executeKernel(reduceForcesKernel, paddedNumAtoms, 128);
}

void OpenCLContext::reduceBuffer(OpenCLArray& array, OpenCLArray& longBuffer, int numBuffers) {
    int bufferSize = array.getSize()/numBuffers;
    reduceReal4Kernel.setArg<MTL::Buffer>(0, array.getDeviceBuffer());
    reduceReal4Kernel.setArg<MTL::Buffer>(1, longBuffer.getDeviceBuffer());
    reduceReal4Kernel.setArg<cl_int>(2, bufferSize);
    reduceReal4Kernel.setArg<cl_int>(3, numBuffers);
    executeKernel(reduceReal4Kernel, bufferSize, 128);
}

double OpenCLContext::reduceEnergy() {
    int workGroupSize  = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    if (workGroupSize > 512)
        workGroupSize = 512;
    reduceEnergyKernel.setArg<MTL::Buffer>(0, energyBuffer.getDeviceBuffer());
    reduceEnergyKernel.setArg<MTL::Buffer>(1, energySum.getDeviceBuffer());
    reduceEnergyKernel.setArg<cl_int>(2, energyBuffer.getSize());
    reduceEnergyKernel.setArg<cl_int>(3, workGroupSize);
    reduceEnergyKernel.setArg(4, workGroupSize*energyBuffer.getElementSize(), NULL);
    executeKernel(reduceEnergyKernel, workGroupSize, workGroupSize);
    if (getUseDoublePrecision() || getUseMixedPrecision()) {
        double energy;
        energySum.download(&energy);
        return energy;
    }
    else {
        float energy;
        energySum.download(&energy);
        return energy;
    }
}

void OpenCLContext::setCharges(const vector<double>& charges) {
    if (!chargeBuffer.isInitialized())
        chargeBuffer.initialize(*this, numAtoms, useDoublePrecision ? sizeof(double) : sizeof(float), "chargeBuffer");
    vector<double> c(numAtoms);
    for (int i = 0; i < numAtoms; i++)
        c[i] = charges[i];
    chargeBuffer.upload(c, true);
    setChargesKernel.setArg<MTL::Buffer>(0, chargeBuffer.getDeviceBuffer());
    setChargesKernel.setArg<MTL::Buffer>(1, posq.getDeviceBuffer());
    setChargesKernel.setArg<MTL::Buffer>(2, atomIndexDevice.getDeviceBuffer());
    setChargesKernel.setArg<cl_int>(3, numAtoms);
    executeKernel(setChargesKernel, numAtoms);
}

bool OpenCLContext::requestPosqCharges() {
    bool allow = !hasAssignedPosqCharges;
    hasAssignedPosqCharges = true;
    return allow;
}

void OpenCLContext::addEnergyParameterDerivative(const string& param) {
    // See if this parameter has already been registered.
    
    for (int i = 0; i < energyParamDerivNames.size(); i++)
        if (param == energyParamDerivNames[i])
            return;
    energyParamDerivNames.push_back(param);
}

void OpenCLContext::flushQueue() {
    getQueue().flush();
}
