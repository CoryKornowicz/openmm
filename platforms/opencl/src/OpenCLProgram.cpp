/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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

#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

using namespace OpenMM;
using namespace std;

OpenCLProgram::OpenCLProgram(OpenCLContext& context, MTL::Library* program) : context(context) {
    this->program = NS::TransferPtr(program);
}

ComputeKernel OpenCLProgram::createKernel(const string& name) {
    NS::Error* error;
    auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(program->newFunction(ns_name, &error));
    if (error) {
        const char* error_description =
            error->localizedDescription()->cString(NS::UTF8StringEncoding);
        throw OpenMMException(
            "Error creating function: " + std::string(error_description));
    }
    
    // Set function name here, so we can recall it later.
    auto desc = NS::TransferPtr(
        MTL::ComputePipelineDescriptor::alloc()->init());
    desc->setLabel(ns_name);
    desc->setComputeFunction(function.get())
    desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
    desc->setMaxCallStackDepth(3);
    
    MTL::ComputePipelineState* pipeline =
        device->newComputePipelineState(desc.get(), 0, &error);
    if (error) {
        const char* error_description =
            error->localizedDescription()->cString(NS::UTF8StringEncoding);
        throw OpenMMException(
            "Error creating pipeline: " + std::string(error_description));
    }
    #if defined(__aarch64__)
    if (pipeline->maxThreadsPerThreadgroup() < 1024) {
        // Our assumptions were wrong; adjust the target threadgroup size.
        throw OpenMMException(
            "Compute pipeline on M1 with unexpected register pressure.")
    }
    #endif
    return shared_ptr<ComputeKernelImpl>(new OpenCLKernel(context, pipeline));
}
