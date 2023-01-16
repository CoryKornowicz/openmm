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

#include "OpenCLKernel.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace OpenMM;
using namespace std;

OpenCLKernel::OpenCLKernel(OpenCLContext& context, MTL::ComputePipelineState* pipeline) : context(context) {
    this->pipeline = NS::TransferPtr(pipeline);
}

string OpenCLKernel::getName() const {
    // Function name was stored here during pipeline creation.
    NS::String* functionName = pipeline->label();
    const char* c_str = functionName->cString(NS::UTF8StringEncoding);
    return std::string(c_str);
}

int OpenCLKernel::getMaxBlockSize() const {
    return pipeline->maxTotalThreadsPerThreadgroup();
}

void OpenCLKernel::execute(int threads, int blockSize) {
    // Set args that are specified by OpenCLArrays.  We can't do this earlier, because it's
    // possible resize() will get called on an array, causing its internal storage to be
    // recreated.
    
    if (blockSize % context.getSIMDWidth() != 0)
        throw OpenMMException("Threadgroup size not multiple of execution width.")
    
    MTL::ComputeCommandEncoder* encoder = context.nextComputeCommand();
    encoder->setComputePipelineState(pipeline.get());
    for (int i = 0; i < arrayArgs.size(); i++) {
        auto argument = arrayArgs[i];
        if (argument.maybeArray) {
            encoder->setBuffer(argument.maybeArray->getDeviceBuffer(), 0, i);
        } else if (argument.maybePrimitive.size() > 0) {
            auto primitive = argument.maybePrimitive;
            auto size = argument.maybePrimitiveSize;
            encoder->setBytes(primitive.data(), primitive.size(), i);
        }
    }
    encoder->dispatchThreads({threads, 1, 1}, {blockSize, 1, 1});
}

void OpenCLKernel::addArrayArg(ArrayInterface& value) {
    int index = arrayArgs.size();
    addEmptyArg();
    setArrayArg(index, value);
}

void OpenCLKernel::addPrimitiveArg(const void* value, int size) {
    int index = arrayArgs.size();
    addEmptyArg();
    setPrimitiveArg(index, value, size);
}

void OpenCLKernel::addEmptyArg() {
    Argument emptyArgument{ NULL, std::vector<uint8_t>() };
    arrayArgs.push_back(emptyArgument);
}

void OpenCLKernel::setArrayArg(int index, ArrayInterface& value) {
    ASSERT_VALID_INDEX(index, arrayArgs);
    Argument argument{ &context.unwrap(value), std::vector<uint8_t>() };
    arrayArgs[index] = argument;
}

void OpenCLKernel::setPrimitiveArg(int index, const void* value, int size) {
    ASSERT_VALID_INDEX(index, arrayArgs);
    std::vector<uint8_t> primitive;
    primitive.assign((const uint8_t*)value, (const uint8_t*)value + size);
    Argument argument{ NULL, primitive };
    arrayArgs[index] = argument;
}
