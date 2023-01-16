/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2012-2022 Stanford University and the Authors.      *
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

#include "OpenCLArray.h"
#include "OpenCLContext.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace OpenMM;

OpenCLArray::OpenCLArray() : buffer(NULL), ownsBuffer(false) {
}

OpenCLArray::OpenCLArray(OpenCLContext& context, size_t size, int elementSize, const std::string& name, MTL::ResourceOptions flags) : buffer(NULL) {
    initialize(context, size, elementSize, name, flags);
}

OpenCLArray::OpenCLArray(OpenCLContext& context, MTL::Buffer* buffer, size_t size, int elementSize, const std::string& name) : buffer(NULL) {
    initialize(context, buffer, size, elementSize, name);
}

OpenCLArray::~OpenCLArray() {
    if (buffer != NULL && ownsBuffer)
        buffer->release();
}

void OpenCLArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
    initialize(dynamic_cast<OpenCLContext&>(context), size, elementSize, name, MTL::ResourceStorageModePrivate);
}

void OpenCLArray::initialize(OpenCLContext& context, size_t size, int elementSize, const std::string& name, MTL::ResourceOptions flags) {
    if (buffer != NULL)
        throw OpenMMException("OpenCLArray has already been initialized");
    this->context = &context;
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    this->flags = flags;
    ownsBuffer = true;
    try {
        MTL::Device* device = context->getDevice();
        buffer = device->makeBuffer(size * elementSize, flags);
        
        // Copy name to Metal resource label, to help with debugging.
        auto string = NS::String::string(name.c_str, NS::UTF8StringEncoding);
        buffer->setLabel(string);
    }
    catch (cl::Error err) {
        std::stringstream str;
        str<<"Error creating array "<<name<<": "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }
}

void OpenCLArray::initialize(OpenCLContext& context, MTL::Buffer* buffer, size_t size, int elementSize, const std::string& name) {
    if (this->buffer != NULL)
        throw OpenMMException("OpenCLArray has already been initialized");
    this->context = &context;
    this->buffer = buffer;
    this->size = size;
    this->elementSize = elementSize;
    this->name = name;
    ownsBuffer = false;
}

void OpenCLArray::resize(size_t size) {
    if (buffer == NULL)
        throw OpenMMException("OpenCLArray has not been initialized");
    if (!ownsBuffer)
        throw OpenMMException("Cannot resize an array that does not own its storage");
    buffer->release();
    buffer = NULL;
    initialize(*context, size, elementSize, name, flags);
}

ComputeContext& OpenCLArray::getContext() {
    return *context;
}

void OpenCLArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
    if (buffer == NULL)
        throw OpenMMException("OpenCLArray has not been initialized");
    if (offset < 0 || offset+elements > getSize())
        throw OpenMMException("uploadSubArray: data exceeds range of array");
    try {
        int64_t actual_offset = offset * elementSize;
        int64_t actual_elements = elements * elementSize
        int64_t temp_offset;
        auto temp_buffer = NS::TransferPtr(context->newTemporaryBuffer(
           ((const char*)data) + actual_offset, actual_elements, &temp_offset));
        
        MTL::BlitCommandEncoder* encoder = context->nextBlitCommand();
        encoder->copyFromBuffer(
            temp_buffer.get(), temp_offset, buffer, actual_offset,
            actual_elements);
        if (blocking) {
            context->maybeFlushCommands(true, true);
        }
    }
    catch (cl::Error err) {
        std::stringstream str;
        str<<"Error uploading array "<<name<<": "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }
}

void OpenCLArray::download(void* data, bool blocking) const {
    if (buffer == NULL)
        throw OpenMMException("OpenCLArray has not been initialized");
    try {
        int64_t actual_offset = 0;
        int64_t actual_elements = size * elementSize;
        int64_t temp_offset;
        auto temp_buffer = NS::TransferPtr(context->newTemporaryBuffer(
           ((const char*)data) + actual_offset, actual_elements, &temp_offset));
        
        MTL::BlitCommandEncoder* encoder = context->nextBlitCommand();
        encoder->copyFromBuffer(
            buffer, actual_offset, temp_buffer.get(), temp_offset,
            actual_elements);
        if (blocking) {
            context->maybeFlushCommands(true, true);
        }
    }
    catch (cl::Error err) {
        std::stringstream str;
        str<<"Error downloading array "<<name<<": "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }
}

void OpenCLArray::copyTo(ArrayInterface& dest) const {
    if (buffer == NULL)
        throw OpenMMException("OpenCLArray has not been initialized");
    if (dest.getSize() != size || dest.getElementSize() != elementSize)
        throw OpenMMException("Error copying array "+name+" to "+dest.getName()+": The destination array does not match the size of the array");
    OpenCLArray& clDest = context->unwrap(dest);
    try {
        int64_t actual_offset = 0;
        int64_t actual_elements = size * elementSize;
        
        MTL::BlitCommandEncoder* encoder = context->nextBlitCommand();
        encoder->copyFromBuffer(
            buffer, actual_offset, clDest->buffer, actual_offset,
            actual_elements);
    }
    catch (cl::Error err) {
        std::stringstream str;
        str<<"Error copying array "<<name<<" to "<<dest.getName()<<": "<<err.what()<<" ("<<err.err()<<")";
        throw OpenMMException(str.str());
    }
}
