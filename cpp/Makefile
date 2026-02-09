# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-20/build/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -Wall -Werror -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = $(HOME)/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources
DIALECT_SOURCES := c_api.cpp ImplResolution.cpp Instantiation.cpp Passes.cpp Trait.cpp TraitAttributes.cpp TraitOps.cpp TraitTypes.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Trait.hpp.inc Trait.cpp.inc \
	     TraitAttributes.hpp.inc TraitAttributes.cpp.inc \
	     TraitOps.hpp.inc TraitOps.cpp.inc \
	     TraitOpInterfaces.hpp.inc TraitOpInterfaces.cpp.inc \
	     TraitTypes.hpp.inc TraitTypes.cpp.inc \
	     TraitTypeInterfaces.hpp.inc TraitTypeInterfaces.cpp.inc

.PHONY: all clean

all: libtrait_dialect.a libtrait_dialect.so

# TableGen rules
Trait.hpp.inc: Trait.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Trait.cpp.inc: Trait.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

TraitAttributes.hpp.inc: TraitAttributes.td
	$(TBLGEN) --gen-attrdef-decls $(INCLUDES) $< -o $@

TraitAttributes.cpp.inc: TraitAttributes.td
	$(TBLGEN) --gen-attrdef-defs $(INCLUDES) $< -o $@

TraitOps.hpp.inc: TraitOps.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

TraitOps.cpp.inc: TraitOps.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

TraitOpInterfaces.hpp.inc: TraitOpInterfaces.td
	$(TBLGEN) --gen-op-interface-decls $(INCLUDES) $< -o $@

TraitOpInterfaces.cpp.inc: TraitOpInterfaces.td
	$(TBLGEN) --gen-op-interface-defs $(INCLUDES) $< -o $@

TraitTypes.hpp.inc: TraitTypes.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

TraitTypes.cpp.inc: TraitTypes.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

TraitTypeInterfaces.hpp.inc: TraitTypeInterfaces.td
	$(TBLGEN) --gen-type-interface-decls $(INCLUDES) $< -o $@

TraitTypeInterfaces.cpp.inc: TraitTypeInterfaces.td
	$(TBLGEN) --gen-type-interface-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

libtrait_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libtrait_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: libtrait_dialect.so
	@echo "Running trait dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
