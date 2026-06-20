# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ifndef MLIR_SYS_220_PREFIX
$(error MLIR_SYS_220_PREFIX must be set; point it at your LLVM/MLIR 22 install)
endif

BUILD_DIR := build
LLVM_BIN_PATH := $(MLIR_SYS_220_PREFIX)/bin
LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen

CXX := clang++
CXXFLAGS := -Wall -Werror -fPIC $(shell $(LLVM_CONFIG) --cxxflags)
INCLUDES := -I$(BUILD_DIR) -I. -I$(MLIR_SYS_220_PREFIX)/include

DIALECT_SOURCES := c_api.cpp ImplResolution.cpp Specialization.cpp Passes.cpp Trait.cpp TraitAttributes.cpp TraitOps.cpp TraitTypes.cpp
DIALECT_OBJECTS := $(addprefix $(BUILD_DIR)/,$(DIALECT_SOURCES:.cpp=.o))
PLUGIN_OBJECTS := $(DIALECT_OBJECTS) $(BUILD_DIR)/Plugin.o
TD_FILES := $(wildcard *.td)

GENERATED := \
  $(BUILD_DIR)/Trait.hpp.inc \
  $(BUILD_DIR)/Trait.cpp.inc \
  $(BUILD_DIR)/TraitAttributes.hpp.inc \
  $(BUILD_DIR)/TraitAttributes.cpp.inc \
  $(BUILD_DIR)/TraitOps.hpp.inc \
  $(BUILD_DIR)/TraitOps.cpp.inc \
  $(BUILD_DIR)/TraitTypes.hpp.inc \
  $(BUILD_DIR)/TraitTypes.cpp.inc \
  $(BUILD_DIR)/TraitTypeInterfaces.hpp.inc \
  $(BUILD_DIR)/TraitTypeInterfaces.cpp.inc

.PHONY: all clean test

all: $(BUILD_DIR)/libtrait_dialect.a $(BUILD_DIR)/libtrait_dialect.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/Trait.hpp.inc: Trait.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Trait.cpp.inc: Trait.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitAttributes.hpp.inc: TraitAttributes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-attrdef-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitAttributes.cpp.inc: TraitAttributes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-attrdef-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitOps.hpp.inc: TraitOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitOps.cpp.inc: TraitOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitTypes.hpp.inc: TraitTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitTypes.cpp.inc: TraitTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitTypeInterfaces.hpp.inc: TraitTypeInterfaces.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-type-interface-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TraitTypeInterfaces.cpp.inc: TraitTypeInterfaces.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-type-interface-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(GENERATED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libtrait_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

$(BUILD_DIR)/libtrait_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

test: $(BUILD_DIR)/libtrait_dialect.so
	TRAIT_DIALECT_PLUGIN=$(CURDIR)/$(BUILD_DIR)/libtrait_dialect.so lit tests

clean:
	rm -rf $(BUILD_DIR)
