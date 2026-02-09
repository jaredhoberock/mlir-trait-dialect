// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"

#include "Trait.cpp.inc"

namespace mlir::trait {

void TraitDialect::initialize() {
  registerAttributes();

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "TraitOps.cpp.inc"
  >();
}

}
