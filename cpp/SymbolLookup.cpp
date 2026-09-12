// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "SymbolLookup.hpp"
#include <llvm/ADT/DenseMap.h>

namespace mlir::trait {

/// A symbol table operation and a name it was asked about, paired with what it
/// answered.
struct HeldSymbolAnswers {
  llvm::DenseMap<std::pair<Operation *, StringAttr>, Operation *> answers;
};

namespace {
/// The answers reads on this thread go through, null where no scope is
/// installed and every read is a scan.
thread_local HeldSymbolAnswers *installed = nullptr;
} // namespace

SymbolLookupScope::SymbolLookupScope() {
  if (installed)
    return;
  held = std::make_unique<HeldSymbolAnswers>();
  installed = held.get();
}

SymbolLookupScope::~SymbolLookupScope() {
  if (held)
    installed = nullptr;
}

void forgetHeldSymbols() {
  if (installed)
    installed->answers.clear();
}

Operation *lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name) {
  if (!module || !name)
    return nullptr;

  StringAttr leaf = name.getAttr();
  std::pair<Operation *, StringAttr> asked{module.getOperation(), leaf};
  if (installed) {
    auto held = installed->answers.find(asked);
    if (held != installed->answers.end())
      return held->second;
  }

  if (Operation *here =
          SymbolTable::lookupSymbolIn(module.getOperation(), leaf)) {
    if (installed)
      installed->answers.insert({asked, here});
    return here;
  }

  Operation *around = module->getParentWithTrait<OpTrait::SymbolTable>();
  return around ? SymbolTable::lookupNearestSymbolFrom(around, name) : nullptr;
}

} // namespace mlir::trait
