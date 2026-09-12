// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "SymbolLookup.hpp"
#include <llvm/ADT/DenseMap.h>

namespace mlir::trait {

/// A symbol table operation and a name it was asked about, paired with what it
/// answered.
struct HeldSymbolAnswers {
  llvm::DenseMap<std::pair<Operation *, StringAttr>, Operation *> answers;

  /// The symbol tables a caller handed this scope, which are indexed and so
  /// answer in place of a scan, a name no table binds included.
  SymbolTableCollection *tables = nullptr;

  /// The one symbol table `tables` is asked about: the table the verifier's
  /// walk is over, whose names it checked to be unique before walking it. A
  /// read anchored anywhere else is a scan.
  Operation *checkedTable = nullptr;
};

namespace {
/// The answers reads on this thread go through, null where no scope is
/// installed and every read is a scan.
thread_local HeldSymbolAnswers *installed = nullptr;

/// Whether `answer` is what `table` binds `leaf` to: the operation still stands
/// in `table` and still carries that name. An answer held from before its
/// operation moved or was renamed fails this, so what a name once bound is
/// never given back as what it binds.
bool stillBinds(Operation *answer, Operation *table, StringAttr leaf) {
  return answer->getParentOp() == table &&
         answer->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()) ==
             leaf;
}
} // namespace

SymbolLookupScope::SymbolLookupScope() {
  if (installed)
    return;
  held = std::make_unique<HeldSymbolAnswers>();
  installed = held.get();
}

SymbolLookupScope::SymbolLookupScope(Operation *op,
                                     SymbolTableCollection &tables) {
  if (installed)
    return;
  held = std::make_unique<HeldSymbolAnswers>();
  held->tables = &tables;
  if (Operation *around = op ? op->getParentOp() : nullptr)
    held->checkedTable = SymbolTable::getNearestSymbolTable(around);
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

  Operation *table = module.getOperation();
  StringAttr leaf = name.getAttr();

  if (installed && installed->tables && installed->checkedTable == table)
    return installed->tables->lookupSymbolIn(table, leaf);

  std::pair<Operation *, StringAttr> asked{table, leaf};
  if (installed) {
    auto held = installed->answers.find(asked);
    if (held != installed->answers.end()) {
      if (stillBinds(held->second, table, leaf))
        return held->second;
      installed->answers.erase(held);
    }
  }

  Operation *here = SymbolTable::lookupSymbolIn(table, leaf);
  if (here && installed)
    installed->answers.insert({asked, here});
  return here;
}

} // namespace mlir::trait
