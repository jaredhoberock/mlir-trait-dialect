// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "SymbolLookup.hpp"

namespace mlir::trait {

struct InstalledSymbolTables {
  SymbolTableCollection *tables;

  /// The one symbol table `tables` is asked about under a verifier: the table
  /// the verifier's walk is over, whose names it checked to be unique before
  /// walking it. A read anchored anywhere else is a scan. Null where every
  /// table is asked.
  Operation *checkedTable = nullptr;
};

namespace {
/// The tables reads on this thread ask, null where no scope is installed and
/// every read is a scan.
thread_local InstalledSymbolTables *installed = nullptr;

/// The module whose table `op` is an entry of, null where `op` is no symbol
/// standing directly in a module.
ModuleOp moduleTableOf(Operation *op) {
  if (!op->hasAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return {};
  return dyn_cast_or_null<ModuleOp>(op->getParentOp());
}
} // namespace

void SymbolTableKeeper::notifyOperationInserted(Operation *op,
                                                OpBuilder::InsertPoint) {
  if (ModuleOp module = moduleTableOf(op))
    tables.getSymbolTable(module).insert(op);
}

void SymbolTableKeeper::notifyOperationErased(Operation *op) {
  if (ModuleOp module = moduleTableOf(op))
    tables.getSymbolTable(module).remove(op);
}

SymbolLookupScope::SymbolLookupScope(SymbolTableCollection &tables) {
  if (installed)
    return;
  held = std::make_unique<InstalledSymbolTables>(
      InstalledSymbolTables{&tables, nullptr});
  installed = held.get();
}

SymbolLookupScope::SymbolLookupScope(Operation *op,
                                     SymbolTableCollection &tables) {
  if (installed)
    return;
  Operation *around = op ? op->getParentOp() : nullptr;
  // A verifier reaching an op outside every symbol table asks no table.
  Operation *checked =
      around ? SymbolTable::getNearestSymbolTable(around) : nullptr;
  if (!checked)
    return;
  held = std::make_unique<InstalledSymbolTables>(
      InstalledSymbolTables{&tables, checked});
  installed = held.get();
}

SymbolLookupScope::~SymbolLookupScope() {
  if (held)
    installed = nullptr;
}

Operation *lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name) {
  if (!module || !name)
    return nullptr;
  Operation *table = module.getOperation();
  if (installed &&
      (!installed->checkedTable || installed->checkedTable == table))
    return installed->tables->lookupSymbolIn(table, name.getAttr());
  return SymbolTable::lookupSymbolIn(table, name.getAttr());
}

} // namespace mlir::trait
