// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::trait {

/// The operation `name` names in `module`'s own symbol table, which is what
/// `SymbolTable::lookupSymbolIn` answers. A table around `module` is never
/// asked: a name is answered in the module it was read in, so a nested module
/// that spells a name the module around it also spells means its own, and a
/// name `module` does not bind is unresolved here.
///
/// A module holds no index of its names, so a read with no table to ask is a
/// scan of the whole module. Under a `SymbolLookupScope` the read asks the
/// scope's symbol table collection instead.
Operation *lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name);

/// The above, as the operation kind the caller expects, null where the name
/// names something else.
template <typename OpT>
OpT lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name) {
  return dyn_cast_or_null<OpT>(lookupSymbolFrom(module, name));
}

/// Keeps the module tables of `tables` true across a span that writes: a
/// symbol inserted into a module through a builder or rewriter this listens to
/// is entered into the module's table, which names it afresh where the module
/// binds its name already (`SymbolTable::insert`), and one erased from a module
/// is taken out of it.
class SymbolTableKeeper : public RewriterBase::Listener {
public:
  explicit SymbolTableKeeper(SymbolTableCollection &tables) : tables(tables) {}

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;
  void notifyOperationErased(Operation *op) override;

private:
  SymbolTableCollection &tables;
};

/// The symbol tables reads on this thread ask while it stands.
struct InstalledSymbolTables;

/// Installs a symbol table collection for the reads of a span.
///
/// Scopes nest, and one entered under another reads the tables already
/// installed, so what a caller built serves the reads its callees make. The
/// install is per thread, so a verifier running on a worker thread holds its
/// own and shares none.
class SymbolLookupScope {
public:
  /// A scope whose every read asks `tables`, for a span that writes symbols
  /// only through a builder or rewriter a `SymbolTableKeeper` of `tables`
  /// listens to.
  explicit SymbolLookupScope(SymbolTableCollection &tables);

  /// A scope that reads through `tables`, for a span in which nothing at all is
  /// written. A verifier is handed `op` and the tables its driver built for the
  /// walk it is one step of, so what an earlier step resolved serves this one,
  /// and what this one resolves serves the steps after it.
  ///
  /// That walk is over the symbol table enclosing `op`, and `tables` is asked
  /// about that one table alone. A name read in a module standing above it is
  /// scanned for instead: the verifier reaches an enclosed symbol table before
  /// the one around it, so the module above has not had its own names checked
  /// yet, and a table built over it would abort on the duplicate the verifier
  /// is about to diagnose.
  SymbolLookupScope(Operation *op, SymbolTableCollection &tables);

  ~SymbolLookupScope();

  SymbolLookupScope(const SymbolLookupScope &) = delete;
  SymbolLookupScope &operator=(const SymbolLookupScope &) = delete;

private:
  /// The tables this scope installed, null when a scope was already installed
  /// and this one reads through that one.
  std::unique_ptr<InstalledSymbolTables> held;
};

} // namespace mlir::trait
