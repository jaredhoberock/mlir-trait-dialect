// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::trait {

/// The operation `name` names, read from `module` outward: `module`'s own
/// symbol table first and then the tables around it, which is what
/// `SymbolTable::lookupNearestSymbolFrom` answers.
///
/// A symbol table holds no index of its names, so each read is a scan of the
/// whole module, and reading the evidence at one site resolves the same handful
/// of names once per call site and once per node of every proof tree it walks.
/// Under a `SymbolLookupScope` the answers `module`'s own table gives are held
/// for that scope's span, so a name is scanned for once there.
Operation *lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name);

/// The above, as the operation kind the caller expects, null where the name
/// names something else.
template <typename OpT>
OpT lookupSymbolFrom(ModuleOp module, FlatSymbolRefAttr name) {
  return dyn_cast_or_null<OpT>(lookupSymbolFrom(module, name));
}

/// The answers a symbol lookup scope holds.
struct HeldSymbolAnswers;

/// Forgets every answer the installed scope holds, for a caller that has taken
/// a symbol out of the IR or renamed one.
///
/// A rewrite driver reports every operation it erases, and that report is where
/// this is called from, so a scope spanning a stage that rewrites holds nothing
/// an erasure has moved.
void forgetHeldSymbols();

/// Holds the name-to-operation answers taken over a span of reads.
///
/// A symbol table's names are unique, so the operation a name binds there is
/// what it binds until something erases that operation or renames it.
/// Appending a symbol moves no answer: a name a table already binds keeps
/// binding what it bound. That is what lets a scope span a stage that writes,
/// and it is why only the answers `module`'s own table gives are held -- an
/// answer a table around it gives is taken again each time, because a symbol
/// appended to `module` would bind the name there instead and a held answer
/// could not say so. A stage takes a symbol back out only through a rewrite
/// driver, which reports the erasure to `forgetHeldSymbols`.
///
/// Scopes nest, and one entered under another reads and writes the answers
/// already installed, so what a caller took serves the reads its callees make.
/// The install is per thread, so a verifier running on a worker thread holds
/// its own and shares none.
class SymbolLookupScope {
public:
  SymbolLookupScope();
  ~SymbolLookupScope();

  SymbolLookupScope(const SymbolLookupScope &) = delete;
  SymbolLookupScope &operator=(const SymbolLookupScope &) = delete;

private:
  /// The answers this scope installed, null when a scope was already installed
  /// and this one reads through that one.
  std::unique_ptr<HeldSymbolAnswers> held;
};

} // namespace mlir::trait
