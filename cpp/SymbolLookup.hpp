// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::trait {

/// The operation `name` names in `module`'s own symbol table, which is what
/// `SymbolTable::lookupSymbolIn` answers. A table around `module` is never
/// asked: a name is answered in the module it was read in, so a nested module
/// that spells a name the module around it also spells means its own, and a
/// name `module` does not bind is unresolved here.
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
/// what it binds until something erases that operation, renames it, or moves it
/// elsewhere. Appending a symbol moves no answer: a name a table already binds
/// keeps binding what it bound, and a read that found nothing is held by
/// nothing, so a symbol appended under a name nothing bound is found by the
/// read after it. That is what lets a scope span a stage that writes. A held
/// answer is given back only while the operation it names still stands in that
/// module under that name, which is read off the operation itself; a stage
/// takes a symbol back out only through a rewrite driver, which reports the
/// erasure to `forgetHeldSymbols`.
///
/// Scopes nest, and one entered under another reads and writes the answers
/// already installed, so what a caller took serves the reads its callees make.
/// The install is per thread, so a verifier running on a worker thread holds
/// its own and shares none.
class SymbolLookupScope {
public:
  /// A scope that holds what it reads, for a span that may append symbols.
  SymbolLookupScope();

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
  /// The answers this scope installed, null when a scope was already installed
  /// and this one reads through that one.
  std::unique_ptr<HeldSymbolAnswers> held;
};

} // namespace mlir::trait
