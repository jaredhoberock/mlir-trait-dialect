// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::trait {

struct MonomorphizePass : PassWrapper<MonomorphizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MonomorphizePass);

  inline StringRef getArgument() const final { return "monomorphize-trait"; }
  inline StringRef getDescription() const final { return "Instantiate monomorphs for trait calls and erase all polymorphs."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createMonomorphizePass();

struct InstantiateMonomorphsPass : PassWrapper<InstantiateMonomorphsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstantiateMonomorphsPass);

  inline StringRef getArgument() const final { return "instantiate-monomorphs-trait"; }
  inline StringRef getDescription() const final { return "Instantiate monomorphs for trait calls."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createInstantiateMonomorphsPass();

/// Erases all residual polymorphism from the module, the second half of
/// monomorphization: it runs after instantiate-monomorphs has proved every
/// monomorphic claim, and erases the trait templates, the claims and
/// projections, and the polymorphic function signatures they stood on.
struct ErasePolymorphsPass : PassWrapper<ErasePolymorphsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ErasePolymorphsPass);

  inline StringRef getArgument() const final { return "erase-polymorphs-trait"; }
  inline StringRef getDescription() const final { return "Erase all residual polymorphism from the module."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createErasePolymorphsPass();

/// Monomorph instantiation with a pattern that puts a claim a function's
/// signature declares to impl selection from inside the instantiation driver.
///
/// The freeze standing over that driver turns any such ask into a fatal, and
/// the driver's own patterns never make one, so this is what exercises the
/// freeze. Only the dialect's plugin registers it: nothing the compiler creates
/// can reach it, and nothing it does belongs in a compilation.
struct AskImplSelectionDuringInstantiationPass
    : PassWrapper<AskImplSelectionDuringInstantiationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AskImplSelectionDuringInstantiationPass);

  inline StringRef getArgument() const final { return "ask-impl-selection-during-instantiation-trait"; }
  inline StringRef getDescription() const final { return "Instantiate monomorphs, asking impl selection for an impl from inside the driver."; }

  void runOnOperation() override;
};

/// Round zero on its own, for the rows that drive it through `mlir-opt`.
///
/// A compilation reaches round zero through instantiate-monomorphs, which runs
/// it before its first round and keeps the resolver it built; this pass runs it
/// alone and discards that resolver.
///
/// XXX TODO: this housing exists for those rows and for nothing the compiler
/// builds. It goes when round zero dissolves into the round loop and the rows
/// drive the loop instead.
struct ResolveImplsPass : PassWrapper<ResolveImplsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveImplsPass);

  inline StringRef getArgument() const final { return "resolve-impls-trait"; }
  inline StringRef getDescription() const final { return "Elaborate claims into proofs of implementations by resolving impls."; }

  void runOnOperation() override;
};

struct VerifyAcyclicTraitsPass : PassWrapper<VerifyAcyclicTraitsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAcyclicTraitsPass);

  inline StringRef getArgument() const final { return "verify-acyclic-traits"; }
  inline StringRef getDescription() const final { return "Verify that the trait dependency graph is acyclic."; }

  void runOnOperation() override;
};

}
