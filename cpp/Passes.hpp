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

struct ResolveImplsPass : PassWrapper<ResolveImplsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveImplsPass);

  inline StringRef getArgument() const final { return "resolve-impls-trait"; }
  inline StringRef getDescription() const final { return "Elaborate claims into proofs of implementations by resolving impls."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createResolveImplsPass();

struct VerifyMonomorphsPass : PassWrapper<VerifyMonomorphsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyMonomorphsPass);

  inline StringRef getArgument() const final { return "verify-monomorphs-trait"; }
  inline StringRef getDescription() const final { return "Check that monomorphic free functions do not leak polymorphic trait types."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createVerifyMonomorphsPass();

struct VerifyAcyclicTraitsPass : PassWrapper<VerifyAcyclicTraitsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAcyclicTraitsPass);

  inline StringRef getArgument() const final { return "verify-acyclic-traits"; }
  inline StringRef getDescription() const final { return "Verify that the trait dependency graph is acyclic."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createVerifyAcyclicTraitsPass();

struct ConvertToTraitPass : PassWrapper<ConvertToTraitPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToTraitPass);

  inline StringRef getArgument() const final { return "convert-to-trait"; }
  inline StringRef getDescription() const final { return "Convert operations of participating dialects into trait operations."; }

  void runOnOperation() override;
};

}
