// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::trait {

/// Whether `op` is a generic trait call instantiation can rewrite now -- the one
/// readiness law both call-lowering patterns gate on, spelled here once: for a
/// trait.func.call, monomorphic operands, proven operand claims, module scope,
/// and a callee with a signature; for a trait.method.call, monomorphic operands,
/// a proven receiver claim, proven argument claims, and a method with a
/// signature. False for any other op. This is the predicate the instantiate step
/// qualifies its discharge by, so the step is present exactly on the calls a
/// pattern would fire on.
bool isRewritableGenericCall(Operation *op);

/// Whether `op` is foreign code this compilation carries to no target: a
/// template (a trait, impl, or proof declaration, or a still-polymorphic
/// function) or code inside one. The readiness and leftover discipline reads
/// this so it never serves or judges what leaves with a template.
bool isForeign(Operation *op);

/// Whether `module` still carries instantiation work outside a template (a
/// trait, impl, or proof body, or a polymorphic function): a rewritable generic
/// call, or an unproven monomorphic application claim or an unresolved ground
/// projection instantiation has not yet discharged. This is the erase step's
/// readiness -- it may run only when this is false, the condition under which
/// nothing standing can still mention a template.
bool isPendingExpansion(ModuleOp module);

/// Runs monomorphization to completion in one pass: instantiates the monomorphs
/// every trait call needs and erases all residual polymorphism. The compiler
/// runs the two halves as separate passes (instantiate-monomorphs then
/// erase-polymorphs), but this fused housing survives because it is the only way
/// to run monomorphization to completion under a per-pass verifier: the state
/// between the two halves — monomorphs standing over the templates they were cut
/// from — does not recursively verify, so the split passes cannot compose under
/// a verifying pass manager. The lit rows that drive `monomorphize-trait`
/// through `mlir-opt`, and the integration test, run monomorphization this way.
struct MonomorphizePass : PassWrapper<MonomorphizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MonomorphizePass);

  inline StringRef getArgument() const final { return "monomorphize-trait"; }
  inline StringRef getDescription() const final { return "Instantiate monomorphs for trait calls and erase all polymorphs."; }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createMonomorphizePass();

/// The first half of monomorphization: instantiates the monomorphs every trait
/// call needs and proves the monomorphic claims, leaving the polymorphic
/// templates standing for erase-polymorphs to remove. Its output is
/// mid-transformation and is not guaranteed to verify — monomorphs can stand
/// over their templates, which the verifier rejects, as inputs across the srcc
/// corpus exhibit — so it runs either with its pass manager's verifier off or
/// immediately before erase-polymorphs.
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

/// The structural acyclicity screen, split from the full check's verify tail.
/// Walks the trait-to-trait `where`-clause edges and reports a cycle, resolving
/// each edge's target trait by name and refusing a dangling reference through a
/// diagnostic rather than the aborting trait accessor. It is safe to run on
/// unverified IR ahead of the full verifier -- a launch screening a frozen blob
/// runs it before `module.verify()`.
LogicalResult verifyAcyclicTraitsStructure(ModuleOp module);

/// The full acyclicity check: the structural screen above followed by a full
/// module verify. `verify-acyclic-traits` and the monomorphize pass run this.
LogicalResult verifyAcyclicTraits(ModuleOp module);

struct VerifyAcyclicTraitsPass : PassWrapper<VerifyAcyclicTraitsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAcyclicTraitsPass);

  inline StringRef getArgument() const final { return "verify-acyclic-traits"; }
  inline StringRef getDescription() const final { return "Verify that the trait dependency graph is acyclic."; }

  void runOnOperation() override;
};

}
