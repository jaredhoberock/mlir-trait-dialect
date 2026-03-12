// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Instantiation.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::trait {

//===----------------------------------------------------------------------===//
// ConvertToTraitPass
//===----------------------------------------------------------------------===//

LogicalResult convertToTrait(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateConvertToTraitPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return success();
}

void ConvertToTraitPass::runOnOperation() {
  if (failed(convertToTrait(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertToTraitPass() {
  return std::make_unique<ConvertToTraitPass>();
}


//===----------------------------------------------------------------------===//
// VerifyMonomorphsPass
//===----------------------------------------------------------------------===//

LogicalResult verifyMonomorphs(ModuleOp module) {
  // forbid monomorphic functions from mentioning !trait.claim in their signatures
  for (auto f : module.getOps<func::FuncOp>()) {
    auto fnTy = f.getFunctionType();
    if (isMonomorphicType(fnTy)) {
      if (containsType<ClaimType>(fnTy))
        return f.emitOpError() << "free function monomorphs may not contain !trait.claim types";
    }
  }
  return success();
}

void VerifyMonomorphsPass::runOnOperation() {
  if (failed(verifyMonomorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createVerifyMonomorphsPass() {
  return std::make_unique<VerifyMonomorphsPass>();
}


//===----------------------------------------------------------------------===//
// VerifyAcyclicTraitsPass
//===----------------------------------------------------------------------===//

LogicalResult verifyAcyclicTraits(ModuleOp module) {
  enum class Status : uint8_t { NotSeen = 0, InPath, Done };
  DenseMap<TraitOp, Status> status;
  SmallVector<TraitOp, 16> stack;

  std::function<LogicalResult(TraitOp)> dfs = [&](TraitOp u) -> LogicalResult {
    Status &s = status[u];
    if (s == Status::InPath) {
      // back-edge: report the cycle u ... u
      auto it = llvm::find(stack, u);
      auto diag = u.emitError("cycle in trait `where` clause: ");
      for (auto i = it; i != stack.end(); ++i)
        diag << "@" << i->getSymName() << " -> ";
      diag << "@" << u.getSymName();
      return failure();
    }

    if (s == Status::Done) return success();

    s = Status::InPath;
    stack.push_back(u);

    for (auto &app : u.getRequirements()) {
      auto v = app.getTraitOrAbort(module, "verifyAcyclicTraits");
      // A requirement like @Trait[!trait.proj<@Trait[!S], "Assoc">] is a
      // syntactic self-reference, but not a real cycle: the projection resolves
      // to a concrete type during monomorphization, breaking the edge. Skip it
      // so that traits with bounded associated types (e.g. `type Assoc: Trait`)
      // don't falsely trigger the acyclicity check.
      if (v == u && containsType<ProjectionType>(app.getTypeArgs().front()))
        continue;
      if (failed(dfs(v))) return failure();
    }

    stack.pop_back();
    s = Status::Done;
    return success();
  };

  for (TraitOp t : module.getOps<TraitOp>()) {
    if (status.lookup(t) == Status::Done) continue;
    if (failed(dfs(t))) {
      return failure();
    }
  }

  return module.verify();
}

void VerifyAcyclicTraitsPass::runOnOperation() {
  if (failed(verifyAcyclicTraits(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createVerifyAcyclicTraitsPass() {
  return std::make_unique<VerifyAcyclicTraitsPass>();
}


//===----------------------------------------------------------------------===//
// ResolveImplsPass
//===----------------------------------------------------------------------===//

namespace {

static void applySubstitutionInPlace(const DenseMap<Type,Type>& subst, Operation* root) {
  if (subst.empty()) return;
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst);
  replacer.recursivelyReplaceElementsIn(root,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

struct ProveClaimPattern : OpRewritePattern<AllegeOp> {
  using OpRewritePattern::OpRewritePattern;

  // one ImplResolver per module; owned by the pass, passed by ref into this pattern
  ImplResolver& resolver;

  ProveClaimPattern(MLIRContext* ctx,
                    ImplResolver &resolver)
    : OpRewritePattern<AllegeOp>(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(AllegeOp op, PatternRewriter& rewriter) const override {
    auto errFn = [&] { return op.emitOpError(); };

    // build or reuse canonical evidence for this claim
    auto sym = resolver.resolveAndEnsureProofFor(op.getClaim(), rewriter, errFn);
    if (failed(sym))
      return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");

    // replace the allegation with a witness
    rewriter.replaceOpWithNewOp<WitnessOp>(
      op,
      *sym,
      op.getClaim().getTraitApplication()
    );

    return success();
  }
};

} // end namespace

FailureOr<ImplResolver> resolveImpls(ModuleOp module) {
  // run convert-to-trait patterns
  if (failed(convertToTrait(module)))
    return failure();

  // verify that monomorphs are legal
  if (failed(verifyMonomorphs(module)))
    return failure();

  // verify traits are acyclic
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // an ImplResolver for this module
  ImplResolver resolver(module);

  MLIRContext *ctx = module.getContext();

  // apply rewrite patterns
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimPattern>(ctx, resolver);

    // rewrite trait.allege -> trait.witness
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
  }

  // assert that no trait.allege remain
  bool hasLeftovers = false;
  module.walk([&](AllegeOp op) {
    hasLeftovers = true;
    op.emitError() << "unresolved trait.allege after ProveClaimPattern";
  });
  if (hasLeftovers) return failure();

  // Normalize claim types: after allege→witness, a proof's type parameter
  // may itself contain a claim that was just proven.  Substitute all
  // unproven claims with their proven forms so that downstream instantiation
  // sees consistent types.
  applySubstitutionInPlace(resolver.buildClaimSubstitutionFromMemo(), module);

  return resolver;
}

void ResolveImplsPass::runOnOperation() {
  if (failed(resolveImpls(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createResolveImplsPass() {
  return std::make_unique<ResolveImplsPass>();
}


//===----------------------------------------------------------------------===//
// InstantiateMonomorphsPass
//===----------------------------------------------------------------------===//

namespace {

/// Extend `subst` with bindings that resolve concrete `!trait.proj` types.
///
/// After buildParameterSpecialization maps generic types to concrete types, applying
/// the substitution to the formal signature may produce concrete projection
/// types like `!trait.proj<@Trait[i64], "Assoc">`. This function walks the
/// given types after substitution, resolves each concrete projection via the
/// ImplResolver, and records the binding in `subst` so that a subsequent
/// `applySubstitutionToFixedPoint` resolves projections in one shot.
static void addProjectionBindings(DenseMap<Type,Type> &subst,
                                  TypeRange types,
                                  ImplResolver &resolver,
                                  PatternRewriter &rewriter) {
  for (Type ty : types) {
    applySubstitutionToFixedPoint(subst, ty).walk([&](Type t) {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj) || subst.count(proj)) return;
      if (auto resolved = resolver.resolveProjectionType(proj, rewriter);
          succeeded(resolved))
        subst[proj] = *resolved;
    });
  }
}

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  ImplResolver &resolver;

  FuncCallOpLowering(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    // if any of the call's operand types are polymorphic, this call can't be resolved yet
    for (auto op : callOp.getOperands()) {
      if (isPolymorphicType(op.getType()))
        return rewriter.notifyMatchFailure(callOp, "operands are still polymorphic");
    }

    // func.call requires the call and callee to be in the same scope
    // we will instantiate the callee at module scope,
    // so only lower if the callOp's nearest symbol table is the module
    Operation *nearestTable = SymbolTable::getNearestSymbolTable(callOp);
    if (!nearestTable || !isa<ModuleOp>(nearestTable))
      return rewriter.notifyMatchFailure(callOp, "call is still nested in a method");

    // build a substitution that can fully concretize the result types
    auto subst = callOp.buildParameterSpecialization();
    if (failed(subst))
      return rewriter.notifyMatchFailure(callOp, "couldn't build substitution");
    addProjectionBindings(*subst, callOp.getResultTypes(), resolver, rewriter);
    addProjectionBindings(*subst, callOp.getOperandTypes(), resolver, rewriter);

    SmallVector<Type> concreteResults;
    for (Type r : callOp.getResultTypes()) {
      Type newR = applySubstitutionToFixedPoint(*subst, r);
      if (isPolymorphicType(newR))
        return rewriter.notifyMatchFailure(callOp, "result type is still polymorphic");
      concreteResults.push_back(newR);
    }

    // instantiate the callee with the enriched substitution
    auto callee = callOp.getOrInstantiateCallee(rewriter, *subst);
    if (failed(callee))
      return rewriter.notifyMatchFailure(callOp, "couldn't get or instantiate callee");

    // Unwrap any trait.proj.cast operands to get the concrete values.
    SmallVector<Value> concreteOperands;
    for (Value operand : callOp.getOperands()) {
      if (auto pcOp = operand.getDefiningOp<ProjCastOp>())
        concreteOperands.push_back(pcOp.getInput());
      else
        concreteOperands.push_back(operand);
    }

    // replace with a func.call to the instanced callee
    rewriter.replaceOpWithNewOp<func::CallOp>(
      callOp,
      callee->getSymName(),
      concreteResults,
      concreteOperands
    );

    return success();
  }
};

struct MethodCallOpLowering : public OpRewritePattern<MethodCallOp> {
  ImplResolver &resolver;

  MethodCallOpLowering(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(MethodCallOp op, PatternRewriter &rewriter) const override {
    // all operands must be monomorphic and the claim must be proven
    for (auto o : op.getOperands()) {
      if (isPolymorphicType(o.getType()))
        return rewriter.notifyMatchFailure(op, "operands are still polymorphic");
    }
    if (!op.getClaimType().isProven()) {
      return rewriter.notifyMatchFailure(op, "claim is still unproven");
    }

    // build a substitution that can fully concretize the result types
    auto subst = op.buildParameterSpecialization();
    if (failed(subst))
      return rewriter.notifyMatchFailure(op, "couldn't build substitution for call");
    addProjectionBindings(*subst, op.getResultTypes(), resolver, rewriter);
    addProjectionBindings(*subst, op.getOperandTypes(), resolver, rewriter);

    SmallVector<Type> concreteResults;
    for (Type r : op.getResultTypes()) {
      Type newR = applySubstitutionToFixedPoint(*subst, r);
      if (isPolymorphicType(newR))
        return rewriter.notifyMatchFailure(op, "result type is still polymorphic");
      concreteResults.push_back(newR);
    }

    // instantiate the callee with the enriched substitution
    auto callee = op.getOrInstantiateCallee(rewriter, *subst);
    if (failed(callee))
      return rewriter.notifyMatchFailure(op, "couldn't get or instantiate callee");

    // pass the claim as the first argument to the instantiated callee
    SmallVector<Value> args;
    args.push_back(op.getClaim());
    llvm::append_range(args, op.getArguments());

    // replace with a trait.func.call to the instantiated callee
    rewriter.replaceOpWithNewOp<FuncCallOp>(
      op,
      concreteResults,
      callee->getSymName(),
      args
    );

    return success();
  }
};

/// Rewrites a monomorphic `trait.derive` into a `trait.witness` backed by a
/// resolved proof.  Fires only when the derived claim's types are fully
/// concrete, then delegates to ImplResolver to find (or mint) the canonical
/// proof and replaces the derive with a witness referencing that proof.
struct DeriveToWitnessPattern : public OpRewritePattern<DeriveOp> {
  ImplResolver &resolver;

  DeriveToWitnessPattern(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(DeriveOp op, PatternRewriter &rewriter) const override {
    // only fire when result type is monomorphic
    if (!op.getDerivedClaim().isMonomorphic())
      return rewriter.notifyMatchFailure(op, "claim still polymorphic");

    auto errFn = [&] { return op.emitOpError(); };
    auto sym = resolver.resolveAndEnsureProofFor(op.getDerivedClaim(), rewriter, errFn);
    if (failed(sym))
      return failure();

    rewriter.replaceOpWithNewOp<WitnessOp>(op, *sym, op.getTraitApplication());
    return success();
  }
};

/// Monomorphizes result types for any op implementing
/// ResultTypeSpecializationOpInterface once all operands are monomorphic.
///
/// When all operands have concrete (non-polymorphic) types, the op's
/// `specializeTypeFromOperands` method computes the concrete result types.
/// If they differ from the op's current result types, the pattern updates
/// them in-place under the rewriter.
struct MonomorphizeResultTypesPattern
    : public OpInterfaceRewritePattern<ResultTypeSpecializationOpInterface> {
  using OpInterfaceRewritePattern<
      ResultTypeSpecializationOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ResultTypeSpecializationOpInterface iface,
                                PatternRewriter &rewriter) const override {
    // only run when all operands are monomorphic
    for (Type ty : iface->getOperandTypes()) {
      if (isPolymorphicType(ty))
        return rewriter.notifyMatchFailure(iface, "operands are still polymorphic");
    }

    // try to compute specialized monomorphic result types
    auto specializedTypes = iface.specializeResultTypes();
    if (failed(specializedTypes))
      return rewriter.notifyMatchFailure(iface, "cannot specialize result types from operands");

    // the arity of results must match
    if (specializedTypes->size() != iface->getNumResults())
      return rewriter.notifyMatchFailure(iface, "specialized result type count mismatch");

    // check if anything actually changes
    if (llvm::equal(iface->getResultTypes(), *specializedTypes))
      return rewriter.notifyMatchFailure(iface, "result types unchanged");

    // mutate result types in-place
    rewriter.modifyOpInPlace(iface, [&] {
      for (auto [result, newType] : llvm::zip(iface->getResults(), *specializedTypes))
        result.setType(newType);
    });

    return success();
  }
};

/// Returns true if `replacer.replaceElementsIn(op, ...)` with the given
/// options would modify anything on `op` (not recursing into children).
static bool wouldReplace(AttrTypeReplacer &replacer, Operation *op,
                         bool replaceAttrs, bool replaceLocs, bool replaceTypes) {
  if (replaceTypes) {
    for (Type t : op->getResultTypes())
      if (replacer.replace(t) != t) return true;
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          if (replacer.replace(arg.getType()) != arg.getType()) return true;
  }
  if (replaceAttrs)
    for (NamedAttribute attr : op->getAttrs())
      if (replacer.replace(attr.getValue()) != attr.getValue()) return true;
  if (replaceLocs)
    if (replacer.replace(op->getLoc()) != op->getLoc()) return true;
  return false;
}

/// Propagates proofs from the resolver's memo into claim types of any op
/// that carries unproven claims the resolver can now prove.
///
/// When DeriveToWitnessPattern proves a claim during the greedy rewrite and
/// FuncCallOpLowering subsequently instantiates a callee that expects that
/// claim, the newly created ops carry unproven claim types.  This pattern
/// substitutes those with their proven counterparts, unblocking
/// MethodCallOpLowering within the same rewrite pass.
///
/// Only replaces types owned by the matched op itself (result types, block
/// argument types, and attributes).  Operand types are SSA-determined and
/// update automatically once the defining value carries the proven type.
/// Child ops are visited independently by the greedy driver.
struct PropagateProofsPattern : public RewritePattern {
  ImplResolver &resolver;

  PropagateProofsPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      resolver(resolver) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto subst = resolver.buildClaimSubstitutionFromMemo();
    if (subst.empty())
      return failure();

    if (!opMentionsType<ClaimType>(op))
      return failure();

    AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst);
    if (!wouldReplace(replacer, op,
                      /*replaceAttrs=*/true,
                      /*replaceLocs=*/false,
                      /*replaceTypes=*/true))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      replacer.replaceElementsIn(op,
                                 /*replaceAttrs=*/true,
                                 /*replaceLocs=*/false,
                                 /*replaceTypes=*/true);
    });
    return success();
  }
};

/// Resolves concrete `!trait.proj` types to their bound types by looking up
/// the matching `trait.impl`'s associated type binding.
struct ResolveProjectionsPattern : public RewritePattern {
  ImplResolver &resolver;

  ResolveProjectionsPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip trait infrastructure ops and their children -
    // they may legitimately contain projections
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return failure();
    if (op->getParentOfType<TraitOp>() || op->getParentOfType<ImplOp>())
      return failure();

    if (!opMentionsType<ProjectionType>(op))
      return failure();

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](Type t) -> std::optional<Type> {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj)) return std::nullopt;
      auto resolved = resolver.resolveProjectionType(proj, rewriter);
      if (failed(resolved)) return std::nullopt;
      return *resolved;
    });
    if (!wouldReplace(replacer, op,
                      /*replaceAttrs=*/true,
                      /*replaceLocs=*/false,
                      /*replaceTypes=*/true))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      replacer.replaceElementsIn(op,
                                 /*replaceAttrs=*/true,
                                 /*replaceLocs=*/false,
                                 /*replaceTypes=*/true);
    });
    return success();
  }
};

} // end namespace

LogicalResult instantiateMonomorphs(ModuleOp module) {
  // resolve impls first
  auto resolver = resolveImpls(module);
  if (failed(resolver))
    return failure();

  MLIRContext* ctx = module.getContext();

  // rewrite trait.func.call, trait.method.call, trait.derive,
  // resolve projections, and any generic op whose results become monomorphic
  RewritePatternSet patterns(ctx);
  patterns.add<MonomorphizeResultTypesPattern>(ctx);
  patterns.add<FuncCallOpLowering>(ctx, *resolver);
  patterns.add<MethodCallOpLowering>(ctx, *resolver);
  patterns.add<ResolveProjectionsPattern>(ctx, *resolver);
  patterns.add<DeriveToWitnessPattern>(ctx, *resolver);
  patterns.add<PropagateProofsPattern>(ctx, *resolver);

  // collect instantiate-monomorphs patterns from other dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateInstantiateMonomorphsPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return module.verify();
}

void InstantiateMonomorphsPass::runOnOperation() {
  if (failed(instantiateMonomorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createInstantiateMonomorphsPass() {
  return std::make_unique<InstantiateMonomorphsPass>();
}


//===----------------------------------------------------------------------===//
// MonomorphizePass
//===----------------------------------------------------------------------===//

namespace {

struct EraseWitnessOp : public OpRewritePattern<WitnessOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WitnessOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseProjectOp : public OpRewritePattern<ProjectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ProjectOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseProjCastOp : public OpConversionPattern<ProjCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ProjCastOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // The claim operand is erased (mapped to nothing) by the TypeConverter.
    // The input operand survives; after projection resolution, input and
    // result have the same concrete type — just forward the input.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


/// Erases all residual polymorphism from the module.
///
/// This runs in two phases because no single MLIR mechanism can handle
/// both kinds of work:
///
/// Phase 1 (applyPartialConversion): Structural op rewrites that erase
///   SSA values.  Claim types map to zero results (1:0 erasure), so ops
///   that carry claims need their operand lists, indices, and signatures
///   rewritten.  Only applyPartialConversion can do this — it manages
///   the value-level bookkeeping (dropping operands, remapping uses).
///   The tuple dialect adjusts tuple.get indices and tuple.make operands;
///   the func dialect rewrites function signatures and call sites.
///
/// Phase 2 (recursivelyReplaceElementsIn): Bulk type rewriting.
///   applyPartialConversion only touches operand/result types on ops
///   matched by patterns.  Types inside attributes (e.g. the body
///   TypeAttr on nominal.def) are invisible to it.  This sweep rewrites
///   all remaining types everywhere.  The nominal dialect registers its
///   NominalType name mangling here.
///
/// Each dialect contributes to both phases via populateErasePolymorphsPatterns.
static LogicalResult erasePolymorphs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  // Delete trait symbol infrastructure upfront — these are templates that
  // have already been instantiated, and their regions contain polymorphic
  // types that would trip the legality check if left for
  // applyPartialConversion.
  for (Operation &op : llvm::make_early_inc_range(*module.getBody())) {
    if (isa<ProofOp, ImplOp, TraitOp>(op))
      op.erase();
    else if (auto f = dyn_cast<func::FuncOp>(op))
      if (isPolymorphicType(f.getFunctionType()))
        f.erase();
  }

  // Phase 1: structural op rewrites via applyPartialConversion.
  // ClaimType maps to zero results (the SSA value disappears).
  TypeConverter opConverter;
  opConverter.addConversion([](Type ty) { return ty; });
  opConverter.addConversion([](ClaimType ty, SmallVectorImpl<Type> &out) {
    return success();
  });

  AttrTypeReplacer typeSweep;

  // Collect from participating dialects
  RewritePatternSet patterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateErasePolymorphsPatterns(opConverter, patterns, typeSweep);
  }

  // Add trait dialect's own patterns
  patterns.add<EraseProjectOp, EraseWitnessOp>(ctx);
  patterns.add<EraseProjCastOp>(opConverter, ctx);

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, opConverter);
  populateCallOpTypeConversionPattern(patterns, opConverter);
  populateReturnOpTypeConversionPattern(patterns, opConverter);

  // Mark !trait.claim and !trait.proj as illegal
  ConversionTarget target(*ctx);
  target.addIllegalOp<AllegeOp, DeriveOp, ProjectOp, WitnessOp, ProjCastOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return !opMentionsType<ClaimType>(op) && !opMentionsType<ProjectionType>(op);
  });

  // Apply Phase 2
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return failure();

  // Phase 2: bulk type rewriting via recursivelyReplaceElementsIn.
  // The typeSweep replacer was already populated by dialects above
  // (e.g. nominal registered NominalType mangling).  Also forward
  // the opConverter's conversions so ClaimType gets swept out of
  // attributes too.
  typeSweep.addReplacement([&](Type t) -> std::optional<Type> {
    Type converted = opConverter.convertType(t);
    if (!converted || converted == t)
      return std::nullopt;
    return converted;
  });
  typeSweep.recursivelyReplaceElementsIn(module,
                                         /*replaceAttrs=*/true,
                                         /*replaceLocs=*/false,
                                         /*replaceTypes=*/true);
  return module.verify();
}

}

LogicalResult monomorphize(ModuleOp module) {
  if (failed(instantiateMonomorphs(module)))
    return failure();

  return erasePolymorphs(module);
}

void MonomorphizePass::runOnOperation() {
  if (failed(monomorphize(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createMonomorphizePass() {
  return std::make_unique<MonomorphizePass>();
}


} // end mlir::trait
