// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ImplResolution.hpp"
#include <llvm/ADT/ScopeExit.h>

namespace mlir::trait {

LogicalResult
ImplResolver::assumptionsSatisfiableFor(ImplOp impl,
                                        ClaimType concreteSelf,
                                        OpBuilder &builder) {
  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // consult the per-(impl,claim) satisfiability memo
  auto key = std::make_pair(impl, app);
  if (memo.assumptionsKnownSatisfiable.contains(key))
    return success();

  // cycle guard: A(app) -> ... -> A(app) means unsatisfiable
  if (!memo.visiting.insert(app).second)
    return failure();
  auto guard = llvm::scope_exit([&]{ memo.visiting.erase(app); });

  // specialize the impl's assumptions to our concrete claim
  auto assumptions = impl.specializeAssumptionsAsClaimsFor(concreteSelf);
  if (failed(assumptions))
    return failure();

  for (ClaimType assume : *assumptions) {
    // find an impl for the assumption
    auto subImpl = resolveImplFor(assume, builder);
    if (failed(subImpl))
      return failure();

    // that impl's own assumptions must be satisfiable too
    if (failed(assumptionsSatisfiableFor(subImpl->impl, subImpl->selectedClaim,
                                         builder)))
      return failure();
  }

  // record a positive result
  memo.assumptionsKnownSatisfiable.insert(key);

  return success();
}

static LogicalResult diagnoseImplResolutionFailure(
    TraitOp trait,
    ClaimType wanted,
    ArrayRef<ImplOp> goodCandidates,
    ArrayRef<ImplOp> badCandidates,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!err) return failure();

  // if there were no good candidates, note the bad candidates that didn't match
  if (goodCandidates.empty()) {
    InFlightDiagnostic diag = err() << "no impl with satisfiable assumptions for "
                                    << wanted;

    unsigned maxNotes = 16;
    unsigned emitted = 0;
    for (ImplOp impl : badCandidates) {
      if (emitted++ == maxNotes) {
        unsigned remaining = badCandidates.size() - maxNotes;
        diag.attachNote(trait.getLoc())
          << remaining << " more unsatisfiable candidate(s) elided";
        break;
      }

      diag.attachNote(impl.getLoc()) << "unsatisfiable candidate";
    }

    return failure();
  }

  // there were multiple good candidates, note the good candidates that did match
  InFlightDiagnostic diag = err() << "incoherent impls (multiple satisfiable) for "
                                  << wanted;

  unsigned maxNotes = 16;
  unsigned emitted = 0;
  for (ImplOp impl : goodCandidates) {
    if (emitted++ == maxNotes) {
      unsigned remaining = goodCandidates.size() - maxNotes;
      diag.attachNote(trait.getLoc())
        << remaining << " more candidate(s) elided";
      break;
    }

    diag.attachNote(impl.getLoc()) << "candidate";
  }

  return diag;
}

FailureOr<ResolvedImpl> ImplResolver::resolveImplFor(
    ClaimType wanted,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DemandFrame frame{Type(wanted)};

  ClaimType originalWanted = wanted;

  // Resolution resolves a demanded claim's monomorphic projections before it
  // selects an impl and records a proof. Every downstream fact minted here --
  // the resolution memo, the proof memo, the proof op, the witness -- is keyed
  // and spelled by this resolved claim, so those facts read back spelled
  // exactly as their post-resolution demand. Declaration-spelled demands
  // (trait and impl headers still carry their source projections) join that
  // resolved vocabulary here; no other component resolves a demanded claim's
  // spelling before impl selection and proof creation. (The obligation
  // recorder in verifyAndRecordProof normalizes both the demanded obligation
  // and the proven value's spelling before recording, so coherent spellings of
  // one obligation record identically; recorded-proof equivalence then fires
  // only to reject genuinely incoherent proofs, not to reconcile spellings.)
  ClaimType selected = cast<ClaimType>(resolveProjectionsIn(wanted, builder));

  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = selected.getTraitApplication();

  // first check the memo
  if (auto it = memo.chosen.find(app); it != memo.chosen.end()) {
    if (failed(it->second))
      return failure();
    return ResolvedImpl{*it->second, selected};
  }

  // get the trait
  TraitOp trait = app.getTraitOrAbort(module, "resolveImplFor: cannot find trait");

  // collect candidates for wanted from the trait and
  // partition them into good/bad by satisfiable assumptions
  //
  // The partition probes candidates it may then discard, so the demands its
  // sub-resolutions raise are marked speculative for as long as it runs.
  SmallVector<ImplOp> good, bad;
  {
    SpeculationScope speculation;
    for (ImplOp impl : trait.getCandidateImplsFor(selected)) {
      if (succeeded(assumptionsSatisfiableFor(impl, selected, builder)))
        good.push_back(impl);
      else
        bad.push_back(impl);
    }
  }

  // if there aren't any good candidates, try to generate one
  if (good.empty()) {
    // Generated ops reach the enclosing greedy driver's worklist only through
    // the builder's listener, so a listener-less builder here would silently
    // change which ops that driver revisits.
    assert(builder.getListener() &&
           "impl generation requires a builder that notifies its caller of "
           "inserted ops");
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    if (auto impl = generators.generateImpl(trait, selected, builder); succeeded(impl)) {
      SpeculationScope speculation;
      if (succeeded(assumptionsSatisfiableFor(*impl, selected, builder)))
        good.push_back(*impl);
      else
        bad.push_back(*impl);
    }
  }

  // if exactly one good candidate exists, return it
  if (good.size() == 1) {
    memo.chosen[app] = good.front();
    return ResolvedImpl{good.front(), selected};
  }

  // otherwise, diagnose resolution failure
  memo.chosen[app] =
      diagnoseImplResolutionFailure(trait, originalWanted, good, bad, err);
  return failure();
}

// find an existing trait.proof that *explicitly* proves impl by name
// and proves the same application app
static ProofOp findExistingProofFor(ModuleOp module, ImplOp impl, TraitApplicationAttr app) {
  size_t scanned = 0;
  for (ProofOp proof : module.getOps<ProofOp>()) {
    ++scanned;
    if (proof.getImplName() == impl.getSymName() &&
        proof.getTraitApplication() == app) {
      countProofScan(scanned);
      return proof;
    }
  }
  countProofScan(scanned);
  return nullptr;
}

ImplResolver::ImplResolver(ModuleOp m, std::shared_ptr<DemandLedger> ledger)
    : module(m), ledger(std::move(ledger)) {
  // collect ImplGenerators from each dialect with the appropriate interface
  for (Dialect *dialect : module.getContext()->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<GenerateImplsInterface>()) {
      iface->populateImplGenerators(generators);
    }
  }
}

FailureOr<Type> ImplResolver::resolveProjectionType(
    ProjectionType proj,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DemandFrame frame{Type(proj)};

  auto traitApp = proj.getTraitApplication();
  StringRef assocName = proj.getAssocName().getValue();

  ClaimType claim = ClaimType::get(proj.getContext(), traitApp);
  auto resolvedImpl = resolveImplFor(claim, builder, err);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;

  SmallVector<Type> assocTypeArgs;
  for (Type arg : proj.getAssocTypeArgs())
    assocTypeArgs.push_back(resolveProjectionsIn(arg, builder));

  auto binding = impl.specializeAssociatedTypeBinding(assocName, assocTypeArgs, err);
  if (failed(binding)) return failure();

  auto subst =
      impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim, err);
  if (failed(subst)) return failure();

  return applySubstitutionToFixedPoint(subst->toTypeMap(), *binding);
}

Type ImplResolver::resolveProjectionsIn(Type ty, OpBuilder &builder) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([this, &builder](Type t) -> std::optional<Type> {
    auto proj = dyn_cast<ProjectionType>(t);
    if (!proj || isPolymorphicType(proj)) return std::nullopt;
    auto resolved = resolveProjectionType(proj, builder);
    if (failed(resolved)) {
      // The failure is swallowed here -- the projection stays spelled as
      // written and the walk goes on -- so this is where the second engine's
      // unserved demand becomes visible. Its count is a lower bound: an
      // application the negative resolution memo already refuted never reaches
      // this engine a second time.
      recordResolverProjectionMiss(Type(proj));
      return std::nullopt;
    }
    return *resolved;
  });
  return replacer.replace(ty);
}

FailureOr<FlatSymbolRefAttr> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DemandFrame frame{Type(wanted)};

  ClaimType originalWanted = wanted;

  // resolve an impl for wanted first
  auto resolvedImpl = resolveImplFor(wanted, builder, err);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;
  ClaimType selected = resolvedImpl->selectedClaim;

  // build a PolyType -> Type map for this impl's self claim against selected
  auto subst = impl.buildSubstitutionForSelfClaim(selected, err);
  if (failed(subst)) return failure();

  // monomorphize the selected claim with that substitution
  ClaimType monomorphicWanted = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(subst->toTypeMap(), selected));
  if (!monomorphicWanted || !monomorphicWanted.isMonomorphic()) {
    if (err) err() << "could not monomorphize claim: " << originalWanted;
    return failure();
  }

  TraitApplicationAttr app = monomorphicWanted.getTraitApplication();

  // check the proof memo for this monomorphic app
  if (auto it = memo.proofMemo.find(app); it != memo.proofMemo.end())
    return it->second;

  MLIRContext *ctx = module.getContext();

  // check for an unconditional impl
  if (impl.isUnconditional()) {
    auto sym = FlatSymbolRefAttr::get(ctx, impl.getSymName());
    memo.proofMemo[app] = sym;
    return sym;
  }

  // check for an existing proof in the module
  if (ProofOp proof = findExistingProofFor(module, impl, app)) {
    auto sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
    memo.proofMemo[app] = sym;
    return sym;
  }

  // Compute the proof name early so we can use it as the coinductive memo entry.
  std::string proofName = impl.generateMangledName(monomorphicWanted) + "_p";
  auto proofSym = FlatSymbolRefAttr::get(ctx, proofName);
  size_t collisionsScanned = 0;
  auto countCollisionScan =
      llvm::scope_exit([&] { countProofCollisionScan(collisionsScanned); });
  for (ProofOp proof : module.getOps<ProofOp>()) {
    ++collisionsScanned;
    if (proof.getSymName() != proofName)
      continue;

    ClaimType candidate = ClaimType::get(ctx, app, proofSym);
    EvidenceBindings bindings;
    if (succeeded(verifyAndRecordProof(candidate.asUnproven(), candidate,
                                       module, bindings,
                                       DemandOrigin::ProofRecording, err))) {
      memo.proofMemo[app] = proofSym;
      return proofSym;
    }

    if (err)
      err() << "proof symbol collision for @" << proofName;
    return failure();
  }

  // Coinductive cycle guard: optimistically populate the proof memo with the
  // proof symbol before recursing into obligations.  If an obligation (after
  // projection resolution) turns out to be the same claim we are currently
  // proving, the recursive call will hit the memo instead of diverging.
  memo.proofMemo[app] = proofSym;
  auto rollback = llvm::scope_exit([&]{ memo.proofMemo.erase(app); });

  // specialize all obligations against the claim selected during resolution
  auto obligations = impl.specializeObligationsAsClaimsFor(
      selected, DemandOrigin::ProofRecording, err);
  if (failed(obligations)) return failure();

  // recursively prove monomorphic obligations
  SmallVector<Attribute> subproofSymbols;
  for (ClaimType ob : *obligations) {
    auto sym = resolveAndEnsureProofFor(ob, builder, err);
    if (failed(sym)) return failure();
    subproofSymbols.push_back(*sym);
  }

  // create the proof and memoize by the monomorphic app
  //
  // A created proof reaches an enclosing greedy driver's worklist only through
  // the builder's listener, for the same reason a generated impl does.
  assert(builder.getListener() &&
         "proof creation requires a builder that notifies its caller of "
         "inserted ops");
  rollback.release();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  ProofOp proof = ProofOp::create(
    builder,
    builder.getUnknownLoc(),
    StringAttr::get(ctx, proofName),
    FlatSymbolRefAttr::get(ctx, impl.getSymName()),
    app,
    ArrayAttr::get(ctx, subproofSymbols)
  );

  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
  memo.proofMemo[app] = sym;
  return sym;
}

} // end mlir::trait
