// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ImplResolution.hpp"
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::trait {

namespace {

/// Refuses an obligation chain that has reached the depth limit for one trait.
///
/// Every frame on the chain is a distinct application, so the cycle guard never
/// fires on a chain whose obligations keep growing the type they ask about.
/// This is what stops it, and the chain is what says where the growth came
/// from.
LogicalResult checkObligationChainDepth(
    ArrayRef<TraitApplicationAttr> chain, TraitApplicationAttr app,
    ImplOp impl) {
  StringAttr trait = app.getTraitName().getAttr();
  unsigned depth = llvm::count_if(chain, [&](TraitApplicationAttr frame) {
    return frame.getTraitName().getAttr() == trait;
  });
  if (depth < kInstantiationDepthLimit)
    return success();

  InFlightDiagnostic diagnostic =
      emitError(impl.getLoc())
      << "overflow evaluating the requirement '" << app << "': "
      << depth << " obligations of @" << trait.getValue()
      << " stand on the chain that reaches it";
  nameChainEnds<TraitApplicationAttr>(
      diagnostic, chain,
      [](InFlightDiagnostic &d, TraitApplicationAttr frame) {
        d.attachNote() << "required by " << frame;
      });
  return failure();
}

} // namespace

unsigned InstantiationChain::depthAt(Operation *instance,
                                    Attribute templateKey) const {
  unsigned depth = 0;
  for (Operation *current = instance; current;) {
    auto it = frames.find(current);
    if (it == frames.end())
      break;
    if (it->second.templateKey == templateKey)
      ++depth;
    current = it->second.parent;
  }
  return depth;
}

unsigned InstantiationChain::note(Operation *instance, Operation *parent,
                                  Attribute templateKey) {
  // An instance reached twice keeps the chain it was first cut on: the depth it
  // stands at is a property of the instance, not of whichever call asked for it
  // again. An instance that is its own parent is a call that reached the
  // function it stands in, which adds no frame.
  if (instance == parent || frames.count(instance))
    return depthAt(instance, templateKey);
  frames.insert({instance, Frame{parent, templateKey}});
  unsigned depth = depthAt(instance, templateKey);
  maxDepth = std::max(maxDepth, depth);
  return depth;
}

SmallVector<std::pair<Operation *, Attribute>>
InstantiationChain::chainTo(Operation *instance) const {
  SmallVector<std::pair<Operation *, Attribute>> reversed;
  for (Operation *current = instance; current;) {
    auto it = frames.find(current);
    if (it == frames.end())
      break;
    reversed.emplace_back(current, it->second.templateKey);
    current = it->second.parent;
  }
  return SmallVector<std::pair<Operation *, Attribute>>(llvm::reverse(reversed));
}

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
  if (llvm::is_contained(memo.visiting, app))
    return failure();

  // growth bound: a chain whose every step asks about a bigger application
  // repeats no frame, so only the depth stops it.
  if (failed(checkObligationChainDepth(memo.visiting, app, impl)))
    return failure();

  memo.visiting.push_back(app);
  auto guard = llvm::scope_exit([&]{ memo.visiting.pop_back(); });

  // The candidate's arguments as the demanded application and its own where
  // clause determine them, read through what selection has settled so far.
  auto byResolver = [&](Type ty) -> FailureOr<Type> {
    return resolveProjectionsIn(ty, builder);
  };
  TypeArguments args = impl.readTypeArgumentsFor(concreteSelf, byResolver);
  SpecializationMap known = args.toSpecialization();

  MLIRContext *ctx = impl.getContext();
  for (Attribute premise : impl.getAssumptions()) {
    // An application premise is discharged by proving it: a unique impl whose
    // own premises hold in turn.
    if (auto application = dyn_cast<TraitApplicationAttr>(premise)) {
      auto assume = cast<ClaimType>(
          instantiate(Type(ClaimType::get(ctx, application)), known));
      auto subImpl = resolveImplFor(assume, builder);
      if (failed(subImpl))
        return failure();
      if (failed(assumptionsSatisfiableFor(subImpl->impl,
                                           subImpl->selectedClaim, builder)))
        return failure();
      continue;
    }

    // An equality premise is discharged here rather than at the impl: it
    // restricts when the impl applies, and only the demanded application says
    // whether it holds. Each side is read through the candidate's own
    // associated-type bindings first -- a premise may project through the very
    // application being selected, which selection cannot ask itself about --
    // and then through what selection has settled elsewhere.
    auto equality = cast<TypeEqualityAttr>(premise);
    NormalizationContext ownBindings;
    ownBindings.addLocalProjectionRule(impl, app, known);
    auto reduce = [&](Type ty) {
      Type instantiated = instantiate(ty, known);
      auto reduced = ownBindings.normalize(instantiated, /*err=*/nullptr);
      return resolveProjectionsIn(succeeded(reduced) ? *reduced : instantiated,
                                  builder);
    };
    if (reduce(equality.getLhs()) != reduce(equality.getRhs()))
      return failure();
  }

  // An impl whose arguments the header and the where clause together leave
  // open is no candidate: selection would have nothing to specialize its
  // methods and associated-type bindings with.
  if (!args.complete())
    return failure();

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
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<RefutationArm> *refusedOn) {
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
    if (it->second.isRefusal()) {
      if (refusedOn)
        *refusedOn = it->second.getRefutationArm();
      return failure();
    }
    return ResolvedImpl{it->second.getImpl(), selected};
  }

  // get the trait
  TraitOp trait = app.getTraitOrAbort(module, "resolveImplFor: cannot find trait");

  // collect candidates for wanted from the trait and
  // partition them into good/bad by satisfiable assumptions
  //
  // The partition probes candidates it may then discard, so the demands its
  // sub-resolutions raise are marked speculative for as long as it runs.
  //
  // The context a candidate's header is read through: what selection has
  // settled, and then the impls the module binds where exactly one does. A
  // header spelling a projection reproduces a demand spelling the resolution
  // through this, and it mints nothing.
  RecordedProjectionLookup byRecord(*this);

  SmallVector<ImplOp> good, bad;
  {
    SpeculationScope speculation;
    for (ImplOp impl : trait.getCandidateImplsFor(selected, byRecord)) {
      if (succeeded(assumptionsSatisfiableFor(impl, selected, builder)))
        good.push_back(impl);
      else
        bad.push_back(impl);
    }
  }

  // if there aren't any good candidates, try to generate one
  if (good.empty()) {
    // Whoever hears about an inserted op is what decides whether anything
    // revisits it, and a generated impl that nothing revisits is IR the caller
    // never sees. What the listener has to do with the news is the caller's --
    // it is stated in the ImplGenerator contract -- but that there is one is
    // checkable here.
    assert(builder.getListener() &&
           "impl generation requires a builder whose insertions someone "
           "observes");
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    if (auto impl = getImplGenerators().generateImpl(trait, selected, builder);
        succeeded(impl)) {
      noteFactWritten();
      SpeculationScope speculation;
      if (succeeded(assumptionsSatisfiableFor(*impl, selected, builder)))
        good.push_back(*impl);
      else
        bad.push_back(*impl);
    }
  }

  // if exactly one good candidate exists, return it
  //
  // A nested resolution of this same application may already have settled it
  // under the cycle guard, which refuses every candidate it re-enters; this
  // call resolved it without that guard in the way, so its outcome replaces
  // whatever the nested one left.
  if (good.size() == 1) {
    memo.chosen.insert_or_assign(app,
                                 ResolutionOutcome::selected(good.front()));
    noteRecordWritten();
    return ResolvedImpl{good.front(), selected};
  }

  // otherwise, diagnose resolution failure, recording which of the two ways to
  // miss a unique satisfiable candidate this application missed on.
  //
  // A refusal is what selection will not have to derive again, and no answer a
  // read of the record is given: a read fails on a refused application exactly
  // as it fails on one selection has never been asked about. So the record
  // epoch stands still for it, as it does for the flush that drops it again.
  RefutationArm arm = good.empty()
                          ? RefutationArm::NoSatisfiableCandidate
                          : RefutationArm::MultipleSatisfiableCandidates;
  memo.chosen.insert_or_assign(app, ResolutionOutcome::refused(arm));
  if (refusedOn)
    *refusedOn = arm;
  return diagnoseImplResolutionFailure(trait, originalWanted, good, bad, err);
}

// find an existing trait.proof that *explicitly* proves impl by name
// and proves the same application app
static ProofOp findExistingProofFor(ModuleOp module, ImplOp impl, TraitApplicationAttr app) {
  for (ProofOp proof : module.getOps<ProofOp>()) {
    if (proof.getImplName() == impl.getSymName() &&
        proof.getTraitApplication() == app) {
      return proof;
    }
  }
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
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<RefutationArm> *refusedOn) {
  DemandFrame frame{Type(proj)};

  auto traitApp = proj.getTraitApplication();
  StringRef assocName = proj.getAssocName().getValue();

  ClaimType claim = ClaimType::get(proj.getContext(), traitApp);
  auto resolvedImpl = resolveImplFor(claim, builder, err, refusedOn);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;

  SmallVector<Type> assocTypeArgs;
  for (Type arg : proj.getAssocTypeArgs())
    assocTypeArgs.push_back(resolveProjectionsIn(arg, builder));

  auto binding = impl.specializeAssociatedTypeBinding(assocName, assocTypeArgs, err);
  if (failed(binding)) return failure();

  // The arguments carrying this impl's header to the claim selection chose it
  // for, read through the same context selection chose it under.
  RecordedProjectionLookup byRecord(*this);
  auto subst = impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim,
                                                  byRecord, err);
  if (failed(subst)) return failure();

  return instantiate(*binding, *subst);
}

ImplResolver::DemandDisposition
ImplResolver::serveDemand(ProjectionType demand, OpBuilder &builder) {
  DemandFrame frame{Type(demand)};

  // What selection settles is recorded by selection itself, so the resolved
  // type is not wanted here -- the answer this call is for is whether asking
  // again could settle it differently.
  std::optional<RefutationArm> refusedOn;
  if (succeeded(resolveProjectionType(demand, builder, /*err=*/nullptr,
                                      &refusedOn)))
    return DemandDisposition::Served;

  // A refusal for two or more satisfiable candidates is the one refusal no
  // later resolution overturns. Every other way of not serving -- no candidate
  // yet, or a binding whose own arguments have still to resolve -- is one the
  // facts can move under.
  return refusedOn == RefutationArm::MultipleSatisfiableCandidates
             ? DemandDisposition::Refused
             : DemandDisposition::Deferred;
}

ImplResolver::DemandDisposition
ImplResolver::serveDemand(ClaimType demand, OpBuilder &builder) {
  DemandFrame frame{Type(demand)};

  // Proving the claim is what serves it: the demander could read the record
  // and not write it, so what it was waiting for is the proof this mints.
  std::optional<RefutationArm> refusedOn;
  if (succeeded(resolveAndEnsureProofFor(demand, builder, /*err=*/nullptr,
                                         &refusedOn)))
    return DemandDisposition::Served;

  // The same reading as for a projection: two or more satisfiable candidates is
  // the one refusal no later resolution overturns, and every other way of not
  // serving is one the facts can move under.
  return refusedOn == RefutationArm::MultipleSatisfiableCandidates
             ? DemandDisposition::Refused
             : DemandDisposition::Deferred;
}

Type ImplResolver::resolveProjectionsIn(Type ty, OpBuilder &builder) {
  AttrTypeReplacer replacer = makeGroundProjectionReplacer(
      [this, &builder](ProjectionType proj) -> std::optional<Type> {
    auto resolved = resolveProjectionType(proj, builder);
    if (failed(resolved)) {
      // Preserve the unresolved demand for a later preparation boundary even
      // though this walk leaves its projection spelled as written.
      recordResolverProjectionMiss(Type(proj));
      return std::nullopt;
    }
    return *resolved;
  });
  return normalizeProjectionsToFixedPoint(
      ty, module, [&](Type t) { return replacer.replace(t); });
}

AttrTypeReplacer ImplResolver::makeProvenClaimReplacer() const {
  MLIRContext *ctx = module.getContext();
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement(
      [this, ctx, recorded = memo.proofMemo.size()](ClaimType claim)
          -> std::optional<std::pair<Type, WalkResult>> {
        assert(memo.proofMemo.size() == recorded &&
               "a proof was recorded while a replacer reading the memo was in "
               "use");
        // A claim that already names its proof is what respelling produces, so
        // it is left alone rather than looked up again.
        if (claim.isProven())
          return std::nullopt;
        // The memo is keyed by trait application, and only the application arm
        // carries one. An equality-arm claim holds a type equality, never an
        // impl-resolution proof, so it is never respelled here; the arm is
        // dispatched before the application is read, which would otherwise
        // assert. It stands unchanged with its interior skipped: an endpoint
        // that received a stamped proof is the state the equality constructor
        // refuses.
        if (!claim.isApplication())
          return std::make_pair(Type(claim), WalkResult::skip());
        auto it = memo.proofMemo.find(claim.getTraitApplication());
        if (it == memo.proofMemo.end())
          return std::nullopt;
        // The proven spelling names the same application, whose type arguments
        // can spell claims of their own, so the walk continues into the result
        // instead of stopping at it.
        return std::make_pair(Type(ClaimType::get(ctx, it->first, it->second)),
                              WalkResult::advance());
      });
  return replacer;
}

FailureOr<FlatSymbolRefAttr> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<RefutationArm> *refusedOn) {
  DemandFrame frame{Type(wanted)};

  ClaimType originalWanted = wanted;

  // resolve an impl for wanted first
  auto resolvedImpl = resolveImplFor(wanted, builder, err, refusedOn);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;
  ClaimType selected = resolvedImpl->selectedClaim;

  // the arguments carrying this impl's header to the selected claim, read
  // through the same context selection chose it under
  RecordedProjectionLookup byRecord(*this);
  auto subst = impl.buildSubstitutionForSelfClaim(selected, byRecord, err);
  if (failed(subst)) return failure();

  // monomorphize the selected claim with that substitution
  ClaimType monomorphicWanted = dyn_cast_or_null<ClaimType>(instantiate(Type(selected), *subst));
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
    recordProof(app, sym);
    return sym;
  }

  // A proof already standing for this impl at this application answers for it
  // only when deriving it succeeds. Naming the impl and the application is
  // where a proof stands, not evidence that the subproofs it cites discharge
  // the impl's obligations, and selection must not hand back a proof it has not
  // seen derive. One that does not derive leaves selection to build its own
  // below, and the standing proof is refused where it is written.
  if (ProofOp proof = findExistingProofFor(module, impl, app)) {
    auto sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
    ClaimType standing = ClaimType::get(ctx, app, sym);
    EvidenceBindings bindings;
    if (succeeded(verifyAndRecordProof(standing.asUnproven(), standing, module,
                                       bindings, DemandOrigin::ProofRecording,
                                       &derivations, /*err=*/nullptr))) {
      recordProof(app, sym);
      return sym;
    }
  }

  // Compute the proof name early so we can use it as the coinductive memo entry.
  std::string proofName = impl.generateMangledName(*subst) + "_p";
  auto proofSym = FlatSymbolRefAttr::get(ctx, proofName);
  for (ProofOp proof : module.getOps<ProofOp>()) {
    if (proof.getSymName() != proofName)
      continue;

    ClaimType candidate = ClaimType::get(ctx, app, proofSym);
    EvidenceBindings bindings;
    if (succeeded(verifyAndRecordProof(candidate.asUnproven(), candidate,
                                       module, bindings,
                                       DemandOrigin::ProofRecording,
                                       &derivations, err))) {
      recordProof(app, proofSym);
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
  recordProof(app, proofSym);
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
  // A created proof is IR nothing revisits unless someone hears about it, for
  // the same reason a generated impl is.
  assert(builder.getListener() &&
         "proof creation requires a builder whose insertions someone observes");
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
  recordProof(app, sym);
  return sym;
}

//===----------------------------------------------------------------------===//
// Forgetting what a later resolution can answer differently
//===----------------------------------------------------------------------===//

void ImplResolver::forgetRetriableRefusals() {
  auto &chosen = memo.resolutionMemo.chosen;
  SmallVector<TraitApplicationAttr> retriable;
  for (const auto &entry : chosen)
    if (entry.second.isRefusal() &&
        entry.second.getRefutationArm() ==
            RefutationArm::NoSatisfiableCandidate)
      retriable.push_back(entry.first);
  // The drops move no record epoch, for the same reason writing the refusal
  // did not: what is erased here is a question impl selection will have to
  // answer again, never an answer a read of the record was given.
  for (TraitApplicationAttr app : retriable)
    chosen.erase(app);
}

//===----------------------------------------------------------------------===//
// Freezing impl generation
//===----------------------------------------------------------------------===//

ImplGenerationFreeze::ImplGenerationFreeze(ImplResolver &resolver,
                                          StringRef span)
    : resolver(resolver), span(span.str()),
      displaced(resolver.installedOverride) {
  resolver.installedOverride = this;
}

ImplGenerationFreeze::~ImplGenerationFreeze() {
  // Stand-ins nest, so the one going out of scope is the one now installed:
  // restoring what this freeze displaced is only the previous state if nothing
  // installed after it is still standing.
  assert(resolver.installedOverride == this &&
         "a freeze must be the innermost stand-in installed when it ends");
  resolver.installedOverride = displaced;
}

FailureOr<ImplOp> ImplGenerationFreeze::generateImpl(TraitOp trait,
                                                     ClaimType wanted,
                                                     OpBuilder &builder) const {
  // Being asked to generate at all is the fault this reports: the stage
  // standing over the driver reads recorded facts and puts nothing to
  // selection, so an ask from under it reached past the record it must read.
  // Reported as a failure rather than a process abort so a demand raised on
  // hostile IR refuses cleanly at the point selection asked. The span's owner
  // reads `wasAsked` after its driver to fail the stage, since a greedy driver
  // swallows this failure as a non-applied pattern.
  generationAsked = true;
  emitError(trait.getLoc())
      << "impl generation is frozen for " << span
      << ", but impl selection demanded an impl of @" << trait.getSymName()
      << " for " << wanted;
  return failure();
}

//===----------------------------------------------------------------------===//
// Reading the recorded facts
//===----------------------------------------------------------------------===//

LogicalResult ReadOnlyImplResolver::decline(ProjectionType demand) const {
  recordReadOnlyResolverMiss(Type(demand));
  return failure();
}

LogicalResult ReadOnlyImplResolver::decline(ClaimType demand) const {
  recordReadOnlyResolverMiss(Type(demand));
  return failure();
}

FailureOr<ResolvedImpl>
ReadOnlyImplResolver::getRecordedImplFor(ClaimType wanted) const {
  DemandFrame frame{Type(wanted)};

  ClaimType selected = cast<ClaimType>(resolveProjectionsIn(wanted));
  auto outcome = getRecordedOutcome(selected.getTraitApplication());
  if (!outcome || outcome->isRefusal())
    return failure();
  return ResolvedImpl{outcome->getImpl(), selected};
}

FailureOr<Type>
ReadOnlyImplResolver::resolveProjectionType(ProjectionType proj) const {
  DemandFrame frame{Type(proj)};

  ClaimType claim = ClaimType::get(proj.getContext(), proj.getTraitApplication());
  auto resolvedImpl = getRecordedImplFor(claim);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;

  SmallVector<Type> assocTypeArgs;
  for (Type arg : proj.getAssocTypeArgs())
    assocTypeArgs.push_back(resolveProjectionsIn(arg));

  auto binding = impl.specializeAssociatedTypeBinding(
      proj.getAssocName().getValue(), assocTypeArgs);
  if (failed(binding)) return failure();

  // The arguments carrying this impl's header to the claim selection chose it
  // for, read through the same context selection chose it under.
  RecordedProjectionLookup byRecord(*this);
  auto subst = impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim,
                                                  byRecord,
                                                  /*errFn=*/nullptr);
  if (failed(subst)) return failure();

  return instantiate(*binding, *subst);
}

Type ReadOnlyImplResolver::resolveProjectionsIn(Type ty) const {
  AttrTypeReplacer replacer = makeGroundProjectionReplacer(
      [this](ProjectionType proj) -> std::optional<Type> {
    auto resolved = resolveProjectionType(proj);
    if (succeeded(resolved))
      return *resolved;
    // Selection settles a projection only for an application some round put to
    // it, so a spelling nothing has asked about yet has no recorded fact to
    // read. One exactly one impl in the module binds is one selection would
    // settle the same way, so the module answers it here; where no impl or
    // several bind it, the lookup declines and says which, and the projection
    // stays spelled as written for the step that can make selection answer it.
    Type byLookup = resolveProjectionsByLookup(
        Type(proj), resolver.module, DemandOrigin::RecordedFactRead,
        LookupScope::Ground);
    if (byLookup == Type(proj))
      return std::nullopt;
    return byLookup;
  });
  return normalizeProjectionsToFixedPoint(
      ty, resolver.module, [&](Type t) { return replacer.replace(t); });
}

FailureOr<FlatSymbolRefAttr>
ReadOnlyImplResolver::getRecordedProofFor(ClaimType claim) const {
  DemandFrame frame{Type(claim)};

  auto resolvedImpl = getRecordedImplFor(claim);
  if (failed(resolvedImpl)) return failure();

  // The arguments carrying this impl's header to the claim selection chose it
  // for, read through the same context selection chose it under.
  RecordedProjectionLookup byRecord(*this);
  auto subst = resolvedImpl->impl.buildSubstitutionForSelfClaim(
      resolvedImpl->selectedClaim, byRecord, /*errFn=*/nullptr);
  if (failed(subst)) return failure();

  auto monomorphic = dyn_cast_or_null<ClaimType>(
      instantiate(Type(resolvedImpl->selectedClaim), *subst));
  if (!monomorphic || !monomorphic.isMonomorphic())
    return failure();

  auto proof = getRecordedProof(monomorphic.getTraitApplication());
  if (!proof) return failure();
  return *proof;
}

} // end mlir::trait
