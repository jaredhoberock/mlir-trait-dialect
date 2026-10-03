// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ImplResolution.hpp"
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/ErrorHandling.h>

namespace mlir::trait {

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

void InstantiationChain::note(Operation *instance, Operation *parent,
                              Attribute templateKey) {
  // An instance reached twice keeps the chain it was first cut on: the depth it
  // stands at is a property of the instance, not of whichever call asked for it
  // again. An instance that is its own parent is a call that reached the
  // function it stands in, which adds no frame.
  if (instance == parent || frames.count(instance))
    return;
  frames.insert({instance, Frame{parent, templateKey}});
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
                                        ModuleOp scope,
                                        OpBuilder &builder) {
  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // consult the per-(impl,claim) satisfiability memo
  auto key = std::make_pair(impl, app);
  if (memo.assumptionsKnownSatisfiable.contains(key))
    return success();

  // cycle guard: A(app) -> ... -> A(app) means unsatisfiable
  if (llvm::any_of(memo.visiting, [&](ObligationFrame frame) {
        return frame.application == app;
      }))
    return failure();

  // growth bound: a chain whose every step asks about a bigger application
  // repeats no frame, so only the depth stops it.
  if (failed(checkObligationChainDepth(memo.visiting, app, impl.getLoc())))
    return failure();

  // Selection descends the candidate's where clause, which no proof mediates.
  memo.visiting.push_back({app, SymbolRefAttr()});
  auto guard = llvm::scope_exit([&]{ memo.visiting.pop_back(); });

  // The candidate's arguments as the demanded application and its own where
  // clause determine them, read through what selection has settled so far.
  auto byResolver = [&](Type ty) -> FailureOr<Type> {
    return resolveProjectionsIn(ty, scope, builder);
  };
  TypeArguments args = impl.readTypeArgumentsFor(concreteSelf, byResolver);
  SpecializationMap known = args.toSpecialization();

  for (ClaimType premise : impl.getWhereClaims()) {
    // An application premise is discharged by proving it: a unique impl whose
    // own premises hold in turn.
    if (premise.isApplication()) {
      auto assume = cast<ClaimType>(instantiate(Type(premise), known));
      auto subImpl = resolveImplFor(assume, scope, builder);
      if (failed(subImpl))
        return failure();
      if (failed(assumptionsSatisfiableFor(subImpl->impl,
                                           subImpl->selectedClaim, scope,
                                           builder)))
        return failure();
      continue;
    }

    // An equality premise is discharged here rather than at the impl: it
    // restricts when the impl applies, and only the demanded application says
    // whether it holds. Each side is read through the candidate's own
    // associated-type bindings first -- a premise may project through the very
    // application being selected, which selection cannot ask itself about --
    // and then through what selection has settled elsewhere. A reading carrying
    // a type variable is left to the instances that fill it.
    TypeEqualityAttr equality = premise.getEqualityAttr();
    NormalizationContext ownBindings;
    ownBindings.addLocalProjectionRule(impl, app, known);
    auto reduce = [&](Type ty) {
      Type instantiated = instantiate(ty, known);
      auto reduced = ownBindings.normalize(instantiated, /*err=*/nullptr);
      return resolveProjectionsIn(succeeded(reduced) ? *reduced : instantiated,
                                  scope, builder);
    };
    Type lhs = reduce(equality.getLhs());
    Type rhs = reduce(equality.getRhs());
    if (premiseDefersToInstances(lhs, rhs))
      continue;
    if (lhs != rhs)
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

/// How many candidates a refusal names one by one. Past this a reader learns
/// nothing more from another impl of the same shape, so the rest are counted.
constexpr unsigned kCandidatesNamed = 16;

/// Attaches one note per impl in `candidates` to `diagnostic`, each reading
/// `label`, with a note at `elidedAt` counting the ones past the limit.
static void nameCandidates(InFlightDiagnostic &diagnostic,
                           ArrayRef<ImplOp> candidates, Location elidedAt,
                           StringRef label) {
  for (ImplOp impl : candidates.take_front(kCandidatesNamed))
    diagnostic.attachNote(impl.getLoc()) << label;
  if (candidates.size() > kCandidatesNamed)
    diagnostic.attachNote(elidedAt)
        << candidates.size() - kCandidatesNamed << " more " << label
        << "(s) elided";
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
    nameCandidates(diag, badCandidates, trait.getLoc(),
                   "unsatisfiable candidate");
    return failure();
  }

  // there were multiple good candidates, note the good candidates that did match
  InFlightDiagnostic diag = err() << "incoherent impls (multiple satisfiable) for "
                                  << wanted;
  nameCandidates(diag, goodCandidates, trait.getLoc(), "candidate");
  return diag;
}

FailureOr<ResolvedImpl> ImplResolver::resolveImplFor(
    ClaimType wanted,
    ModuleOp scope,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<Refutation> *refusedOn) {
  DemandFrame frame{Type(wanted)};

  ClaimType originalWanted = wanted;

  // Resolution resolves a demanded claim's monomorphic projections before it
  // selects an impl and records a proof. Every downstream fact minted here --
  // the resolution memo, the proof memo, the proof op, the witness -- is keyed
  // and spelled by this resolved claim, so those facts read back spelled
  // exactly as their post-resolution demand. Declaration-spelled demands
  // (trait and impl headers still carry their source projections) join that
  // resolved vocabulary here; no other component resolves a demanded claim's
  // spelling before impl selection and proof creation.
  ClaimType selected =
      cast<ClaimType>(resolveProjectionsIn(wanted, scope, builder));

  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = selected.getTraitApplication();

  // first check the memo
  if (auto it = memo.chosen.find({scope, app}); it != memo.chosen.end()) {
    if (it->second.isRefusal()) {
      // The record keeps the arm and not the candidates: an application
      // selection has already refused is one nothing asks the trait about
      // again.
      if (refusedOn)
        *refusedOn = Refutation{it->second.getRefutationArm(), {}};
      return failure();
    }
    return ResolvedImpl{it->second.getImpl(), selected};
  }

  // get the trait. The demand's spelling names it in the module the demand was
  // read in, and the impls that trait holds are that module's, so a demand
  // raised inside a nested module is served by the impls standing there.
  //
  // A spelling this scope does not declare has no candidate here, the same
  // standing a trait whose impls all miss has: an op in a nested module may
  // name a trait only that module declares, and the demand it raises reaches
  // this scope keyed by that spelling.
  auto declaredTrait = app.getTrait(scope, err);
  if (failed(declaredTrait)) {
    memo.chosen.insert_or_assign(
        {scope, app},
        ResolutionOutcome::refused(RefutationArm::NoSatisfiableCandidate));
    if (refusedOn)
      *refusedOn = Refutation{RefutationArm::NoSatisfiableCandidate, {}};
    return failure();
  }
  TraitOp trait = *declaredTrait;

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
  RecordedProjectionLookup byRecord(*this, scope);

  SmallVector<ImplOp> good, bad;
  {
    SpeculationScope speculation;
    for (ImplOp impl : trait.getCandidateImplsFor(selected, byRecord)) {
      if (succeeded(assumptionsSatisfiableFor(impl, selected, scope, builder)))
        good.push_back(impl);
      else
        bad.push_back(impl);
    }
  }

  // if there aren't any good candidates, try to generate one. An application a
  // generator has already supplied an impl for is not asked again: that impl
  // stands in the module and the scan above has just judged it, so asking
  // would only publish a second op under the name the first holds.
  if (good.empty() && !memo.generatedFor.contains({scope, app})) {
    // Whoever hears about an inserted op is what decides whether anything
    // revisits it, and a generated impl that nothing revisits is IR the caller
    // never sees. What the listener has to do with the news is the caller's --
    // it is stated in the ImplGenerator contract -- but that there is one is
    // checkable here.
    assert(builder.getListener() &&
           "impl generation requires a builder whose insertions someone "
           "observes");
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(scope.getBody());
    if (auto impl = getImplGenerators().generateImpl(trait, selected, builder);
        succeeded(impl)) {
      memo.generatedFor.insert({scope, app});
      noteFactWritten();
      SpeculationScope speculation;
      if (succeeded(assumptionsSatisfiableFor(*impl, selected, scope, builder)))
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
    memo.chosen.insert_or_assign({scope, app},
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
  memo.chosen.insert_or_assign({scope, app}, ResolutionOutcome::refused(arm));
  if (refusedOn)
    *refusedOn = Refutation{arm, good};
  return diagnoseImplResolutionFailure(trait, originalWanted, good, bad, err);
}

ImplResolver::StandingProofs &
ImplResolver::getStandingProofs(ModuleOp scope) const {
  auto [entry, inserted] = standingProofs.try_emplace(scope.getOperation());
  // Read once per module, in module order, so the first proof of an impl at an
  // application is the one found. The impl is matched by identity rather than
  // by name: a name is resolved in one symbol table, and two modules can each
  // hold an impl of that name meaning two different impls.
  if (inserted)
    for (ProofOp proof : scope.getOps<ProofOp>())
      entry->second.note(proof);
  return entry->second;
}

void ImplResolver::StandingProofs::note(ProofOp proof) {
  byClaim[{proof.getImpl(), proof.getTraitApplication()}].push_back(proof);
}

/// Writes at the end of `scope` the proof `name` whose body derives `app` from
/// `impl` over one premise per entry of `entries`, `impl`'s where entries at
/// the citation: a witness of the next of `subproofs` for an application entry
/// and the evidence the next of `equalitySteps` build for an equality entry.
/// Where a symbol of `scope` holds `name` already, the proof is named as the
/// module's symbol table renames it (`SymbolTable::insert`): mangled names are
/// not one-to-one, so the table, not the mangling, makes a proof's name unique.
static ProofOp
writeProofBody(OpBuilder &builder, ModuleOp scope, StringRef name, ImplOp impl,
               TraitApplicationAttr app, ArrayRef<ClaimType> entries,
               ArrayRef<FlatSymbolRefAttr> subproofs,
               ArrayRef<SmallVector<ResolutionStep>> equalitySteps) {
  // A created proof is IR nothing revisits unless someone hears about it, for
  // the same reason a generated impl is.
  assert(builder.getListener() &&
         "proof creation requires a builder whose insertions someone observes");
  MLIRContext *ctx = scope.getContext();
  // The table is read before the proof stands, so the proof's name is the one
  // it checks.
  SymbolTable symbols(scope);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(scope.getBody());
  Location loc = builder.getUnknownLoc();
  ProofOp proof = ProofOp::create(builder, loc, name);
  symbols.insert(proof);
  builder.setInsertionPointToEnd(&proof.getBody().front());
  SmallVector<Value> premises;
  auto nextSubproof = subproofs.begin();
  auto nextSteps = equalitySteps.begin();
  for (ClaimType entry : entries) {
    if (entry.isApplication())
      premises.push_back(WitnessOp::create(builder, loc, *nextSubproof++,
                                           entry.getTraitApplication()));
    else
      premises.push_back(buildEqualityEvidence(
          builder, loc, entry.getEqualityAttr(), *nextSteps++));
  }
  auto derived = DeriveOp::create(builder, loc, ClaimType::get(ctx, app),
                                  FlatSymbolRefAttr::get(ctx, impl.getSymName()),
                                  premises);
  ReturnOp::create(builder, loc, derived.getResult());
  return proof;
}

ClaimType ImplResolver::findProof(ModuleOp scope, ImplOp impl,
                                  TraitApplicationAttr app,
                                  ArrayRef<FlatSymbolRefAttr> subproofs) const {
  MLIRContext *ctx = scope.getContext();
  // An impl with no parameters and no where entries is its own proof.
  if (impl.isUnconditional())
    return ClaimType::get(ctx, app,
                          FlatSymbolRefAttr::get(ctx, impl.getSymName()));
  // A proof is identified by the evidence it derives its claim from: the impl,
  // the application, and the proof each application premise names.
  auto citesSubproofs = [&](ProofOp proof) {
    auto next = subproofs.begin();
    for (Value premise : proof.getDerive().getAssumptions()) {
      auto claim = cast<ClaimType>(premise.getType());
      if (claim.isApplication() && claim.getProof() != *next++)
        return false;
    }
    return true;
  };
  StandingProofs &standing = getStandingProofs(scope);
  if (auto it = standing.byClaim.find({impl, app}); it != standing.byClaim.end())
    for (ProofOp proof : it->second)
      if (citesSubproofs(proof))
        return ClaimType::get(
            ctx, app, FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr()));
  return {};
}

ClaimType ImplResolver::writeProof(
    ModuleOp scope, ImplOp impl, TraitApplicationAttr app,
    const SpecializationMap &arguments, ArrayRef<ClaimType> entries,
    ArrayRef<FlatSymbolRefAttr> subproofs,
    ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
    OpBuilder &builder) const {
  MLIRContext *ctx = scope.getContext();
  ProofOp proof =
      writeProofBody(builder, scope, impl.generateMangledName(arguments) + "_p",
                     impl, app, entries, subproofs, equalitySteps);
  getStandingProofs(scope).note(proof);
  return ClaimType::get(ctx, app,
                        FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr()));
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

/// The arguments carrying `resolved`'s impl header to the claim selection chose
/// it for, read through `record`, the context selection chose it under.
static FailureOr<SpecializationMap>
argumentsOf(const ResolvedImpl &resolved, const ReadOnlyImplResolver &record,
            llvm::function_ref<InFlightDiagnostic()> err) {
  ImplOp impl = resolved.impl;
  return impl.buildSubstitutionForSelfClaim(
      resolved.selectedClaim, RecordedProjectionLookup(record), err);
}

FailureOr<ProjectionResolution> ProjectionResolution::get(
    ProjectionType projection,
    llvm::function_ref<FailureOr<ResolvedImpl>(ClaimType)> select,
    const ReadOnlyImplResolver &record,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto resolved = select(projection.asClaim());
  if (failed(resolved)) return failure();
  auto arguments = argumentsOf(*resolved, record, err);
  if (failed(arguments)) return failure();
  ImplOp impl = resolved->impl;
  auto binding = impl.specializeAssociatedTypeBinding(
      projection.getAssocName().getValue(), projection.getAssocTypeArgs(),
      *arguments, err);
  if (failed(binding)) return failure();
  return ProjectionResolution(projection, impl, std::move(*arguments),
                              *binding);
}

/// The monomorphic application `resolved`'s impl header states at `arguments`,
/// the ones it takes at the claim selection chose it for, which is what a proof
/// of that claim is recorded under.
static FailureOr<TraitApplicationAttr>
monomorphicApplicationOf(const ResolvedImpl &resolved,
                         const SpecializationMap &arguments) {
  auto instance = dyn_cast_or_null<ClaimType>(
      instantiate(Type(resolved.selectedClaim), arguments));
  if (!instance || !instance.isMonomorphic())
    return failure();
  return instance.getTraitApplication();
}

FailureOr<ProjectionResolution> ImplResolver::resolveProjection(
    ProjectionType proj,
    ModuleOp scope,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<Refutation> *refusedOn) {
  DemandFrame frame{Type(proj)};

  auto select = [&](ClaimType claim) {
    return resolveImplFor(claim, scope, builder, err, refusedOn);
  };
  return ProjectionResolution::get(proj, select,
                                   ReadOnlyImplResolver(*this, scope), err);
}

/// What putting `demand` to impl selection settled, given what selection
/// refused it on when it did not serve it.
///
/// Two or more satisfiable candidates is the one refusal no later resolution
/// overturns: candidates are only appended. Every other way of not serving --
/// no candidate yet, or a binding whose own arguments have still to resolve --
/// is one the facts can move under.
///
/// The ambiguity is named where the demand stands, and here and nowhere else.
/// It is a refusal the demand's own spelling need not carry: an obligation read
/// off a trait's where clause at a ground application is spelled in no
/// operation, so the stage's leftover walks have nothing to find and the demand
/// would go unreported.
static ImplResolver::DemandDisposition
refusalDisposition(Type demand, ModuleOp scope,
                   const std::optional<Refutation> &refusedOn) {
  if (!refusedOn ||
      refusedOn->arm != RefutationArm::MultipleSatisfiableCandidates)
    return ImplResolver::DemandDisposition::Deferred;
  InFlightDiagnostic diagnostic =
      emitError(currentDemandAnchor().value_or(scope.getLoc()))
      << "incoherent impls (multiple satisfiable) for " << demand;
  nameCandidates(diagnostic, refusedOn->satisfiable, scope.getLoc(),
                 "candidate");
  return ImplResolver::DemandDisposition::Refused;
}

ImplResolver::DemandDisposition
ImplResolver::serveDemand(ProjectionType demand, ModuleOp scope,
                          OpBuilder &builder) {
  DemandFrame frame{Type(demand)};

  // What selection settles is recorded by selection itself, so the resolved
  // type is not wanted here -- the answer this call is for is whether asking
  // again could settle it differently.
  std::optional<Refutation> refusedOn;
  if (succeeded(resolveProjection(demand, scope, builder, /*err=*/nullptr,
                                  &refusedOn)))
    return DemandDisposition::Served;
  DemandDisposition disposition =
      refusalDisposition(Type(demand), scope, refusedOn);
  // A demand put to selection is one the stage has undertaken to serve,
  // wherever it was found spelled, and the stage's exit check reads the
  // recorded demands. So one selection could not serve yet is recorded here,
  // whether an engine recorded it before or the round found it spelled.
  if (disposition == DemandDisposition::Deferred)
    recordResolverProjectionMiss(Type(demand), scope);
  return disposition;
}

ImplResolver::DemandDisposition
ImplResolver::serveDemand(ClaimType demand, ModuleOp scope,
                          OpBuilder &builder) {
  DemandFrame frame{Type(demand)};

  // Proving the claim is what serves it: the demander could read the record
  // and not write it, so what it was waiting for is the proof this mints.
  std::optional<Refutation> refusedOn;
  if (succeeded(resolveAndEnsureProofFor(demand, scope, builder,
                                         /*err=*/nullptr, &refusedOn)))
    return DemandDisposition::Served;
  return refusalDisposition(Type(demand), scope, refusedOn);
}

Type ImplResolver::resolveProjectionsIn(Type ty, ModuleOp scope,
                                        OpBuilder &builder) {
  AttrTypeReplacer replacer = makeGroundProjectionReplacer(
      [this, scope, &builder](ProjectionType proj) -> std::optional<Type> {
    auto resolved = resolveProjection(proj, scope, builder);
    if (failed(resolved)) {
      // Preserve the unresolved demand for a later preparation boundary even
      // though this walk leaves its projection spelled as written.
      recordResolverProjectionMiss(Type(proj), scope);
      return std::nullopt;
    }
    return resolved->getBinding();
  });
  return normalizeProjectionsToFixedPoint(
      ty, scope, [&](Type t) { return replacer.replace(t); });
}

AttrTypeReplacer ImplResolver::makeProvenClaimReplacer(ModuleOp scope) const {
  MLIRContext *ctx = scope.getContext();
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement(
      [this, ctx, scope, recorded = memo.proofMemo.size()](ClaimType claim)
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
        auto it = memo.proofMemo.find({scope, claim.getTraitApplication()});
        if (it == memo.proofMemo.end())
          return std::nullopt;
        // The proven spelling names the same application, whose type arguments
        // can spell claims of their own, so the walk continues into the result
        // instead of stopping at it.
        return std::make_pair(
            Type(ClaimType::get(ctx, it->first.second, it->second)),
            WalkResult::advance());
      });
  return replacer;
}

FailureOr<ClaimType> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    ModuleOp scope,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err,
    std::optional<Refutation> *refusedOn) {
  DemandFrame frame{Type(wanted)};

  ClaimType originalWanted = wanted;

  // resolve an impl for wanted first
  auto resolvedImpl = resolveImplFor(wanted, scope, builder, err, refusedOn);
  if (failed(resolvedImpl)) return failure();
  ImplOp impl = resolvedImpl->impl;

  auto subst = argumentsOf(*resolvedImpl, ReadOnlyImplResolver(*this, scope), err);
  if (failed(subst)) return failure();
  auto monomorphic = monomorphicApplicationOf(*resolvedImpl, *subst);
  if (failed(monomorphic)) {
    if (err) err() << "could not monomorphize claim: " << originalWanted;
    return failure();
  }
  TraitApplicationAttr app = *monomorphic;
  MLIRContext *ctx = scope.getContext();

  // check the proof memo for this monomorphic app, as read here
  if (auto it = memo.proofMemo.find({scope, app}); it != memo.proofMemo.end())
    return ClaimType::get(ctx, app, it->second);

  // The evidence for each where entry at the arguments selection chose, in
  // order: the proof of an application entry, and the resolution of an
  // equality entry, whose sides selection carries to one spelling. The trait's
  // requirements are the impl's to return, read at the proof's derive.
  // Selection chose `impl` only once these entries held through a chain that
  // refuses an application it meets again (`assumptionsSatisfiableFor`), so
  // proving them never asks for `app` itself.
  SmallVector<ClaimType> entries = impl.getWhereClaimsAt(*subst);
  SmallVector<FlatSymbolRefAttr> subproofs;
  SmallVector<SmallVector<ResolutionStep>> equalitySteps;
  auto hop = [&](ProjectionType proj) {
    return resolveProjection(proj, scope, builder);
  };
  auto proofOf = [&](ClaimType claim) {
    return resolveAndEnsureProofFor(claim, scope, builder);
  };
  EqualitySource source{hop, proofOf, scope};
  for (ClaimType entry : entries) {
    if (entry.isApplication()) {
      auto subproof = resolveAndEnsureProofFor(entry, scope, builder, err);
      if (failed(subproof)) return failure();
      subproofs.push_back(subproof->getProof());
      continue;
    }
    SmallVector<ResolutionStep> steps;
    auto sides = resolveEquality(entry.getEqualityAttr(), source, steps);
    if (failed(sides) || sides->first != sides->second) {
      if (err) err() << "impl '@" << impl.getSymName() << "' applies where "
                     << entry << ", which selection does not settle at "
                     << originalWanted;
      return failure();
    }
    equalitySteps.push_back(std::move(steps));
  }

  // A proof is identified by the evidence its derive cites: one standing over
  // these premises answers, and otherwise the proof is written; either is
  // memoized by the monomorphic app.
  if (ClaimType found = findProof(scope, impl, app, subproofs))
    return recordProof(scope, app, found.getProof());
  ClaimType written = writeProof(scope, impl, app, *subst, entries, subproofs,
                                 equalitySteps, builder);
  return recordProof(scope, app, written.getProof());
}

//===----------------------------------------------------------------------===//
// Equality evidence
//===----------------------------------------------------------------------===//

FailureOr<std::pair<Type, Type>>
resolveEquality(TypeEqualityAttr eq, const EqualitySource &source,
                SmallVectorImpl<ResolutionStep> &steps, unsigned depth) {
  if (eq.getLhs() == eq.getRhs())
    return std::make_pair(eq.getLhs(), eq.getRhs());
  MLIRContext *ctx = eq.getContext();
  // A step whose resolution fails is left standing; the failure is carried out
  // past the fixed point so that it, and not the standing projection, is what
  // refuses. One replacer serves both sides and every round of the fixed
  // point, and it answers a projection it has already met from its cache, so a
  // projection spelled twice is one step. An equality entry of a resolving
  // impl recurses, bounded as every obligation chain is.
  bool stepFailed = depth >= kInstantiationDepthLimit;
  AttrTypeReplacer replacer = makeGroundProjectionReplacer(
      [&](ProjectionType proj) -> std::optional<Type> {
        if (stepFailed)
          return std::nullopt;
        FailureOr<ProjectionResolution> resolved = source.hop(proj);
        if (failed(resolved)) {
          stepFailed = true;
          return std::nullopt;
        }
        ResolutionStep step;
        step.equality = TypeEqualityAttr::get(ctx, Type(proj),
                                              resolved->getBinding());
        step.impl = FlatSymbolRefAttr::get(
            ctx, resolved->getImpl().getSymNameAttr());
        for (ClaimType entry :
             resolved->getImpl().getWhereClaimsAt(resolved->getArguments())) {
          if (entry.isApplication()) {
            FailureOr<ClaimType> proven = source.proofOf(entry);
            if (failed(proven)) {
              stepFailed = true;
              return std::nullopt;
            }
            step.premises.push_back(ClaimType::get(
                ctx, entry.getTraitApplication(), proven->getProof()));
            continue;
          }
          auto nested = std::make_shared<EqualityResolution>();
          nested->equality = entry.getEqualityAttr();
          auto sides = resolveEquality(nested->equality, source, nested->steps,
                                       depth + 1);
          if (failed(sides) || sides->first != sides->second) {
            stepFailed = true;
            return std::nullopt;
          }
          step.premises.push_back(std::move(nested));
        }
        steps.push_back(std::move(step));
        return resolved->getBinding();
      });
  // The shared normalizer owns the fixed point's bound and stops the
  // compilation at a chain that never grounds out, the same refusal every
  // ground resolver makes.
  auto resolve = [&](Type side) {
    return normalizeProjectionsToFixedPoint(
        side, source.module,
        [&](Type current) { return replacer.replace(current); });
  };
  Type lhs = resolve(eq.getLhs());
  Type rhs = resolve(eq.getRhs());
  bool standing = false;
  for (Type side : {lhs, rhs})
    side.walk([&](ProjectionType proj) {
      standing |= !isPolymorphicType(Type(proj));
    });
  if (stepFailed || standing)
    return failure();
  return std::make_pair(lhs, rhs);
}

Value buildEqualityEvidence(OpBuilder &builder, Location loc,
                            TypeEqualityAttr eq,
                            ArrayRef<ResolutionStep> steps) {
  if (eq.getLhs() == eq.getRhs())
    return WitnessOp::create(builder, loc, eq).getResult();
  assert(!steps.empty() &&
         "two spellings of one ground type differ in a projection they spell");
  SmallVector<Value> witnesses;
  for (const ResolutionStep &step : steps) {
    SmallVector<Value> premises;
    for (const auto &premise : step.premises) {
      if (auto *proven = std::get_if<ClaimType>(&premise)) {
        premises.push_back(WitnessOp::create(builder, loc, proven->getProof(),
                                             proven->getTraitApplication()));
        continue;
      }
      const auto &nested = std::get<std::shared_ptr<EqualityResolution>>(premise);
      premises.push_back(buildEqualityEvidence(builder, loc, nested->equality,
                                               nested->steps));
    }
    witnesses.push_back(
        WitnessOp::create(builder, loc, step.equality, step.impl, premises)
            .getResult());
  }
  if (witnesses.size() == 1 && steps.front().equality == eq)
    return witnesses.front();
  return WitnessOp::create(builder, loc, eq, ValueRange(witnesses)).getResult();
}

//===----------------------------------------------------------------------===//
// Forgetting what a later resolution can answer differently
//===----------------------------------------------------------------------===//

void ImplResolver::forgetRetriableRefusals() {
  auto &chosen = memo.resolutionMemo.chosen;
  SmallVector<ScopedApplication> retriable;
  for (const auto &entry : chosen)
    if (entry.second.isRefusal() &&
        entry.second.getRefutationArm() ==
            RefutationArm::NoSatisfiableCandidate)
      retriable.push_back(entry.first);
  // The drops move no record epoch, for the same reason writing the refusal
  // did not: what is erased here is a question impl selection will have to
  // answer again, never an answer a read of the record was given.
  for (const ScopedApplication &application : retriable)
    chosen.erase(application);
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
  recordReadOnlyResolverMiss(Type(demand), scope);
  return failure();
}

LogicalResult ReadOnlyImplResolver::decline(ClaimType demand) const {
  recordReadOnlyResolverMiss(Type(demand), scope);
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

FailureOr<ProjectionResolution>
ReadOnlyImplResolver::resolveProjection(ProjectionType proj) const {
  DemandFrame frame{Type(proj)};

  auto select = [this](ClaimType claim) { return getRecordedImplFor(claim); };
  return ProjectionResolution::get(proj, select, *this, /*err=*/nullptr);
}

Type ReadOnlyImplResolver::resolveProjectionsIn(Type ty) const {
  // A recorded fact answers a projection by its head: selection settled which
  // impl serves that application, and what that impl binds is a function of the
  // projection's own associated-type arguments. A head selection has settled is
  // therefore read here whatever those arguments still spell.
  AttrTypeReplacer replacer = makeGroundHeadProjectionReplacer(
      [this](ProjectionType proj) -> std::optional<Type> {
    auto resolved = resolveProjection(proj);
    if (succeeded(resolved))
      return resolved->getBinding();
    // Selection settles a projection only for an application some round put to
    // it, so a spelling nothing has asked about yet has no recorded fact to
    // read. One exactly one impl in the module binds is one selection would
    // settle the same way, so the module answers it here; where no impl or
    // several bind it, the lookup declines and says which, and the projection
    // stays spelled as written for the step that can make selection answer it.
    Type byLookup = resolveProjectionsByLookup(
        Type(proj), scope, DemandOrigin::RecordedFactRead,
        LookupScope::Ground);
    if (byLookup == Type(proj))
      return std::nullopt;
    return byLookup;
  });
  return normalizeProjectionsToFixedPoint(
      ty, scope, [&](Type t) { return replacer.replace(t); });
}

FailureOr<ClaimType>
ReadOnlyImplResolver::getRecordedProofFor(ClaimType claim) const {
  DemandFrame frame{Type(claim)};

  auto resolvedImpl = getRecordedImplFor(claim);
  if (failed(resolvedImpl)) return failure();

  auto subst = argumentsOf(*resolvedImpl, *this, /*err=*/nullptr);
  if (failed(subst)) return failure();
  auto monomorphic = monomorphicApplicationOf(*resolvedImpl, *subst);
  if (failed(monomorphic)) return failure();

  auto proof = getRecordedProof(*monomorphic);
  if (!proof) return failure();
  return ClaimType::get(claim.getContext(), *monomorphic, *proof);
}

} // end mlir::trait
