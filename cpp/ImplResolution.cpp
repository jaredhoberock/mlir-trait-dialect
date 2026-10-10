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

Answer<ImplOp>
ImplResolver::assumptionsSatisfiableFor(ImplOp impl,
                                        ClaimType concreteSelf,
                                        const SelectionSite &site,
                                        OpBuilder &builder) {
  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // consult the per-(impl,claim) satisfiability memo
  auto key = std::make_pair(impl, app);
  if (memo.assumptionsKnownSatisfiable.contains(key))
    return impl;

  // The candidate's arguments as the demanded application and its own where
  // clause determine them, each projection they spell read through selection.
  bool readOverflowed = false;
  auto byResolver = [&](Type ty) -> FailureOr<Type> {
    Answer<Type> read = resolveProjectionsIn(ty, site, builder);
    readOverflowed |= read.isOverflow();
    return read.orFailure();
  };
  TypeArguments args = impl.readTypeArgumentsFor(concreteSelf, byResolver);
  if (readOverflowed)
    return Answer<ImplOp>::overflow();
  SpecializationMap known = args.toSpecialization();

  for (ClaimType premise : impl.getWhereClaims()) {
    // An application premise is discharged by proving it: a unique impl whose
    // own premises hold in turn.
    if (premise.isApplication()) {
      auto assume = cast<ClaimType>(instantiate(Type(premise), known));
      Answer<ResolvedImpl> subImpl = resolveImplFor(assume, site, builder);
      if (!subImpl.isAnswer())
        return subImpl.stop<ImplOp>();
      Answer<ImplOp> held = assumptionsSatisfiableFor(
          subImpl->impl, subImpl->selectedClaim, site, builder);
      if (!held.isAnswer())
        return held;
      continue;
    }

    // An equality premise is discharged here rather than at the impl: it
    // restricts when the impl applies, and only the demanded application says
    // whether it holds. Each side is read through the candidate's own
    // associated-type bindings first -- a premise may project through the very
    // application being selected, which selection cannot ask itself about --
    // and then through what selection has settled elsewhere. A reading carrying
    // a type variable is left to the instances that fill it.
    // A side whose projections still change after the depth limit's worth of
    // steps is the premise's overflow.
    TypeEqualityAttr equality = premise.getEqualityAttr();
    NormalizationContext ownBindings;
    ownBindings.addLocalProjectionRule(impl, app, known);
    auto reduce = [&](Type ty) {
      Type instantiated = instantiate(ty, known);
      auto reduced = ownBindings.normalize(instantiated, /*err=*/nullptr);
      return settleThroughSelection(
          succeeded(reduced) ? *reduced : instantiated, site, builder);
    };
    Answer<std::optional<Type>> lhs = reduce(equality.getLhs());
    Answer<std::optional<Type>> rhs = reduce(equality.getRhs());
    if (lhs.isOverflow() || rhs.isOverflow())
      return Answer<ImplOp>::overflow();
    if (!*lhs || !*rhs) {
      overflow(Overflow::projectionSteps(instantiate(Type(premise), known)),
               site);
      return Answer<ImplOp>::overflow();
    }
    if (premiseDefersToInstances(**lhs, **rhs))
      continue;
    if (**lhs != **rhs)
      return Answer<ImplOp>::refusal();
  }

  // An impl whose arguments the header and the where clause together leave
  // open is no candidate: selection would have nothing to specialize its
  // methods and associated-type bindings with.
  if (!args.complete())
    return Answer<ImplOp>::refusal();

  // record a positive result
  memo.assumptionsKnownSatisfiable.insert(key);

  return impl;
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
    Type wanted,
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

Answer<ResolvedImpl> ImplResolver::resolveImplFor(
    ClaimType wanted,
    const SelectionSite &site,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  ModuleOp scope = site.scope;
  ClaimType originalWanted = wanted;
  using Selected = Answer<ResolvedImpl>;

  // Resolution resolves a demanded claim's monomorphic projections before it
  // selects an impl and records a proof. Every downstream fact minted here --
  // the resolution memo, the proof memo, the proof op, the witness -- is keyed
  // and spelled by this resolved claim, so those facts read back spelled
  // exactly as their post-resolution demand. Declaration-spelled demands
  // (trait and impl headers still carry their source projections) join that
  // resolved vocabulary here; no other component resolves a demanded claim's
  // spelling before impl selection and proof creation.
  // An overflow is a hard error: the stage fails on it, so once one is met
  // selection answers nothing more, rather than search on under it.
  if (overflowed)
    return Selected::overflow();

  Answer<Type> normalized = resolveProjectionsIn(wanted, site, builder);
  if (!normalized.isAnswer())
    return normalized.stop<ResolvedImpl>();
  ClaimType selected = cast<ClaimType>(*normalized);

  ResolutionMemo &memo = this->memo.resolutionMemo;
  TraitApplicationAttr app = selected.getTraitApplication();

  // first check the memo. A refusal asked about again is named as the first
  // ask named it, over the candidates that refused it: which op asks first is
  // no part of what the program says. A selection read back stands as high
  // above this chain as its derivation did.
  if (auto it = memo.chosen.find({scope, app}); it != memo.chosen.end()) {
    if (failed(checkObligationChainDepth(memo.visiting, it->second.height))) {
      overflow(Overflow::obligations(app, memo.visiting, it->second.height),
               site);
      return Selected::overflow();
    }
    memo.heightBelow = std::max(memo.heightBelow, it->second.height);
    return ResolvedImpl{it->second.impl, selected};
  }
  if (auto it = memo.refused.find({scope, app}); it != memo.refused.end()) {
    if (err) {
      if (auto trait = app.getTrait(scope, err); succeeded(trait))
        (void)diagnoseImplResolutionFailure(*trait, originalWanted,
                                            it->second.satisfiable,
                                            it->second.unsatisfiable, err);
    }
    return Selected::refusal();
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
    memo.refused.try_emplace({scope, app});
    return Selected::refusal();
  }
  TraitOp trait = *declaredTrait;

  // cycle guard: a selection that meets its own application again -- through a
  // candidate's where clause, or a projection a candidate's header spells --
  // means unsatisfiable there. The refusal leans on the frame it met again,
  // which is still part-way through.
  if (auto repeat = llvm::find_if(memo.visiting, [&](ObligationFrame frame) {
        return frame.application == app;
      });
      repeat != memo.visiting.end()) {
    memo.provisionalBelow = std::min<unsigned>(
        memo.provisionalBelow, repeat - memo.visiting.begin());
    return Selected::refusal();
  }

  // growth bound: a chain whose every step asks about a bigger application
  // repeats no frame, so only the depth stops it. The overflow stands at the
  // demand: every impl on the chain was asked about because something wanted
  // the application in hand, and naming one would name an impl with nothing
  // wrong with it.
  if (failed(checkObligationChainDepth(memo.visiting))) {
    overflow(Overflow::obligations(app, memo.visiting), site);
    return Selected::overflow();
  }

  // A refusal reached while a cycle guard refused a candidate at a frame below
  // this one leans on that frame, which is still part-way through, so it is
  // not entered (`provisionalBelow`). What the computation below learns is
  // carried out to the selections around it, its height among it: the
  // derivation of the candidate it chooses, or, refused, everything it
  // explored.
  unsigned depth = memo.visiting.size();
  unsigned provisionalAround = memo.provisionalBelow;
  unsigned heightAround = memo.heightBelow;
  unsigned height = 0;
  memo.provisionalBelow = UINT_MAX;
  memo.heightBelow = 0;
  auto restoreAround = llvm::scope_exit([&] {
    memo.provisionalBelow = std::min(provisionalAround, memo.provisionalBelow);
    memo.heightBelow = std::max(heightAround, height);
  });

  // The selection stands on the chain while it judges its candidates: their
  // headers, their where clauses, and a generated impl's.
  memo.visiting.push_back({app, SymbolRefAttr()});
  auto guard = llvm::scope_exit([&] { memo.visiting.pop_back(); });

  // collect candidates for wanted from the trait and
  // partition them into good/bad by satisfiable assumptions
  //
  // A candidate's header is read as selection reads any spelling: each ground
  // projection it spells is resolved through selection, which generates an
  // impl where a generator serves that projection's application. A header is
  // so settled before it is judged, so an answer entered below is one no later
  // minting reads otherwise; a projection whose own selection leans on a frame
  // still running leaves the refusal provisional.
  // A header with no normal form makes its impl no candidate; where the
  // selection is refused without it, the refusal is that overflow.
  std::optional<Type> unsettledHeader;
  bool headerOverflowed = false;
  auto bySelection = [&](Type ty) -> FailureOr<Type> {
    Answer<std::optional<Type>> read = settleThroughSelection(ty, site, builder);
    headerOverflowed |= read.isOverflow();
    if (read.isAnswer() && *read)
      return **read;
    if (read.isAnswer() && !unsettledHeader)
      unsettledHeader = ty;
    return failure();
  };
  SmallVector<ImplOp> candidates =
      trait.getCandidateImplsFor(selected, bySelection);
  if (headerOverflowed)
    return Selected::overflow();
  // The headers were read for every candidate, and the derivation of the one
  // chosen stands on them.
  unsigned headersHeight = memo.heightBelow;
  unsigned exploredHeight = headersHeight;

  // Each candidate's where clause is judged on its own: the height the chosen
  // candidate's derivation reaches is what the memo carries, not that of a
  // candidate refused beside it.
  SmallVector<ImplOp> good, bad;
  unsigned goodHeight = 0;
  auto judge = [&](ImplOp impl) -> LogicalResult {
    memo.heightBelow = 0;
    Answer<ImplOp> held =
        assumptionsSatisfiableFor(impl, selected, site, builder);
    if (held.isOverflow())
      return failure();
    exploredHeight = std::max(exploredHeight, memo.heightBelow);
    if (held.isAnswer()) {
      good.push_back(impl);
      goodHeight = memo.heightBelow;
    } else {
      bad.push_back(impl);
    }
    return success();
  };
  for (ImplOp impl : candidates)
    if (failed(judge(impl)))
      return Selected::overflow();

  // if there aren't any good candidates, try to generate one. An application a
  // generator has already supplied an impl for is not asked again: that impl
  // stands in the module and the scan above has just judged it, so asking
  // would only publish a second op under the name the first holds. A demand
  // still carrying a type variable reaches no generator: a generator answers
  // one application, which such a demand is not.
  if (good.empty() && selected.isMonomorphic() &&
      !memo.generatedFor.contains({scope, app})) {
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
    if (auto impl = generators.generateImpl(trait, selected, builder);
        succeeded(impl)) {
      memo.generatedFor.insert({scope, app});
      // A generated impl answers the application asked and no other: its
      // header is that application. A refusal the memo keeps is final only
      // because no impl generated later serves an application refused
      // earlier, so an impl whose header is any other stops selection where
      // the generator wrote it.
      ClaimType header = impl->getSelfClaim();
      if (header.getTraitApplication() != app) {
        overflow(Overflow::inexactImpl(*impl, app), site);
        return Selected::overflow();
      }
      if (failed(judge(*impl)))
        return Selected::overflow();
    }
  }

  // if exactly one good candidate exists, return it
  if (good.size() == 1) {
    height = std::max(headersHeight, goodHeight) + 1;
    memo.chosen.insert_or_assign({scope, app}, ChosenImpl{good.front(), height});
    return ResolvedImpl{good.front(), selected};
  }

  // otherwise, diagnose resolution failure, entering the refusal where it is
  // final. A refusal leaning on a candidate header with no normal form is that
  // header's overflow.
  height = exploredHeight + 1;
  if (unsettledHeader && good.empty()) {
    overflow(Overflow::projectionSteps(*unsettledHeader), site);
    return Selected::overflow();
  }
  if (memo.provisionalBelow >= depth)
    memo.refused.insert_or_assign({scope, app}, Refusal{good, bad});
  (void)diagnoseImplResolutionFailure(trait, originalWanted, good, bad, err);
  return Selected::refusal();
}

/// The witness of each of `steps`, built at `builder`'s insertion point: each
/// cites its impl with one claim per where entry, the evidence of the premise
/// discharging an application entry and the evidence built for the equality at
/// an equality entry.
static SmallVector<Value> buildStepWitnesses(OpBuilder &builder, Location loc,
                                             ArrayRef<ResolutionStep> steps) {
  SmallVector<Value> witnesses;
  for (const ResolutionStep &step : steps) {
    SmallVector<Value> premises;
    for (const auto &premise : step.premises) {
      if (auto *proven = std::get_if<std::shared_ptr<ProvenPremise>>(&premise)) {
        premises.push_back(buildPremiseEvidence(builder, loc, **proven));
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
  return witnesses;
}

/// `value` spelled as `spelling`: `value` itself where it is spelled so
/// already, else its coercion citing the witness of each of `steps`, the steps
/// carrying the one spelling to the other.
static Value respell(OpBuilder &builder, Location loc, Value value,
                     ClaimType spelling, ArrayRef<ResolutionStep> steps) {
  if (value.getType() == Type(spelling))
    return value;
  return CoerceOp::create(builder, loc, spelling, value,
                          buildStepWitnesses(builder, loc, steps))
      .getResult();
}

Value buildPremiseEvidence(OpBuilder &builder, Location loc,
                           const ProvenPremise &premise) {
  Value witness =
      WitnessOp::create(builder, loc, premise.proven.getProof(),
                        premise.proven.getTraitApplication());
  return respell(builder, loc, witness, premise.entry, premise.respelling);
}

/// Writes at the end of `scope` the proof `name` of `app`, whose body derives
/// `header`, `impl`'s header at the citation, over one premise per entry of
/// `entries`, `impl`'s where entries at the citation -- the evidence of the
/// next of `applicationPremises` at an application entry and the evidence the
/// next of `equalitySteps` build at an equality entry -- and returns it
/// respelled as `app` by `headerSteps`. Where a symbol of `scope` holds `name`
/// already, the proof is named as the module's symbol table renames it
/// (`SymbolTable::insert`): mangled names are not one-to-one, so the table,
/// not the mangling, makes a proof's name unique.
static ProofOp
writeProofBody(OpBuilder &builder, ModuleOp scope, StringRef name, ImplOp impl,
               ClaimType header, TraitApplicationAttr app,
               ArrayRef<ClaimType> entries,
               ArrayRef<ProvenPremise> applicationPremises,
               ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
               ArrayRef<ResolutionStep> headerSteps) {
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
  auto nextPremise = applicationPremises.begin();
  auto nextSteps = equalitySteps.begin();
  for (ClaimType entry : entries) {
    if (entry.isApplication())
      premises.push_back(buildPremiseEvidence(builder, loc, *nextPremise++));
    else
      premises.push_back(buildEqualityEvidence(
          builder, loc, entry.getEqualityAttr(), *nextSteps++));
  }
  auto derived = DeriveOp::create(builder, loc, header,
                                  FlatSymbolRefAttr::get(ctx, impl.getSymName()),
                                  premises);
  ReturnOp::create(builder, loc,
                   respell(builder, loc, derived.getResult(),
                           ClaimType::get(ctx, app), headerSteps));
  return proof;
}

ClaimType ImplResolver::findProof(ModuleOp scope, ImplOp impl,
                                  TraitApplicationAttr app,
                                  ArrayRef<FlatSymbolRefAttr> subproofs) const {
  MLIRContext *ctx = scope.getContext();
  // An impl with no parameters and no where entries is its own proof where its
  // header spells the application; one whose header spells it otherwise is
  // proven by a proof respelling its header.
  if (impl.isUnconditional() && impl.getSelfApplication() == app)
    return ClaimType::get(ctx, app,
                          FlatSymbolRefAttr::get(ctx, impl.getSymName()));
  // A proof is identified by the evidence it derives its claim from: the impl,
  // the application, and the proof each application premise names. The
  // module's proofs are read as they stand, in module order, so the first
  // proof of an impl at an application over those premises is the one found.
  // The impl is matched by identity rather than by name: a name is resolved
  // in one symbol table, and two modules can each hold an impl of that name
  // meaning two different impls.
  auto citesSubproofs = [&](ProofOp proof) {
    auto next = subproofs.begin();
    for (Value premise : proof.getDerive().getAssumptions()) {
      auto claim = cast<ClaimType>(premise.getType());
      if (claim.isApplication() && claim.getProof() != *next++)
        return false;
    }
    return true;
  };
  for (ProofOp proof : scope.getOps<ProofOp>())
    if (proof.getTraitApplication() == app && proof.getImpl() == impl &&
        citesSubproofs(proof))
      return ClaimType::get(
          ctx, app, FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr()));
  return {};
}

Answer<ClaimType> ImplResolver::writeProof(
    ModuleOp scope, ImplOp impl, TraitApplicationAttr app,
    const SpecializationMap &arguments, ArrayRef<ClaimType> entries,
    ArrayRef<FlatSymbolRefAttr> subproofs,
    ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
    const SelectionSite &site, OpBuilder &builder) {
  MLIRContext *ctx = scope.getContext();
  // Every respelling is resolved before the proof is begun: resolving one can
  // write the proofs its steps cite, and those stand before this one.
  auto header = cast<ClaimType>(instantiate(Type(impl.getSelfClaim()), arguments));
  Answer<SmallVector<ResolutionStep>> headerSteps = resolveRespelling(
      header.getTraitApplication(), app, site, builder, /*depth=*/0);
  if (!headerSteps.isAnswer())
    return headerSteps.stop<ClaimType>();
  SmallVector<ProvenPremise> applicationPremises;
  auto nextSubproof = subproofs.begin();
  for (ClaimType entry : entries) {
    if (!entry.isApplication())
      continue;
    Answer<ProvenPremise> premise =
        resolvePremise(entry, *nextSubproof++, site, builder, /*depth=*/0);
    if (!premise.isAnswer())
      return premise.stop<ClaimType>();
    applicationPremises.push_back(std::move(*premise));
  }
  ProofOp proof = writeProofBody(
      builder, scope, impl.generateMangledName(arguments) + "_p", impl, header,
      app, entries, applicationPremises, equalitySteps, *headerSteps);
  return ClaimType::get(ctx, app,
                        FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr()));
}

Answer<SmallVector<ResolutionStep>>
ImplResolver::resolveRespelling(TraitApplicationAttr spelled,
                                TraitApplicationAttr resolved,
                                const SelectionSite &site, OpBuilder &builder,
                                unsigned depth) {
  SmallVector<ResolutionStep> steps;
  for (auto [from, to] :
       llvm::zip_equal(spelled.getTypeArgs(), resolved.getTypeArgs())) {
    if (from == to)
      continue;
    auto sides =
        resolveEquality(TypeEqualityAttr::get(from.getContext(), from, to),
                        site, builder, steps, /*err=*/nullptr, depth);
    if (!sides.isAnswer())
      return sides.stop<SmallVector<ResolutionStep>>();
    if (sides->first != sides->second)
      return Answer<SmallVector<ResolutionStep>>::refusal();
  }
  return steps;
}

Answer<ProvenPremise>
ImplResolver::resolvePremise(ClaimType entry, FlatSymbolRefAttr proof,
                             const SelectionSite &site, OpBuilder &builder,
                             unsigned depth) {
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(site.scope, proof);
  if (failed(cited))
    return Answer<ProvenPremise>::refusal();
  auto proofOp = dyn_cast<ProofOp>(*cited);
  TraitApplicationAttr own = proofOp ? proofOp.getTraitApplication()
                                     : cast<ImplOp>(*cited).getSelfApplication();
  Answer<SmallVector<ResolutionStep>> respelling = resolveRespelling(
      own, entry.getTraitApplication(), site, builder, depth);
  if (!respelling.isAnswer())
    return respelling.stop<ProvenPremise>();
  MLIRContext *ctx = entry.getContext();
  return ProvenPremise{ClaimType::get(ctx, own, proof),
                       ClaimType::get(ctx, entry.getTraitApplication(), proof),
                       std::move(*respelling)};
}

ImplResolver::ImplResolver(ModuleOp m) : module(m) {
  // collect ImplGenerators from each dialect with the appropriate interface
  for (Dialect *dialect : module.getContext()->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<GenerateImplsInterface>()) {
      iface->populateImplGenerators(generators);
    }
  }
}

/// The arguments carrying `resolved`'s impl header to the claim selection chose
/// it for, the header read through `readHeader`, the context selection chose it
/// under.
static FailureOr<SpecializationMap>
argumentsOf(const ResolvedImpl &resolved, Normalizer readHeader,
            llvm::function_ref<InFlightDiagnostic()> err) {
  ImplOp impl = resolved.impl;
  return impl.buildSubstitutionForSelfClaim(resolved.selectedClaim, readHeader,
                                            err);
}

Answer<ProjectionResolution> ProjectionResolution::get(
    ProjectionType projection,
    llvm::function_ref<Answer<ResolvedImpl>(ClaimType)> select,
    Normalizer readHeader,
    llvm::function_ref<InFlightDiagnostic()> err) {
  Answer<ResolvedImpl> resolved = select(projection.asClaim());
  if (!resolved.isAnswer())
    return resolved.stop<ProjectionResolution>();
  auto arguments = argumentsOf(*resolved, readHeader, err);
  if (failed(arguments))
    return Answer<ProjectionResolution>::refusal();
  ImplOp impl = resolved->impl;
  auto binding = impl.specializeAssociatedTypeBinding(
      projection.getAssocName().getValue(), projection.getAssocTypeArgs(),
      *arguments, err);
  if (failed(binding))
    return Answer<ProjectionResolution>::refusal();
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

Answer<ProjectionResolution> ImplResolver::resolveProjection(
    ProjectionType proj,
    const SelectionSite &site,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto select = [&](ClaimType claim) {
    return resolveImplFor(claim, site, builder, err);
  };
  auto readHeader = [&](Type ty) -> FailureOr<Type> {
    return resolveProjectionsIn(ty, site, builder).orFailure();
  };
  return ProjectionResolution::get(proj, select, readHeader, err);
}

Answer<std::optional<Type>>
ImplResolver::settleThroughSelection(
    Type ty, const SelectionSite &site, OpBuilder &builder,
    decltype(&makeGroundProjectionReplacer) replacerFor) {
  bool stepOverflowed = false;
  AttrTypeReplacer replacer = replacerFor(
      [&](ProjectionType proj) -> std::optional<Type> {
    Answer<ProjectionResolution> resolved =
        resolveProjection(proj, site, builder);
    stepOverflowed |= resolved.isOverflow();
    if (!resolved.isAnswer())
      return std::nullopt;
    return resolved->getBinding();
  });
  Type out;
  bool settles = succeeded(tryNormalizeProjectionsToFixedPoint(
      ty, [&](Type t) { return replacer.replace(t); }, out));
  if (stepOverflowed)
    return Answer<std::optional<Type>>::overflow();
  if (!settles)
    return std::optional<Type>();
  return std::optional<Type>(out);
}

Answer<Type> ImplResolver::resolveProjectionsIn(
    Type ty, const SelectionSite &site, OpBuilder &builder,
    decltype(&makeGroundProjectionReplacer) replacerFor) {
  Answer<std::optional<Type>> settled =
      settleThroughSelection(ty, site, builder, replacerFor);
  if (!settled.isAnswer())
    return settled.stop<Type>();
  if (!*settled) {
    overflow(Overflow::projectionSteps(ty), site);
    return Answer<Type>::overflow();
  }
  return **settled;
}

void Overflow::emit(Location anchor) const {
  if (ImplOp generated = impl) {
    generated->emitError() << "an impl generated for " << app
                           << " must state that application; it states "
                           << generated.getSelfClaim().getTraitApplication();
    return;
  }
  if (app) {
    emitObligationOverflow(anchor, app, chain, height);
    return;
  }
  emitError(anchor) << "overflow evaluating the requirement " << spelled
                    << ": " << kInstantiationDepthLimit
                    << " projection steps stand on the chain";
}

void ImplResolver::overflow(const Overflow &what, const SelectionSite &site) {
  memo.resolutionMemo.provisionalBelow = 0;
  overflowed = true;
  if (overflowSites.insert(site.cause).second)
    what.emit(site.cause);
}

Answer<ClaimType> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    const SelectionSite &site,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  ModuleOp scope = site.scope;
  ClaimType originalWanted = wanted;
  using Proven = Answer<ClaimType>;

  // resolve an impl for wanted first
  Answer<ResolvedImpl> resolvedImpl = resolveImplFor(wanted, site, builder, err);
  if (!resolvedImpl.isAnswer())
    return resolvedImpl.stop<ClaimType>();
  ImplOp impl = resolvedImpl->impl;

  auto readHeader = [&](Type ty) -> FailureOr<Type> {
    return resolveProjectionsIn(ty, site, builder).orFailure();
  };
  auto subst = argumentsOf(*resolvedImpl, readHeader, err);
  if (failed(subst))
    return Proven::refusal();
  auto monomorphic = monomorphicApplicationOf(*resolvedImpl, *subst);
  if (failed(monomorphic)) {
    if (err) err() << "could not monomorphize claim: " << originalWanted;
    return Proven::refusal();
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
  // refuses an application it meets again (the cycle guard of
  // `resolveImplFor`), so proving them never asks for `app` itself.
  SmallVector<ClaimType> entries = impl.getWhereClaimsAt(*subst);
  SmallVector<FlatSymbolRefAttr> subproofs;
  SmallVector<SmallVector<ResolutionStep>> equalitySteps;
  for (ClaimType entry : entries) {
    if (entry.isApplication()) {
      Proven subproof = resolveAndEnsureProofFor(entry, site, builder, err);
      if (!subproof.isAnswer())
        return subproof;
      subproofs.push_back(subproof->getProof());
      continue;
    }
    SmallVector<ResolutionStep> steps;
    auto sides = resolveEquality(entry.getEqualityAttr(), site, builder, steps);
    if (sides.isOverflow())
      return Proven::overflow();
    if (!sides.isAnswer() || sides->first != sides->second) {
      if (err) err() << "impl '@" << impl.getSymName() << "' applies where "
                     << entry << ", which selection does not settle at "
                     << originalWanted;
      return Proven::refusal();
    }
    equalitySteps.push_back(std::move(steps));
  }

  // A proof is identified by the evidence its derive cites: one standing over
  // these premises answers, and otherwise the proof is written; either is
  // memoized by the monomorphic app.
  if (ClaimType found = findProof(scope, impl, app, subproofs))
    return recordProof(scope, app, found.getProof());
  Proven written = writeProof(scope, impl, app, *subst, entries, subproofs,
                              equalitySteps, site, builder);
  if (!written.isAnswer())
    return written;
  return recordProof(scope, app, written->getProof());
}

//===----------------------------------------------------------------------===//
// Equality evidence
//===----------------------------------------------------------------------===//

Answer<std::pair<Type, Type>>
ImplResolver::resolveEquality(TypeEqualityAttr eq, const SelectionSite &site,
                              OpBuilder &builder,
                              SmallVectorImpl<ResolutionStep> &steps,
                              llvm::function_ref<InFlightDiagnostic()> err,
                              unsigned depth) {
  using Sides = Answer<std::pair<Type, Type>>;
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
  bool stepOverflowed = false;
  auto stop = [&](bool overflowedHere) {
    stepFailed = true;
    stepOverflowed |= overflowedHere;
    return std::nullopt;
  };
  AttrTypeReplacer replacer = makeGroundProjectionReplacer(
      [&](ProjectionType proj) -> std::optional<Type> {
        if (stepFailed)
          return std::nullopt;
        Answer<ProjectionResolution> resolved =
            resolveProjection(proj, site, builder, err);
        if (!resolved.isAnswer())
          return stop(resolved.isOverflow());
        // The step is stated at the application the impl's header spells at
        // its arguments, which its witness's citation rebuilds; the steps
        // resolving the projections either spelling holds carry the
        // projection as spelled to it.
        TraitApplicationAttr header =
            cast<ClaimType>(instantiate(Type(resolved->getImpl().getSelfClaim()),
                                        resolved->getArguments()))
                .getTraitApplication();
        ProjectionType stated = proj;
        if (header != proj.getTraitApplication()) {
          Answer<SmallVector<ResolutionStep>> respelling = resolveRespelling(
              proj.getTraitApplication(), header, site, builder, depth + 1);
          if (!respelling.isAnswer())
            return stop(respelling.isOverflow());
          for (ResolutionStep &carried : *respelling)
            steps.push_back(std::move(carried));
          stated = ProjectionType::get(ctx, header, proj.getAssocName(),
                                       proj.getAssocTypeArgs());
        }
        ResolutionStep step;
        step.equality = TypeEqualityAttr::get(ctx, Type(stated),
                                              resolved->getBinding());
        step.impl = FlatSymbolRefAttr::get(
            ctx, resolved->getImpl().getSymNameAttr());
        for (ClaimType entry :
             resolved->getImpl().getWhereClaimsAt(resolved->getArguments())) {
          if (entry.isApplication()) {
            Answer<ClaimType> proven =
                resolveAndEnsureProofFor(entry, site, builder, err);
            if (!proven.isAnswer())
              return stop(proven.isOverflow());
            Answer<ProvenPremise> premise = resolvePremise(
                entry, proven->getProof(), site, builder, depth + 1);
            if (!premise.isAnswer())
              return stop(premise.isOverflow());
            step.premises.push_back(
                std::make_shared<ProvenPremise>(std::move(*premise)));
            continue;
          }
          auto nested = std::make_shared<EqualityResolution>();
          nested->equality = entry.getEqualityAttr();
          Sides sides = resolveEquality(nested->equality, site, builder,
                                        nested->steps, err, depth + 1);
          if (!sides.isAnswer() || sides->first != sides->second)
            return stop(sides.isOverflow());
          step.premises.push_back(std::move(nested));
        }
        steps.push_back(std::move(step));
        return resolved->getBinding();
      });
  // A side whose projections still change after the depth limit's worth of
  // steps is an overflow, named once, as every resolution selection runs
  // names its own.
  bool settles = true;
  auto resolve = [&](Type side) {
    Type out;
    settles &= succeeded(tryNormalizeProjectionsToFixedPoint(
        side, [&](Type current) { return replacer.replace(current); }, out));
    return out;
  };
  Type lhs = resolve(eq.getLhs());
  Type rhs = resolve(eq.getRhs());
  if (stepOverflowed)
    return Sides::overflow();
  if (!settles) {
    overflow(Overflow::projectionSteps(ClaimType::getEquality(ctx, eq)), site);
    return Sides::overflow();
  }
  bool standing = false;
  for (Type side : {lhs, rhs})
    side.walk([&](ProjectionType proj) {
      standing |= !isPolymorphicType(Type(proj));
    });
  if (stepFailed || standing)
    return Sides::refusal();
  return std::make_pair(lhs, rhs);
}

Value buildEqualityEvidence(OpBuilder &builder, Location loc,
                            TypeEqualityAttr eq,
                            ArrayRef<ResolutionStep> steps) {
  if (eq.getLhs() == eq.getRhs())
    return WitnessOp::create(builder, loc, eq).getResult();
  assert(!steps.empty() &&
         "two spellings of one ground type differ in a projection they spell");
  SmallVector<Value> witnesses = buildStepWitnesses(builder, loc, steps);
  if (witnesses.size() == 1 && steps.front().equality == eq)
    return witnesses.front();
  return WitnessOp::create(builder, loc, eq, ValueRange(witnesses)).getResult();
}

void ImplResolver::nameRefusal(
    Type obligation, ModuleOp scope,
    llvm::function_ref<InFlightDiagnostic()> err) const {
  auto projection = dyn_cast<ProjectionType>(obligation);
  ClaimType claim =
      projection ? projection.asClaim() : cast<ClaimType>(obligation);
  auto selected = cast<ClaimType>(readSettledProjectionsIn(claim, scope));
  TraitApplicationAttr app = selected.getTraitApplication();
  auto refusal = memo.resolutionMemo.refused.find({scope, app});
  if (refusal == memo.resolutionMemo.refused.end())
    return;
  auto trait = app.getTrait(scope, /*errFn=*/nullptr);
  if (succeeded(trait))
    (void)diagnoseImplResolutionFailure(*trait, obligation,
                                        refusal->second.satisfiable,
                                        refusal->second.unsatisfiable, err);
}

//===----------------------------------------------------------------------===//
// Reading what selection has settled
//===----------------------------------------------------------------------===//

FailureOr<ProjectionResolution>
ImplResolver::readSettledProjection(ProjectionType proj, ModuleOp scope) const {
  auto select = [&](ClaimType wanted) -> Answer<ResolvedImpl> {
    // Selection keys what it settles by the claim whose projections it
    // resolved, so the spelling is put through the same resolution first.
    ClaimType selected = cast<ClaimType>(readSettledProjectionsIn(wanted, scope));
    auto it = memo.resolutionMemo.chosen.find(
        {scope, selected.getTraitApplication()});
    if (it == memo.resolutionMemo.chosen.end())
      return Answer<ResolvedImpl>::refusal();
    return ResolvedImpl{it->second.impl, selected};
  };
  auto readHeader = [&](Type ty) -> FailureOr<Type> {
    return readSettledProjectionsIn(ty, scope);
  };
  return ProjectionResolution::get(proj, select, readHeader, /*err=*/nullptr)
      .orFailure();
}

Type ImplResolver::readSettledProjectionsIn(Type ty, ModuleOp scope) const {
  // A settled fact answers a projection by its head: selection settled which
  // impl serves that application, and what that impl binds is a function of the
  // projection's own associated-type arguments. A head selection has settled is
  // therefore read here whatever those arguments still spell.
  AttrTypeReplacer replacer = makeGroundHeadProjectionReplacer(
      [this, scope](ProjectionType proj) -> std::optional<Type> {
    auto resolved = readSettledProjection(proj, scope);
    if (failed(resolved))
      return std::nullopt;
    return resolved->getBinding();
  });
  // A reading still changing at the depth limit is handed back as its partial
  // normal form: it reads only answers selection settled, and selection named
  // the overflow wherever it met one.
  Type out;
  (void)tryNormalizeProjectionsToFixedPoint(
      ty, [&](Type t) { return replacer.replace(t); }, out);
  return out;
}

} // end mlir::trait
