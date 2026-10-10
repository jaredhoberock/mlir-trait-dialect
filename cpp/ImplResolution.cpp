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

Answer<SpecializationMap>
ImplResolver::assumptionsSatisfiableFor(ImplOp impl,
                                        ClaimType concreteSelf,
                                        const SelectionSite &site,
                                        OpBuilder &builder) {
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // consult the per-(impl,claim) satisfiability memo
  auto key = std::make_pair(impl, app);
  if (auto known = memo.assumptionsKnownSatisfiable.find(key);
      known != memo.assumptionsKnownSatisfiable.end())
    return known->second;

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
    return Answer<SpecializationMap>::overflow();
  SpecializationMap known = args.toSpecialization();

  for (ClaimType premise : impl.getWhereClaims()) {
    // An application premise is discharged by proving it: a unique impl whose
    // own premises hold in turn.
    if (premise.isApplication()) {
      auto assume = cast<ClaimType>(instantiate(Type(premise), known));
      Answer<ResolvedImpl> subImpl = resolveImplFor(assume, site, builder);
      if (!subImpl.isAnswer())
        return subImpl.stop<SpecializationMap>();
      Answer<SpecializationMap> held = assumptionsSatisfiableFor(
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
    auto reduce = [&](Type ty) {
      Type instantiated = instantiate(ty, known);
      FailureOr<Type> reduced = impl.readOwnBindings(instantiated, known);
      return settleThroughSelection(
          succeeded(reduced) ? *reduced : instantiated, site, builder);
    };
    Answer<std::optional<Type>> lhs = reduce(equality.getLhs());
    Answer<std::optional<Type>> rhs = reduce(equality.getRhs());
    if (lhs.isOverflow() || rhs.isOverflow())
      return Answer<SpecializationMap>::overflow();
    if (!*lhs || !*rhs) {
      overflow(Overflow::projectionSteps(instantiate(Type(premise), known)),
               site);
      return Answer<SpecializationMap>::overflow();
    }
    if (premiseDefersToInstances(**lhs, **rhs))
      continue;
    if (**lhs != **rhs)
      return Answer<SpecializationMap>::refusal();
  }

  // An impl whose arguments the header and the where clause together leave
  // open is no candidate: selection would have nothing to specialize its
  // methods and associated-type bindings with.
  if (!args.complete())
    return Answer<SpecializationMap>::refusal();

  // record a positive result
  memo.assumptionsKnownSatisfiable.try_emplace(key, known);

  return known;
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
    return ResolvedImpl{it->second.impl, selected, it->second.arguments};
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
  SpecializationMap goodArguments;
  auto judge = [&](ImplOp impl) -> LogicalResult {
    memo.heightBelow = 0;
    Answer<SpecializationMap> held =
        assumptionsSatisfiableFor(impl, selected, site, builder);
    if (held.isOverflow())
      return failure();
    exploredHeight = std::max(exploredHeight, memo.heightBelow);
    if (held.isAnswer()) {
      good.push_back(impl);
      goodHeight = memo.heightBelow;
      goodArguments = std::move(*held);
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
    memo.chosen.insert_or_assign({scope, app},
                                 ChosenImpl{good.front(), goodArguments, height});
    return ResolvedImpl{good.front(), selected, std::move(goodArguments)};
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
    witnesses.push_back(WitnessOp::create(builder, loc, step.equality,
                                          step.impl, step.arguments, premises)
                            .getResult());
  }
  return witnesses;
}

/// `value` spelled as `spelling`, an unproven claim: `value` itself where it
/// is spelled so already, else its coercion citing the witness of each of
/// `steps`, the steps carrying the one spelling to the other. The coercion's
/// result names no proof: a claim names the proof of exactly its spelling, and
/// the proof `value` carries proves another one.
static Value respell(OpBuilder &builder, Location loc, Value value,
                     ClaimType spelling, ArrayRef<ResolutionStep> steps) {
  if (cast<ClaimType>(value.getType()).asUnproven() == spelling)
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

/// A proof named `name` standing empty at the end of `scope`, its body's
/// block the insertion point `builder` is left at. Where a symbol of `scope`
/// holds `name` already, the proof is named as the module's symbol table
/// renames it when the builder's listener enters it there
/// (`SymbolTableKeeper`): mangled names are not one-to-one, so the table, not
/// the mangling, makes a proof's name unique.
static ProofOp createProof(OpBuilder &builder, ModuleOp scope, StringRef name) {
  // A created proof is IR nothing revisits unless someone hears about it, for
  // the same reason a generated impl is.
  assert(builder.getListener() &&
         "proof creation requires a builder whose insertions someone observes");
  builder.setInsertionPointToEnd(scope.getBody());
  ProofOp proof = ProofOp::create(builder, builder.getUnknownLoc(), name);
  builder.setInsertionPointToEnd(&proof.getBody().front());
  return proof;
}

/// Writes at the end of `scope` the proof `name` of `app`, whose body derives
/// `header`, `impl`'s header at `arguments`, citing `impl` at them over one
/// premise per entry of `entries`, `impl`'s where entries there -- the
/// evidence of the next of `applicationPremises` at an application entry and
/// the evidence the next of `equalitySteps` build at an equality entry -- and
/// returns it respelled as `app` by `headerSteps`.
static ProofOp
writeProofBody(OpBuilder &builder, ModuleOp scope, StringRef name, ImplOp impl,
               const SpecializationMap &arguments, ClaimType header,
               TraitApplicationAttr app,
               ArrayRef<ClaimType> entries,
               ArrayRef<ProvenPremise> applicationPremises,
               ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
               ArrayRef<ResolutionStep> headerSteps) {
  MLIRContext *ctx = scope.getContext();
  OpBuilder::InsertionGuard guard(builder);
  ProofOp proof = createProof(builder, scope, name);
  Location loc = proof.getLoc();
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
                                  impl.stateArguments(arguments), premises);
  ReturnOp::create(builder, loc,
                   respell(builder, loc, derived.getResult(),
                           ClaimType::get(ctx, app), headerSteps));
  return proof;
}

FlatSymbolRefAttr ImplResolver::lookupProof(const ProofIdentity &identity) {
  auto scope = cast<ModuleOp>(std::get<0>(identity));
  if (indexedScopes.insert(scope).second)
    for (ProofOp proof : scope.getOps<ProofOp>())
      indexProof(scope, proof);
  return proofs.lookup(identity);
}

void ImplResolver::indexProof(ModuleOp scope, ProofOp proof) {
  // A proof is identified by what it cites. The impl is held by identity
  // rather than by name: a name is resolved in one symbol table, and two
  // modules can each hold an impl of that name meaning two different impls.
  TraitApplicationAttr app = proof.getTraitApplication();
  FlatSymbolRefAttr name =
      FlatSymbolRefAttr::get(proof.getContext(), proof.getSymNameAttr());
  if (Operation *source = proof.getCastSource()) {
    proofs.try_emplace({scope, source, app, ArrayAttr()}, name);
    return;
  }
  SmallVector<Attribute> subproofs = llvm::map_to_vector(
      proof.getSubproofs(),
      [](ClaimType subproof) -> Attribute { return subproof.getProof(); });
  proofs.try_emplace({scope, proof.getImpl(), app,
                      ArrayAttr::get(proof.getContext(), subproofs)},
                     name);
}

ClaimType ImplResolver::findProof(ModuleOp scope, ImplOp impl,
                                  TraitApplicationAttr app,
                                  ArrayRef<FlatSymbolRefAttr> subproofs) {
  MLIRContext *ctx = scope.getContext();
  // An impl with no parameters and no where entries is its own proof where its
  // header spells the application; one whose header spells it otherwise is
  // proven by a cast of it.
  if (impl.isUnconditional() && impl.getSelfApplication() == app)
    return ClaimType::get(ctx, app,
                          FlatSymbolRefAttr::get(ctx, impl.getSymName()));
  SmallVector<Attribute> cited(subproofs.begin(), subproofs.end());
  if (FlatSymbolRefAttr proof =
          lookupProof({scope, impl, app, ArrayAttr::get(ctx, cited)}))
    return ClaimType::get(ctx, app, proof);
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
      builder, scope, impl.generateMangledName(arguments) + "_p", impl,
      arguments, header, app, entries, applicationPremises, equalitySteps,
      *headerSteps);
  indexProof(scope, proof);
  return ClaimType::get(ctx, app,
                        FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr()));
}

Answer<ClaimType> ImplResolver::respellProof(ClaimType proven,
                                             TraitApplicationAttr to,
                                             const SelectionSite &site,
                                             OpBuilder &builder) {
  if (proven.getTraitApplication() == to)
    return proven;
  using Respelled = Answer<ClaimType>;
  ModuleOp scope = site.scope;
  MLIRContext *ctx = scope.getContext();
  // The source of the cast: the root `proven` rests on.
  auto root = ProofOp::getRootOf(scope, proven.getProof());
  if (failed(root))
    return Respelled::refusal();
  Operation *source = *root;
  auto proof = dyn_cast<ProofOp>(source);
  TraitApplicationAttr from = proof ? proof.getTraitApplication()
                                    : cast<ImplOp>(source).getSelfApplication();
  FlatSymbolRefAttr sourceName = FlatSymbolRefAttr::get(
      ctx, cast<SymbolOpInterface>(source).getNameAttr());
  if (from == to)
    return ClaimType::get(ctx, to, sourceName);
  if (FlatSymbolRefAttr standing = lookupProof({scope, source, to, ArrayAttr()}))
    return ClaimType::get(ctx, to, standing);

  // Every respelling step is resolved before the cast is begun: resolving one
  // can write the proofs its steps cite, and those stand before this one.
  Answer<SmallVector<ResolutionStep>> steps =
      resolveRespelling(from, to, site, builder, /*depth=*/0);
  if (!steps.isAnswer())
    return steps.stop<ClaimType>();
  OpBuilder::InsertionGuard guard(builder);
  ProofOp cast = createProof(builder, scope,
                             (sourceName.getValue() + "_as").str());
  Location loc = cast.getLoc();
  Value witness = WitnessOp::create(builder, loc, sourceName, from).getResult();
  ReturnOp::create(builder, loc,
                   respell(builder, loc, witness, ClaimType::get(ctx, to),
                           *steps));
  indexProof(scope, cast);
  return ClaimType::get(ctx, to,
                        FlatSymbolRefAttr::get(ctx, cast.getSymNameAttr()));
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
  return ProvenPremise{ClaimType::get(ctx, own, proof), entry.asUnproven(),
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

Answer<ProjectionResolution> ProjectionResolution::get(
    ProjectionType projection,
    llvm::function_ref<Answer<ResolvedImpl>(ClaimType)> select,
    llvm::function_ref<InFlightDiagnostic()> err) {
  Answer<ResolvedImpl> resolved = select(projection.asClaim());
  if (!resolved.isAnswer())
    return resolved.stop<ProjectionResolution>();
  ImplOp impl = resolved->impl;
  auto binding = impl.specializeAssociatedTypeBinding(
      projection.getAssocName().getValue(), projection.getAssocTypeArgs(),
      resolved->arguments, err);
  if (failed(binding))
    return Answer<ProjectionResolution>::refusal();
  return ProjectionResolution(projection, impl, resolved->arguments, *binding);
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
  return ProjectionResolution::get(proj, select, err);
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
  memo.provisionalBelow = 0;
  overflowed = true;
  if (overflowSites.insert(site.cause).second)
    what.emit(site.cause);
}

Answer<ClaimType> ImplResolver::proveResolution(
    ClaimType wanted, const SelectionSite &site, OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  using Proven = Answer<ClaimType>;
  Answer<ResolvedImpl> resolvedImpl = resolveImplFor(wanted, site, builder, err);
  if (!resolvedImpl.isAnswer())
    return resolvedImpl.stop<ClaimType>();
  const SpecializationMap &subst = resolvedImpl->arguments;
  auto monomorphic = monomorphicApplicationOf(*resolvedImpl, subst);
  if (failed(monomorphic)) {
    if (err) err() << "could not monomorphize claim: " << wanted;
    return Proven::refusal();
  }
  return proofAtResolution(resolvedImpl->impl, *monomorphic, subst, site,
                           builder, wanted, err);
}

Answer<ClaimType> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    const SelectionSite &site,
    OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // A proof is identified by the evidence its derive cites at the application
  // selection resolves the claim to, as rustc and GHC key evidence by the
  // normalized predicate. A claim spelled otherwise names a cast of that proof
  // to its own spelling (`respellProof`): a claim names the proof of exactly
  // its spelling (`verifyCitation`), and every spelling of one application
  // names one proof.
  Answer<ClaimType> atResolution = proveResolution(wanted, site, builder, err);
  if (!atResolution.isAnswer())
    return atResolution;
  return respellProof(*atResolution, wanted.getTraitApplication(), site,
                      builder);
}

Answer<ProvenPremise> ImplResolver::resolveEvidenceFor(
    ClaimType claim, const SelectionSite &site, OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  Answer<ClaimType> atResolution = proveResolution(claim, site, builder, err);
  if (!atResolution.isAnswer())
    return atResolution.stop<ProvenPremise>();
  return resolvePremise(claim, atResolution->getProof(), site, builder,
                        /*depth=*/0);
}

Answer<ClaimType> ImplResolver::proofAtResolution(
    ImplOp impl, TraitApplicationAttr app, const SpecializationMap &arguments,
    const SelectionSite &site, OpBuilder &builder, ClaimType wanted,
    llvm::function_ref<InFlightDiagnostic()> err) {
  using Proven = Answer<ClaimType>;
  ModuleOp scope = site.scope;
  // Selection's record of the application holds its proof once one stands.
  auto chosen = memo.chosen.find({scope, app});
  if (chosen != memo.chosen.end() && chosen->second.proof)
    return ClaimType::get(scope.getContext(), app, chosen->second.proof);

  // The evidence for each where entry at the arguments selection chose, in
  // order: the proof of an application entry at the application selection
  // resolves it to, and the resolution of an equality entry, whose sides
  // selection carries to one spelling. The trait's requirements are the
  // impl's to return, read at the proof's derive. Selection chose `impl` only
  // once these entries held through a chain that refuses an application it
  // meets again (the cycle guard of `resolveImplFor`), so proving them never
  // asks for `app` itself.
  SmallVector<ClaimType> entries = impl.getWhereClaimsAt(arguments);
  SmallVector<FlatSymbolRefAttr> subproofs;
  SmallVector<SmallVector<ResolutionStep>> equalitySteps;
  for (ClaimType entry : entries) {
    if (entry.isApplication()) {
      Answer<Type> resolvedEntry = resolveProjectionsIn(entry, site, builder);
      if (!resolvedEntry.isAnswer())
        return resolvedEntry.stop<ClaimType>();
      Proven subproof = resolveAndEnsureProofFor(
          cast<ClaimType>(*resolvedEntry), site, builder, err);
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
                     << wanted;
      return Proven::refusal();
    }
    equalitySteps.push_back(std::move(steps));
  }

  // A proof is identified by the evidence its derive cites: one standing over
  // these premises answers, and otherwise the proof is written.
  Proven proven = findProof(scope, impl, app, subproofs);
  if (!*proven)
    proven = writeProof(scope, impl, app, arguments, entries, subproofs,
                        equalitySteps, site, builder);
  if (proven.isAnswer())
    if (auto entry = memo.chosen.find({scope, app}); entry != memo.chosen.end())
      entry->second.proof = proven->getProof();
  return proven;
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
        step.arguments =
            resolved->getImpl().stateArguments(resolved->getArguments());
        for (ClaimType entry :
             resolved->getImpl().getWhereClaimsAt(resolved->getArguments())) {
          if (entry.isApplication()) {
            Answer<Type> resolvedEntry =
                resolveProjectionsIn(entry, site, builder);
            if (!resolvedEntry.isAnswer())
              return stop(resolvedEntry.isOverflow());
            Answer<ClaimType> proven = resolveAndEnsureProofFor(
                cast<ClaimType>(*resolvedEntry), site, builder, err);
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
  auto refusal = memo.refused.find({scope, app});
  if (refusal == memo.refused.end())
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
    auto it = memo.chosen.find({scope, selected.getTraitApplication()});
    if (it == memo.chosen.end())
      return Answer<ResolvedImpl>::refusal();
    return ResolvedImpl{it->second.impl, selected, it->second.arguments};
  };
  return ProjectionResolution::get(proj, select, /*err=*/nullptr).orFailure();
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
