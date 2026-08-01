// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ImplResolution.hpp"
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/xxhash.h>
#include <cinttypes>
#include <cstdlib>

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

  auto subst =
      impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim, err);
  if (failed(subst)) return failure();

  return applySubstitutionToFixedPoint(subst->toTypeMap(), *binding);
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

AttrTypeReplacer ImplResolver::makeProvenClaimReplacer() const {
  MLIRContext *ctx = module.getContext();
  AttrTypeReplacer replacer;
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
    recordProof(app, sym);
    return sym;
  }

  // check for an existing proof in the module
  if (ProofOp proof = findExistingProofFor(module, impl, app)) {
    auto sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
    recordProof(app, sym);
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
// Reporting the recorded facts
//===----------------------------------------------------------------------===//

namespace {

/// What a refused application renders as, whichever arm refused it.
constexpr const char *refusalToken = "refused";

/// Collects the renderings of a resolver's facts, one line each, so that facts
/// held in pointer-keyed maps can be put in a determinate order before they are
/// digested.
class FactRendering {
public:
  /// Opens a line. The caller writes the fact and closes it with `end`.
  llvm::raw_ostream &begin() {
    pending.clear();
    return stream;
  }
  void end() { lines.push_back(pending); }

  /// The renderings in an order that does not depend on where the maps holding
  /// the facts happened to put them.
  ArrayRef<std::string> sorted() {
    llvm::sort(lines);
    return lines;
  }

private:
  SmallVector<std::string> lines;
  std::string pending;
  llvm::raw_string_ostream stream{pending};
};

/// How many recorded selections were of each kind, for the line that reports
/// the digest.
struct RecordedFactCounts {
  uint64_t selections = 0;
  uint64_t refusalsByArm[numRefutationArms] = {};
};

/// Renders everything `memo` holds, one fact per line.
void renderRecordedFacts(const ProofResolutionMemo &memo, FactRendering &facts,
                         RecordedFactCounts &counts) {
  const ResolutionMemo &resolution = memo.resolutionMemo;

  for (const auto &entry : resolution.chosen) {
    const ResolutionOutcome &outcome = entry.second;
    llvm::raw_ostream &os = facts.begin();
    os << "impl " << entry.first << " = ";
    if (outcome.isRefusal()) {
      ++counts.refusalsByArm[static_cast<unsigned>(outcome.getRefutationArm())];
      os << refusalToken;
    } else {
      ++counts.selections;
      os << outcome.getImpl().getSymName();
    }
    facts.end();
  }

  for (const auto &entry : resolution.assumptionsKnownSatisfiable) {
    ImplOp impl = entry.first;
    facts.begin() << "assumptions " << impl.getSymName() << " for "
                  << entry.second;
    facts.end();
  }

  for (const auto &entry : memo.proofMemo) {
    facts.begin() << "proof " << entry.first << " = " << entry.second;
    facts.end();
  }
}

} // namespace

uint64_t ImplResolver::getRecordedFactsDigest() const {
  FactRendering facts;
  RecordedFactCounts counts;
  renderRecordedFacts(memo, facts, counts);

  std::string rendered;
  for (const std::string &fact : facts.sorted()) {
    rendered += fact;
    rendered += '\n';
  }
  return llvm::xxh3_64bits(rendered);
}

void ImplResolver::reportRecordedFacts() const {
  const ResolutionMemo &resolution = memo.resolutionMemo;

  // Every application selection opened it has closed, so nothing is part-way
  // resolved here and the facts below are the whole of what this resolver knows.
  assert(resolution.visiting.empty() &&
         "impl selection must not be part-way through an application when its "
         "facts are read");

  FactRendering facts;
  RecordedFactCounts counts;
  renderRecordedFacts(memo, facts, counts);

  std::string rendered;
  for (const std::string &fact : facts.sorted()) {
    llvm::errs() << stageRecordFactPrefix << " " << fact << "\n";
    rendered += fact;
    rendered += '\n';
  }

  llvm::errs() << stageRecordDigestPrefix
               << llvm::format(" value=0x%016" PRIx64,
                               llvm::xxh3_64bits(rendered))
               << " selected-impls=" << counts.selections
               << " refusals-no-candidate="
               << counts.refusalsByArm[static_cast<unsigned>(
                      RefutationArm::NoSatisfiableCandidate)]
               << " refusals-ambiguous="
               << counts.refusalsByArm[static_cast<unsigned>(
                      RefutationArm::MultipleSatisfiableCandidates)]
               << " assumption-facts="
               << resolution.assumptionsKnownSatisfiable.size()
               << " proofs=" << memo.proofMemo.size() << "\n";
}

//===----------------------------------------------------------------------===//
// Forgetting what a later resolution can answer differently
//===----------------------------------------------------------------------===//

ImplResolver::RefusalCounts ImplResolver::forgetRetriableRefusals() {
  DenseMap<TraitApplicationAttr, ResolutionOutcome> &chosen =
      memo.resolutionMemo.chosen;
  RefusalCounts counts;

  // What became of the refusals the last call dropped. An application selection
  // has not been asked about since is neither, and is the remainder.
  for (TraitApplicationAttr app : lastForgotten) {
    auto it = chosen.find(app);
    if (it == chosen.end())
      continue;
    if (it->second.isRefusal())
      ++counts.reEarned;
    else
      ++counts.overturned;
  }

  lastForgotten.clear();
  for (const auto &entry : chosen) {
    if (!entry.second.isRefusal())
      continue;
    if (entry.second.getRefutationArm() ==
        RefutationArm::NoSatisfiableCandidate) {
      lastForgotten.push_back(entry.first);
      ++counts.forgotten;
    } else {
      ++counts.kept;
    }
  }
  // The drops move no record epoch, for the same reason writing the refusal
  // did not: what is erased here is a question impl selection will have to
  // answer again, never an answer a read of the record was given.
  for (TraitApplicationAttr app : lastForgotten)
    chosen.erase(app);
  return counts;
}

bool ImplResolver::recordsOnlyRealizedProofs() const {
  SymbolTable symbols(module);
  for (const auto &entry : memo.proofMemo) {
    Operation *defining = symbols.lookup(entry.second.getValue());
    if (!defining)
      return false;
    // The two shapes a recorded proof takes: a trait.proof, or the impl that
    // proves an application with no obligations of its own. Which application
    // the named op spells is not compared, because the memo is keyed by the
    // spelling selection recorded and the commit respells the module's copy of
    // it -- the two are the same application under different spellings from the
    // first commit that reaches the proof onwards.
    if (isa<ProofOp>(defining))
      continue;
    auto impl = dyn_cast<ImplOp>(defining);
    if (!impl || !impl.isUnconditional())
      return false;
  }
  return true;
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
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "impl generation is frozen for " << span
         << ", but impl selection demanded an impl of @" << trait.getSymName()
         << " for " << wanted;
  llvm::report_fatal_error(Twine(message));
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

  auto subst = impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim);
  if (failed(subst)) return failure();

  countReadOnlyResolverServe();
  return applySubstitutionToFixedPoint(subst->toTypeMap(), *binding);
}

Type ReadOnlyImplResolver::resolveProjectionsIn(Type ty) const {
  AttrTypeReplacer replacer;
  replacer.addReplacement([this](Type t) -> std::optional<Type> {
    auto proj = dyn_cast<ProjectionType>(t);
    if (!proj || isPolymorphicType(proj)) return std::nullopt;
    auto resolved = resolveProjectionType(proj);
    if (succeeded(resolved))
      return *resolved;
    // Selection settles a projection only for an application some round put to
    // it, so a spelling nothing has asked about yet has no recorded fact to
    // read. One exactly one impl in the module binds is one selection would
    // settle the same way, so the module answers it here; where no impl or
    // several bind it, the lookup declines and says which, and the projection
    // stays spelled as written for the step that can make selection answer it.
    Type byLookup = resolveGroundProjectionsByLookup(
        Type(proj), resolver.module, DemandOrigin::RecordedFactRead);
    if (byLookup == Type(proj))
      return std::nullopt;
    return byLookup;
  });
  return replacer.replace(ty);
}

FailureOr<FlatSymbolRefAttr>
ReadOnlyImplResolver::getRecordedProofFor(ClaimType claim) const {
  DemandFrame frame{Type(claim)};

  auto resolvedImpl = getRecordedImplFor(claim);
  if (failed(resolvedImpl)) return failure();

  auto subst =
      resolvedImpl->impl.buildSubstitutionForSelfClaim(resolvedImpl->selectedClaim);
  if (failed(subst)) return failure();

  auto monomorphic = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(
      subst->toTypeMap(), resolvedImpl->selectedClaim));
  if (!monomorphic || !monomorphic.isMonomorphic())
    return failure();

  auto proof = getRecordedProof(monomorphic.getTraitApplication());
  if (!proof) return failure();
  countReadOnlyResolverServe();
  return *proof;
}

} // end mlir::trait
