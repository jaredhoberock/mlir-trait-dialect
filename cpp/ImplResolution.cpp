#include "ImplResolution.hpp"
#include <llvm/ADT/ScopeExit.h>

namespace mlir::trait {

// forward declaration
static FailureOr<ImplOp> resolveImplFor(ClaimType wanted,
                                        ModuleOp module,
                                        ResolutionMemo &memo,
                                        const ImplGenerator &gen,
                                        PatternRewriter& rewriter,
                                        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

static LogicalResult
assumptionsSatisfiableFor(ImplOp impl,
                          ClaimType concreteSelf, // concrete self claim, proven or not
                          ModuleOp module,
                          ResolutionMemo& memo,
                          const ImplGenerator& gen,
                          PatternRewriter &rewriter) {
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // consult the per-(impl,claim) satisfiability memo
  auto key = std::make_pair(impl, app);
  if (memo.assumptionsKnownSatisfiable.contains(key))
    return success();

  // cycle guard: A(app) -> ... -> A(app) means unsatisfiable
  if (!memo.visiting.insert(app).second)
    return failure();
  auto guard = llvm::make_scope_exit([&]{ memo.visiting.erase(app); });

  // specialize the impl's assumptions to our concrete claim
  auto assumptions = impl.specializeAssumptionsAsClaimsFor(concreteSelf);
  if (failed(assumptions))
    return failure();

  for (ClaimType assume : *assumptions) {
    // find an impl for the assumption
    auto subImpl = resolveImplFor(assume, module, memo, gen, rewriter);
    if (failed(subImpl))
      return failure();

    // that impl's own assumptions must be satisfiable too
    if (failed(assumptionsSatisfiableFor(*subImpl, assume, module, memo, gen, rewriter)))
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

      std::string header;
      llvm::raw_string_ostream os(header);
      os << "unsatisfiable candidate";

      diag.attachNote(impl.getLoc()) << os.str();
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

    std::string header;
    llvm::raw_string_ostream os(header);
    os << "candidate";

    diag.attachNote(impl.getLoc()) << os.str();
  }

  return diag;
}


// pick the unique ImplOp for the wanted claim. Coherence assumed; error on 0 or >1.
static FailureOr<ImplOp> resolveImplFor(
    ClaimType wanted,
    ModuleOp module,
    ResolutionMemo &memo,
    const ImplGenerator &gen,
    PatternRewriter &rewriter,
    llvm::function_ref<InFlightDiagnostic()> err) {
  TraitApplicationAttr app = wanted.getTraitApplication();

  // first check the memo
  if (auto it = memo.chosen.find(app); it != memo.chosen.end())
    return it->second;

  // get the trait
  TraitOp trait = app.getTraitOrAbort(module, "resolveImplFor: cannot find trait");

  // collect candidates for wanted from the trait and
  // partition them into good/bad by satisfiable assumptions
  SmallVector<ImplOp> good, bad;
  for (ImplOp impl : trait.getCandidateImplsFor(wanted)) {
    if (succeeded(assumptionsSatisfiableFor(impl, wanted, module, memo, gen, rewriter)))
      good.push_back(impl);
    else
      bad.push_back(impl);
  }

  // if there aren't any good candidates, try to generate one
  if (good.empty()) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    if (auto impl = gen.generateImpl(trait, wanted, rewriter); succeeded(impl)) {
      if (succeeded(assumptionsSatisfiableFor(*impl, wanted, module, memo, gen, rewriter)))
        good.push_back(*impl);
      else
        bad.push_back(*impl);
    }
  }

  // if exactly one good candidate exists, return it
  if (good.size() == 1)
    return memo.chosen[app] = good.front();

  // otherwise, diagnose resolution failure
  return memo.chosen[app] = diagnoseImplResolutionFailure(trait, wanted, good, bad, err);
}

// find an existing trait.proof that *explicitly* proves impl by name
// and proves the same application app
static ProofOp findExistingProofFor(ModuleOp module, ImplOp impl, TraitApplicationAttr app) {
  auto uses = mlir::SymbolTable::getSymbolUses(impl, module);
  if (!uses) return nullptr;

  // traverse all ProofOps that use impl 
  for (const SymbolTable::SymbolUse& use : *uses) {
    if (auto proof = dyn_cast<ProofOp>(use.getUser())) {
      // ensure both:
      // 1. proof refers to exactly this impl by name, and
      // 2. the trait application matches
      if (proof.getImplName() == impl.getSymName() &&
          proof.getTraitApplication() == app)
        return proof;
    }
  }
  return nullptr;
}

ImplResolver::ImplResolver(ModuleOp m) : module(m) {
  // collect ImplGenerators from each dialect with the appropriate interface
  for (Dialect *dialect : module.getContext()->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<GenerateImplsInterface>()) {
      iface->populateImplGenerators(generators);
    }
  }
}

FailureOr<FlatSymbolRefAttr> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    PatternRewriter &rewriter,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // resolve an impl for wanted first
  auto implOr = resolveImplFor(wanted, module, memo.resolutionMemo, generators, rewriter, err);
  if (failed(implOr)) return failure();
  ImplOp impl = *implOr;

  // build a PolyType -> Type map for this impl's self claim against wanted
  auto subst = impl.buildSubstitutionForSelfClaim(wanted, err);
  if (failed(subst)) return failure();

  // monomorphize the wanted claim with that substitution
  ClaimType monomorphicWanted = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(*subst, wanted));
  if (!monomorphicWanted || !monomorphicWanted.isMonomorphic()) {
    if (err) err() << "could not monomorphize claim: " << wanted;
    return failure();
  }

  TraitApplicationAttr app = monomorphicWanted.getTraitApplication();

  // check the proof memo for this monomorphic app
  if (auto it = memo.proofMemo.find(app); it != memo.proofMemo.end())
    return it->second;

  MLIRContext *ctx = module.getContext();

  // check for a self-proof
  if (impl.isSelfProof()) {
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

  // specialize all obligations against wanted
  auto obligations = impl.specializeObligationsAsClaimsFor(wanted, err);
  if (failed(obligations)) return failure();

  // recursively prove monomorphic obligations
  SmallVector<Attribute> subproofSymbols;
  for (ClaimType ob : *obligations) {
    auto sym = resolveAndEnsureProofFor(ob, rewriter, err);
    if (failed(sym)) return failure();
    subproofSymbols.push_back(*sym);
  }

  // create the proof and memoize by the monomorphic app 
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  // generate a mangled name for the proof based on the monomorphic wanted claim
  std::string proofName = impl.generateMangledName(monomorphicWanted) + "_p";

  ProofOp proof = rewriter.create<ProofOp>(
    rewriter.getUnknownLoc(),
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
