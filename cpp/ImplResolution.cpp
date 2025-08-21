#include "ImplResolution.hpp"

namespace mlir::trait {

// forward declaration
static FailureOr<ImplOp> resolveImplFor(ClaimType wanted,
                                        ModuleOp module,
                                        ResolutionMemo &memo,
                                        const ImplGenerator &gen,
                                        PatternRewriter& rewriter);

static LogicalResult
assumptionsSatisfiableFor(ImplOp impl,
                          ClaimType concreteSelf, // concrete self claim, proven or not
                          ModuleOp module,
                          ResolutionMemo& memo,
                          const ImplGenerator& gen,
                          PatternRewriter &rewriter) {
  TraitApplicationAttr app = concreteSelf.getTraitApplication();

  // check the memo
  if (memo.knownSatisfiable.contains(app))
    return success();

  // cycle guard: A(app) -> ... -> A(app) means unsatisfiable
  if (!memo.visiting.insert(app).second)
    return failure();

  auto subst = impl.buildSubstitutionFor(concreteSelf);

  // specialize the impl's assumptions to our concrete claim
  for (ClaimType assume : impl.getAssumptionsAsClaimsWith(subst)) {
    // find an impl for the assumption
    auto subImpl = resolveImplFor(assume, module, memo, gen, rewriter);
    if (failed(subImpl)) {
      memo.visiting.erase(app);
      return failure();
    }

    // that impl's own assumptions must be satisfiable too
    if (failed(assumptionsSatisfiableFor(*subImpl, assume, module, memo, gen, rewriter))) {
      memo.visiting.erase(app);
      return failure();
    }
  }

  memo.visiting.erase(app);

  // record a positive result
  memo.knownSatisfiable.insert(app);

  return success();
}

// pick the unique ImplOp for the wanted claim. Coherence assumed; error on 0 or >1.
static FailureOr<ImplOp> resolveImplFor(
    ClaimType wanted,
    ModuleOp module,
    ResolutionMemo &memo,
    const ImplGenerator &gen,
    PatternRewriter &rewriter) {
  TraitApplicationAttr app = wanted.getTraitApplication();

  // first check the memo
  if (auto it = memo.chosen.find(app); it != memo.chosen.end())
    return it->second;

  // get the trait
  TraitOp trait = app.getTraitOrAbort(module, "resolveImplFor: cannot find trait");

  // allow the generator to generate the wanted impl
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // we don't care if it is successful or not
    (void)gen.generate(trait, wanted, rewriter);
  }

  // collect candidate impls whose self claim can be substituted with our wanted claim
  SmallVector<ImplOp> candidates;
  for (ImplOp impl : trait.getImpls()) {
    if (succeeded(substituteWith(impl.getSelfClaim(), wanted, module)))
      candidates.push_back(impl);
  }

  if (candidates.empty())
    return memo.chosen[app] =
      trait.emitError() << "no coherent impl found for " << wanted;

  // keep only candidates whose assumptions are satisfiable
  SmallVector<ImplOp> ok;
  for (ImplOp impl : candidates)
    if (succeeded(assumptionsSatisfiableFor(impl, wanted, module, memo, gen, rewriter)))
      ok.push_back(impl);

  if (ok.size() == 1)
    return memo.chosen[app] = ok.front();

  if (ok.empty())
    return memo.chosen[app] =
      trait.emitError() << "no impl with satisfiable assumptions for " << wanted;

  return memo.chosen[app] =
    trait.emitError() << "incoherent impls (multiple satisfiable) for " << wanted;
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
  // collect ImplGeneators from each dialect with the appropriate interface
  for (Dialect *dialect : module.getContext()->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<GenerateImplsInterface>()) {
      iface->populateImplGenerators(generators);
    }
  }
}

FailureOr<FlatSymbolRefAttr> ImplResolver::resolveAndEnsureProofFor(
    ClaimType wanted,
    PatternRewriter &rewriter) {
  if (!wanted.isMonomorphic())
    llvm::report_fatal_error("resolveAndEnsureProofFor called on polymorphic ClaimType");

  TraitApplicationAttr app = wanted.getTraitApplication();

  // first check the proof memo
  if (auto it = memo.proofMemo.find(app); it != memo.proofMemo.end())
    return it->second;

  MLIRContext* ctx = module.getContext();

  // resolve the unique impl for this concrete application
  auto implOr = resolveImplFor(wanted, module, memo.resolutionMemo, generators, rewriter);
  if (failed(implOr))
    return failure();
  ImplOp impl = *implOr;

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

  // first compute all subproof symbols (requirements & assumptions)
  SmallVector<Attribute> subproofSymbols;

  // ask the impl to build a substitution for the wanted claim 
  auto subst = impl.buildSubstitutionFor(wanted);

  // recursively collect proofs for impl obligations
  for (ClaimType ob : impl.getObligationsAsClaimsWith(subst)) {
    auto sym = resolveAndEnsureProofFor(ob, rewriter);
    if (failed(sym)) return failure();
    subproofSymbols.push_back(*sym);
  }

  // create a new proof op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  // generate a mangled name for the proof based on the wanted claim
  std::string proofName = impl.generateMangledName(wanted) + "_p";

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
