#include "ImplResolution.hpp"

namespace mlir::trait {

// pick the unique ImplOp for app. Coherence assumed; error on 0 or >1.
static FailureOr<ImplOp> resolveImplFor(TraitApplicationAttr app, ModuleOp module) {
  // find the trait op
  TraitOp trait = SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, app.getTrait());
  if (!trait)
    return module.emitError() << "unknown trait symbol '" << app.getTrait()
                              << "'";

  // collect users of the trait
  auto uses = mlir::SymbolTable::getSymbolUses(trait, module);
  if (!uses)
    return trait.emitError()
           << "no impls found for trait '" << trait.getSymName() << "'";

  ImplOp symbolicImpl = nullptr;

  // search through all ImplOps with use this trait
  for (const auto& use : *uses) {
    auto impl = dyn_cast<ImplOp>(use.getUser());
    if (!impl) continue;

    // first check for an exact match on the impl's application
    if (impl.getSelfApplication() == app)
      return impl;

    // XXX TODO What do we do if ImplOp has assumptions?

    // otherwise, check if our wanted claim can unify with the
    // impl's claim, indicating a "symbolic" match
    ClaimType wanted = ClaimType::get(module.getContext(), app);
    if (succeeded(unifyTypes(impl.getSelfClaim(), wanted, module))) {
      // if there is more than one symbolic match, that's ambiguous, and an error
      if (symbolicImpl)
        return trait.emitError()
               << "incoherent impls found for " << app;
      symbolicImpl = impl;
    }
  }

  if (!symbolicImpl)
    return trait.emitError()
           << "no coherent impl found for " << app;

  return symbolicImpl;
}

// find an existing trait.proof that *explicitly* proves impl by name
// and proves the same application app
static ProofOp findExistingProofFor(ModuleOp module, ImplOp impl, TraitApplicationAttr app) {
  auto uses = mlir::SymbolTable::getSymbolUses(impl, module);
  if (!uses) return nullptr;

  // traverse all ProofOps that use of impl 
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

FailureOr<FlatSymbolRefAttr>
resolveAndEnsureProofFor(TraitApplicationAttr app,
                         ModuleOp module,
                         llvm::DenseMap<TraitApplicationAttr, FlatSymbolRefAttr> &memo,
                         PatternRewriter &rewriter) {
  // first check the memo
  if (auto it = memo.find(app); it != memo.end())
    return it->second;

  MLIRContext* ctx = module.getContext();

  // resolve the unique impl for this concrete application
  auto implOr = resolveImplFor(app, module);
  if (failed(implOr))
    return failure();
  ImplOp impl = *implOr;

  // check for a self-proof
  if (impl.isSelfProof()) {
    auto sym = FlatSymbolRefAttr::get(ctx, impl.getSymName());
    memo[app] = sym;
    return sym;
  }

  // check for an existing proof in the module
  if (ProofOp proof = findExistingProofFor(module, impl, app)) {
    auto sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
    memo[app] = sym;
    return sym;
  }

  // first compute all subproof symbols (requirements & assumptions)
  SmallVector<Attribute> subproofSymbols;

  // collect applications for trait requirements
  for (ClaimType req : impl.getRequirementsAsClaims()) {
    auto sym = resolveAndEnsureProofFor(req.getTraitApplication(), module, memo, rewriter);
    if (failed(sym)) return failure();
    subproofSymbols.push_back(*sym);
  }

  // XXX TODO collect applications for impl assumptions
  if (!impl.getAssumptions().getApplications().empty())
    llvm_unreachable("resolveAndEnsureProofFor: unimplemented");

  // create a new proof op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  StringAttr name = rewriter.getStringAttr(impl.getSymName().str() + "_p");
  ProofOp proof = rewriter.create<ProofOp>(
    rewriter.getUnknownLoc(),
    name,
    FlatSymbolRefAttr::get(ctx, impl.getSymName()),
    app,
    ArrayAttr::get(ctx, subproofSymbols)
  );

  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(ctx, proof.getSymNameAttr());
  memo[app] = sym;
  return sym;
}

} // end mlir::trait
