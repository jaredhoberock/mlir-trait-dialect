// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The judgments shared across the trait dialect's equality-evidence checkers:
// projection resolution with its obligation discharge, the ground-congruence
// entailment a witness composition and a proven coerce both appeal to, and the
// pending unification a marked coerce stands in.

#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/SymbolTable.h>
#include <functional>
#include <optional>

using namespace mlir;
using namespace mlir::trait;

// Rewrite `ty` modulo the cited equality premises: the premises relate the types
// they mention, and every member of one class rewrites to the one member the
// class stands for. When a projection-headed impl self-application cannot be
// aligned by structural matching, verification matches modulo these equalities;
// both sides of every such comparison are rewritten here, so the verdict does
// not turn on how any premise was oriented.
//
// One rewrite is the whole reading. The member a class is headed by spells no
// more projections and no more types than any other member, so it cannot have
// another member of its own class standing inside it; rewriting a member to it
// therefore mints nothing this reading would rewrite again. Two premises of
// opposite orientation, and a premise whose one endpoint stands inside the other
// (!S = tuple<!S>), each reach the member their class is headed by in that one
// rewrite rather than growing the spelling.
static Type applyEqualityPremises(Type ty,
                                  ArrayRef<TypeEqualityAttr> premises) {
  if (premises.empty())
    return ty;
  TypeEquivalence classes;
  for (TypeEqualityAttr eq : premises)
    classes.assumeEqual(eq.getLhs(), eq.getRhs());
  return applySubstitutionOnce(classes.substitutionToCanonicalMembers(), ty);
}

// verifyProjectionResolutionAtUse and verifyProjectionResolutionAtImpl share
// the static core below; their contract -- the binding and the obligation
// discharge -- is stated in full at their declarations in TraitOps.hpp.

// Specializes `impl`'s own application assumptions for `selfClaim` through
// `subst` -- the head-match substitution verification already built -- rather
// than rebuilding one module-grade. Keeping the same rigid substitution here as
// at the head match is what makes the assumptions the discharge check inspects
// agree with the head the match produced.
static SmallVector<ClaimType> specializeAssumptionsThroughSubst(
    ImplOp impl, const SpecializationMap &subst) {
  return llvm::map_to_vector(impl.getAssumptionsAsClaims(), [&](ClaimType a) {
    return cast<ClaimType>(instantiate(a, subst));
  });
}

// What an obligation discharge reads but never
// varies as it recurses: the module the citations resolve in, the equality
// premises the comparisons run modulo, the citing impl's own where-clause cover
// and the discharge citations the two arms consult, and the diagnostic sink.
// The verification core owns one and threads it through the recursion; only the
// obligation under discharge and the active-citation guard vary per call.
struct ObligationDischargeContext {
  ModuleOp module;
  ArrayRef<TypeEqualityAttr> premises;
  ArrayRef<TraitApplicationAttr> obligationPremises;
  ArrayRef<WitnessAttr> dischargeWitnesses;
  llvm::function_ref<InFlightDiagnostic()> err;
  // Whether a premise spelled as a ground projection's resolution discharges an
  // obligation spelled as the projection (and the reverse). Set only by the
  // use-site entry, which already resolves the actual side by module lookup; the
  // impl-verification entry keeps its rigid, module-free comparison so an impl's
  // verdict cannot turn on the unrelated impls the module carries.
  bool resolveGround;
};

// Whether two ground applications, already read modulo the context's equality
// premises, denote the same claim: their spellings match, or -- at the use-site
// entry -- their ground normal forms do. This is the equivalence the clone rule
// and the commit sweep apply, local to the witness's own operands and citation;
// a non-converging projection chain refuses.
static bool groundApplicationsMatch(const ObligationDischargeContext &ctx,
                                    Type have, Type want) {
  if (have == want)
    return true;
  if (!ctx.resolveGround)
    return false;
  // XXX TODO a projection a declaration spells must be over its own self
  // application, a where-clause application, a trait requirement or a declared
  // witness (Rust's projection well-formedness rule), so every projection has
  // evidence at a known index and this module read deletes with LookupScope and
  // the verifier DemandOrigins.
  auto haveGround = resolveProjectionsByLookup(
      have, ctx.module, DemandOrigin::ProofVerification, LookupScope::Ground,
      ctx.err);
  auto wantGround = resolveProjectionsByLookup(
      want, ctx.module, DemandOrigin::ProofVerification, LookupScope::Ground,
      ctx.err);
  return succeeded(haveGround) && succeeded(wantGround) &&
         *haveGround == *wantGround;
}

// Whether `want` -- a ground application obligation, already read modulo the
// context's equality premises -- is discharged. Arm (i): a hypothetical cover
// among the citing impl's own where-clause premises. Arm (ii): a discharge
// citation whose spelled application is `want` and whose named impl,
// instantiated ONLY over its own generics for that application, has each of its
// own assumptions discharged in turn.
//
// Termination: arm (ii) recurses only into a citation whose application is not
// already on the active `inProgress` stack; the declared citation list is
// finite, so each recursion pushes a distinct application and the depth is
// bounded by the list length. A citation that would re-enter an application
// under resolution is a cycle and discharges nothing along that path.
static bool dischargeApplicationObligation(
    const ObligationDischargeContext &ctx, Type want,
    SmallVectorImpl<TraitApplicationAttr> &inProgress) {
  ModuleOp module = ctx.module;
  MLIRContext *mlirCtx = module.getContext();

  // Arm (i): the citing impl's own where clause covers the obligation.
  for (TraitApplicationAttr premiseApp : ctx.obligationPremises) {
    ClaimType premiseClaim = ClaimType::get(mlirCtx, premiseApp);
    Type have = applyEqualityPremises(Type(premiseClaim), ctx.premises);
    if (groundApplicationsMatch(ctx, have, want))
      return true;
  }

  // Arm (ii): a declared discharge citation names the obligation and an impl
  // that supplies it.
  for (WitnessAttr citation : ctx.dischargeWitnesses) {
    ClaimType citedApp = ClaimType::get(mlirCtx, citation.getApplication());
    Type cited = applyEqualityPremises(Type(citedApp), ctx.premises);
    if (!groundApplicationsMatch(ctx, cited, want))
      continue;
    if (llvm::is_contained(inProgress, citation.getApplication()))
      continue; // cycle: this path grounds nothing

    auto dischargerOp = SymbolTable::lookupNearestSymbolFrom<ImplOp>(
        module, citation.getImplRef());
    if (!dischargerOp)
      continue;

    // The named impl must genuinely supply the application: read its own
    // parameters off the application and require its header rebuilt at them to
    // be that application (rigid actual side, no established context).
    ClaimType appClaim = ClaimType::get(mlirCtx, citation.getApplication());
    auto subst = dischargerOp.buildSubstitutionForSelfClaim(appClaim);
    if (failed(subst))
      continue;

    // Its own assumptions, specialized through that same substitution, must each
    // discharge in turn.
    inProgress.push_back(citation.getApplication());
    bool allDischarged = true;
    for (ClaimType assumption :
         specializeAssumptionsThroughSubst(dischargerOp, *subst)) {
      Type subWant =
          applyEqualityPremises(Type(assumption.asUnproven()), ctx.premises);
      if (!dischargeApplicationObligation(ctx, subWant, inProgress)) {
        allDischarged = false;
        break;
      }
    }
    inProgress.pop_back();
    if (allDischarged)
      return true;
  }

  return false;
}

// The binding check and obligation discharge, written once. On success it
// returns the head-match substitution; `rigidHeadMatch` selects the head-match
// mode. `witness` must be equality-armed.
static FailureOr<SpecializationMap> verifyProjectionResolutionCore(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    ArrayRef<WitnessAttr> dischargeWitnesses,
    bool rigidHeadMatch,
    llvm::function_ref<InFlightDiagnostic()> err,
    TypeEqualityAttr currentEquality = {}) {
  assert(isa<TypeEqualityAttr>(witness.getPredicate()) &&
         "projection-resolution verification requires an equality-armed witness");
  Type projection = witness.getProjection();
  Type resolved = witness.getResolved();
  FlatSymbolRefAttr citedImpl = witness.getImplRef();

  auto projectionTy = dyn_cast<ProjectionType>(projection);
  if (!projectionTy) {
    if (err) err() << "a projection-resolution witness must name a projection, found "
                   << projection;
    return failure();
  }

  auto implOp =
      SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, citedImpl);
  if (!implOp) {
    if (err) err() << "cannot find trait.impl '" << citedImpl << "' cited by the witness";
    return failure();
  }

  // Head match the cited impl against the projection's application. The impl-
  // verification entry passes rigidHeadMatch: it instantiates only the cited impl's own
  // generics against a null module, so a projection spelled in the projection's
  // application stays rigid and is never resolved by a module-visible impl --
  // an impl's verdict cannot then turn on the unrelated impls the module carries.
  // The use-site entry leaves it clear and resolves the actual side's ground
  // projections by module lookup.
  ClaimType selfClaim =
      ClaimType::get(module.getContext(), projectionTy.getTraitApplication());
  // XXX TODO a projection a declaration spells must be over its own self
  // application, a where-clause application, a trait requirement or a declared
  // witness (Rust's projection well-formedness rule), so every projection has
  // evidence at a known index and this module read deletes with LookupScope and
  // the verifier DemandOrigins.
  GroundProjectionLookup byGroundLookup(module, DemandOrigin::ProofVerification);
  auto subst = implOp.buildSubstitutionForSelfClaim(
      selfClaim, rigidHeadMatch ? Normalizer() : Normalizer(byGroundLookup),
      err);
  if (failed(subst))
    return failure();

  auto bound = implOp.specializeAssociatedTypeBinding(
      projectionTy.getAssocName().getValue(), projectionTy.getAssocTypeArgs());
  if (failed(bound)) {
    if (err) err() << "impl '" << citedImpl
                   << "' does not bind associated type '"
                   << projectionTy.getAssocName().getValue() << "'";
    return failure();
  }
  Type actual = subst->apply(*bound);

  // Proof-blind exact comparison. When a projection-headed impl self-application
  // cannot be aligned by structural matching, the comparison runs modulo the
  // cited equality premises, applied to both the impl's binding and the certified
  // resolution before comparison.
  actual = applyEqualityPremises(actual, premises);
  if (actual != applyEqualityPremises(resolved, premises)) {
    if (err) err() << "impl '" << citedImpl << "' binds the projection to "
                   << actual << ", not the certified resolution " << resolved;
    return failure();
  }

  // The op's current endpoints may be a substitution instance of the witness's
  // stored ones -- a clone specializes the stored projection and resolved into
  // its own equality. The assumptions to discharge are the stored impl's
  // assumptions carried to that instance, so the premises a clone supplies at
  // its own spelling match. Without a current equality the stored endpoints
  // stand in and the instance substitution is the identity.
  SpecializationMap instanceSubst;
  if (currentEquality) {
    Type stored = TupleType::get(
        module.getContext(), {witness.getProjection(), witness.getResolved()});
    Type current = TupleType::get(
        module.getContext(),
        {currentEquality.getLhs(), currentEquality.getRhs()});
    auto match = matchDeclaration(getTypeParametersIn(stored), stored, current,
                                  /*normalize=*/Normalizer(), /*err=*/nullptr);
    if (succeeded(match))
      instanceSubst = *match;
  }

  // Obligation-discharge check. The cited impl's own assumptions -- specialized
  // through the same rigid head-match substitution, then carried to the op's
  // current endpoints -- must each be discharged, proof-stripped and modulo the
  // cited equality premises, by a hypothetical cover (arm i) or a declared
  // discharge citation (arm ii). The impl's trait requirements are deliberately
  // not reached here (they may quantify over GAT variables with no ground
  // instance at the witness).
  ObligationDischargeContext dischargeCtx{module,
                                          premises,
                                          obligationPremises,
                                          dischargeWitnesses,
                                          err,
                                          /*resolveGround=*/!rigidHeadMatch};
  for (ClaimType assumption :
       specializeAssumptionsThroughSubst(implOp, *subst)) {
    Type want = Type(assumption.asUnproven());
    want = instantiate(want, instanceSubst);
    // At the use-site entry, read the obligation modulo the module's ground
    // impls, so an assumption spelling a ground projection is compared as its
    // resolution -- a non-converging chain refuses.
    if (dischargeCtx.resolveGround) {
      // XXX TODO a projection a declaration spells must be over its own self
      // application, a where-clause application, a trait requirement or a
      // declared witness (Rust's projection well-formedness rule), so every
      // projection has evidence at a known index and this module read deletes
      // with LookupScope and the verifier DemandOrigins.
      auto wantGround = resolveProjectionsByLookup(
          want, module, DemandOrigin::ProofVerification, LookupScope::Ground,
          err);
      if (failed(wantGround))
        return failure();
      want = *wantGround;
    }
    want = applyEqualityPremises(want, premises);
    SmallVector<TraitApplicationAttr> inProgress;
    if (!dischargeApplicationObligation(dischargeCtx, want, inProgress)) {
      if (err) err() << "cited impl '" << citedImpl
                     << "' has an undischarged assumption " << assumption
                     << "; the witness premises do not supply it";
      return failure();
    }
  }
  return *subst;
}

LogicalResult mlir::trait::verifyProjectionResolutionAtUse(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    llvm::function_ref<InFlightDiagnostic()> err,
    TypeEqualityAttr currentEquality) {
  if (failed(verifyProjectionResolutionCore(module, witness, premises,
                                            obligationPremises,
                                            /*dischargeWitnesses=*/{},
                                            /*rigidHeadMatch=*/false, err,
                                            currentEquality)))
    return failure();
  return success();
}

FailureOr<SpecializationMap> mlir::trait::verifyProjectionResolutionAtImpl(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    ArrayRef<WitnessAttr> dischargeWitnesses,
    llvm::function_ref<InFlightDiagnostic()> err) {
  return verifyProjectionResolutionCore(module, witness, premises,
                                        obligationPremises, dischargeWitnesses,
                                        /*rigidHeadMatch=*/true, err);
}

// A distinct sentinel type per child position. A shell is only ever compared
// against another shell and children are compared separately, so a sentinel
// coinciding with a real leaf type is harmless: it merely marks that a child
// occupied that position.
static Type positionPlaceholder(MLIRContext *ctx, unsigned position) {
  return IntegerType::get(ctx, position + 1);
}

TermShape mlir::trait::decomposeTerm(Type t) {
  TermShape s;
  MLIRContext *ctx = t.getContext();
  if (auto claim = dyn_cast<ClaimType>(t)) {
    if (auto eq = claim.getEqualityAttr()) {
      s.key = StringAttr::get(ctx, "trait.claim.eq");
      s.children.push_back(eq.getLhs());
      s.children.push_back(eq.getRhs());
      return s;
    }
    // Application claims are compared modulo the proof, so the key ignores it.
    auto app = claim.getTraitApplication();
    s.key = ArrayAttr::get(
        ctx, {StringAttr::get(ctx, "trait.claim.app"), app.getTraitName()});
    for (Type a : app.getTypeArgs())
      s.children.push_back(a);
    return s;
  }
  if (auto proj = dyn_cast<ProjectionType>(t)) {
    auto app = proj.getTraitApplication();
    s.key = ArrayAttr::get(
        ctx, {StringAttr::get(ctx, "trait.proj"), app.getTraitName(),
              proj.getAssocName(),
              IntegerAttr::get(IntegerType::get(ctx, 64),
                               (int64_t)proj.getAssocTypeArgs().size())});
    for (Type a : app.getTypeArgs())
      s.children.push_back(a);
    for (Type a : proj.getAssocTypeArgs())
      s.children.push_back(a);
    return s;
  }

  SmallVector<Attribute> subAttrs;
  SmallVector<Type> subTypes;
  t.walkImmediateSubElements([&](Attribute a) { subAttrs.push_back(a); },
                             [&](Type ty) { subTypes.push_back(ty); });
  if (subTypes.empty()) {
    s.key = TypeAttr::get(t);
    return s;
  }
  SmallVector<Type> placeholders;
  for (unsigned i = 0, n = subTypes.size(); i < n; ++i)
    placeholders.push_back(positionPlaceholder(ctx, i));
  // A partial constructor declines the placeholder arguments -- its inference
  // fails on them, as a weak product with no result does -- and returns a null
  // shell. Such a type is keyed atomically: its own TypeAttr, no children
  // enumerated, exactly as a leaf is. Congruence and the position-paired
  // proof-swap walk both read children from here, so neither descends past this
  // constructor's shell. Completeness across it is deliberately forgone, not
  // lost by accident: a coerce that needs the crossing refuses with the ordinary
  // not-equal diagnostic rather than crashing on the null shell.
  Type shell = t.replaceImmediateSubElements(subAttrs, placeholders);
  if (!shell) {
    s.key = TypeAttr::get(t);
    return s;
  }
  s.key = TypeAttr::get(shell);
  s.children = std::move(subTypes);
  return s;
}

namespace {

// Ground congruence closure over the subterm DAG of a coerce's endpoints and
// its cited equalities. It seeds the classes with the equalities, then closes
// under congruence: two terms with the same constructor and pairwise equal
// children are united. It only unites -- it never decomposes, so
// f(a) = f(b) is not read backwards to a = b at projection heads or anywhere
// else. It also closes across normalizing type constructors: a composite is
// united with the normal form its own constructor yields when a united class
// member is substituted into it, so an equality a constructor establishes by
// normalizing its arguments is not missed. Child enumeration and constructor
// identity both come from decomposeTerm, which reads the type-bearing trait
// attributes directly rather than through a generic walk.
class GroundCongruence {
public:
  // Seed an equality between two endpoints (and intern their subterms).
  void seed(Type a, Type b) { classes.unite(intern(a), intern(b)); }

  // Intern a type and all its subterms; returns its term id.
  //
  // The classes hand out the ids, and a term's constructor key and children sit
  // at its own id here, so an id already carrying a key is one already
  // decomposed.
  unsigned intern(Type t) {
    unsigned id = classes.intern(t);
    if (id < ctorKey.size())
      return id;
    ctorKey.resize(id + 1);
    children.resize(id + 1);

    TermShape shape = decomposeTerm(t);
    ctorKey[id] = shape.key;
    SmallVector<unsigned> childIds;
    for (Type c : shape.children)
      childIds.push_back(intern(c));
    children[id] = std::move(childIds);
    return id;
  }

  // Close under congruence and constructor normalization to a fixed point.
  void close() {
    // A backstop for the rebuild's termination guarantee: with the
    // free-application filter in place the rebuild mints only normal forms, a
    // finite set, so the DAG stays far under this bound. A future constructor
    // that normalized without a fixed point could mint without bound; the
    // assert below then aborts a build that compiles asserts rather than
    // looping forever. It is generous and never bears on a verdict.
    const size_t mintCeiling = classes.size() * 8 + 256;
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0, n = classes.size(); i != n; ++i)
        for (unsigned j = i + 1; j != n; ++j) {
          if (classes.findCanonical(i) == classes.findCanonical(j))
            continue;
          if (ctorKey[i] != ctorKey[j] ||
              children[i].size() != children[j].size())
            continue;
          bool allEqual = true;
          for (auto [ci, cj] : llvm::zip(children[i], children[j]))
            if (classes.findCanonical(ci) != classes.findCanonical(cj)) {
              allEqual = false;
              break;
            }
          if (allEqual) {
            classes.unite(i, j);
            changed = true;
          }
        }
      if (rebuildNormalizedParents(mintCeiling))
        changed = true;
    }
  }

  bool equal(Type a, Type b) {
    return classes.findCanonical(intern(a)) == classes.findCanonical(intern(b));
  }

private:
  // Extend the closure across type constructors that normalize their arguments
  // when a type is built. Each parent a type constructor built is rebuilt
  // through that same constructor with a united class member substituted for one
  // child; a normalizing constructor folds the rebuilt form to its normal form,
  // and uniting that form with the parent adds only what congruence and the
  // constructor's own definitional law already entail.
  //
  // The invariant this depends on: a type constructor may normalize purely as a
  // context-free, deterministic function of its arguments -- the rebuilt object
  // IS the normal form the constructor names. An identification that turns on
  // facts outside the arguments must never enter construction; it belongs to the
  // surrounding environment, and this rule would otherwise import it as if a
  // constructor had settled it.
  //
  // The invariant behind the filter: a rebuild that merely re-applies the
  // constructor -- same key, children exactly the substituted list -- is
  // dropped. Such free applications state no equality congruence does not already
  // decide over the existing terms, and minting them has no fixed point over a
  // cyclic cited equality: the closure would build ever-larger terms and never
  // terminate.
  bool rebuildNormalizedParents([[maybe_unused]] size_t mintCeiling) {
    bool changed = false;
    // Terms minted below join the next pass, so the parent set rebuilt this pass
    // is fixed and the loop bounds stay valid as the classes grow.
    unsigned n = classes.size();
    for (unsigned i = 0; i != n; ++i) {
      if (children[i].empty())
        continue;
      // Only type constructors normalize; claim and projection keys are not
      // TypeAttr and carry no construction-time law to reapply.
      if (!isa<TypeAttr>(ctorKey[i]))
        continue;
      SmallVector<Attribute> subAttrs;
      SmallVector<Type> subTypes;
      classes.termAt(i).walkImmediateSubElements(
          [&](Attribute a) { subAttrs.push_back(a); },
          [&](Type t) { subTypes.push_back(t); });
      for (unsigned pos = 0; pos != subTypes.size(); ++pos) {
        unsigned childId = children[i][pos];
        for (unsigned m = 0; m != n; ++m) {
          if (m == childId ||
              classes.findCanonical(m) != classes.findCanonical(childId))
            continue;
          SmallVector<Type> repl(subTypes.begin(), subTypes.end());
          repl[pos] = classes.termAt(m);
          // Rebuild through the real constructor: get() applies whatever
          // normalization the type defines. A partial constructor returns null
          // and an unchanged rebuild carries nothing new -- skip both.
          Type r = classes.termAt(i).replaceImmediateSubElements(subAttrs, repl);
          if (!r || r == classes.termAt(i))
            continue;
          TermShape rs = decomposeTerm(r);
          bool freeReapplication =
              rs.key == ctorKey[i] && rs.children.size() == repl.size();
          for (unsigned k = 0; freeReapplication && k != repl.size(); ++k)
            if (rs.children[k] != repl[k])
              freeReapplication = false;
          if (freeReapplication)
            continue;
          unsigned rid = intern(r);
          assert(classes.size() <= mintCeiling &&
                 "ground congruence rebuild minted past its budget: a "
                 "constructor is normalizing without a fixed point");
          if (classes.findCanonical(i) != classes.findCanonical(rid)) {
            classes.unite(i, rid);
            changed = true;
          }
        }
      }
    }
    return changed;
  }

  TypeEquivalence classes;
  SmallVector<Attribute> ctorKey;
  SmallVector<SmallVector<unsigned>> children;
};

} // namespace

// The one ground-entailment decision the witness composition arm and
// trait.coerce's proven arm share: whether `lhs` and `rhs` fall in one class of
// the ground congruence closure seeded by the premise equalities. Application-
// claim proofs are stripped from every endpoint first (comparison is modulo
// the proof, permanently). For the composition arm the transitivity and
// congruence that carry the premises to the result are derived here at verify
// and never stored, so the witness holds only its leaf premises and only
// definitional leaves are ever stored.
bool mlir::trait::entailedByGroundCongruence(Type lhs, Type rhs,
                                             ArrayRef<TypeEqualityAttr> premises) {
  lhs = stripClaimProofs(lhs);
  rhs = stripClaimProofs(rhs);

  GroundCongruence closure;
  closure.intern(lhs);
  closure.intern(rhs);
  for (TypeEqualityAttr eq : premises)
    closure.seed(stripClaimProofs(eq.getLhs()),
                 stripClaimProofs(eq.getRhs()));
  closure.close();

  return closure.equal(lhs, rhs);
}

Type mlir::trait::stripClaimProofs(Type type) {
  // An endpoint carries no proven claim by construction, so this rewrite finds
  // nothing to strip inside one; the seal states that rather than relying on it.
  AttrTypeReplacer strip = makeEndpointSealedReplacer();
  strip.addReplacement([](ClaimType claim) -> std::optional<Type> {
    if (claim.isProven())
      return Type(claim.asUnproven());
    return std::nullopt;
  });
  return strip.replace(type);
}

// The pending judgment a marked (unproven) coerce carries. Its reconciling
// equalities are not yet citable -- the impl that supplies them is minted at
// monomorphization -- so the endpoints are judged twice: here, where the
// spellings may still be open, and again at the erase barrier, where they are
// not. Endpoints identical after proof stripping are already reconciled.
// Endpoints where either side still spells a projection or a type variable are
// open: what each denotes is settled by instantiation and by the impls
// monomorphization mints, so this verifier has nothing to decide and leaves the
// judgment to the barrier, which refuses a coerce whose ground endpoints stand
// apart. Two ground endpoints that differ are settled here and now: no later
// step can bring them together, so they are refused. Endpoints arrive with
// proofs already stripped.
LogicalResult mlir::trait::verifyPendingCoerceEndpoints(
    Type input, Type result,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (input == result)
    return success();

  auto stillOpen = [](Type ty) {
    return containsType<ProjectionType>(ty) || containsType<PolyType>(ty);
  };
  if (stillOpen(input) || stillOpen(result))
    return success();

  if (emitError)
    emitError() << "input type " << input << " and result type " << result
                << " are not consistent as a pending coerce";
  return failure();
}
