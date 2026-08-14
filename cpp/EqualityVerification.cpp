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

// Does `needle` occur as a subterm of `haystack`? Equality-claim endpoints are
// sealed from Type::walk, so this classifies through the deep walk: a needle
// reachable only inside an endpoint is still found.
static bool typeOccursIn(Type needle, Type haystack) {
  return walkTypesDeep(haystack, [&](Type sub) {
           return sub == needle ? WalkResult::interrupt()
                                : WalkResult::advance();
         }).wasInterrupted();
}

// Rewrite `ty` by the cited equality premises: each premise's lhs endpoint
// rewrites to its rhs. When a projection-headed impl self-application cannot be
// aligned by structural matching, verification matches modulo these equalities.
//
// A premise set whose rewrite relation cycles has no finite solution and its
// fixed point would not terminate, so it is refused. The relation orders key a
// before key b when b occurs in the value bound to a; a self-referential premise
// such as !S = tuple<!S> is the degenerate self-loop. Refusing here keeps the
// verifier total on spellable IR.
static FailureOr<Type> applyEqualityPremises(
    Type ty, ArrayRef<TypeEqualityAttr> premises,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DenseMap<Type, Type> subst;
  for (TypeEqualityAttr eq : premises)
    subst[eq.getLhs()] = eq.getRhs();
  if (subst.empty())
    return ty;

  SmallVector<Type> keys;
  for (auto &kv : subst)
    keys.push_back(kv.first);
  // A depth-first walk of the rewrite relation reporting a back edge. The color
  // marks are unvisited, on the current path, and finished.
  DenseMap<Type, unsigned> color;
  std::function<Type(Type)> findCycle = [&](Type key) -> Type {
    color[key] = 1;
    Type value = subst.lookup(key);
    for (Type other : keys)
      if (typeOccursIn(other, value)) {
        unsigned c = color.lookup(other);
        if (c == 1)
          return other;
        if (c == 0)
          if (Type hit = findCycle(other))
            return hit;
      }
    color[key] = 2;
    return Type();
  };
  for (Type key : keys)
    if (color.lookup(key) == 0)
      if (Type cyclic = findCycle(key)) {
        if (err)
          err() << "a self-referential equality premise (" << cyclic
                << " occurs in its own rewrite) has no finite solution";
        return failure();
      }

  return applySubstitutionToFixedPoint(subst, ty);
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
  auto typeMap = subst.toTypeMap();
  return llvm::map_to_vector(impl.getAssumptionsAsClaims(), [&](ClaimType a) {
    return cast<ClaimType>(applySubstitutionToFixedPoint(typeMap, a));
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
};

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

  // Arm (i): the citing impl's own where clause covers the obligation. The
  // equality premises are already known non-cyclic here (verification rewrote its
  // endpoints through them before reaching this check), so the rewrite cannot
  // fail on a well-formed premise set.
  for (TraitApplicationAttr premiseApp : ctx.obligationPremises) {
    ClaimType premiseClaim = ClaimType::get(mlirCtx, premiseApp);
    auto haveOr =
        applyEqualityPremises(Type(premiseClaim), ctx.premises, ctx.err);
    if (succeeded(haveOr) && *haveOr == want)
      return true;
  }

  // Arm (ii): a declared discharge citation names the obligation and an impl
  // that supplies it.
  for (WitnessAttr citation : ctx.dischargeWitnesses) {
    ClaimType citedApp = ClaimType::get(mlirCtx, citation.getApplication());
    auto citedOr =
        applyEqualityPremises(Type(citedApp), ctx.premises, ctx.err);
    if (failed(citedOr) || *citedOr != want)
      continue;
    if (llvm::is_contained(inProgress, citation.getApplication()))
      continue; // cycle: this path grounds nothing

    auto dischargerOp = SymbolTable::lookupNearestSymbolFrom<ImplOp>(
        module, citation.getImplRef());
    if (!dischargerOp)
      continue;

    // The named impl must genuinely supply the application: instantiate only
    // its own generics for the application (rigid actual side, no module scan).
    ClaimType appClaim = ClaimType::get(mlirCtx, citation.getApplication());
    auto subst = buildSpecialization(dischargerOp.getSelfClaim(), Type(appClaim),
                                     ModuleOp());
    if (failed(subst))
      continue;

    // Its own assumptions, specialized through that same substitution, must each
    // discharge in turn.
    inProgress.push_back(citation.getApplication());
    bool allDischarged = true;
    for (ClaimType assumption :
         specializeAssumptionsThroughSubst(dischargerOp, *subst)) {
      auto subWantOr = applyEqualityPremises(Type(assumption.asUnproven()),
                                             ctx.premises, ctx.err);
      if (failed(subWantOr) ||
          !dischargeApplicationObligation(ctx, *subWantOr, inProgress)) {
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
    llvm::function_ref<InFlightDiagnostic()> err) {
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
  auto subst = rigidHeadMatch
                   ? buildSpecialization(implOp.getSelfClaim(), Type(selfClaim),
                                         ModuleOp(), err)
                   : implOp.buildSubstitutionForSelfClaim(selfClaim, err);
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
  auto actualOr = applyEqualityPremises(actual, premises, err);
  if (failed(actualOr))
    return failure();
  actual = *actualOr;
  auto resolvedOr = applyEqualityPremises(resolved, premises, err);
  if (failed(resolvedOr))
    return failure();
  if (actual != *resolvedOr) {
    if (err) err() << "impl '" << citedImpl << "' binds the projection to "
                   << actual << ", not the certified resolution " << resolved;
    return failure();
  }

  // Obligation-discharge check. The cited impl's own assumptions -- specialized
  // through the same rigid head-match substitution -- must each be discharged,
  // proof-stripped and modulo the cited equality premises, by a hypothetical
  // cover (arm i) or a declared discharge citation (arm ii). The impl's trait
  // requirements are deliberately not reached here (they may quantify over GAT
  // variables with no ground instance at the witness).
  ObligationDischargeContext dischargeCtx{module, premises, obligationPremises,
                                          dischargeWitnesses, err};
  for (ClaimType assumption :
       specializeAssumptionsThroughSubst(implOp, *subst)) {
    auto wantOr =
        applyEqualityPremises(Type(assumption.asUnproven()), premises, err);
    if (failed(wantOr))
      return failure();
    SmallVector<TraitApplicationAttr> inProgress;
    if (!dischargeApplicationObligation(dischargeCtx, *wantOr, inProgress)) {
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
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (failed(verifyProjectionResolutionCore(module, witness, premises,
                                            obligationPremises,
                                            /*dischargeWitnesses=*/{},
                                            /*rigidHeadMatch=*/false, err)))
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
// its cited equalities. It seeds the union-find with the equalities, then
// closes under congruence: two terms with the same constructor and pairwise
// equal children are united. It only unites -- it never decomposes, so
// f(a) = f(b) is not read backwards to a = b at projection heads or anywhere
// else. It also closes across normalizing type constructors: a composite is
// united with the normal form its own constructor yields when a united class
// member is substituted into it, so an equality a constructor establishes by
// normalizing its arguments is not missed. Child enumeration and constructor
// identity both come from decomposeTerm, which reads through the type-bearing
// trait attributes the generic walkers are opaque to.
class GroundCongruence {
public:
  // Seed an equality between two endpoints (and intern their subterms).
  void seed(Type a, Type b) { unite(intern(a), intern(b)); }

  // Intern a type and all its subterms; returns its term id.
  unsigned intern(Type t) {
    auto it = ids.find(t);
    if (it != ids.end())
      return it->second;
    unsigned id = terms.size();
    ids[t] = id;
    terms.push_back(t);
    parent.push_back(id);
    ctorKey.push_back(Attribute());
    children.emplace_back();

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
    const size_t mintCeiling = terms.size() * 8 + 256;
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0, n = terms.size(); i != n; ++i)
        for (unsigned j = i + 1; j != n; ++j) {
          if (find(i) == find(j))
            continue;
          if (ctorKey[i] != ctorKey[j] ||
              children[i].size() != children[j].size())
            continue;
          bool allEqual = true;
          for (auto [ci, cj] : llvm::zip(children[i], children[j]))
            if (find(ci) != find(cj)) {
              allEqual = false;
              break;
            }
          if (allEqual) {
            unite(i, j);
            changed = true;
          }
        }
      if (rebuildNormalizedParents(mintCeiling))
        changed = true;
    }
  }

  bool equal(Type a, Type b) { return find(intern(a)) == find(intern(b)); }

private:
  unsigned find(unsigned x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(unsigned a, unsigned b) {
    a = find(a);
    b = find(b);
    if (a != b)
      parent[a] = b;
  }

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
    // is fixed and the loop bounds stay valid as terms grows.
    unsigned n = terms.size();
    for (unsigned i = 0; i != n; ++i) {
      if (children[i].empty())
        continue;
      // Only type constructors normalize; claim and projection keys are not
      // TypeAttr and carry no construction-time law to reapply.
      if (!isa<TypeAttr>(ctorKey[i]))
        continue;
      SmallVector<Attribute> subAttrs;
      SmallVector<Type> subTypes;
      terms[i].walkImmediateSubElements(
          [&](Attribute a) { subAttrs.push_back(a); },
          [&](Type t) { subTypes.push_back(t); });
      for (unsigned pos = 0; pos != subTypes.size(); ++pos) {
        unsigned childId = children[i][pos];
        for (unsigned m = 0; m != n; ++m) {
          if (m == childId || find(m) != find(childId))
            continue;
          SmallVector<Type> repl(subTypes.begin(), subTypes.end());
          repl[pos] = terms[m];
          // Rebuild through the real constructor: get() applies whatever
          // normalization the type defines. A partial constructor returns null
          // and an unchanged rebuild carries nothing new -- skip both.
          Type r = terms[i].replaceImmediateSubElements(subAttrs, repl);
          if (!r || r == terms[i])
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
          assert(terms.size() <= mintCeiling &&
                 "ground congruence rebuild minted past its budget: a "
                 "constructor is normalizing without a fixed point");
          if (find(i) != find(rid)) {
            unite(i, rid);
            changed = true;
          }
        }
      }
    }
    return changed;
  }

  DenseMap<Type, unsigned> ids;
  SmallVector<Type> terms;
  SmallVector<unsigned> parent;
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
  AttrTypeReplacer strip;
  strip.addReplacement([](ClaimType claim) -> std::optional<Type> {
    if (claim.isProven())
      return Type(claim.asUnproven());
    return std::nullopt;
  });
  return strip.replace(type);
}

// The pending judgment a marked (unproven) coerce carries. Its reconciling
// equalities are not yet citable -- the impl that supplies them is minted at
// monomorphization -- so instead of ground congruence over cited leaves the
// endpoints must UNIFY, with every !trait.proj term treated as a shared
// unification variable keyed by the projection itself: the same projection is
// one variable and cannot stand for two types, every other constructor position
// is rigid, and a claim's (or any other composite's) predicate arguments are
// descended through decomposeTerm, whose enumeration reaches the hand-written
// attribute storage the generic type walkers are opaque to. A whole projection
// is one opaque variable: its own trait-application and associated-type
// arguments are NOT descended during reconciliation, so two projections meet as
// whole variables -- the same variable, or a pair aliased and owed one grounding
// at discharge -- never unified by matching their arguments. Reflexive endpoints
// pass. A projection may resolve to a projection-free position (the ground type
// the minted impl supplies), or stand for itself, or alias another bare
// projection --
// two lookups asserted to denote one type, each still owed a projection-free
// grounding at discharge. What it may NOT resolve to is a composite still
// carrying a projection: that would equate two distinct projections inside a
// rigid constructor, a shape this form never licensed. Binding a projection to a
// type that contains the projection itself is an unfoundable infinite type; it is
// refused by an occurs check that also keeps the binding acyclic so the
// resolution walks below terminate. Endpoints arrive with proofs already
// stripped.
LogicalResult mlir::trait::verifyPendingProjectionUnification(
    Type input, Type result,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  // Each projection stands for at most one type; a projection absent from the
  // map is unbound and stands for itself. The occurs check below keeps the map
  // acyclic, so `resolve` and the descent walks always terminate.
  DenseMap<ProjectionType, Type> binding;

  std::function<Type(Type)> resolve = [&](Type t) -> Type {
    while (auto proj = dyn_cast<ProjectionType>(t)) {
      auto it = binding.find(proj);
      if (it == binding.end() || it->second == t)
        return t;
      t = it->second;
    }
    return t;
  };

  // Whether the projection `p` occurs anywhere in `t` once bindings resolve to a
  // fixed point. Binding `p` to such a `t` would close a cycle (an infinite
  // type), so it is refused before the binding is made; the acyclic invariant
  // this preserves is what bounds the recursion here and in `carriesProjection`.
  std::function<bool(ProjectionType, Type)> occursIn =
      [&](ProjectionType p, Type t) -> bool {
    t = resolve(t);
    if (auto pt = dyn_cast<ProjectionType>(t))
      return pt == p;
    for (Type child : decomposeTerm(t).children)
      if (occursIn(p, child))
        return true;
    return false;
  };

  // Whether a type still carries a projection once its bindings resolve to a
  // fixed point. A bound projection must reach a projection-free type.
  std::function<bool(Type)> carriesProjection = [&](Type t) -> bool {
    t = resolve(t);
    if (isa<ProjectionType>(t))
      return true;
    for (Type child : decomposeTerm(t).children)
      if (carriesProjection(child))
        return true;
    return false;
  };

  std::function<LogicalResult(Type, Type)> unifyPending =
      [&](Type a, Type b) -> LogicalResult {
    a = resolve(a);
    b = resolve(b);
    if (a == b)
      return success();
    if (auto pa = dyn_cast<ProjectionType>(a)) {
      if (occursIn(pa, b)) {
        if (emitError) emitError() << "input type " << input << " and result type "
                                   << result << " are not consistent as a pending coerce";
        return failure();
      }
      binding[pa] = b;
      return success();
    }
    if (auto pb = dyn_cast<ProjectionType>(b)) {
      if (occursIn(pb, a)) {
        if (emitError) emitError() << "input type " << input << " and result type "
                                   << result << " are not consistent as a pending coerce";
        return failure();
      }
      binding[pb] = a;
      return success();
    }
    // Both sides are rigid here: same constructor identity, children paired.
    TermShape sa = decomposeTerm(a);
    TermShape sb = decomposeTerm(b);
    if (sa.key != sb.key || sa.children.size() != sb.children.size()) {
      if (emitError) emitError() << "input type " << input << " and result type "
                                 << result << " are not consistent as a pending coerce";
      return failure();
    }
    for (auto [ca, cb] : llvm::zip(sa.children, sb.children))
      if (failed(unifyPending(ca, cb)))
        return failure();
    return success();
  };

  if (failed(unifyPending(input, result)))
    return failure();

  // Final licensing check (the header states the rule): a bare-projection
  // terminal is a direct alias, still owed a grounding at discharge, and stays
  // pending; a terminal still carrying a projection would equate two distinct
  // projections in a rigid constructor and is refused.
  for (auto &[proj, bound] : binding) {
    Type terminal = resolve(bound);
    if (isa<ProjectionType>(terminal))
      continue;
    if (carriesProjection(terminal)) {
      if (emitError) emitError() << "input type " << input << " and result type "
                                 << result
                                 << " equate distinct projections in a pending coerce";
      return failure();
    }
  }

  return success();
}
