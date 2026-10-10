// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The ground-congruence entailment a witness composition and a coerce both
// appeal to, and the term decomposition it keys on.

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

  // A parameter occurrence is a variable, a leaf for shape purposes, keyed by
  // its own spelling. A kind-constraining wrapper (such as
  // `!coord.poly<!trait.poly<0>>`) carries its label as an immediate
  // sub-element whose reconstruction declines a position placeholder, so it
  // must be keyed here rather than decomposed structurally below.
  if (getParameterOccurrence(t)) {
    s.key = TypeAttr::get(t);
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
// the ground congruence closure seeded by the premise equalities. The
// comparison is modulo application-claim proofs, permanently: a premise's
// endpoints carry none (`TypeEqualityAttr::verify`), and the caller strips the
// two compared types. For the composition arm the transitivity and
// congruence that carry the premises to the result are derived here at verify
// and never stored, so the witness holds only its leaf premises and only
// definitional leaves are ever stored.
bool mlir::trait::entailedByGroundCongruence(Type lhs, Type rhs,
                                             ArrayRef<TypeEqualityAttr> premises) {
  GroundCongruence closure;
  closure.intern(lhs);
  closure.intern(rhs);
  for (TypeEqualityAttr eq : premises)
    closure.seed(eq.getLhs(), eq.getRhs());
  closure.close();

  return closure.equal(lhs, rhs);
}

Type mlir::trait::stripClaimProofs(Type type) {
  if (auto claim = dyn_cast<ClaimType>(type))
    return claim.asUnproven();
  // A claim stands inside a container type -- a signature, a tuple of claims --
  // only as a value's own type, never inside another claim's predicate (the
  // trait application's symbol-use verifier refuses one there), so the rewrite
  // strips each claim it reaches and never enters one.
  AttrTypeReplacer strip = makeEndpointSealedReplacer();
  strip.addReplacement([](ClaimType claim) -> std::pair<Type, WalkResult> {
    return {claim.asUnproven(), WalkResult::skip()};
  });
  return strip.replace(type);
}

