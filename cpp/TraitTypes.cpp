/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "TraitTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TraitTypes.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TraitTypes.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

int nextPolyTypeId() {
  static std::atomic<int> counter{-1};
  return counter.fetch_sub(1, std::memory_order_relaxed);
}

PolyType PolyType::getUnique(MLIRContext* ctx) {
  return PolyType::get(ctx, nextPolyTypeId());
}

Type PolyType::instantiate(DenseMap<Type,Type> &inst, uint64_t &idCounter) {
  // check memo first - if we've already instantiated this PolyType, return it
  if (auto it = inst.find(*this); it != inst.end()) {
    return it->second;
  }

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++, getUniqueId());
  inst[*this] = fresh;
  return fresh;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <unique> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    uniqueId = nextPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'unique'");
      return Type();
    }
    
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>'");
    return Type();
  }

  return PolyType::get(ctx, uniqueId);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getUniqueId() << ">";
}

//===----------------------------------------------------------------------===//
// InferenceType
//===----------------------------------------------------------------------===//

LogicalResult InferenceType::unify(
  Type other,
  ModuleOp /*module*/,
  DenseMap<Type,Type> &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;

  // normalize
  other = applySubstitution(subst, other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto it = subst.find(self); it != subst.end()) {
    if (it->second != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << it->second
                            << ", cannot bind to " << other;
      return failure();
    }
    return success();
  }

  // occurs check: forbid T := f(..., T, ...) to avoid cycles
  auto occursIn = [](Type needle, Type haystack) {
    bool hit = false;
    haystack.walk([&](Type t) {
      if (!hit && t == needle) hit = true;
    });
    return hit;
  };

  if (occursIn(self, other)) {
    if (err) err() << "recursive substitution: " << self
                   << " occurs in " << other;
    return failure();
  }

  // bind the variable
  subst[self] = other;
  return success();
}


//===----------------------------------------------------------------------===//
// ClaimType
//===----------------------------------------------------------------------===//

LogicalResult ClaimType::verify(function_ref<InFlightDiagnostic()> emitError,
                                TraitApplicationAttr app,
                                FlatSymbolRefAttr proof) {
  bool polymorphic = llvm::any_of(app.getTypeArgs(), [](Type ty) {
    return isPolymorphicType(ty);
  });

  if (polymorphic && proof) {
    if (emitError) emitError() << "A polymorphic !trait.claim cannot have a proof";
    return failure();
  }

  return success();
}

LogicalResult ClaimType::verifySymbolUses(ModuleOp module, llvm::function_ref<InFlightDiagnostic()> err) {
  // verify trait application
  if (failed(getTraitApplication().verifyTraitApplication(module, err)))
    return failure();

  // if there's a proof, verify that it points to a valid symbol
  if (auto proof = getProof()) {
    if (failed(ProofOp::getProofOpOrSelfProofImplOp(module, proof, err)))
      return failure();
  }

  return success();
}

Type ClaimType::parse(AsmParser& p) {
  MLIRContext *ctx = p.getContext();

  if (p.parseLess())
    return {};

  TraitApplicationAttr app = mlir::dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app)
    return {};

  FlatSymbolRefAttr proof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proof))
      return {};
  }

  if (p.parseGreater())
    return {};

  auto emitError = [&] { return p.emitError(p.getCurrentLocation()); };
  return ClaimType::getChecked(emitError, ctx, app, proof);
}

void ClaimType::print(AsmPrinter& p) const {
  p << "<";
  getTraitApplication().print(p);
  if (isProven()) {
    p << " by " << getProof();
  }
  p << ">";
}

bool ClaimType::isPolymorphic() const {
  // a !trait.claim<@Trait[Types...]> is polymorphic if any of its type arguments are polymorphic
  return llvm::any_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  });
}

LogicalResult verifyAndRecordProof(
    ClaimType unproven,
    ClaimType proven,
    ModuleOp module,
    DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // the proven side must carry a proof
  if (!proven.isProven()) {
    if (err) err() << "expected proven claim, but found " << proven;
    return failure();
  }

  // early exit if we've already recorded this proof
  if (auto it = subst.find(unproven); it != subst.end()) {
    if (it->second != proven) {
      if (err) err() << "inconsistent proof mapping: " << unproven
                     << " is already bound to " << it->second
                     << ", but attempted to bind " << proven;
      return failure();
    }
    return success();
  }

  // the "unproven" parameter we're verifying might have already been
  // normalized once by previous calls: helpers like
  //   getRequirementsAsClaimsWith(subst)
  // apply `subst` to obligations before giving them back.
  // That means an obligation we thought was "unproven" when we first
  // saw it can come back already carrying the same proof symbol
  // as `proven`. In that case, there's nothing left to check.
  // We just accept it and stop: the claim is already proven and
  // agrees with what we're trying to record.
  if (unproven.isProven()) {
    if (unproven != proven) {
      // The claim came back proven but *with a different proof symbol*.
      // That's an inconsistency: some other path claimed to prove the
      // same trait with a different proof.
      if (err) err() << "incoherent proofs for obligation "
                     << unproven
                     << ": " << unproven.getProof() << " vs " << proven.getProof();
      return failure();
    }
    // already proven with the same proof -- nothing to do.
    return success();
  }

  // look up the trait and its requirements using the unproven claim
  auto trait = unproven.getTraitApplication().getTrait(module, err);
  if (failed(trait)) return failure();

  // inspect the proof symbol on the proven side
  auto symOp = ProofOp::getProofOpOrSelfProofImplOp(module, proven.getProof(), err);
  if (failed(symOp)) return failure();

  // if it's an impl op, check that the trait has no requirements
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (trait->hasRequirements()) {
      if (err) err() << "self-proving impl provides no subproof for trait requirements";
      return failure();
    }

    // success: bind the whole claim so that later normalization keeps the proof
    subst[unproven] = proven;
    return success();
  }

  // otherwise the symbol must be a ProofOp
  auto proof = dyn_cast<ProofOp>(*symOp);

  // check that the proof's claim can unify with proven
  if (failed(proof.getProvenClaim().unify(proven, module, subst, err))) 
    return failure();

  // specialize obligations for the unproven claim
  auto obligations = proof.getImpl().specializeObligationsAsClaimsFor(unproven, err);
  if (failed(obligations)) return failure();

  // get the subproof claims
  auto subproofs = proof.verifyAndGetSubproofClaims(err);
  if (failed(subproofs)) return failure();

  // the number of subproofs must match the number of obligations 
  if (subproofs->size() != obligations->size()) {
    if (err) err() << "arity mismatch: expected " << obligations->size()
                   << " subproofs, but found " << subproofs->size();
    return failure();
  }

  // recurse over obligations
  for (auto [ob, sub] : llvm::zip(*obligations, *subproofs)) {
    if (failed(verifyAndRecordProof(ob, sub, module, subst, err)))
      return failure();
  }

  // success: bind the whole claim so that later normalization keeps the proof
  subst[unproven] = proven;
  return success();
}

LogicalResult recordProofBindingsIn(
    Type root,
    ModuleOp module,
    DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  LogicalResult status = success();

  // walk down to ClaimType nodes and call verifyAndRecordProof
  root.walk([&](Type node) {
    if (status.failed()) return;

    if (auto claim = dyn_cast<ClaimType>(node)) {
      if (claim.isProven()) {
        // found a proven claim, delegate to verifyAndRecordProof
        if (failed(verifyAndRecordProof(claim.asUnproven(), claim, module, subst, err))) {
          status = failure();
        }
      }
    }
  });

  return status;
}

void ClaimType::getProjections(
    ModuleOp module,
    SmallVectorImpl<ClaimType>& result,
    ClaimType::ProjectionKind kinds) {
  // identity?
  if (kinds & Identity) {
    result.push_back(*this);
  }

  // trait requirements?
  if (kinds & TraitRequirements) {
    auto trait = getTraitApplication().getTraitOrAbort(module, "ClaimType::getProjections: couldn't find trait");
    auto specRequirements = trait.specializeRequirementsAsClaimsFor(*this);
    if (succeeded(specRequirements))
      result.append(*specRequirements);
  }

  // proven impl assumptions?
  if (kinds & ProvenImplAssumptions and isProven()) {
    if (auto proof = SymbolTable::lookupNearestSymbolFrom<ProofOp>(module, getProof())) {
      auto specAssumptions = proof.getImpl().specializeAssumptionsAsClaimsFor(*this);
      if (succeeded(specAssumptions))
        result.append(*specAssumptions);
    }
  }
}

LogicalResult ClaimType::unify(
    Type other,
    ModuleOp module,
    DenseMap<Type,Type>& subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize formal first
  Type formalNormTy = applySubstitution(subst, *this);
  ClaimType formal = mlir::dyn_cast<ClaimType>(formalNormTy);

  // if formal is no longer a ClaimType, delegate to generic path
  if (!formal)
    return trait::unify(formalNormTy, other, module, subst, err);

  // normalize actual second
  Type normActualTy = applySubstitution(subst, other);
  ClaimType actual = mlir::dyn_cast<ClaimType>(normActualTy);

  // if actual isn't a claim, it's an immediate mismatch
  if (!actual) {
    if (err) {
      err() << "expected !trait.claim, but found " << normActualTy;
    }
    return failure();
  }

  // do claim-specific checks below

  auto formalApp = formal.getTraitApplication();
  auto actualApp = actual.getTraitApplication();

  // same trait?
  if (formalApp.getTraitName() != actualApp.getTraitName()) {
    if (err) err() << "trait mismatch: expected " << formalApp.getTraitName()
                   << ", but found " << actualApp.getTraitName();
    return failure();
  }

  // same arity?
  auto formalArgs = formalApp.getTypeArgs();
  auto actualArgs = actualApp.getTypeArgs();
  if (formalArgs.size() != actualArgs.size()) {
    if (err) err() << "arity mismatch: expected " << formalArgs.size()
                   << " type arguments, but found " << actualArgs.size();
    return failure();
  }

  // check proofs
  auto formalProof = formal.getProof();
  auto actualProof = actual.getProof();
  if (formalProof && actualProof && formalProof != actualProof) {
    if (err) err() << "proof mismatch: expected " << formalProof
                   << ", but found " << actualProof;
    return failure();
  }
  if (formalProof && !actualProof) {
    if (err) err() << "cannot unify proven claim with unproven claim";
    return failure();
  }

  // recurse on each argument pair
  for (auto [f, a] : llvm::zip(formalArgs, actualArgs)) {
    if (failed(trait::unify(f, a, module, subst, err)))
      return failure();
  }

  return success();
}


//===----------------------------------------------------------------------===//
// unify
//===----------------------------------------------------------------------===//

/// Collect exactly the immediate child Types and Attributes of `ty`. If `ty` has no sub‐elements,
/// returns empty vectors.
static std::pair<SmallVector<Type, 4>, SmallVector<Attribute, 4>> getImmediateSubElements(Type ty) {
  SmallVector<Type, 4> childTypes;
  SmallVector<Attribute, 4> childAttrs;
  ty.walkImmediateSubElements(
      /*walkAttrsFn=*/[&](Attribute subAttr) {
        childAttrs.push_back(subAttr);
      },
      /*walkTypesFn=*/[&](Type subTy) {
        childTypes.push_back(subTy);
      });
  return std::pair(childTypes, childAttrs);
}

static LogicalResult unifyStructurally(Type formal,
                                       Type actual,
                                       ModuleOp module,
                                       llvm::DenseMap<Type,Type> &subst,
                                       llvm::function_ref<InFlightDiagnostic()> err) {
  if (formal == actual) return success();

  // check for same
  // 1. type constructor
  // 2. subelement arity
  // 3. attribute equality
  // and then recurse on children, if there are any
  auto [formalSubTys, formalSubAttrs] = getImmediateSubElements(formal);
  auto [actualSubTys, actualSubAttrs] = getImmediateSubElements(actual);

  bool formalHasSubs = !formalSubTys.empty() || !formalSubAttrs.empty();
  bool actualHasSubs = !actualSubTys.empty() || !actualSubAttrs.empty();

  // if neither side is decomposable, they're unequal leaves -> mismatch
  // if only one side is decomposable, constructors differ in structure -> mismatch
  if (!formalHasSubs || !actualHasSubs) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // the constructor and arity of subelements of both types must match before recursing
  if (formal.getTypeID() != actual.getTypeID() ||
      formalSubTys.size() != actualSubTys.size() ||
      formalSubAttrs.size() != actualSubAttrs.size()) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // the attributes of both types must match exactly before recursing on child types
  for (auto [f, a] : llvm::zip(formalSubAttrs, actualSubAttrs)) {
    if (f != a) {
      if (err) err() << "attribute mismatch: expected " << f
                     << " but found " << a;
      return failure();
    }
  }

  // Recurse on each sub type pair
  for (auto [f, a] : llvm::zip(formalSubTys, actualSubTys)) {
    if (failed(unify(f, a, module, subst, err)))
      return failure();
  }

  return success();
}

/// Attempt to unify `formal` with `actual`, extending `subst` with any
/// new bindings that make them equal under substitution.
///
/// Both sides are first normalized by applying `subst` to a fixed point.
/// After that we check for trivial equality and then choose how to drive
/// unification:
///
/// Priority of unifiers:
///  1. **Type variables first** — If either side is a `TypeVariableInterface`
///     (a leaf existential like `!trait.poly`), we let that side decide how
///     to bind itself (recording `poly := other`). We prefer to drive with
///     a type variable rather than try to break it down structurally.
///     If both `formal` and `actual` are `TypeVariableInterface`, `formal` wins.
///  2. **Formal unification types** — If the formal side implements
///     `UnificationTypeInterface`, we let it attempt to match/record
///     substitutions for its own structure.
///  3. **Structural fallback** — Otherwise we fall back to generic
///     shape-by-shape unification for non-monomorphizable, non-variable types.
///
/// The order matters: giving TVI “first refusal” ensures existential type
/// vars bind eagerly and prevents infinite ping-pong (e.g. `poly` vs `poly`)
/// while still letting unification types manage their own internal
/// unification when present.
///
/// Returns success if the two types can be made equal under an extended `subst`.
/// On failure, nothing is recorded and `err` (if provided) will be invoked to
/// emit a diagnostic.
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize both types by applying the current substitution
  formal = applySubstitutionToFixedPoint(subst, formal);
  actual = applySubstitutionToFixedPoint(subst, actual);

  // if the normalized types are equal, unification succeeds
  if (formal == actual)
    return success();

  // prefer a substituteWith method call if possible, in this priority:
  // 1. formal is TypeVariableInterface
  // 2. actual is TypeVariableInterface
  // 3. formal is UnificationTypeInterface

  // case 1.
  if (isa<TypeVariableInterface>(formal)) {
    // we assume every TypeVariableInterface is also UnificationTypeInterface
    auto formalUnifier = cast<UnificationTypeInterface>(formal);
    return formalUnifier.unify(actual, module, subst, err);
  }

  // case 2.
  if (isa<TypeVariableInterface>(actual)) {
    // we assume every TypeVariableInterface is also UnificationTypeInterface
    auto actualUnifier = cast<UnificationTypeInterface>(actual);
    return actualUnifier.unify(formal, module, subst, err);
  }

  // case 3.
  if (auto formalUnifier = dyn_cast<UnificationTypeInterface>(formal)) {
    return formalUnifier.unify(actual, module, subst, err);
  }

  // otherwise, unify structurally
  return unifyStructurally(formal, actual, module, subst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::DenseMap<Type,Type> &subst) {
  auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
  return unify(formal, actual, module, subst, errFn);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DenseMap<Type,Type> discardedSubst;
  return unify(formal, actual, module, discardedSubst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module) {
  DenseMap<Type,Type> discardedSubst;
  return unify(formal, actual, module, discardedSubst);
}


//===----------------------------------------------------------------------===//
// instantiate
//===----------------------------------------------------------------------===//

Type instantiate(Type root, DenseMap<Type,Type> &inst, uint64_t &idCounter) {
  AttrTypeReplacer r;
  r.addReplacement([&](Type t) -> std::optional<Type> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      return generic.instantiate(inst, idCounter);
    }
    return std::nullopt;
  });

  // this walks into types nested inside attributes (e.g., trait applications)
  // and replaces all GenericTypeInterface types according to (and extending) inst
  return r.replace(root);
}


} // end mlir::trait
