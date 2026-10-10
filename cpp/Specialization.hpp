// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TraitOps.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::trait {

/// What a clone is cut as: a template, whose spelling resolves when it is
/// itself cloned for a concrete instance, or an instance.
enum class CloneKind { Template, Instance };

/// Builds a type replacer that stamps `variables`, and `projections` the
/// ground projections a call's substitution closes over, into a clone of kind
/// `kind`: a template's clone receives each binding once, an instance's the
/// closed call substitution chased to its fixed point; a claim's predicate
/// receives the variable bindings alone (`respellClaimPredicate`). It resolves
/// nothing: a ground projection the substitution mints (a concrete argument
/// substituted into a projection spelling) stays spelled, and the op carrying
/// it asks impl selection for it where it stands, so a clone is normalized by
/// the one solver every other spelling is.
AttrTypeReplacer makeTypeReplacerFromSubstitution(
    const SpecializationMap &variables, CloneKind kind,
    const ProjectionBindings &projections = ProjectionBindings());

/// Builds a type replacer that stamps `variables` and nothing else: no
/// projection binding, so a spelling it stamps keeps every projection it
/// spells. The result of a call computing evidence is stamped by it, since the
/// variables of the requirement the call computes are read off that spelling
/// (`MethodCallOp::inlineEvidence`).
AttrTypeReplacer makeSpellingReplacerFromSubstitution(
    const SpecializationMap &variables);

/// Clones `source`'s blocks into `dest` before `before` under `mapping`, then
/// stamps every block argument, op result and attribute of the clones by
/// `typeReplacer`, except the result of a call computing evidence, which
/// `spellingReplacer` stamps: that resolves nothing, since the call's result
/// spelling is where the variables of the requirement it computes are read
/// (`MethodCallOp::inlineEvidence`). Block arguments are stamped before the
/// ops reading them; `builder`'s listener hears of every clone.
void cloneRegionStampedBefore(OpBuilder &builder, Region &source, Region &dest,
                              Region::iterator before, IRMapping &mapping,
                              AttrTypeReplacer &typeReplacer,
                              AttrTypeReplacer &spellingReplacer);

/// Clones `polymorph` at `rewriter`'s insertion point as `instanceName`, its
/// signature, attributes and body stamped under `substitution`, and answers the
/// clone, or null when `polymorph` has no body to clone.
///
/// The clone is the function the block it is inserted into holds: a
/// `trait.method` inside a trait or impl and a `func.func` anywhere else. A
/// method cut into a `func.func` has every `trait.return` ending a block of its
/// body become a `func.return` over the same operands; nothing else in the body
/// changes kind. A method carries no visibility, so a clone into a trait or impl
/// takes none, and a `func.func` takes the polymorph's.
FunctionOpInterface specializePolymorph(
    RewriterBase &rewriter, FunctionOpInterface polymorph,
    StringRef instanceName, const SpecializationMap &variables,
    const ProjectionBindings &projections = ProjectionBindings());

/// Clones `polymorph` into the empty `monomorph`, stamped under `variables`: as
/// an instance's body, or as a template's where the builder stands inside one.
void specializePolymorphicRegion(OpBuilder &builder, Region &polymorph,
                                 Region &monomorph,
                                 const SpecializationMap &variables);

/// The identity of one instance a template is cut into: the template, the
/// arguments its type parameters take, and the evidence each of its formal
/// positions receives, in parameter order.
///
/// An instance is named by the evidence it was made with, never by its claim
/// types alone. A claim type states a fact, and two uses can supply one fact for
/// different reasons -- different proofs, selecting different impls -- while an
/// instance's body runs the impls its own proofs select. So two uses share an
/// instance exactly when they supply equal type arguments and, position by
/// position, equal evidence. A proven claim names its proof and the proof names
/// each of its subproofs, so the type a use supplies at a position that takes a
/// claim is the whole of the evidence there, compared by interned identity. A
/// position whose formal type mentions no claim takes no evidence.
///
/// Type arguments are read at the spelling the instance is stamped in -- the
/// substitution its body is cut under, with ground projections resolved -- so
/// one type supplied under two spellings is one argument. Evidence is read at
/// the spelling the instance's parameter has: the formal stamped, whose claim
/// keeps its predicate (`respellClaimPredicate`). A claim supplied under that
/// spelling is the evidence there; one supplied under another is carried to
/// it by its proof respelled (`ImplResolver::respellProof`), which the call
/// passes in its place: a claim names the proof of its own spelling, so the
/// supplied proof proves another one. The parameters of the instance a key
/// names are spelled as the key holds them.
///
/// A key exists only for a use whose every application claim names its proof:
/// an unproven claim states a fact without a reason, which identifies no
/// instance.
class InstanceKey {
public:
  /// The key of a use supplying `actualInputs` to the template `templateRef`,
  /// whose formal inputs are `formalInputs` and whose type parameters take
  /// `typeArguments`, each argument and input read through `stamp`, the type
  /// replacement the instance is cut under; `respell` carries a proven claim
  /// to another spelling of its application. Fails when the two input lists
  /// differ in length, when what the use supplies at a position that takes a
  /// claim holds an unproven application, or when a claim the use supplies
  /// under another spelling is not carried to the parameter's.
  static FailureOr<InstanceKey>
  get(SymbolRefAttr templateRef, ArrayRef<Type> typeArguments,
      TypeRange formalInputs, TypeRange actualInputs, AttrTypeReplacer &stamp,
      llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell);

  /// The symbol the instance is cut under: the template's root name, a hash of
  /// the type arguments followed by the evidence, then for a method the
  /// method's name. Equal keys spell one name in every process; unequal keys of
  /// one template spell different names but for a collision of the 64-bit hash.
  std::string getSymbolName() const;

  /// What each formal position received, null at a position that takes no
  /// evidence.
  ArrayRef<Type> getEvidence() const { return evidence; }

private:
  InstanceKey(SymbolRefAttr templateRef, ArrayRef<Type> typeArguments,
              SmallVector<Type> evidence)
      : templateRef(templateRef),
        typeArguments(typeArguments.begin(), typeArguments.end()),
        evidence(std::move(evidence)) {}

  SymbolRefAttr templateRef;
  SmallVector<Type> typeArguments;
  SmallVector<Type> evidence;
};

/// The instance `key` names among `module`'s symbols: the one already cut for
/// it, or the one `cut` makes under the key's symbol name.
///
/// A fresh instance takes, at each position that takes evidence, exactly the
/// evidence the key holds there, and a projection its body reads off an
/// operand takes the evidence the source's proof determines at its index.
/// `cut` answers null when the template has no body to clone, and so does
/// this.
func::FuncOp getOrCutInstance(RewriterBase &rewriter, ModuleOp module,
                              const InstanceKey &key,
                              llvm::function_ref<func::FuncOp(StringRef)> cut);

}
