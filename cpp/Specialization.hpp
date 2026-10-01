// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TraitOps.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::trait {

/// Builds a type replacer that chases each stamped type to the substitution's
/// fixed point. When `module` is non-null it also resolves the ground
/// projections the substitution mints (a concrete argument substituted
/// into a projection spelling) by module-visible impl lookup, so a specialized
/// monomorph carries no ground projection that a unique module-visible impl
/// resolves; generator-pending and multi-candidate ground projections survive
/// unchanged. A null `module` performs no such lookup.
AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst,
                                                  ModuleOp module);

func::FuncOp specializePolymorph(OpBuilder& builder,
                                 func::FuncOp polymorph,
                                 StringRef instanceName,
                                 const DenseMap<Type,Type> &substitution);

void specializePolymorphicRegion(OpBuilder& builder,
                                 Region& polymorph,
                                 Region& monomorph,
                                 const DenseMap<Type,Type> &substitution);

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
/// Type arguments and evidence are read at the spelling the instance is
/// stamped in -- the substitution its body is cut under, with ground
/// projections resolved -- so one type or one proof supplied under two
/// spellings is one argument or one piece of evidence, and the parameters of
/// the instance a key names are spelled as the key holds them.
///
/// A key exists only for a use whose every application claim names its proof:
/// an unproven claim states a fact without a reason, which identifies no
/// instance.
class InstanceKey {
public:
  /// The key of a use supplying `actualInputs` to the template `templateRef`,
  /// whose formal inputs are `formalInputs` and whose type parameters take
  /// `typeArguments`, each argument and input read through `stamp`, the type
  /// replacement the instance is cut under. Fails when the two input lists
  /// differ in length, or when what the use supplies at a position that takes a
  /// claim holds an unproven application.
  static FailureOr<InstanceKey> get(SymbolRefAttr templateRef,
                                    ArrayRef<Type> typeArguments,
                                    TypeRange formalInputs,
                                    TypeRange actualInputs,
                                    AttrTypeReplacer &stamp);

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
/// evidence the key holds there. The substitution `cut` stamps the template
/// under spells a claim the same way at every position, so where two positions
/// receive one claim through different proofs only the position says which
/// proof each parameter carries. `cut` answers null when the template has no
/// body to clone, and so does this.
func::FuncOp getOrCutInstance(RewriterBase &rewriter, ModuleOp module,
                              const InstanceKey &key,
                              llvm::function_ref<func::FuncOp(StringRef)> cut);

}
