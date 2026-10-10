// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::trait {
  // forward declarations for Attributes.td/Attributes.hpp.inc
  class TraitOp;
  class SpecializationMap;
}


#define GET_ATTRDEF_CLASSES
#include <TraitAttributes.hpp.inc>

namespace mlir { class AsmParser; class AsmPrinter; }

namespace mlir::trait {

/// Parse the bracketed type-argument list of a trait application `@Trait[...]`
/// whose leading symbol `traitName` has already been read, and build the checked
/// application. This is the single grammar for the application body; the entry
/// token that precedes it -- a required symbol, or the optional symbol that
/// distinguishes an application from an equality in a claim's predicate -- is
/// the caller's to read. Fails, having emitted a diagnostic, on a malformed
/// argument list.
FailureOr<TraitApplicationAttr>
parseTraitApplicationBody(AsmParser &parser, FlatSymbolRefAttr traitName);

/// Whether a claim type stands anywhere inside `type`. A proof lives at the root
/// of a value's claim type, so a claim is never a type argument of a trait
/// application or of a projection: their symbol-use verifiers refuse one at any
/// depth, and a comparison modulo proofs strips only the root.
bool containsClaim(Type type);

} // end mlir::trait
