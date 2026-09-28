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
/// distinguishes an application from an equality in a where-clause predicate --
/// is the caller's to read. Fails, having emitted a diagnostic, on a malformed
/// argument list.
FailureOr<TraitApplicationAttr>
parseTraitApplicationBody(AsmParser &parser, FlatSymbolRefAttr traitName);

/// Parse a where-clause entry: an application (`@Trait[...]`), an equality
/// (`!A = !B`), or a bound predicate (`forall [...] where [...] -> ...`).
FailureOr<Attribute> parseWherePredicate(AsmParser &parser);

/// Print a where-clause entry as `parseWherePredicate` reads it.
void printWherePredicate(AsmPrinter &printer, Attribute predicate);

/// Parse the arguments an impl citation carries, `[!P = T, ...]`, one binding
/// per parameter of the impl, into `arguments`, and return whether the list is
/// present. The one grammar for the arguments every citation of an impl
/// carries; whether an absent list means an empty one or no statement at all
/// is the citing form's to decide.
FailureOr<bool> parseImplArguments(AsmParser &parser,
                                   SmallVectorImpl<TypeBindingAttr> &arguments);

/// Print `arguments` as the list `parseImplArguments` reads, `[]` when empty.
void printImplArguments(AsmPrinter &printer,
                        ArrayRef<TypeBindingAttr> arguments);

} // end mlir::trait
