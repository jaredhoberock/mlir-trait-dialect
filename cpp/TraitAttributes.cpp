// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>


#define GET_ATTRDEF_CLASSES
#include <TraitAttributes.cpp.inc>

namespace mlir::trait {

// Whether any claim nested in the type is proven -- a proven claim spelled
// into a position that forbids one. The equality arm freezes endpoints that
// contain no proven claim.
static bool containsProvenClaim(Type type) {
  bool found = false;
  type.walk([&](Type sub) {
    if (auto claim = dyn_cast<ClaimType>(sub))
      if (claim.isProven())
        found = true;
  });
  return found;
}

// Structural well-formedness of an equality proposition. An endpoint must not
// contain a proven claim: a proven claim spelled into an endpoint would carry a
// proof into the arm that forbids one, re-creating the asymmetric proof
// comparison the equality arm exists to avoid. Constructing the attribute does
// not assert the equality; only a value of the enclosing claim type is
// evidence.
LogicalResult TypeEqualityAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Type lhs, Type rhs) {
  if (!lhs || !rhs) {
    if (emitError) emitError() << "type equality requires two endpoint types";
    return failure();
  }

  if (containsProvenClaim(lhs) || containsProvenClaim(rhs)) {
    if (emitError) emitError() << "a type-equality endpoint must not contain a proven claim";
    return failure();
  }

  return success();
}

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include <TraitAttributes.cpp.inc>
  >();
}

// The generic parser hands a dialect the text between `#trait<` and `>` and
// reads nothing of what the dialect leaves unread, so a suffix past the
// attribute's own text would be silently dropped. Every attribute of this
// dialect reads its whole text or is refused.
Attribute TraitDialect::parseAttribute(DialectAsmParser &parser,
                                       Type type) const {
  SMLoc tagLoc = parser.getCurrentLocation();
  StringRef tag;
  Attribute attr;
  OptionalParseResult parsed =
      generatedAttributeParser(parser, &tag, type, attr);
  if (!parsed.has_value()) {
    parser.emitError(tagLoc) << "unknown attribute `" << tag
                             << "` in dialect `" << getNamespace() << "`";
    return {};
  }
  if (failed(*parsed))
    return {};
  // The text ends where the dialect's text does: at the `>` closing the
  // verbose form, which the lexer reads next, or past a pretty form's own
  // closing delimiter.
  SMLoc next = parser.getCurrentLocation();
  if (next.getPointer() < parser.getFullSymbolSpec().end()) {
    parser.emitError(next, "expected the end of the attribute");
    return {};
  }
  return attr;
}

void TraitDialect::printAttribute(Attribute attr,
                                  DialectAsmPrinter &printer) const {
  (void)generatedAttributePrinter(attr, printer);
}

template<class T>
static T cantFail(FailureOr<T> f, const char* message) {
  if (failed(f))
    llvm_unreachable(message);
  return *f;
}

FailureOr<TraitOp> TraitApplicationAttr::getTrait(
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError
) const {
  TraitOp traitOp = lookupSymbolFrom<TraitOp>(module, getTraitName());
  if (!traitOp) {
    if (emitError) emitError() << "cannot find trait '" << getTraitName() << "'";
    return failure();
  }
  return traitOp;
}

TraitOp TraitApplicationAttr::getTraitOrAbort(
    ModuleOp module,
    const char* msg
) const {
  return cantFail(getTrait(module), msg);
}

// A trait application references a trait symbol applied to a fixed number of
// type arguments; verify the trait exists and its arity is respected. These
// attributes appear as inherent operation arguments, which the automatic
// symbol-user driver never walks -- it visits only discardable attributes -- so
// their owning ops verify them by delegating to this entry point. The interface
// is adopted for uniformity with symbol-using types; the same method would also
// verify a trait application encountered in a discardable position.
LogicalResult TraitApplicationAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  ModuleOp module = getAnchorModule(op);
  if (!module)
    return op->emitError()
           << "cannot verify trait application '" << getTraitName()
           << "': anchor operation is not nested in a module";
  auto err = [&] { return op->emitError(); };

  auto trait = getTrait(module, err);
  if (failed(trait))
    return failure();

  auto expectedArity = trait->getTypeParams().size();
  if (getTypeArgs().size() != expectedArity)
    return err() << "trait '" << getTraitName() << "' expects " << expectedArity
                 << " type arguments, found " << getTypeArgs().size();

  for (Type arg : getTypeArgs())
    if (containsClaim(arg))
      return err() << "trait application " << *this
                   << " takes a claim as a type argument; a claim is a value's "
                      "evidence, never a type argument";

  return success();
}

Attribute ImplArgumentsAttr::parse(AsmParser &parser, Type) {
  SmallVector<Type> types;
  if (parser.parseLess() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     [&]() -> ParseResult {
                                       Type type;
                                       if (parser.parseType(type))
                                         return failure();
                                       types.push_back(type);
                                       return success();
                                     }) ||
      parser.parseGreater())
    return {};
  return ImplArgumentsAttr::get(parser.getContext(), types);
}

void ImplArgumentsAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  llvm::interleaveComma(getTypes(), printer);
  printer << "]>";
}

bool containsClaim(Type type) {
  return type
      .walk([](ClaimType) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

// The single grammar for a trait application's `[!T1, !T2, ...]` body, shared by
// TraitApplicationAttr::parse and by the claim predicate parser; only the entry
// token that reads the leading symbol differs between them.
FailureOr<TraitApplicationAttr>
parseTraitApplicationBody(AsmParser &parser, FlatSymbolRefAttr traitName) {
  // Parse required type arguments in brackets.
  if (parser.parseLSquare())
    return failure();

  SmallVector<Type> typeArgs;
  do {
    Type ty;
    if (parser.parseType(ty))
      return failure();
    typeArgs.push_back(ty);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare())
    return failure();

  TraitApplicationAttr app = TraitApplicationAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), traitName, typeArgs);
  if (!app)
    return failure();
  return app;
}

Attribute TraitApplicationAttr::parse(AsmParser &parser, Type type) {
  // Expect: @TraitName[!T1, !T2, ...]
  FlatSymbolRefAttr traitName;
  if (parser.parseAttribute(traitName))
    return {};

  FailureOr<TraitApplicationAttr> app =
      parseTraitApplicationBody(parser, traitName);
  if (failed(app))
    return {};
  return *app;
}

void TraitApplicationAttr::print(mlir::AsmPrinter &printer) const {
  printer << getTraitName(); // print the trait symbol name

  printer << '[';
  llvm::interleaveComma(getTypeArgs(), printer);
  printer << ']';
}

} // namespace mlir::trait
