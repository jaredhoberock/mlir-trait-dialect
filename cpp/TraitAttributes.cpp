// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_ATTRDEF_CLASSES
#include <TraitAttributes.cpp.inc>

namespace mlir::trait {

// Whether any claim nested in the type is proven -- a proven claim spelled
// into a position that forbids one. The equality arm and the
// projection-resolution witness both freeze endpoints that contain no
// proven claim.
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

// A binding's parameter is a label a declaration binds: a type parameter that
// is its own parameter occurrence, not a wrapper constraining one, since a
// substitution is keyed by the labels themselves.
LogicalResult TypeBindingAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Type parameter, Type argument) {
  if (!parameter || !argument)
    return emitError() << "a type binding pairs a parameter with an argument";
  if (Type(getParameterOccurrence(parameter)) != parameter)
    return emitError() << "a type binding's key must be a type parameter, found "
                       << parameter;
  return success();
}

// Structural well-formedness of a witness: the predicate is one of the two arms
// and an impl is named. An equality predicate's own invariant -- it contains no
// proven claim -- is enforced when the `TypeEqualityAttr` is constructed. An
// application-armed witness carries no arguments: its impl is read at the
// application it names. Whether an equality-armed witness's keys are exactly
// the cited impl's parameters needs the impl, so it is checked where the
// witness is verified.
LogicalResult WitnessAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Attribute predicate, FlatSymbolRefAttr impl,
    ArrayRef<TypeBindingAttr> arguments) {
  if (!predicate)
    return emitError() << "a witness pairs a predicate with an impl";
  if (!isa<TraitApplicationAttr, TypeEqualityAttr>(predicate))
    return emitError() << "a witness predicate must be a trait application or "
                          "a type equality, found " << predicate;
  if (!impl)
    return emitError() << "a witness must name the impl that witnesses it";
  if (isa<TraitApplicationAttr>(predicate) && !arguments.empty())
    return emitError() << "an application witness names its impl alone; the "
                          "impl is read at the application it discharges";
  return success();
}

std::optional<std::pair<Attribute, WalkResult>> respellWitness(
    WitnessAttr witness, llvm::function_ref<Type(Type)> respell) {
  auto equality = dyn_cast<TypeEqualityAttr>(witness.getPredicate());
  if (!equality)
    return std::nullopt;
  MLIRContext *ctx = witness.getContext();
  auto rebuiltEquality = TypeEqualityAttr::getChecked(
      /*emitError=*/nullptr, ctx, respell(equality.getLhs()),
      respell(equality.getRhs()));
  if (!rebuiltEquality)
    return std::nullopt;
  SmallVector<TypeBindingAttr> arguments;
  for (TypeBindingAttr binding : witness.getArguments())
    arguments.push_back(TypeBindingAttr::get(ctx, binding.getParameter(),
                                             respell(binding.getArgument())));
  auto rebuilt = WitnessAttr::getChecked(/*emitError=*/nullptr, ctx,
                                         Attribute(rebuiltEquality),
                                         witness.getImplRef(),
                                         ArrayRef<TypeBindingAttr>(arguments));
  if (!rebuilt)
    return std::nullopt;
  return std::make_pair(Attribute(rebuilt), WalkResult::skip());
}

// Reach every symbol a witness names as a symbol reference, which no type walk
// reaches: the impl the witness cites, and the trait an application predicate
// names. An equality predicate names symbols only through the types in its
// endpoints, and those are ordinary sub-elements the framework's own type walk
// reaches wherever this attribute rides.
LogicalResult WitnessAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  Operation *impl = symbolTable.lookupNearestSymbolFrom(op, getImplRef());
  if (!isa_and_nonnull<ImplOp>(impl))
    return op->emitError() << "witness names '" << getImplRef()
                           << "', which does not resolve to an impl";

  if (auto app = dyn_cast<TraitApplicationAttr>(getPredicate()))
    return app.verifySymbolUses(op, symbolTable);
  return success();
}

ParseResult parseImplArguments(AsmParser &parser,
                               SmallVectorImpl<TypeBindingAttr> &arguments) {
  if (failed(parser.parseOptionalLSquare()))
    return success();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        Type parameter, argument;
        if (parser.parseType(parameter) || parser.parseEqual() ||
            parser.parseType(argument))
          return failure();
        auto err = [&]() { return parser.emitError(parser.getNameLoc()); };
        auto binding = TypeBindingAttr::getChecked(err, parser.getContext(),
                                                   parameter, argument);
        if (!binding)
          return failure();
        arguments.push_back(binding);
        return success();
      }))
    return failure();
  return parser.parseRSquare();
}

void printImplArguments(AsmPrinter &printer,
                        ArrayRef<TypeBindingAttr> arguments) {
  if (arguments.empty())
    return;
  printer << '[';
  llvm::interleaveComma(arguments, printer, [&](TypeBindingAttr binding) {
    printer << binding.getParameter() << " = " << binding.getArgument();
  });
  printer << ']';
}

Attribute WitnessAttr::parse(AsmParser &parser, Type) {
  FailureOr<Attribute> predicate = parseApplicationOrEqualityPredicate(parser);
  if (failed(predicate))
    return {};
  FlatSymbolRefAttr impl;
  SmallVector<TypeBindingAttr> arguments;
  if (parser.parseKeyword("by") || parser.parseAttribute(impl) ||
      parseImplArguments(parser, arguments))
    return {};
  auto err = [&]() { return parser.emitError(parser.getNameLoc()); };
  return WitnessAttr::getChecked(err, parser.getContext(), *predicate, impl,
                                 arguments);
}

void WitnessAttr::print(AsmPrinter &printer) const {
  if (auto app = dyn_cast<TraitApplicationAttr>(getPredicate()))
    app.print(printer);
  else
    cast<TypeEqualityAttr>(getPredicate()).print(printer);
  printer << " by " << getImplRef();
  printImplArguments(printer, getArguments());
}

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include <TraitAttributes.cpp.inc>
  >();
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

  return success();
}

// The single grammar for a trait application's `[!T1, !T2, ...]` body, shared by
// TraitApplicationAttr::parse and by the where-clause predicate parser; only the
// entry token that reads the leading symbol differs between them.
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

LogicalResult PredicateArrayAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Attribute> predicates) {
  for (Attribute p : predicates)
    if (!mlir::isa<TraitApplicationAttr, TypeEqualityAttr>(p))
      return emitError() << "a trait requirement must be a trait application "
                            "or a type equality";
  return success();
}

// Verify each predicate. An application entry names a trait as a symbol
// reference, which no type walk reaches, so it is checked here. An equality
// entry names symbols only through the types in its endpoints, and the
// framework's own type walk over the owning operation's attributes reaches
// those, so there is nothing left for this entry point to check.
LogicalResult PredicateArrayAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  for (Attribute p : getPredicates())
    if (auto app = mlir::dyn_cast<TraitApplicationAttr>(p))
      if (failed(app.verifySymbolUses(op, symbolTable)))
        return failure();
  return success();
}

Attribute PredicateArrayAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto errFn = [&]{ return p.emitError(p.getCurrentLocation()); };

  SmallVector<Attribute> preds;

  if (p.parseLSquare())
    return {};
  if (succeeded(p.parseOptionalRSquare()))
    return PredicateArrayAttr::getChecked(errFn, ctx, preds);

  // Each entry is an application (`@Trait[...]`) or an equality (`!A = !B`).
  do {
    FailureOr<Attribute> pred = parseApplicationOrEqualityPredicate(p);
    if (failed(pred))
      return {};
    preds.push_back(*pred);
  } while (succeeded(p.parseOptionalComma()));

  if (p.parseRSquare())
    return {};

  return PredicateArrayAttr::getChecked(errFn, ctx, preds);
}

void PredicateArrayAttr::print(mlir::AsmPrinter &printer) const {
  printer << "[";
  llvm::interleaveComma(getPredicates(), printer, [&](Attribute p) {
    if (auto app = mlir::dyn_cast<TraitApplicationAttr>(p))
      app.print(printer);
    else
      mlir::cast<TypeEqualityAttr>(p).print(printer);
  });
  printer << ']';
}

} // end mlir::trait
