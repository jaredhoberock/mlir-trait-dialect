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

// The parameters a predicate spells, each counted once. Claims carry their
// type arguments in an attribute, so the predicate is read through the claim
// that states it, whose reader descends into them.
static SmallVector<GenericTypeInterface, 4>
parametersSpelledBy(Attribute predicate) {
  return getTypeParametersIn(
      Type(ClaimType::get(predicate.getContext(), predicate, nullptr)));
}

LogicalResult BoundPredicateAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> parameters, ArrayRef<Attribute> premises,
    Attribute conclusion) {
  if (parameters.empty())
    return emitError() << "a bound predicate binds at least one parameter";
  DenseSet<Type> bound;
  for (Type parameter : parameters) {
    if (!parameter || Type(getParameterOccurrence(parameter)) != parameter)
      return emitError() << "a bound predicate binds type parameters, found "
                         << parameter;
    if (!bound.insert(parameter).second)
      return emitError() << "a bound predicate binds " << parameter << " twice";
  }
  for (Attribute premise : premises)
    if (!isa_and_nonnull<TraitApplicationAttr, TypeEqualityAttr>(premise))
      return emitError() << "a bound predicate's premise must be a trait "
                            "application or a type equality";
  if (!isa_and_nonnull<TraitApplicationAttr, TypeEqualityAttr>(conclusion))
    return emitError() << "a bound predicate's conclusion must be a trait "
                          "application or a type equality";
  if (llvm::none_of(parametersSpelledBy(conclusion),
                    [&](GenericTypeInterface spelled) {
                      return bound.contains(Type(spelled));
                    }))
    return emitError() << "a bound predicate's conclusion spells none of the "
                          "parameters it binds";
  return success();
}

SpecializationMap
BoundPredicateAttr::bindingFor(ArrayRef<Type> arguments) const {
  assert(arguments.size() == getParameters().size() &&
         "one argument per parameter the binder introduces");
  SpecializationMap binding;
  for (auto [parameter, argument] : llvm::zip(getParameters(), arguments))
    binding.bind(getParameterOccurrence(parameter), argument);
  return binding;
}

// A bound predicate's premises and conclusion name traits as symbol references,
// which no type walk reaches, so they are checked here as a where clause's
// application entries are.
LogicalResult BoundPredicateAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  for (Attribute premise : getPremises())
    if (auto app = dyn_cast<TraitApplicationAttr>(premise))
      if (failed(app.verifySymbolUses(op, symbolTable)))
        return failure();
  if (auto app = dyn_cast<TraitApplicationAttr>(getConclusion()))
    return app.verifySymbolUses(op, symbolTable);
  return success();
}

// `forall [params] where [premises] -> conclusion`, following the keyword the
// caller has already read.
static FailureOr<BoundPredicateAttr> parseBoundPredicateBody(AsmParser &p) {
  SmallVector<Type> parameters;
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        Type parameter;
        if (p.parseType(parameter))
          return failure();
        parameters.push_back(parameter);
        return success();
      }))
    return failure();

  SmallVector<Attribute> premises;
  if (succeeded(p.parseOptionalKeyword("where")) &&
      p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        FailureOr<Attribute> premise = parseApplicationOrEqualityPredicate(p);
        if (failed(premise))
          return failure();
        premises.push_back(*premise);
        return success();
      }))
    return failure();

  if (p.parseArrow())
    return failure();
  FailureOr<Attribute> conclusion = parseApplicationOrEqualityPredicate(p);
  if (failed(conclusion))
    return failure();

  auto err = [&]() { return p.emitError(p.getCurrentLocation()); };
  auto bound = BoundPredicateAttr::getChecked(err, p.getContext(), parameters,
                                              premises, *conclusion);
  if (!bound)
    return failure();
  return bound;
}

static void printApplicationOrEquality(AsmPrinter &printer, Attribute p) {
  if (auto app = dyn_cast<TraitApplicationAttr>(p))
    app.print(printer);
  else
    cast<TypeEqualityAttr>(p).print(printer);
}

static void printBoundPredicateBody(AsmPrinter &printer,
                                    BoundPredicateAttr bound) {
  printer << "forall [";
  llvm::interleaveComma(bound.getParameters(), printer);
  printer << "]";
  if (!bound.getPremises().empty()) {
    printer << " where [";
    llvm::interleaveComma(bound.getPremises(), printer, [&](Attribute p) {
      printApplicationOrEquality(printer, p);
    });
    printer << "]";
  }
  printer << " -> ";
  printApplicationOrEquality(printer, bound.getConclusion());
}

Attribute BoundPredicateAttr::parse(AsmParser &p, Type) {
  if (p.parseKeyword("forall"))
    return {};
  FailureOr<BoundPredicateAttr> bound = parseBoundPredicateBody(p);
  if (failed(bound))
    return {};
  return *bound;
}

void BoundPredicateAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printBoundPredicateBody(printer, *this);
}

LogicalResult BoundBodyAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    std::optional<unsigned> premise, std::optional<unsigned> whereEntry,
    bool refl, FlatSymbolRefAttr impl, ArrayRef<TypeBindingAttr> arguments,
    ArrayRef<BoundBodyAttr> discharges) {
  unsigned forms = premise.has_value() + whereEntry.has_value() + refl +
                   static_cast<bool>(impl);
  if (forms != 1)
    return emitError() << "evidence under a binder is exactly one of a "
                          "premise, a where-clause entry, reflexivity, or an "
                          "impl";
  if (!impl && (!arguments.empty() || !discharges.empty()))
    return emitError() << "only evidence citing an impl carries arguments and "
                          "discharges";
  return success();
}

// An impl the body cites, at any depth, is a symbol reference no type walk
// reaches, so it is checked here.
LogicalResult BoundBodyAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  if (FlatSymbolRefAttr impl = getImplRef()) {
    Operation *cited = symbolTable.lookupNearestSymbolFrom(op, impl);
    if (!isa_and_nonnull<ImplOp>(cited))
      return op->emitError() << "evidence names '" << impl
                             << "', which does not resolve to an impl";
  }
  for (BoundBodyAttr discharge : getDischarges())
    if (failed(discharge.verifySymbolUses(op, symbolTable)))
      return failure();
  return success();
}

Attribute BoundBodyAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto err = [&]() { return p.emitError(p.getCurrentLocation()); };
  auto leaf = [&](std::optional<unsigned> premise,
                  std::optional<unsigned> whereEntry, bool refl) {
    return BoundBodyAttr::getChecked(err, ctx, premise, whereEntry, refl,
                                     FlatSymbolRefAttr(), {}, {});
  };

  unsigned position;
  if (succeeded(p.parseOptionalKeyword("premise"))) {
    if (p.parseInteger(position))
      return {};
    return leaf(position, std::nullopt, false);
  }
  if (succeeded(p.parseOptionalKeyword("where"))) {
    if (p.parseInteger(position))
      return {};
    return leaf(std::nullopt, position, false);
  }
  if (succeeded(p.parseOptionalKeyword("refl")))
    return leaf(std::nullopt, std::nullopt, true);

  FlatSymbolRefAttr impl;
  SmallVector<TypeBindingAttr> arguments;
  if (p.parseAttribute(impl) || parseImplArguments(p, arguments))
    return {};
  SmallVector<BoundBodyAttr> discharges;
  if (succeeded(p.parseOptionalKeyword("given")) &&
      p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        auto discharge =
            dyn_cast_or_null<BoundBodyAttr>(BoundBodyAttr::parse(p, Type()));
        if (!discharge)
          return failure();
        discharges.push_back(discharge);
        return success();
      }))
    return {};
  return BoundBodyAttr::getChecked(err, ctx, std::nullopt, std::nullopt, false,
                                   impl, arguments, discharges);
}

// A body as `BoundBodyAttr::parse` reads it, with no leading space: the form an
// enclosing attribute prints it in.
static void printBoundBody(AsmPrinter &printer, BoundBodyAttr body) {
  if (std::optional<unsigned> premise = body.getPremise()) {
    printer << "premise " << *premise;
    return;
  }
  if (std::optional<unsigned> whereEntry = body.getWhereEntry()) {
    printer << "where " << *whereEntry;
    return;
  }
  if (body.getRefl()) {
    printer << "refl";
    return;
  }
  printer << body.getImplRef();
  printImplArguments(printer, body.getArguments());
  if (body.getDischarges().empty())
    return;
  printer << " given [";
  llvm::interleaveComma(body.getDischarges(), printer,
                        [&](BoundBodyAttr discharge) {
                          printBoundBody(printer, discharge);
                        });
  printer << "]";
}

void BoundBodyAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printBoundBody(printer, *this);
}

LogicalResult BoundEvidenceAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  if (failed(getPredicate().verifySymbolUses(op, symbolTable)))
    return failure();
  return getBody().verifySymbolUses(op, symbolTable);
}

Attribute BoundEvidenceAttr::parse(AsmParser &p, Type) {
  unsigned requirement;
  if (p.parseInteger(requirement) || p.parseColon() ||
      p.parseKeyword("forall"))
    return {};
  FailureOr<BoundPredicateAttr> predicate = parseBoundPredicateBody(p);
  if (failed(predicate) || p.parseKeyword("by"))
    return {};
  auto body = dyn_cast_or_null<BoundBodyAttr>(BoundBodyAttr::parse(p, Type()));
  if (!body)
    return {};
  return BoundEvidenceAttr::get(p.getContext(), requirement, *predicate, body);
}

void BoundEvidenceAttr::print(AsmPrinter &printer) const {
  printer << ' ' << getRequirement() << ": ";
  printBoundPredicateBody(printer, getPredicate());
  printer << " by ";
  printBoundBody(printer, getBody());
}

FailureOr<Attribute> parseWherePredicate(AsmParser &p) {
  if (succeeded(p.parseOptionalKeyword("forall"))) {
    FailureOr<BoundPredicateAttr> bound = parseBoundPredicateBody(p);
    if (failed(bound))
      return failure();
    return Attribute(*bound);
  }
  return parseApplicationOrEqualityPredicate(p);
}

void printWherePredicate(AsmPrinter &printer, Attribute predicate) {
  if (auto bound = dyn_cast<BoundPredicateAttr>(predicate))
    printBoundPredicateBody(printer, bound);
  else
    printApplicationOrEquality(printer, predicate);
}

LogicalResult PredicateArrayAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Attribute> predicates) {
  for (Attribute p : predicates)
    if (!mlir::isa<TraitApplicationAttr, TypeEqualityAttr, BoundPredicateAttr>(
            p))
      return emitError() << "a trait requirement must be a trait application, "
                            "a type equality, or a bound predicate";
  return success();
}

// Verify each predicate. An application entry, and every application a bound
// entry states, names a trait as a symbol reference, which no type walk
// reaches, so it is checked here. An equality entry names symbols only through
// the types in its endpoints, and the framework's own type walk over the owning
// operation's attributes reaches those, so there is nothing left for this entry
// point to check.
LogicalResult PredicateArrayAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  for (Attribute p : getPredicates()) {
    if (auto app = mlir::dyn_cast<TraitApplicationAttr>(p))
      if (failed(app.verifySymbolUses(op, symbolTable)))
        return failure();
    if (auto bound = mlir::dyn_cast<BoundPredicateAttr>(p))
      if (failed(bound.verifySymbolUses(op, symbolTable)))
        return failure();
  }
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

  // Each entry is an application (`@Trait[...]`), an equality (`!A = !B`), or
  // a bound predicate (`forall [...] ... -> ...`).
  do {
    FailureOr<Attribute> pred = parseWherePredicate(p);
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
    printWherePredicate(printer, p);
  });
  printer << ']';
}

} // end mlir::trait
