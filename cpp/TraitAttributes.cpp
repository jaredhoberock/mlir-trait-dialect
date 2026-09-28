// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <TraitAttrInterfaces.cpp.inc>

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

// Whether `body` is the body of a bound requirement's witness: an impl
// citation whose discharges are such bodies in turn, a requirement hop whose
// body and premises are such bodies in turn, a binder premise, an impl
// premise, reflexivity or an allegation.
static LogicalResult verifyWitnessBody(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute body) {
  if (auto citation = dyn_cast_or_null<ImplCitationAttr>(body)) {
    if (!citation.getImplRef())
      return emitError() << "a witness body citing an impl names it";
    for (Attribute discharge : citation.getDischarges())
      if (failed(verifyWitnessBody(emitError, discharge)))
        return failure();
    return success();
  }
  if (auto hop = dyn_cast_or_null<RequirementHopAttr>(body)) {
    if (failed(verifyWitnessBody(emitError, hop.getOf())))
      return failure();
    for (Attribute premise : hop.getPremises())
      if (failed(verifyWitnessBody(emitError, premise)))
        return failure();
    return success();
  }
  if (auto allegation = dyn_cast_or_null<AllegationAttr>(body)) {
    if (!allegation.getApplication())
      return emitError() << "an allegation states a trait application";
    return success();
  }
  if (!isa_and_nonnull<BinderPremiseAttr, ImplPremiseAttr, UnitAttr>(body))
    return emitError() << "a witness body is an impl citation, a requirement "
                          "hop, a binder premise, an impl premise, reflexivity "
                          "or an allegation, found "
                       << body;
  return success();
}

// Structural well-formedness of a witness: the predicate is one of the three
// arms and the body fit for it. A bound requirement's body is any of the four
// arms. An application or equality witness cites the impl that witnesses it and
// discharges nothing: those arms read the cited impl's premises where they are
// verified. An equality predicate's own invariant -- it contains no proven
// claim -- is enforced when the `TypeEqualityAttr` is constructed. An
// application-armed witness's citation carries no arguments: its impl is read
// at the application it names. Whether a citation's keys are exactly the cited
// impl's parameters needs the impl, so it is checked where the witness is
// verified.
LogicalResult WitnessAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute predicate,
    Attribute body) {
  if (auto position = dyn_cast_or_null<IntegerAttr>(predicate)) {
    if (position.getInt() < 0)
      return emitError() << "a requirement position is non-negative";
    return verifyWitnessBody(emitError, body);
  }
  if (!isa_and_nonnull<TraitApplicationAttr, TypeEqualityAttr>(predicate))
    return emitError() << "a witness predicate must be a trait application, "
                          "a type equality or a requirement position, found "
                       << predicate;
  auto citation = dyn_cast_or_null<ImplCitationAttr>(body);
  if (!citation || !citation.getImplRef())
    return emitError() << "a witness of an application or an equality cites "
                          "the impl that witnesses it";
  if (!citation.getDischarges().empty())
    return emitError() << "only a bound requirement's witness discharges the "
                          "premises of the impl it cites";
  if (isa<TraitApplicationAttr>(predicate) && !citation.getArguments().empty())
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

// Reach every symbol a witness body names, as symbol references no type walk
// reaches: the impl a citation names, the trait an allegation names, and the
// symbols of every body a citation or a requirement hop holds in turn.
static LogicalResult verifyBodySymbolUses(Attribute body, Operation *op,
                                          SymbolTableCollection &symbolTable) {
  if (auto citation = dyn_cast<ImplCitationAttr>(body))
    return citation.verifySymbolUses(op, symbolTable);
  if (auto hop = dyn_cast<RequirementHopAttr>(body)) {
    if (failed(verifyBodySymbolUses(hop.getOf(), op, symbolTable)))
      return failure();
    for (Attribute premise : hop.getPremises())
      if (failed(verifyBodySymbolUses(premise, op, symbolTable)))
        return failure();
    return success();
  }
  if (auto allegation = dyn_cast<AllegationAttr>(body))
    return allegation.getApplication().verifySymbolUses(op, symbolTable);
  return success();
}

// Reach every impl a citation names, and the symbols its discharges name in
// turn, as symbol references no type walk reaches.
LogicalResult ImplCitationAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  Operation *impl = symbolTable.lookupNearestSymbolFrom(op, getImplRef());
  if (!isa_and_nonnull<ImplOp>(impl))
    return op->emitError() << "witness names '" << getImplRef()
                           << "', which does not resolve to an impl";
  for (Attribute discharge : getDischarges())
    if (failed(verifyBodySymbolUses(discharge, op, symbolTable)))
      return failure();
  return success();
}

// Reach every symbol a witness names as a symbol reference, which no type walk
// reaches: the impls its body cites, and the trait an application predicate
// names. An equality predicate names symbols only through the types in its
// endpoints, and those are ordinary sub-elements the framework's own type walk
// reaches wherever this attribute rides.
LogicalResult WitnessAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(op, symbolTable);

  if (failed(verifyBodySymbolUses(getBody(), op, symbolTable)))
    return failure();
  if (auto app = dyn_cast<TraitApplicationAttr>(getPredicate()))
    return app.verifySymbolUses(op, symbolTable);
  return success();
}

FailureOr<bool> parseImplArguments(AsmParser &parser,
                                   SmallVectorImpl<TypeBindingAttr> &arguments) {
  if (failed(parser.parseOptionalLSquare()))
    return false;
  if (succeeded(parser.parseOptionalRSquare()))
    return true;
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
      }) ||
      parser.parseRSquare())
    return failure();
  return true;
}

void printImplArguments(AsmPrinter &printer,
                        ArrayRef<TypeBindingAttr> arguments) {
  printer << '[';
  llvm::interleaveComma(arguments, printer, [&](TypeBindingAttr binding) {
    printer << binding.getParameter() << " = " << binding.getArgument();
  });
  printer << ']';
}

/// Parse a comma-separated, bracketed list of witness bodies.
static ParseResult parseWitnessBodyList(AsmParser &parser,
                                        SmallVectorImpl<Attribute> &bodies);

/// Parse a witness body: `@impl[!P = T, ...] given [body, ...]`, `premise N`,
/// `where N`, `refl`, `requirement N for [types] given [body, ...] of body`
/// or `allege @Trait[...]`. Every arm is read here and nowhere else.
static FailureOr<Attribute> parseWitnessBody(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  unsigned position;
  if (succeeded(parser.parseOptionalKeyword("requirement"))) {
    if (parser.parseInteger(position))
      return failure();
    SmallVector<Type> typeArgs;
    if (succeeded(parser.parseOptionalKeyword("for")) &&
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
          Type type;
          if (parser.parseType(type))
            return failure();
          typeArgs.push_back(type);
          return success();
        }))
      return failure();
    SmallVector<Attribute> premises;
    if (succeeded(parser.parseOptionalKeyword("given")) &&
        parseWitnessBodyList(parser, premises))
      return failure();
    if (parser.parseKeyword("of"))
      return failure();
    FailureOr<Attribute> of = parseWitnessBody(parser);
    if (failed(of))
      return failure();
    return Attribute(RequirementHopAttr::get(ctx, position, *of, typeArgs, premises));
  }
  if (succeeded(parser.parseOptionalKeyword("allege"))) {
    auto application =
        dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(parser, {}));
    if (!application)
      return failure();
    RuleAttrInterface rule;
    if (succeeded(parser.parseOptionalKeyword("by"))) {
      Attribute named;
      llvm::SMLoc loc = parser.getCurrentLocation();
      if (parser.parseAttribute(named))
        return failure();
      rule = dyn_cast<RuleAttrInterface>(named);
      if (!rule)
        return parser.emitError(loc)
               << named << " names no impl rule: a rule is an attribute "
                           "implementing RuleAttrInterface";
    }
    return Attribute(AllegationAttr::get(ctx, application, rule));
  }
  if (succeeded(parser.parseOptionalKeyword("premise"))) {
    if (parser.parseInteger(position))
      return failure();
    return Attribute(BinderPremiseAttr::get(ctx, position));
  }
  if (succeeded(parser.parseOptionalKeyword("where"))) {
    if (parser.parseInteger(position))
      return failure();
    return Attribute(ImplPremiseAttr::get(ctx, position));
  }
  if (succeeded(parser.parseOptionalKeyword("refl")))
    return Attribute(UnitAttr::get(ctx));

  FlatSymbolRefAttr impl;
  SmallVector<TypeBindingAttr> arguments;
  if (parser.parseAttribute(impl) ||
      failed(parseImplArguments(parser, arguments)))
    return failure();
  SmallVector<Attribute> discharges;
  if (succeeded(parser.parseOptionalKeyword("given")) &&
      parseWitnessBodyList(parser, discharges))
    return failure();
  return Attribute(ImplCitationAttr::get(ctx, impl, arguments, discharges));
}

static ParseResult parseWitnessBodyList(AsmParser &parser,
                                        SmallVectorImpl<Attribute> &bodies) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
    FailureOr<Attribute> body = parseWitnessBody(parser);
    if (failed(body))
      return failure();
    bodies.push_back(*body);
    return success();
  });
}

/// Print a witness body as `parseWitnessBody` reads it.
static void printWitnessBody(AsmPrinter &printer, Attribute body) {
  if (auto hop = dyn_cast<RequirementHopAttr>(body)) {
    printer << "requirement " << hop.getPosition();
    if (!hop.getTypeArgs().empty()) {
      printer << " for [";
      llvm::interleaveComma(hop.getTypeArgs(), printer);
      printer << "]";
    }
    if (!hop.getPremises().empty()) {
      printer << " given [";
      llvm::interleaveComma(hop.getPremises(), printer, [&](Attribute premise) {
        printWitnessBody(printer, premise);
      });
      printer << "]";
    }
    printer << " of ";
    printWitnessBody(printer, hop.getOf());
    return;
  }
  if (auto allegation = dyn_cast<AllegationAttr>(body)) {
    printer << "allege ";
    allegation.getApplication().print(printer);
    if (RuleAttrInterface rule = allegation.getRule())
      printer << " by " << Attribute(rule);
    return;
  }
  if (auto premise = dyn_cast<BinderPremiseAttr>(body)) {
    printer << "premise " << premise.getPosition();
    return;
  }
  if (auto premise = dyn_cast<ImplPremiseAttr>(body)) {
    printer << "where " << premise.getPosition();
    return;
  }
  if (isa<UnitAttr>(body)) {
    printer << "refl";
    return;
  }
  auto citation = cast<ImplCitationAttr>(body);
  printer << citation.getImplRef();
  if (!citation.getArguments().empty())
    printImplArguments(printer, citation.getArguments());
  if (citation.getDischarges().empty())
    return;
  printer << " given [";
  llvm::interleaveComma(citation.getDischarges(), printer,
                        [&](Attribute discharge) {
                          printWitnessBody(printer, discharge);
                        });
  printer << "]";
}

/// Parse a body arm standing alone, as `T` prints it.
template <typename T>
static Attribute parseWitnessBodyArm(AsmParser &parser) {
  FailureOr<Attribute> body = parseWitnessBody(parser);
  if (failed(body))
    return {};
  if (!isa<T>(*body)) {
    parser.emitError(parser.getNameLoc(), "expected another witness body");
    return {};
  }
  return *body;
}

Attribute ImplCitationAttr::parse(AsmParser &parser, Type) {
  return parseWitnessBodyArm<ImplCitationAttr>(parser);
}
void ImplCitationAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printWitnessBody(printer, *this);
}
Attribute BinderPremiseAttr::parse(AsmParser &parser, Type) {
  return parseWitnessBodyArm<BinderPremiseAttr>(parser);
}
void BinderPremiseAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printWitnessBody(printer, *this);
}
Attribute ImplPremiseAttr::parse(AsmParser &parser, Type) {
  return parseWitnessBodyArm<ImplPremiseAttr>(parser);
}
void ImplPremiseAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printWitnessBody(printer, *this);
}
Attribute RequirementHopAttr::parse(AsmParser &parser, Type) {
  return parseWitnessBodyArm<RequirementHopAttr>(parser);
}
void RequirementHopAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printWitnessBody(printer, *this);
}
Attribute AllegationAttr::parse(AsmParser &parser, Type) {
  return parseWitnessBodyArm<AllegationAttr>(parser);
}
void AllegationAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  printWitnessBody(printer, *this);
}

Attribute WitnessAttr::parse(AsmParser &parser, Type) {
  MLIRContext *ctx = parser.getContext();
  Attribute predicate;
  if (succeeded(parser.parseOptionalKeyword("requirement"))) {
    int64_t position;
    if (parser.parseInteger(position))
      return {};
    predicate = IntegerAttr::get(IntegerType::get(ctx, 64), position);
  } else {
    FailureOr<Attribute> read = parseApplicationOrEqualityPredicate(parser);
    if (failed(read))
      return {};
    predicate = *read;
  }
  if (parser.parseKeyword("by"))
    return {};
  FailureOr<Attribute> body = parseWitnessBody(parser);
  if (failed(body))
    return {};
  auto err = [&]() { return parser.emitError(parser.getNameLoc()); };
  return WitnessAttr::getChecked(err, ctx, predicate, *body);
}

void WitnessAttr::print(AsmPrinter &printer) const {
  printer << ' ';
  if (std::optional<unsigned> position = getRequirement())
    printer << "requirement " << *position;
  else if (auto app = dyn_cast<TraitApplicationAttr>(getPredicate()))
    app.print(printer);
  else
    cast<TypeEqualityAttr>(getPredicate()).print(printer);
  printer << " by ";
  printWitnessBody(printer, getBody());
}

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include <TraitAttributes.cpp.inc>
  >();
}

// The generic parser hands a dialect the text between `#trait<` and `>` and
// reads nothing of what the dialect leaves unread, and a witness body ends
// where its last arm ends: `requirement 0 of where 0 given [where 0]` would
// read as `requirement 0 of where 0`, its suffix silently dropped. So every
// attribute of this dialect reads its whole text or is refused.
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

// The binder variables a predicate spells. Claims carry their type arguments in
// an attribute, so the predicate is read through the claim that states it,
// whose reader descends into them.
static SmallVector<BoundVarType> binderVariablesSpelledBy(Attribute predicate) {
  SmallVector<BoundVarType> variables;
  for (GenericTypeInterface generic : getGenericTypesIn(
           Type(ClaimType::get(predicate.getContext(), predicate, nullptr))))
    if (auto variable = dyn_cast<BoundVarType>(Type(generic)))
      variables.push_back(variable);
  return variables;
}

LogicalResult BoundPredicateAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, unsigned arity,
    ArrayRef<Attribute> premises, Attribute conclusion) {
  if (arity == 0)
    return emitError() << "a bound predicate binds at least one variable";
  SmallVector<Attribute> predicates(premises.begin(), premises.end());
  predicates.push_back(conclusion);
  for (Attribute predicate : predicates) {
    if (!isa_and_nonnull<TraitApplicationAttr, TypeEqualityAttr>(predicate))
      return emitError() << "a bound predicate's premises and conclusion are "
                            "trait applications or type equalities";
    for (BoundVarType variable : binderVariablesSpelledBy(predicate))
      if (variable.getPosition() >= arity)
        return emitError() << "a bound predicate binds " << arity
                           << " variables, and " << predicate << " spells "
                           << Type(variable);
  }
  if (binderVariablesSpelledBy(conclusion).empty())
    return emitError() << "a bound predicate's conclusion spells none of the "
                          "variables it binds";
  return success();
}

SpecializationMap
BoundPredicateAttr::bindingFor(ArrayRef<Type> arguments) const {
  assert(arguments.size() == getArity() &&
         "one argument per variable the binder introduces");
  SpecializationMap binding;
  for (auto [position, argument] : llvm::enumerate(arguments))
    binding.bind(cast<GenericTypeInterface>(Type(BoundVarType::get(
                     getContext(), static_cast<unsigned>(position)))),
                 argument);
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

// `forall [variables] where [premises] -> conclusion`, following the keyword the
// caller has already read.
static FailureOr<BoundPredicateAttr> parseBoundPredicateBody(AsmParser &p) {
  // The variables, listed in position order.
  unsigned arity = 0;
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                [&]() -> ParseResult {
        llvm::SMLoc location = p.getCurrentLocation();
        Type variable;
        if (p.parseType(variable))
          return failure();
        auto bound = dyn_cast<BoundVarType>(variable);
        if (!bound || bound.getPosition() != arity)
          return p.emitError(location)
                 << "a bound predicate lists its variables in position order: "
                    "expected !trait.bound<"
                 << arity << ">, found " << variable;
        ++arity;
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
  auto bound = BoundPredicateAttr::getChecked(err, p.getContext(), arity,
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
  llvm::interleaveComma(llvm::seq(bound.getArity()), printer, [&](unsigned position) {
    printer << Type(BoundVarType::get(bound.getContext(), position));
  });
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
  for (Attribute p : predicates) {
    if (isa<BoundPredicateAttr>(p))
      continue;
    if (!mlir::isa<TraitApplicationAttr, TypeEqualityAttr>(p))
      return emitError() << "a trait requirement must be a trait application, "
                            "a type equality, or a bound predicate";
    // A binder variable stands only inside its binder.
    if (!binderVariablesSpelledBy(p).empty())
      return emitError() << p << " spells a binder variable outside a bound "
                                 "predicate";
  }
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
