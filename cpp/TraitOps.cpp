// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Passes.hpp"
#include "Specialization.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/Support/xxhash.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/RegionUtils.h>
#include <optional>
#include <set>
#include <variant>

namespace mlir::trait {
/// Parse and print the optional visibility keyword a template op leads with,
/// so `trait.proof private @p ...` reads and prints as `func.func` does: the
/// keyword is printed exactly when the attribute is present.
static ::mlir::ParseResult parseVisibilityKeyword(::mlir::OpAsmParser &parser,
                                                  ::mlir::StringAttr &visibility) {
  ::mlir::NamedAttrList attrs;
  // A missing keyword is not an error: the op is public. The helper reports
  // failure when it finds none and consumes nothing, so the outcome is ignored.
  (void)::mlir::impl::parseOptionalVisibilityKeyword(parser, attrs);
  visibility = ::llvm::dyn_cast_or_null<::mlir::StringAttr>(
      attrs.get("sym_visibility"));
  return ::mlir::success();
}

static void printVisibilityKeyword(::mlir::OpAsmPrinter &printer,
                                   ::mlir::Operation *, ::mlir::StringAttr visibility) {
  if (visibility)
    printer << visibility.getValue() << ' ';
}

/// The unproven claim an op names by spelling the predicate it states: an
/// application `@Trait[...]` or an equality `!A = !B`, read as its unproven
/// claim and printed as that predicate. An op whose result admits one arm only
/// refuses the other through its result type's constraint.
static ::mlir::ParseResult parseClaimPredicate(::mlir::OpAsmParser &parser,
                                               ::mlir::Type &claim) {
  ::mlir::FailureOr<::mlir::Attribute> predicate =
      parseApplicationOrEqualityPredicate(parser);
  if (::mlir::failed(predicate))
    return ::mlir::failure();
  claim = ClaimType::get(parser.getContext(), *predicate, /*proof=*/nullptr);
  return ::mlir::success();
}

static void printClaimPredicate(::mlir::OpAsmPrinter &printer,
                                ::mlir::Operation *, ::mlir::Type claim) {
  auto stated = ::llvm::cast<ClaimType>(claim);
  if (auto equality = stated.getEqualityAttr())
    printer << equality.getLhs() << " = " << equality.getRhs();
  else
    stated.getTraitApplication().print(printer);
}

/// The arguments a citation states for its impl's parameters, `[T0, T1]`
/// after the impl's symbol, or nothing for an impl taking none.
static ::mlir::ParseResult parseImplArguments(::mlir::OpAsmParser &parser,
                                              ImplArgumentsAttr &arguments) {
  ::llvm::SmallVector<::mlir::Type> types;
  if (parser.parseCommaSeparatedList(
          ::mlir::OpAsmParser::Delimiter::OptionalSquare,
          [&]() -> ::mlir::ParseResult {
            ::mlir::Type type;
            if (parser.parseType(type))
              return ::mlir::failure();
            types.push_back(type);
            return ::mlir::success();
          }))
    return ::mlir::failure();
  arguments = ImplArgumentsAttr::get(parser.getContext(), types);
  return ::mlir::success();
}

static void printImplArguments(::mlir::OpAsmPrinter &printer,
                               ::mlir::Operation *, ImplArgumentsAttr arguments) {
  if (arguments.getTypes().empty())
    return;
  printer << "[";
  ::llvm::interleaveComma(arguments.getTypes(), printer);
  printer << "]";
}
} // namespace mlir::trait

#define GET_OP_CLASSES
#include <TraitOps.cpp.inc>

namespace mlir::trait {

void AllegeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       ObligationResource::get());
}

void ProjectOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       ObligationResource::get());
}

} // end mlir::trait

using namespace mlir;
using namespace mlir::trait;

namespace {

/// Names a declaration's block arguments after the facts they hold: `self`
/// for its own application, a premise after its trait, and an equality after
/// the associated type it resolves. The names are the printer's; nothing
/// stores them.
void nameDeclarationArguments(Region &body, OpAsmSetValueNameFn setNameFn) {
  if (body.empty())
    return;
  for (BlockArgument argument : body.front().getArguments()) {
    if (argument.getArgNumber() == 0) {
      setNameFn(argument, "self");
      continue;
    }
    auto claim = dyn_cast<ClaimType>(argument.getType());
    if (!claim)
      continue;
    std::string name;
    if (TypeEqualityAttr equality = claim.getEqualityAttr()) {
      auto projection = dyn_cast<ProjectionType>(equality.getLhs());
      name = projection ? projection.getAssocName().getValue().lower() : "eq";
    } else {
      // A trait's symbol may carry a dotted prefix; the name is its last
      // component.
      StringRef trait = claim.getTraitApplication().getTraitName().getValue();
      name = trait.substr(trait.rfind('.') + 1).lower();
    }
    setNameFn(argument, name);
  }
}

/// Verifies that a function's result generics are determined by its inputs.
///
/// Generics supplied by the caller are treated as already determined. Every
/// other generic in a result must also appear in an input type, including
/// claim inputs that encode ordinary where-clause evidence and a trait
/// member's leading self claim, whose arguments are the trait's parameters.
/// Otherwise function monomorphization has no source of evidence for choosing
/// that result type. A claim result is the one exception: a call of the
/// function states its result types, and a claim it returns is evidence the
/// stage proves where the call is monomorphic rather than an instance it cuts,
/// so a generic only a claim result states is determined by the call that
/// states it. This is intentionally a syntactic check: the verifier does not
/// try to invert equality predicates or associated-type bindings to recover
/// missing result generics.
static LogicalResult verifyFunctionResultGenericsAreDetermined(
    FunctionOpInterface function, FunctionType functionType,
    const DenseSet<Type> &providedGenerics) {
  DenseSet<Type> inputGenerics;
  for (Type input : functionType.getInputs()) {
    auto generics = getGenericTypesIn(input);
    inputGenerics.insert(generics.begin(), generics.end());
  }

  DenseSet<Type> seenResultGenerics;
  SmallVector<GenericTypeInterface, 4> resultGenerics;
  for (Type result : functionType.getResults()) {
    if (isa<ClaimType>(result))
      continue;
    for (auto generic : getGenericTypesIn(result)) {
      if (seenResultGenerics.insert(generic).second)
        resultGenerics.push_back(generic);
    }
  }

  for (Type resultGeneric : resultGenerics) {
    if (providedGenerics.contains(resultGeneric) || inputGenerics.contains(resultGeneric))
      continue;

    return function.emitOpError()
           << "function '" << function.getName()
           << "' result type contains type parameter " << resultGeneric
           << " that is not determined by any input type";
  }

  return success();
}

/// Whether `value`, computed in a declaration's body, rests on `self`: whether
/// it is `self` or an op defining it reads `self`, directly or through its own
/// operands. The body is a dominance region, so the walk meets no cycle.
bool restsOn(Value value, Value self) {
  SmallVector<Value> pending{value};
  DenseSet<Operation *> seen;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (current == self)
      return true;
    Operation *producer = current.getDefiningOp();
    if (producer && seen.insert(producer).second)
      llvm::append_range(pending, producer->getOperands());
  }
  return false;
}

/// Whether `value`, computed in `impl`'s body, rests on a projection of `impl`'s
/// own application named by symbol: a witness or a derive of `impl` at its
/// self application, whose requirement the projection reads. Such evidence is
/// read back through the very return it stands in, so it has no base case
/// (GHC's rule for instance superclasses); a derive or witness of the impl
/// that nothing projects is a constructor and stays legal.
bool projectsOwnApplication(Value value, ImplOp impl,
                            SymbolTableCollection &symbolTable) {
  TraitApplicationAttr own = impl.getSelfApplication();
  auto namesOwnApplication = [&](Operation *op) {
    if (auto derive = dyn_cast<DeriveOp>(op))
      return derive.getImpl() == impl.getSymName() &&
             derive.getTraitApplication() == own;
    auto witness = dyn_cast<WitnessOp>(op);
    if (!witness || !witness.getProof())
      return false;
    if (witness.getProof().getValue() == impl.getSymName())
      return true;
    auto proof = symbolTable.lookupNearestSymbolFrom<ProofOp>(
        impl, witness.getProof());
    return proof && proof.getImpl(symbolTable) == impl &&
           proof.getTraitApplication() == own;
  };
  // A producer is visited once as reached and once more as reached from under
  // a projection.
  SmallVector<std::pair<Value, bool>> pending{{value, false}};
  DenseSet<Operation *> seen[2];
  while (!pending.empty()) {
    auto [current, underProjection] = pending.pop_back_val();
    Operation *producer = current.getDefiningOp();
    if (!producer || !seen[underProjection].insert(producer).second)
      continue;
    if (underProjection && namesOwnApplication(producer))
      return true;
    bool below = underProjection || isa<ProjectOp>(producer);
    for (Value operand : producer->getOperands())
      pending.push_back({operand, below});
  }
  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// TraitOp
//===----------------------------------------------------------------------===//

/// Parses a dictionary written `{name = <value>, ...}` whose keys are bare
/// identifiers or strings, each value read by `parseValue`.
static ParseResult
parseNamedEntries(OpAsmParser &parser, NamedAttrList &entries,
                  llvm::function_ref<ParseResult(Attribute &)> parseValue) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Braces, [&]() -> ParseResult {
        std::string name;
        Attribute value;
        if (parser.parseKeywordOrString(&name) || parser.parseEqual() ||
            parseValue(value))
          return failure();
        entries.append(name, value);
        return success();
      });
}

/// Prints `dictionary` as `parseNamedEntries` reads it, each value by
/// `printValue`.
static void printNamedEntries(OpAsmPrinter &p, DictionaryAttr dictionary,
                              llvm::function_ref<void(Attribute)> printValue) {
  p << '{';
  llvm::interleaveComma(dictionary, p, [&](NamedAttribute entry) {
    p.printKeywordOrString(entry.getName().getValue());
    p << " = ";
    printValue(entry.getValue());
  });
  p << '}';
}

/// Prints a declaration's clauses after its header, each on its own line
/// indented under the declaration: `keyword {...}` for a non-empty
/// dictionary, by `printEntry`, and `members [...]` with one member per line
/// in order, by `printMember`, when there are members.
static void printClauses(OpAsmPrinter &p, StringRef keyword,
                         DictionaryAttr dictionary,
                         llvm::function_ref<void(Attribute)> printEntry,
                         ArrayAttr members,
                         llvm::function_ref<void(unsigned)> printMember) {
  p.increaseIndent();
  p.increaseIndent();
  if (!dictionary.empty()) {
    p.printNewline();
    p << keyword << ' ';
    printNamedEntries(p, dictionary, printEntry);
  }
  if (!members.empty()) {
    p.printNewline();
    p << "members [";
    p.increaseIndent();
    for (unsigned position = 0; position < members.size(); ++position) {
      p.printNewline();
      printMember(position);
      if (position + 1 < members.size())
        p << ',';
    }
    p.decreaseIndent();
    p.printNewline();
    p << ']';
  }
  p.decreaseIndent();
  p.decreaseIndent();
}

ParseResult TraitOp::parse(OpAsmParser &parser, OperationState &result) {
  // `<visibility> @name(<self>) [-> requirements] [assoc {...}]
  // [members [...]] [attributes {...}]`
  StringAttr visibility;
  (void)parseVisibilityKeyword(parser, visibility);
  if (visibility)
    result.addAttribute(SymbolTable::getVisibilityAttrName(), visibility);
  StringAttr name;
  Type self;
  SmallVector<Type> requirements;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLParen() || parser.parseType(self) || parser.parseRParen() ||
      parser.parseOptionalArrowTypeList(requirements))
    return failure();
  Builder &builder = parser.getBuilder();
  result.addAttribute(getSelfAttrName(result.name), TypeAttr::get(self));
  result.addAttribute(getRequirementsAttrName(result.name),
                      builder.getTypeArrayAttr(requirements));

  NamedAttrList assoc;
  if (succeeded(parser.parseOptionalKeyword("assoc")) &&
      parseNamedEntries(parser, assoc, [&](Attribute &count) -> ParseResult {
        int64_t value;
        if (parser.parseInteger(value))
          return failure();
        count = builder.getI64IntegerAttr(value);
        return success();
      }))
    return failure();
  result.addAttribute(getAssocAttrName(result.name),
                      assoc.getDictionary(builder.getContext()));

  SmallVector<Attribute> members;
  if (succeeded(parser.parseOptionalKeyword("members")) &&
      parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
            FlatSymbolRefAttr member;
            if (parser.parseAttribute(member))
              return failure();
            members.push_back(member);
            return success();
          }))
    return failure();
  result.addAttribute(getMembersAttrName(result.name),
                      builder.getArrayAttr(members));
  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

void TraitOp::print(OpAsmPrinter &p) {
  p << ' ';
  printVisibilityKeyword(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());
  p << '(' << getSelf() << ')';
  p.printOptionalArrowTypeList(getRequirements().getAsValueRange<TypeAttr>());
  printClauses(
      p, "assoc", getAssoc(),
      [&](Attribute count) { p << cast<IntegerAttr>(count).getInt(); },
      getMembers(), [&](unsigned position) {
        p.printAttribute(getMembers()[position]);
      });
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getSymVisibilityAttrName(), getSelfAttrName(),
       getRequirementsAttrName(), getAssocAttrName(), getMembersAttrName()});
}

LogicalResult TraitOp::verify() {
  // The trait's own application has its parameters as arguments, at least
  // one: parameter `i` is an occurrence of label `i`, its position.
  auto self = dyn_cast<ClaimType>(getSelf());
  if (!self || !self.isApplication() || self.isProven() ||
      self.getTraitApplication().getTraitName().getValue() != getSymName())
    return emitOpError() << "states its own application as the unproven claim "
                            "@"
                         << getSymName() << "[...]";
  for (auto [position, ty] : llvm::enumerate(getTypeParams())) {
    auto label = dyn_cast_or_null<PolyType>(Type(getParameterOccurrence(ty)));
    if (!label || static_cast<size_t>(label.getLabel()) != position)
      return emitOpError() << "parameter " << position << " must be labelled "
                           << position << ", its position, found " << ty;
  }
  if (getTypeParams().empty())
    return emitOpError() << "requires at least one type parameter";

  // An associated type's own parameter count is a count.
  for (NamedAttribute entry : getAssoc()) {
    auto count = dyn_cast<IntegerAttr>(entry.getValue());
    if (!count || count.getType() != IntegerType::get(getContext(), 64) ||
        count.getInt() < 0)
      return emitOpError() << "declares associated type '"
                           << entry.getName().getValue()
                           << "' with " << entry.getValue()
                           << ", which is no count of its own parameters";
  }

  // Each member is named under the trait by the dialect's rule
  // (`getMemberName`) and listed once, at the position `getMemberPosition`
  // answers for it. Neither reads a symbol table, so the trait refuses its own
  // list before any impl pairing one of its members is judged against it.
  for (auto [position, entry] : llvm::enumerate(getMembers())) {
    auto symbol = cast<FlatSymbolRefAttr>(entry);
    if (!getMemberName(getSymName(), symbol.getValue()))
      return emitOpError() << "member " << symbol
                           << " is not named under the trait";
    if (getMemberPosition(symbol) != position)
      return emitOpError() << "lists member " << symbol << " twice";
  }

  // An endpoint mentions a type parameter when any generic hiding inside it is
  // one of the trait's parameters or one an associated type takes of its own,
  // labelled after them; getGenericTypesIn descends the attributes -- a trait
  // application's arguments, an equality's endpoints -- that its own walk over
  // immediate type sub-elements does not reach.
  int64_t ownBound = getTypeParams().size();
  for (NamedAttribute entry : getAssoc())
    ownBound = std::max<int64_t>(
        ownBound,
        getTypeParams().size() + cast<IntegerAttr>(entry.getValue()).getInt());
  auto mentionsParam = [&](Type ty) {
    return llvm::any_of(getGenericTypesIn(ty), [&](GenericTypeInterface g) {
      auto label =
          dyn_cast_or_null<PolyType>(Type(getParameterOccurrence(Type(g))));
      return label && label.getLabel() < ownBound;
    });
  };

  // Each requirement is a claim relating the trait's parameters. A direct
  // self-reference like @Trait[!S] would create a circular obligation that no
  // impl can satisfy. However, a self-reference whose self argument goes
  // through a projection (e.g. @Trait[!trait.proj<...>]) is safe: the
  // projection resolves to a concrete type during monomorphization, so the
  // obligation is discharged against a different impl, not the one being
  // defined.
  for (Type entry : getRequirements().getAsValueRange<TypeAttr>()) {
    auto requirement = dyn_cast<ClaimType>(entry);
    if (!requirement || requirement.isProven())
      return emitOpError() << "requirement " << entry
                           << " must be an unproven claim";
    if (!mentionsParam(entry))
      return emitOpError() << "requirement " << entry
                           << " must mention at least one type parameter";
    if (requirement.isApplication()) {
      TraitApplicationAttr app = requirement.getTraitApplication();
      if (app.getTraitName().getValue() == getSymName() &&
          !containsType<ProjectionType>(app.getTypeArgs().front()))
        return emitOpError() << "requirement " << entry
                             << " must not reference the current trait";
    }
  }
  return success();
}

LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The requirements are types stored in an attribute, which the
  // symbol-use driver does not reach.
  for (Type requirement : getRequirements().getAsValueRange<TypeAttr>())
    if (failed(cast<ClaimType>(requirement).verifySymbolUses(getOperation(),
                                                             symbolTable)))
      return failure();

  // Member k is a function leading with the trait's own application: one
  // with a body is the default an impl may leave the member to, and an
  // external one is required of every impl.
  ClaimType self = getSelfClaim();
  for (auto [position, entry] : llvm::enumerate(getMembers())) {
    auto symbol = cast<FlatSymbolRefAttr>(entry);
    auto member =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(*this, symbol);
    if (!member)
      return emitOpError() << "member " << position << ", " << symbol
                           << ", is no function";
    auto type = dyn_cast<FunctionType>(member.getFunctionType());
    if (!type || type.getNumInputs() == 0 || type.getInput(0) != Type(self))
      return emitOpError() << "member " << symbol
                           << " does not lead with the trait's own application "
                           << Type(self);
    if (failed(verifyFunctionResultGenericsAreDetermined(member, type, {})))
      return failure();
  }
  return success();
}

std::optional<unsigned> TraitOp::getMemberPosition(FlatSymbolRefAttr member) {
  for (auto [position, entry] : llvm::enumerate(getMembers()))
    if (entry == member)
      return position;
  return std::nullopt;
}

std::optional<unsigned> TraitOp::getAssociatedTypeArity(StringRef name) {
  auto count = dyn_cast_or_null<IntegerAttr>(getAssoc().get(name));
  if (!count)
    return std::nullopt;
  return count.getInt();
}

std::string mlir::trait::getMemberSymbol(StringRef owner, StringRef name) {
  return (owner + "." + name).str();
}

std::optional<StringRef> mlir::trait::getMemberName(StringRef owner,
                                                    StringRef symbol) {
  if (!symbol.consume_front(owner) || !symbol.consume_front("."))
    return std::nullopt;
  return symbol;
}

/// The symbol of `owner`'s member function `name` (`getMemberSymbol`).
static FlatSymbolRefAttr memberSymbol(StringAttr owner, StringRef name) {
  return FlatSymbolRefAttr::get(owner.getContext(),
                                getMemberSymbol(owner.getValue(), name));
}

FlatSymbolRefAttr TraitOp::getMemberNamed(StringRef name) {
  FlatSymbolRefAttr wanted = memberSymbol(getSymNameAttr(), name);
  return getMemberPosition(wanted) ? wanted : FlatSymbolRefAttr();
}

func::FuncOp TraitOp::addMember(OpBuilder &builder, Location loc,
                                StringRef name, FunctionType signature,
                                bool withDefault) {
  FlatSymbolRefAttr symbol = memberSymbol(getSymNameAttr(), name);
  SmallVector<Type> inputs{getSelfClaim()};
  llvm::append_range(inputs, signature.getInputs());
  auto member = func::FuncOp::create(
      builder, loc, symbol.getValue(),
      builder.getFunctionType(inputs, signature.getResults()));
  member.setPrivate();
  if (withDefault)
    member.addEntryBlock();
  SmallVector<Attribute> members(getMembers().getValue());
  members.push_back(symbol);
  setMembersAttr(builder.getArrayAttr(members));
  return member;
}

void TraitOp::declareAssociatedType(StringRef name, unsigned arity) {
  NamedAttrList assoc(getAssoc());
  assoc.set(name, Builder(getContext()).getI64IntegerAttr(arity));
  setAssocAttr(assoc.getDictionary(getContext()));
}

FailureOr<SpecializationMap> TraitOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                      llvm::function_ref<InFlightDiagnostic()> errFn) {
  // A trait's parameters are its self application's arguments in order, so an
  // application of this trait at the right arity determines every parameter by
  // position: nothing is read out of the arguments and nothing is compared
  // afterwards.
  TraitApplicationAttr application = actualSelfClaim.getTraitApplication();
  if (application.getTraitName().getValue() != getSymName()) {
    if (errFn)
      errFn() << "trait mismatch: expected @" << getSymName() << ", but found "
              << application.getTraitName();
    return failure();
  }

  ArrayRef<Type> parameters = getTypeParams();
  ArrayRef<Type> arguments = application.getTypeArgs();
  if (parameters.size() != arguments.size()) {
    if (errFn)
      errFn() << "trait '@" << getSymName() << "' takes " << parameters.size()
              << " type arguments, but " << application << " supplies "
              << arguments.size();
    return failure();
  }

  SpecializationMap result;
  for (auto [parameter, argument] : llvm::zip(parameters, arguments)) {
    auto generic = dyn_cast<GenericTypeInterface>(parameter);
    if (!generic) {
      if (errFn)
        errFn() << "trait '@" << getSymName()
                << "' declares a type parameter that is not a type variable";
      return failure();
    }
    result.bind(generic, argument);
  }
  return result;
}

FailureOr<SmallVector<ClaimType>> TraitOp::specializeRequirementsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto spec = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(spec)) return failure();
  // A substitution rewrites the type arguments a claim carries, never the claim
  // wrapper itself: its keys are this trait's type parameters, never a whole
  // ClaimType, so the result is always a claim.
  return llvm::map_to_vector(
      getRequirements().getAsValueRange<TypeAttr>(), [&](Type requirement) {
        return cast<ClaimType>(instantiate(requirement, *spec));
      });
}

FailureOr<ClaimType> TraitOp::specializeRequirementAsClaimFor(
    ClaimType actualSelfClaim, unsigned index,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  ArrayAttr requirements = getRequirements();
  assert(index < requirements.size() && "requirement index out of range");
  auto spec = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(spec)) return failure();
  Type requirement = cast<TypeAttr>(requirements[index]).getValue();
  return cast<ClaimType>(instantiate(requirement, *spec));
}

SmallVector<ImplOp> TraitOp::getImpls() {
  // Impls stand beside the trait in its symbol table, so scan them directly
  // and match this trait's symbol name. This avoids a full symbol-use walk,
  // which materializes every operation's attribute dictionary.
  Operation *table = (*this)->getParentOp();
  if (!table || table->getNumRegions() == 0 || table->getRegion(0).empty())
    return {};
  StringRef traitName = getSymName();
  SmallVector<ImplOp> result;
  for (Operation &op : table->getRegion(0).front()) {
    auto impl = dyn_cast<ImplOp>(op);
    if (impl && impl.getTraitNameAttr().getValue() == traitName)
      result.push_back(impl);
  }
  return result;
}

SmallVector<ImplOp> TraitOp::getCandidateImplsFor(ClaimType wanted,
                                                  Normalizer normalize) {
  SmallVector<ImplOp> result;
  for (auto impl : getImpls()) {
    if (succeeded(impl.buildSubstitutionForSelfClaim(wanted, normalize,
                                                     /*errFn=*/nullptr)))
      result.push_back(impl);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

ParseResult ImplOp::parse(OpAsmParser &parser, OperationState &result) {
  // `<visibility> @name(%self: <claim>, <where entries>...) [bindings {...}]
  // [members [@T.m -> @f, ...]] [attributes {...}] { <evidence> }`
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> results;
  SMLoc headerLoc = parser.getCurrentLocation();
  StringAttr visibility;
  (void)parseVisibilityKeyword(parser, visibility);
  if (visibility)
    result.addAttribute(SymbolTable::getVisibilityAttrName(), visibility);
  StringAttr name;
  bool isVariadic = false;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, arguments, isVariadic, results,
          resultAttrs))
    return failure();
  if (!results.empty())
    return parser.emitError(headerLoc)
           << "an impl states no results: it returns its trait's";
  if (llvm::any_of(arguments, [](const OpAsmParser::Argument &arg) {
        return arg.attrs && !arg.attrs.empty();
      }))
    return parser.emitError(headerLoc)
           << "a declaration's arguments carry no attributes";

  Builder &builder = parser.getBuilder();
  NamedAttrList bindings;
  if (succeeded(parser.parseOptionalKeyword("bindings")) &&
      parseNamedEntries(parser, bindings, [&](Attribute &bound) -> ParseResult {
        Type type;
        if (parser.parseType(type))
          return failure();
        bound = TypeAttr::get(type);
        return success();
      }))
    return failure();
  result.addAttribute(getBindingsAttrName(result.name),
                      bindings.getDictionary(builder.getContext()));

  SmallVector<Attribute> members;
  if (succeeded(parser.parseOptionalKeyword("members")) &&
      parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
            FlatSymbolRefAttr member, implementation;
            if (parser.parseAttribute(member) || parser.parseArrow() ||
                parser.parseAttribute(implementation))
              return failure();
            members.push_back(builder.getArrayAttr({member, implementation}));
            return success();
          }))
    return failure();
  result.addAttribute(getMembersAttrName(result.name),
                      builder.getArrayAttr(members));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/false))
    return failure();
  ensureTerminator(*body, builder, result.location);
  return success();
}

void ImplOp::print(OpAsmPrinter &p) {
  p << ' ';
  printVisibilityKeyword(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());
  Region &body = getBody();
  call_interface_impl::printFunctionSignature(
      p, body.front().getArgumentTypes(), /*argAttrs=*/nullptr,
      /*isVariadic=*/false, /*resultTypes=*/{}, /*resultAttrs=*/nullptr, &body,
      /*printEmptyResult=*/false);
  printClauses(
      p, "bindings", getBindings(),
      [&](Attribute bound) { p.printType(cast<TypeAttr>(bound).getValue()); },
      getMembers(), [&](unsigned position) {
        auto pair = cast<ArrayAttr>(getMembers()[position]);
        p.printAttribute(pair[0]);
        p << " -> ";
        p.printAttribute(pair[1]);
      });
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getSymVisibilityAttrName(),
                            getBindingsAttrName(), getMembersAttrName()});
  p << ' ';
  // An impl of a trait requiring nothing returns nothing, and its empty return
  // is the one the parser supplies.
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/getReturn().getNumOperands() != 0);
}

void ImplOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  nameDeclarationArguments(region, setNameFn);
}

ReturnOp ImplOp::getReturn() {
  return cast<ReturnOp>(getBody().front().getTerminator());
}

/// Whether `expected` and `actual`, two spellings an impl's own declarations
/// are judged by, name one type under the facts in scope there: each read
/// through the impl's own associated-type bindings, which hold of its self
/// application anywhere in its body, then equal under the ground congruence
/// its where clause's equalities seed, modulo proofs (`entailedByGroundCongruence`).
/// The verifier enumerates no candidate impls, so a projection neither reduces
/// is equal to itself alone. Fails, naming why through `errFn`, where reading
/// the bindings does not settle. `own` is the impl's reader of its own
/// bindings (`ImplOp::getOwnBindingsReader`), which the judgments of one impl
/// share.
static FailureOr<bool>
agreeInOwnContext(ImplOp impl, ProjectionNormalizer &own, Type expected,
                  Type actual, llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto read = [&](Type ty) -> FailureOr<Type> {
    Normalized out = own.normalize(ty);
    if (!out.unsettled)
      return out.type;
    if (errFn)
      errFn() << "projection normalization did not converge; check for "
                 "cyclic associated type bindings";
    return failure();
  };
  FailureOr<Type> lhs = read(expected);
  FailureOr<Type> rhs = read(actual);
  if (failed(lhs) || failed(rhs))
    return failure();
  SmallVector<TypeEqualityAttr> hypotheses;
  for (TypeEqualityAttr equality : impl.getEqualityPremises()) {
    FailureOr<Type> left = read(equality.getLhs());
    FailureOr<Type> right = read(equality.getRhs());
    if (failed(left) || failed(right))
      return failure();
    hypotheses.push_back(
        TypeEqualityAttr::get(impl.getContext(), *left, *right));
  }
  return entailedByGroundCongruence(stripClaimProofs(*lhs),
                                    stripClaimProofs(*rhs), hypotheses);
}

/// Verifies that `implemented`, the function `impl` pairs with `member` of
/// `traitOp`, is the member's declaration carried to the impl: rustc's
/// `compare_impl_method`.
///
/// Each signature is read past its leading claims -- the trait's own
/// application on the member, the impl's block arguments on the function --
/// which leaves the member's inputs as the trait declares them and as the impl
/// states them. A member's signature binds its trait's parameters and then its
/// own, labelled after the trait's without a gap; the function's binds the
/// impl's and then equally many of its own: the trait states how many a caller
/// supplies, and a copy with a different count is a different declaration. So
/// the carrying substitution is positional -- the trait's parameters take the
/// impl's self arguments, and the member's own parameter `traitCount + j`
/// becomes the function's `implCount + j`, the rebase rustc's `rebase_onto`
/// performs -- and the check is the one identity it makes true, read through
/// the impl's own bindings and where equalities.
static LogicalResult verifyMemberSignature(
    ImplOp impl, TraitOp traitOp, FlatSymbolRefAttr member,
    FunctionType memberType, FunctionType implementedType,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto rebase =
      traitOp.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(rebase)) return failure();

  MLIRContext *ctx = impl.getContext();
  unsigned traitCount = traitOp.getTypeParams().size();
  unsigned implCount = impl.getNumTypeParams();
  unsigned leading = impl.getBody().front().getNumArguments();
  Type traitMethodTy = FunctionType::get(
      ctx, memberType.getInputs().drop_front(), memberType.getResults());
  Type implMethodTy = FunctionType::get(
      ctx, implementedType.getInputs().drop_front(leading),
      implementedType.getResults());
  FailureOr<unsigned> traitOwn = countDenseLabelsFrom(traitMethodTy, traitCount);
  FailureOr<unsigned> implOwn = countDenseLabelsFrom(implMethodTy, implCount);
  if (failed(traitOwn) || failed(implOwn)) {
    if (errFn)
      errFn() << "member " << member << " labels its own type parameters "
              << "after its declaration's, without a gap";
    return failure();
  }
  if (*traitOwn != *implOwn) {
    if (errFn)
      errFn() << "member " << member << " is implemented binding " << *implOwn
              << " type parameter(s) of its own, but trait '"
              << traitOp.getSymNameAttr() << "' declares it with "
              << *traitOwn;
    return failure();
  }

  // The copy may constrain the kind of an own parameter, so what the trait's
  // parameter becomes is the copy's spelling of its label: its first
  // occurrence in the copy's signature, a kind-constraining wrapper or the
  // label itself.
  SmallVector<GenericTypeInterface, 4> implGenerics =
      getGenericTypesIn(implMethodTy);
  for (unsigned j = 0; j < *traitOwn; ++j) {
    Type label = PolyType::get(ctx, implCount + j);
    auto spelling = llvm::find_if(implGenerics, [&](GenericTypeInterface g) {
      return Type(getParameterOccurrence(Type(g))) == label;
    });
    rebase->bind(cast<GenericTypeInterface>(
                     Type(PolyType::get(ctx, traitCount + j))),
                 Type(*spelling));
  }
  Type expected = instantiate(traitMethodTy, *rebase);
  ProjectionNormalizer own = impl.getOwnBindingsReader();
  FailureOr<bool> agree =
      agreeInOwnContext(impl, own, expected, implMethodTy, errFn);
  if (failed(agree))
    return failure();
  if (!*agree) {
    if (errFn)
      errFn() << "member " << member << " has incompatible signature: "
              << "expected " << expected << " but found " << implMethodTy;
    return failure();
  }
  return success();
}

LogicalResult mlir::trait::verifyImplMemberSignatures(
    ImplOp impl, TraitOp trait, SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return impl.emitOpError(); };
  for (Attribute entry : impl.getMembers()) {
    auto pair = cast<ArrayAttr>(entry);
    auto member = cast<FlatSymbolRefAttr>(pair[0]);
    auto implemented = cast<FlatSymbolRefAttr>(pair[1]);
    auto memberFn =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(trait, member);
    auto implementedFn =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(impl, implemented);
    if (failed(verifyMemberSignature(
            impl, trait, member, cast<FunctionType>(memberFn.getFunctionType()),
            cast<FunctionType>(implementedFn.getFunctionType()), errFn)))
      return failure();
  }
  return success();
}

/// Verifies the evidence this impl returns for its trait's requirements: one
/// claim per requirement, in the trait's order, each the requirement at the
/// impl's self arguments once both are read through the impl's own bindings and
/// where equalities, and none resting on the impl's own application -- an
/// impl's requirement evidence may not assume what it is evidence for (GHC's
/// rule for instance superclasses). A derive of this impl at other arguments
/// cites it by symbol and reads no argument, so it stays legal.
static LogicalResult verifyRequirementEvidence(
    ImplOp impl, TraitOp traitOp, SymbolTableCollection &symbolTable,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto requirements =
      traitOp.specializeRequirementsAsClaimsFor(impl.getSelfClaim(), errFn);
  if (failed(requirements))
    return failure();
  ReturnOp evidence = impl.getReturn();
  if (evidence.getNumOperands() != requirements->size())
    return errFn() << "returns " << evidence.getNumOperands()
                   << " claims, and trait '@" << traitOp.getSymName()
                   << "' requires " << requirements->size();

  Value self = impl.getBody().front().getArgument(0);
  ProjectionNormalizer own = impl.getOwnBindingsReader();
  for (auto [index, operand, requirement] :
       llvm::enumerate(evidence.getOperands(), *requirements)) {
    auto supplied = dyn_cast<ClaimType>(operand.getType());
    if (!supplied)
      return errFn() << "returns " << operand.getType() << " for requirement "
                     << index << ", which is no claim";
    FailureOr<bool> agree = agreeInOwnContext(impl, own, Type(requirement),
                                              Type(supplied), errFn);
    if (failed(agree))
      return failure();
    if (!*agree)
      return errFn() << "returns " << supplied << " for requirement " << index
                     << ", which trait '@" << traitOp.getSymName()
                     << "' states as " << requirement;
    if (restsOn(operand, self))
      return errFn() << "returns evidence for requirement " << index
                     << " that rests on the impl's own application";
    if (projectsOwnApplication(operand, impl, symbolTable))
      return errFn() << "returns evidence for requirement " << index
                     << " that projects the impl's own application";
  }
  return success();
}

static LogicalResult verifyImplParametersAreConstrained(ImplOp impl);

/// Verifies each associated type binding against the two lists a use of it
/// supplies arguments for: the impl header's parameters, labels 0 to n - 1,
/// bound where the impl is selected, and the binding's own parameters, the
/// `arity` labels after them, bound by a projection's associated type
/// arguments. A bound type mentioning a parameter from neither list has
/// nothing to supply it, so the resolved type would carry a parameter no
/// substitution reaches.
static LogicalResult verifyBindingScope(ImplOp impl, StringRef name,
                                        Type bound, unsigned arity) {
  unsigned headerCount = impl.getNumTypeParams();
  for (GenericTypeInterface parameter : getTypeParametersIn(bound)) {
    auto label = dyn_cast_or_null<PolyType>(
        Type(getParameterOccurrence(Type(parameter))));
    if (label && static_cast<unsigned>(label.getLabel()) < headerCount + arity)
      continue;
    return impl.emitOpError()
           << "binds associated type '" << name
           << "' to a type mentioning type parameter " << parameter
           << ", which neither impl '@" << impl.getSymName()
           << "' nor the associated type declares";
  }
  return success();
}

LogicalResult ImplOp::verify() {
  // The first block argument is the impl's own application and every other a
  // where entry: an application it assumes or an equality it assumes.
  Block &body = getBody().front();
  auto self = body.getNumArguments() != 0
                  ? dyn_cast<ClaimType>(body.getArgument(0).getType())
                  : ClaimType();
  if (!self || !self.isApplication() || self.isProven())
    return emitOpError() << "takes the unproven claim of its own application "
                            "as its first block argument";
  for (BlockArgument entry : body.getArguments().drop_front()) {
    auto claim = dyn_cast<ClaimType>(entry.getType());
    if (!claim || claim.isProven())
      return emitOpError() << "where entry " << entry.getArgNumber() - 1
                           << " must be an unproven claim, found "
                           << entry.getType();
  }
  // A parameter's label is its position, so the header and the where clause
  // together spell labels 0 to n - 1 without a gap.
  if (failed(countDenseLabelsFrom(llvm::to_vector(body.getArgumentTypes()), 0)))
    return emitOpError() << "labels its type parameters 0, 1, ... by position, "
                            "and its header and where clause skip one";
  // Each member entry pairs a member of the trait with its implementation, and
  // each binding is a type.
  for (Attribute entry : getMembers()) {
    auto pair = dyn_cast<ArrayAttr>(entry);
    if (!pair || pair.size() != 2 ||
        !llvm::all_of(pair, llvm::IsaPred<FlatSymbolRefAttr>))
      return emitOpError() << "pairs members as " << entry
                           << ", which is no pair of a member and the "
                              "function implementing it";
  }
  for (NamedAttribute binding : getBindings())
    if (!isa<TypeAttr>(binding.getValue()))
      return emitOpError() << "binds associated type '"
                           << binding.getName().getValue() << "' to "
                           << binding.getValue() << ", which is no type";
  return verifyImplParametersAreConstrained(*this);
}

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&]{ return emitOpError(); };

  // The self application and the where entries name traits by symbol; the
  // block arguments holding them are types the symbol-use driver reads no
  // further than their own attributes.
  for (BlockArgument argument : getBody().front().getArguments())
    if (failed(cast<ClaimType>(argument.getType())
                   .verifySymbolUses(getOperation(), symbolTable)))
      return failure();

  TraitOp traitOp = getTrait(symbolTable);

  // The pairs name members of the trait, each once, and a member's
  // implementation leads with this impl's block arguments, by type identity.
  // A member the impl pairs with nothing is left to its trait's default, so
  // its own function must have a body.
  ArrayAttr members = traitOp.getMembers();
  SmallVector<bool> paired(members.size(), false);
  TypeRange leading = getBody().front().getArgumentTypes();
  for (Attribute entry : getMembers()) {
    auto pair = cast<ArrayAttr>(entry);
    auto member = cast<FlatSymbolRefAttr>(pair[0]);
    auto implemented = cast<FlatSymbolRefAttr>(pair[1]);
    std::optional<unsigned> position = traitOp.getMemberPosition(member);
    if (!position)
      return errFn() << "pairs " << member << ", which is no member of trait '"
                     << getTraitNameAttr() << "'";
    if (paired[*position])
      return errFn() << "pairs member " << member << " twice";
    paired[*position] = true;
    auto function =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(*this, implemented);
    if (!function || function.isExternal())
      return errFn() << "implements member " << member << " by "
                     << implemented << ", which is no function with a body";
    ArrayRef<Type> inputs = function.getArgumentTypes();
    if (inputs.size() < leading.size() ||
        !llvm::equal(inputs.take_front(leading.size()), leading))
      return errFn() << "implements member " << member << " by "
                     << implemented
                     << ", whose leading parameters are not the impl's block "
                        "arguments";
  }
  for (auto [position, done] : llvm::enumerate(paired)) {
    if (done)
      continue;
    auto member = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        traitOp, cast<FlatSymbolRefAttr>(members[position]));
    if (!member || member.isExternal())
      return errFn() << "pairs no implementation with member "
                     << members[position] << " of trait '"
                     << getTraitNameAttr() << "'";
  }

  // The bindings are total over the trait's associated types, each spelled
  // over the impl's parameters and the associated type's own.
  for (NamedAttribute binding : getBindings()) {
    std::optional<unsigned> arity =
        traitOp.getAssociatedTypeArity(binding.getName().getValue());
    if (!arity)
      return errFn() << "binds associated type '"
                     << binding.getName().getValue()
                     << "', which trait '" << getTraitNameAttr()
                     << "' does not declare";
    if (failed(verifyBindingScope(*this, binding.getName().getValue(),
                                  cast<TypeAttr>(binding.getValue()).getValue(),
                                  *arity)))
      return failure();
  }
  for (NamedAttribute declared : traitOp.getAssoc())
    if (!getBindings().get(declared.getName()))
      return errFn() << "missing binding for associated type '"
                     << declared.getName().getValue() << "' of trait '"
                     << getTraitNameAttr() << "'";

  return verifyRequirementEvidence(*this, traitOp, symbolTable, errFn);
}

FlatSymbolRefAttr ImplOp::getImplementationOf(FlatSymbolRefAttr member) {
  for (Attribute entry : getMembers()) {
    auto pair = cast<ArrayAttr>(entry);
    if (pair[0] == member)
      return cast<FlatSymbolRefAttr>(pair[1]);
  }
  return {};
}

func::FuncOp ImplOp::addMember(OpBuilder &builder, Location loc,
                               FlatSymbolRefAttr member,
                               FunctionType signature) {
  // The function is named, under this impl, after the member's name within
  // its trait, by the rule naming both; the trait's verifier refuses a member
  // not named under it.
  std::optional<StringRef> name =
      getMemberName(getTraitNameAttr().getValue(), member.getValue());
  assert(name && "a member of the impl's trait is named under it");
  SmallVector<Type> inputs(getBody().front().getArgumentTypes());
  llvm::append_range(inputs, signature.getInputs());
  auto function = func::FuncOp::create(
      builder, loc, getMemberSymbol(getSymName(), *name),
      builder.getFunctionType(inputs, signature.getResults()));
  function.setPrivate();
  function.addEntryBlock();
  SmallVector<Attribute> members(getMembers().getValue());
  members.push_back(builder.getArrayAttr(
      {member, FlatSymbolRefAttr::get(function.getSymNameAttr())}));
  setMembersAttr(builder.getArrayAttr(members));
  return function;
}

void ImplOp::bindAssociatedType(StringRef name, Type type) {
  NamedAttrList bindings(getBindings());
  bindings.set(name, TypeAttr::get(type));
  setBindingsAttr(bindings.getDictionary(getContext()));
}

namespace {
/// One way a where-clause equality of an impl determines its parameters: once
/// every parameter `input` spells is known, so is the type `input` stands for,
/// and the parameters `determined` spells outside every projection are read off
/// that type.
struct EqualityReading {
  Type input;
  Type determined;
};
} // namespace

/// The readings `impl`'s where-clause equalities offer under rustc's
/// constrained-parameter rule (`setup_constraining_clauses` in
/// rustc_hir_analysis/src/constrained_generic_params.rs). Each equality is read
/// in both directions, except that a projection of the impl's own trait
/// application is never an input: it resolves through this impl's own
/// associated-type binding, which is spelled over the very parameters it would
/// determine, so it names no type until they are known (rustc skips "a sneaky
/// attempt to project out an associated type defined by this very trait").
///
/// A projection spells nothing outside every projection, so a reading whose
/// determined side is one determines nothing, and on an equality between a
/// projection and a type the two directions are rustc's one reading, from the
/// projection to the type.
static SmallVector<EqualityReading> getEqualityReadings(ImplOp impl) {
  TraitApplicationAttr own = impl.getSelfApplication();
  auto projectsOwnApplication = [&](Type side) {
    auto projection = dyn_cast<ProjectionType>(side);
    return projection && projection.getTraitApplication() == own;
  };
  SmallVector<EqualityReading> readings;
  for (TypeEqualityAttr equality : impl.getEqualityPremises()) {
    if (!projectsOwnApplication(equality.getLhs()))
      readings.push_back({equality.getLhs(), equality.getRhs()});
    if (!projectsOwnApplication(equality.getRhs()))
      readings.push_back({equality.getRhs(), equality.getLhs()});
  }
  return readings;
}

/// rustc's constrained-parameter rule (E0207,
/// `enforce_impl_non_lifetime_params_are_constrained` in
/// rustc_hir_analysis/src/impl_wf_check.rs): every type parameter an impl binds
/// is constrained. A parameter is constrained when the self application spells
/// it outside every projection, or when an equality reading
/// (`getEqualityReadings`) whose input spells only constrained parameters spells
/// it outside every projection on its determined side, to a fixed point.
///
/// A projection is not injective -- two arguments can reach one resolution --
/// so a parameter standing only inside one is not constrained by it. Every
/// constrained parameter is one `readTypeArgumentsFor` reads off a demanded
/// application, so each use of the impl names one instance of it; a parameter
/// nothing constrains would leave the impl's members and associated-type
/// bindings spelling a variable selection never assigns.
static LogicalResult verifyImplParametersAreConstrained(ImplOp impl) {
  // The parameters the self application determines: those standing somewhere in
  // it outside a projection.
  DenseSet<Type> constrained;
  std::function<void(Type)> readOutsideProjections = [&](Type ty) {
    if (isa<ProjectionType>(ty))
      return;
    if (GenericTypeInterface parameter = getParameterOccurrence(ty)) {
      constrained.insert(Type(parameter));
      return;
    }
    for (Type child : decomposeTerm(ty).children)
      readOutsideProjections(child);
  };
  for (Type argument : impl.getSelfApplication().getTypeArgs())
    readOutsideProjections(argument);

  // Then close over the where clause's equality readings: one whose input
  // spells only constrained parameters constrains what its determined side
  // spells outside every projection.
  SmallVector<EqualityReading> readings = getEqualityReadings(impl);
  for (bool grew = true; grew;) {
    grew = false;
    for (const EqualityReading &reading : readings) {
      if (!llvm::all_of(getTypeParametersIn(reading.input),
                        [&](GenericTypeInterface inside) {
                          return constrained.contains(Type(inside));
                        }))
        continue;
      size_t before = constrained.size();
      readOutsideProjections(reading.determined);
      grew |= constrained.size() != before;
    }
  }

  for (GenericTypeInterface parameter : impl.getTypeParams())
    if (!constrained.contains(Type(parameter)))
      return impl.emitOpError()
             << "type parameter " << Type(parameter)
             << " is not constrained by the impl's trait application or its "
                "where clause, so impl selection cannot determine it";
  return success();
}

void mlir::trait::allegeRequirements(ImplOp impl, TraitOp trait,
                                     OpBuilder &builder) {
  auto requirements =
      trait.specializeRequirementsAsClaimsFor(impl.getSelfClaim());
  if (failed(requirements) || requirements->empty())
    return;
  ReturnOp evidence = impl.getReturn();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(evidence);
  SmallVector<Value> alleged;
  for (ClaimType requirement : *requirements)
    alleged.push_back(
        AllegeOp::create(builder, evidence.getLoc(), requirement).getResult());
  evidence->setOperands(alleged);
}

bool ImplOp::isUnconditional() {
  // A citation supplies an impl its parameters' arguments and one claim per
  // where entry; an impl taking neither stands for one application and assumes
  // nothing there, so naming it is the whole citation. Its trait's requirements
  // are not counted: the impl returns their evidence itself.
  return getNumTypeParams() == 0 && getWhereClaims().empty();
}

LogicalResult ImplOp::verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isUnconditional()) {
    if (err) err() << "impl '@" << getSymName()
                   << "' binds type parameters or has a where clause, so it "
                      "must be cited through a trait.proof";
    return failure();
  }
  return success();
}

unsigned ImplOp::getNumTypeParams() {
  return getLabelBound(llvm::to_vector(getBody().front().getArgumentTypes()));
}

SmallVector<GenericTypeInterface, 4> ImplOp::getTypeParams() {
  // The parameters are labelled by position (`ImplOp::verify`), so they are
  // the labels below the bound the header and the where clause spell.
  SmallVector<GenericTypeInterface, 4> parameters;
  for (unsigned label = 0, count = getNumTypeParams(); label < count; ++label)
    parameters.push_back(
        cast<GenericTypeInterface>(Type(PolyType::get(getContext(), label))));
  return parameters;
}

ImplArgumentsAttr ImplOp::stateArguments(const SpecializationMap &arguments) {
  SmallVector<Type> stated;
  for (GenericTypeInterface parameter : getTypeParams())
    stated.push_back(instantiate(Type(parameter), arguments));
  return ImplArgumentsAttr::get(getContext(), stated);
}

//===----------------------------------------------------------------------===//
// Evidence clones
//===----------------------------------------------------------------------===//

/// Clones at `rewriter`'s insertion point the ops of a declaration's or a
/// proof's body that compute `root`, operands before their users, each op once:
/// a value `mapping` already holds is read from it, so the caller maps the
/// body's block arguments before asking. `stamp` respells every type and
/// attribute the clones carry.
static Value cloneDefiningTree(RewriterBase &rewriter, Value root,
                               IRMapping &mapping, AttrTypeReplacer &stamp) {
  if (Value mapped = mapping.lookupOrNull(root))
    return mapped;
  Operation *producer = root.getDefiningOp();
  assert(producer && "a body's block arguments are mapped before it is read");
  for (Value operand : producer->getOperands())
    (void)cloneDefiningTree(rewriter, operand, mapping, stamp);
  Operation *clone = rewriter.clone(*producer, mapping);
  for (Value result : clone->getResults())
    result.setType(stamp.replace(result.getType()));
  for (NamedAttribute attr : clone->getAttrs())
    clone->setAttr(attr.getName(), stamp.replace(attr.getValue()));
  return mapping.lookup(root);
}

/// Maps in `mapping` each value of `reads` -- the values a projection of
/// `impl`'s return reads from the impl -- to the evidence the receiver's proof
/// `selfProof` supplies there, cloning at `rewriter`'s insertion point what
/// must be computed, each clone of the impl's ops stamped by `stamp`: argument
/// 0 to `self`; argument k to a clone of the claim the receiver's proof derives
/// the impl from at position k - 1, its defining ops copied out of the proof's
/// body, which is ground; a value the impl's body computes to a clone of its
/// defining ops over those. A proof is closed, so the claims its derive is
/// given are computed by its own body alone. An argument the receiver's proof
/// supplies nothing for stays unmapped, and its reader is not cloned.
static void mapDeclarationReads(RewriterBase &rewriter,
                                SymbolTableCollection &symbolTable,
                                ImplOp impl,
                                const llvm::SetVector<Value> &reads,
                                Value self, ClaimType selfProof,
                                AttrTypeReplacer &stamp, IRMapping &mapping) {
  Block &declarationBody = impl.getBody().front();
  mapping.map(declarationBody.getArgument(0), self);
  if (declarationBody.getNumArguments() > 1)
    if (auto proof = symbolTable.lookupNearestSymbolFrom<ProofOp>(
            impl, selfProof.getProof());
        proof && proof.getDerive(symbolTable)) {
      AttrTypeReplacer asWritten;
      IRMapping fromProof;
      for (auto [argument, premise] :
           llvm::zip(declarationBody.getArguments().drop_front(),
                     proof.getDerive(symbolTable).getAssumptions()))
        mapping.map(argument,
                    cloneDefiningTree(rewriter, premise, fromProof, asWritten));
    }
  for (Value read : reads)
    if (!isa<BlockArgument>(read) || mapping.contains(read))
      (void)cloneDefiningTree(rewriter, read, mapping, stamp);
}

bool mlir::trait::producesPositionalEvidence(Operation *op) {
  if (isa<ProjectOp, DeriveOp, CoerceOp>(op))
    return true;
  auto call = dyn_cast<FuncCallOp>(op);
  return call && call.computesEvidence();
}

namespace {

/// Inlines a callee's body at a call: every op is legal to inline, since the
/// body reads nothing from outside it but its parameters, and the call's
/// results take the operands of the body's return, announced to `rewriter`
/// so their users are visited again. An operand spelled otherwise than the
/// call's result is bridged by a coercion, as a projection's inlined evidence
/// is (`ProjectOp::inlineEvidence`).
struct CalleeBodyInliner : public InlinerInterface {
  CalleeBodyInliner(MLIRContext *ctx, RewriterBase &rewriter)
      : InlinerInterface(ctx), rewriter(rewriter) {}

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    return true;
  }
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    for (auto [result, returned] : llvm::zip(valuesToRepl, op->getOperands())) {
      Value replacement = returned;
      if (returned.getType() != result.getType())
        replacement = CoerceOp::create(rewriter, op->getLoc(), result.getType(),
                                       returned, ValueRange{});
      rewriter.replaceAllUsesWith(result, replacement);
    }
  }

  RewriterBase &rewriter;
};

} // namespace

namespace {
/// The impl a claim value commits to: the impl, the application the value
/// proves, and the impl's arguments there, with the derive committing it,
/// whose given operands are the impl's where arguments, read in `enclosing`.
struct CommittedImpl {
  ImplOp impl;
  TraitApplicationAttr application;
  SpecializationMap arguments;
  /// Null for an unconditional impl cited by name, which has no where clause.
  DeriveOp derive;
  /// The impl the derive stands in, as committed; null for a proof's derive,
  /// which stands in no impl.
  std::shared_ptr<const CommittedImpl> enclosing;
};
} // namespace

/// The value a read computes, through the coercions that bridge spellings.
static Value throughCoercions(Value value) {
  while (auto coerce = value.getDefiningOp<CoerceOp>())
    value = coerce.getInput();
  return value;
}

ClaimType mlir::trait::evidenceOf(Value value) {
  for (;;) {
    auto claim = dyn_cast<ClaimType>(value.getType());
    if (claim && claim.isApplication() && claim.isProven())
      return claim;
    auto coerce = value.getDefiningOp<CoerceOp>();
    if (!coerce)
      return {};
    value = coerce.getInput();
  }
}

Type mlir::trait::evidenceTypeOf(Value value) {
  if (ClaimType evidence = evidenceOf(value))
    return evidence;
  return value.getType();
}

/// The impl `value` commits to, read in `context`, the committed impl whose
/// body `value` stands in (none for a value outside every impl): a proven
/// claim's impl through its proof, a derived claim's through the derive, and
/// one of `context`'s where arguments through what its derive was given there.
/// None where the value commits to no impl yet. Names resolve in the symbol
/// table `from` stands in, through `symbolTable`.
static std::optional<CommittedImpl>
committedImplOf(Value value, std::shared_ptr<const CommittedImpl> context,
                SymbolTableCollection &symbolTable, Operation *from) {
  value = throughCoercions(value);
  if (auto argument = dyn_cast<BlockArgument>(value);
      argument && context &&
      argument.getOwner()->getParentOp() == ImplOp(context->impl)) {
    unsigned position = argument.getArgNumber();
    DeriveOp derive = context->derive;
    if (position == 0 || !derive ||
        position > derive.getAssumptions().size())
      return std::nullopt;
    return committedImplOf(derive.getAssumptions()[position - 1],
                           context->enclosing, symbolTable, from);
  }
  SpecializationMap none;
  const SpecializationMap &outer = context ? context->arguments : none;
  auto claim = dyn_cast_or_null<ClaimType>(instantiate(value.getType(), outer));
  if (!claim)
    return std::nullopt;
  if (!claim.isProven()) {
    // A derive states its impl's arguments, spelled in the body it stands in,
    // which `outer` carries to this reading.
    auto derive = value.getDefiningOp<DeriveOp>();
    ImplOp impl = derive ? derive.getImplOp(symbolTable) : ImplOp();
    if (!impl)
      return std::nullopt;
    auto arguments = SpecializationMap::fromPositions(llvm::map_range(
        derive.getImplArgs().getTypes(),
        [&](Type argument) { return instantiate(argument, outer); }));
    return CommittedImpl{impl, claim.getTraitApplication(),
                         std::move(arguments), derive, std::move(context)};
  }
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(
      symbolTable, from, claim.getProof(), /*errFn=*/nullptr);
  if (failed(cited))
    return std::nullopt;
  auto proof = dyn_cast<ProofOp>(*cited);
  ImplOp impl = proof ? proof.getImpl(symbolTable) : cast<ImplOp>(*cited);
  if (!impl)
    return std::nullopt;
  SpecializationMap arguments =
      proof ? proof.getImplArguments(symbolTable) : SpecializationMap();
  return CommittedImpl{impl, claim.getTraitApplication(), std::move(arguments),
                       proof ? proof.getDerive(symbolTable) : DeriveOp(),
                       nullptr};
}

LogicalResult ProjectOp::inlineEvidence(RewriterBase &rewriter,
                                        SymbolTableCollection &symbolTable) {
  ClaimType source = getSourceEvidence();
  if (!source)
    return failure();
  auto committed = committedImplOf(getSource(), nullptr, symbolTable, *this);
  if (!committed)
    return failure();
  ImplOp impl = committed->impl;

  // A trait requirement is the impl's return operand at its index; a where
  // entry, past them, is the impl's block argument there, which the source's
  // proof supplies.
  uint64_t traitCount = impl.getTrait(symbolTable).getRequirements().size();
  Block &body = impl.getBody().front();
  uint64_t index = getIndex();
  if (index >= traitCount && index - traitCount + 1 >= body.getNumArguments())
    return failure();
  Value read = index < traitCount ? impl.getReturn().getOperand(index)
                                  : Value(body.getArgument(index - traitCount + 1));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(*this);
  AttrTypeReplacer stamp =
      makeTypeReplacerFromSubstitution(committed->arguments, CloneKind::Instance);
  IRMapping mapping;
  llvm::SetVector<Value> reads;
  reads.insert(read);
  mapDeclarationReads(rewriter, symbolTable, impl, reads, getSource(), source,
                      stamp, mapping);
  Value inlined = mapping.lookupOrNull(read);
  if (!inlined)
    return failure();
  // The inlined ops are located where the impl wrote them, called from the
  // projection, as an inlined call's are.
  for (auto [original, clone] : mapping.getOperationMap())
    clone->setLoc(CallSiteLoc::get(clone->getLoc(), getLoc()));

  // The evidence is spelled as the impl wrote it, which the result may spell
  // otherwise: the two meet as their projections resolve. The coercion
  // bridging them keeps the evidence beneath the result, which names no proof
  // of its own.
  Value replacement = inlined;
  if (inlined.getType() != getResult().getType())
    replacement = CoerceOp::create(rewriter, getLoc(), getResult().getType(),
                                   inlined, ValueRange{});
  rewriter.replaceOp(*this, replacement);
  return success();
}

/// Whether `value`, standing in `impl`'s body, is computed from one of
/// `impl`'s block arguments: its where arguments, or its own claim.
static bool readsArgumentsOf(Value value, ImplOp impl) {
  SmallVector<Value> pending{value};
  DenseSet<Value> seen;
  while (!pending.empty()) {
    Value next = pending.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    if (auto argument = dyn_cast<BlockArgument>(next)) {
      if (argument.getOwner()->getParentOp() == impl.getOperation())
        return true;
      continue;
    }
    llvm::append_range(pending, next.getDefiningOp()->getOperands());
  }
  return false;
}

/// The derivation `committed` stands for, at its application, as `identity`
/// names it: the impl, where it is derived over no evidence -- cited by name,
/// or derived given nothing, which its application alone determines -- and
/// otherwise the derive committing it and, where what that derive is given
/// depends on the impl it stands in, the derivation committing that impl in
/// turn. Two readings meet one derivation when they name it alike.
static void appendDerivation(const CommittedImpl &committed,
                             SmallVectorImpl<const void *> &identity) {
  ImplOp impl = committed.impl;
  DeriveOp derive = committed.derive;
  if (!derive || derive.getAssumptions().empty()) {
    identity.push_back(impl.getOperation());
    return;
  }
  identity.push_back(derive.getOperation());
  if (!committed.enclosing)
    return;
  ImplOp enclosing = committed.enclosing->impl;
  bool dependsOnEnclosing =
      llvm::any_of(derive.getAssumptions(), [&](Value given) {
        return isPolymorphicType(given.getType()) ||
               readsArgumentsOf(given, enclosing);
      });
  if (dependsOnEnclosing)
    appendDerivation(*committed.enclosing, identity);
}

EvidenceReading ProjectOp::readEvidence(SymbolTableCollection &symbolTable) {
  Operation *from = *this;
  EvidenceReading reading;
  // A reading that meets one requirement of one derivation at one application
  // again has no base. The derivation, not only its impl, is what repeats: one
  // impl derived twice at one application over different evidence is two
  // derivations, and reading through both is finite.
  std::set<SmallVector<const void *, 8>> reached;
  using Context = std::shared_ptr<const CommittedImpl>;
  using Ending = std::pair<Value, Context>;

  // The evidence `project`, read in `context`, reads: followed through the
  // returns of the impls it is read through to the value it ends at -- one
  // that is no projection -- with the committed impl that value stands in.
  // None where the reading stops: at evidence that commits to no impl yet, at
  // a where entry, whose evidence the derive or proof supplies, or at a cycle
  // or the depth limit, which `reading.end` records.
  std::function<std::optional<Ending>(ProjectOp, Context)> follow;

  // The impl `value`, read in `context`, commits to. A value an unproven
  // projection produces commits to whatever the evidence it reads commits to,
  // so a projection standing on another is read through the inner one's
  // evidence first; each hop of that reading is a hop of this one.
  auto commit = [&](Value value,
                    Context context) -> std::optional<CommittedImpl> {
    value = throughCoercions(value);
    if (auto inner = value.getDefiningOp<ProjectOp>();
        inner && !inner.getResultClaim().isProven()) {
      std::optional<Ending> ending = follow(inner, context);
      if (!ending)
        return std::nullopt;
      return committedImplOf(ending->first, ending->second, symbolTable, from);
    }
    return committedImplOf(value, context, symbolTable, from);
  };

  follow = [&](ProjectOp project, Context context) -> std::optional<Ending> {
    std::optional<CommittedImpl> current = commit(project.getSource(), context);
    uint64_t index = project.getIndex();
    // A where entry, past the trait's requirements, is evidence a proof or a
    // derive supplies, which is a base; only a requirement is read through a
    // return.
    while (current &&
           index < current->impl.getTrait(symbolTable).getRequirements().size()) {
      ImplOp impl = current->impl;
      reading.impls.push_back(impl.getSymNameAttr());
      SmallVector<const void *, 8> key{current->application.getAsOpaquePointer(),
                                       reinterpret_cast<const void *>(index)};
      appendDerivation(*current, key);
      if (!reached.insert(key).second) {
        reading.end = EvidenceReading::End::Cycle;
        return std::nullopt;
      }
      reading.chain.push_back({current->application, SymbolRefAttr()});
      auto here = std::make_shared<const CommittedImpl>(std::move(*current));
      Value returned = throughCoercions(impl.getReturn().getOperand(index));
      auto next = returned.getDefiningOp<ProjectOp>();
      if (!next)
        return Ending{returned, here};
      if (reading.chain.size() > kInstantiationDepthLimit) {
        reading.end = EvidenceReading::End::Overflow;
        return std::nullopt;
      }
      current = commit(next.getSource(), here);
      index = next.getIndex();
    }
    return std::nullopt;
  };

  (void)follow(*this, nullptr);
  return reading;
}

/// Generate a deterministic symbol name for an ImplOp.
///
/// The name has the form {TraitName}_impl_h{hash} where the hash is a
/// 64-bit xxHash of the full type argument and where-clause signature. This
/// keeps symbols short and bounded in length.
std::string ImplOp::generateSymName(TraitApplicationAttr selfApp,
                                    ArrayRef<ClaimType> where) {
  // The equality entries follow the application entries, so two impls that
  // differ only in an equality assumption synthesize distinct names.
  std::string signature;
  llvm::raw_string_ostream os(signature);
  for (auto ty : selfApp.getTypeArgs())
    os << "_" << ty;
  bool anyApplication = false;
  for (ClaimType entry : where) {
    if (!entry.isApplication())
      continue;
    if (!anyApplication)
      os << "_where";
    anyApplication = true;
    TraitApplicationAttr app = entry.getTraitApplication();
    os << "_" << app.getTraitName().getValue();
    for (auto typeArg : app.getTypeArgs())
      os << "_" << typeArg;
  }
  os << "_eq";
  for (ClaimType entry : where)
    if (TypeEqualityAttr eq = entry.getEqualityAttr())
      os << "_" << eq.getLhs() << "_" << eq.getRhs();
  os.flush();

  return selfApp.getTraitName().getValue().str() + "_impl" + hashToSuffix(signature);
}

std::string ImplOp::generateMangledName(const SpecializationMap &arguments) {
  return getSymName().str() +
         applySubstitutionAndGenerateMangledNameSuffix(arguments,
                                                       getTypeParams());
}

TraitOp ImplOp::getTrait(SymbolTableCollection &symbolTable) {
  return getSelfApplication().getTraitOrAbort(
      symbolTable, *this, "ImplOp::getTrait: couldn't find trait");
}

TypeArguments ImplOp::readTypeArgumentsFor(ClaimType actualSelfClaim,
                                           Normalizer normalize) {
  TypeArguments args(getNumTypeParams());
  extractTypeArguments(Type(getSelfClaim()), Type(actualSelfClaim), args);

  // A parameter the header leaves open is one an equality reading determines
  // (`verifyImplParametersAreConstrained`): the reading's input, instantiated at
  // what is known and read through `normalize`, is the type its determined side
  // is read against, as the header is read against the demand. Determining one
  // parameter can settle another reading's input, so the reading runs until it
  // stops growing.
  auto settled = [&](Type type) {
    return llvm::all_of(getTypeParametersIn(type),
                        [&](GenericTypeInterface inside) {
                          return !args.binds(inside) || args.lookup(inside);
                        });
  };
  auto settledCount = [&] {
    return llvm::count_if(llvm::seq(args.getBound()), [&](unsigned label) {
      return args.at(label).has_value();
    });
  };
  SmallVector<EqualityReading> readings = getEqualityReadings(*this);
  for (bool grew = true; grew;) {
    grew = false;
    for (const EqualityReading &reading : readings) {
      // A determined side with no open parameter has nothing to learn, so its
      // input is not normalized. Nor is an input still spelling an open
      // parameter: normalizing a projection over one selects among every impl
      // of its trait, this one included, and reading this impl's equalities
      // again recurses without end. The round that settles that parameter reads
      // this one.
      if (settled(reading.determined) || !settled(reading.input))
        continue;
      Type value = instantiate(reading.input, args.toSpecialization());
      if (normalize) {
        FailureOr<Type> normalized = normalize(value);
        if (failed(normalized))
          continue;
        value = *normalized;
      }
      auto before = settledCount();
      extractTypeArguments(reading.determined, value, args);
      grew |= settledCount() != before;
    }
  }
  return args;
}

FailureOr<SpecializationMap> ImplOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                     Normalizer normalize,
                                                                     llvm::function_ref<InFlightDiagnostic()> errFn) {
  // The impl's header is the declaration and the demanded application is the
  // use. Its parameters take the arguments standing opposite them, and the
  // header rebuilt at those arguments must be the demand: a position the header
  // spells as a projection determines nothing, so what carries such a header to
  // a demand spelling the resolution is the caller's context, never a narrowing
  // of the demand.
  SpecializationMap arguments =
      readTypeArgumentsFor(actualSelfClaim, normalize).toSpecialization();
  if (failed(verifyEqualAfterInstantiation(Type(getSelfClaim()), arguments,
                                           Type(actualSelfClaim), normalize,
                                           errFn)))
    return failure();
  return arguments;
}

FailureOr<Type> ImplOp::specializeAssociatedTypeBinding(
    StringRef name,
    ArrayRef<Type> assocTypeArgs,
    const SpecializationMap &headerArguments,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto binding = getAssociatedTypeBinding(name, err);
  if (failed(binding)) return failure();

  // The header's parameters and the binding's own take their arguments in one
  // substitution: the binding's own parameters are labelled after the
  // header's, so the union of the two lists is a function, and one pass never
  // revisits a term it stamped -- neither argument list can be read as the
  // parameters the other list answers for.
  SpecializationMap arguments = headerArguments;
  unsigned headerCount = getNumTypeParams();
  for (auto [position, arg] : llvm::enumerate(assocTypeArgs))
    arguments.bind(cast<GenericTypeInterface>(Type(
                       PolyType::get(getContext(), headerCount + position))),
                   arg);

  return instantiate(*binding, arguments);
}

ProjectionNormalizer ImplOp::getOwnBindingsReader() {
  // The bindings are spelled over the impl's own parameters, so they are read
  // at the arguments those parameters take in its own body: themselves. Only
  // a projection of the impl's own application is read here, so only one of
  // those in its arguments is read before it.
  TraitApplicationAttr own = getSelfApplication();
  return ProjectionNormalizer(
      [own](ProjectionType proj) { return proj.getTraitApplication() == own; },
      [impl = getOperation()](ProjectionType proj) -> std::optional<Type> {
        auto bound = cast<ImplOp>(impl).specializeAssociatedTypeBinding(
            proj.getAssocName().getValue(), proj.getAssocTypeArgs(),
            SpecializationMap());
        if (failed(bound))
          return std::nullopt;
        return *bound;
      });
}

//===----------------------------------------------------------------------===//
// ProofOp
//===----------------------------------------------------------------------===//

/// What a proof's returned claim rests on, read through its one respelling:
/// the derive it records, or the application witness of the proof it casts.
static Operation *respelledFrom(Value proven) {
  if (auto coerce = proven.getDefiningOp<CoerceOp>())
    proven = coerce.getInput();
  if (auto derive = proven.getDefiningOp<DeriveOp>())
    return derive;
  auto witness = proven.getDefiningOp<WitnessOp>();
  if (witness && witness.getResultClaim().isApplication() &&
      witness.getProof())
    return witness;
  return nullptr;
}

LogicalResult ProofOp::verify() {
  // A proof is closed and returns the one claim it proves: the claim a derive
  // in its body derives, which is the decision the proof records, spelled as
  // the derive spells it or respelled by one coercion.
  Block &body = getBody().front();
  if (body.getNumArguments() != 0)
    return emitOpError() << "takes no block arguments: a proof is closed";
  auto returned = body.empty() ? ReturnOp() : dyn_cast<ReturnOp>(body.back());
  if (!returned)
    return emitOpError() << "must end with 'trait.return' of the claim it proves";
  if (returned.getNumOperands() != 1 ||
      !respelledFrom(returned.getOperand(0)))
    return emitOpError() << "returns the one claim a derive in its body "
                            "derives, or that claim or another proof's "
                            "respelled";
  // A proof is ground: the stage writes one only at a monomorphic application,
  // so a citation of it is one comparison (`verifyCitation`). A claim over
  // type variables is proven by a derive in the template that holds them.
  if (getProvenClaim().isPolymorphic())
    return emitOpError() << "proves " << Type(getProvenClaim().asUnproven())
                         << ", which spells a type variable: a proof is ground";
  return success();
}

Value ProofOp::getProven() {
  return getBody().front().getTerminator()->getOperand(0);
}

Operation *ProofOp::getCastSource(SymbolTableCollection &symbolTable) {
  auto witness = dyn_cast_or_null<WitnessOp>(respelledFrom(getProven()));
  if (!witness)
    return nullptr;
  return symbolTable.lookupNearestSymbolFrom(*this, witness.getProof());
}

DeriveOp ProofOp::getDerive(SymbolTableCollection &symbolTable) {
  // A cast's source is a root (`WitnessOp::verifySymbolUses`), so the derive is
  // this proof's own or its source's, one hop away.
  Operation *source = getCastSource(symbolTable);
  if (auto proof = dyn_cast_or_null<ProofOp>(source))
    return dyn_cast_or_null<DeriveOp>(respelledFrom(proof.getProven()));
  if (source)
    return DeriveOp();
  return dyn_cast_or_null<DeriveOp>(respelledFrom(getProven()));
}

ImplOp ProofOp::getImpl(SymbolTableCollection &symbolTable) {
  if (DeriveOp derive = getDerive(symbolTable))
    return derive.getImplOp(symbolTable);
  return dyn_cast_or_null<ImplOp>(getCastSource(symbolTable));
}

TraitOp ProofOp::getTrait(SymbolTableCollection &symbolTable) {
  return getTraitApplication().getTraitOrAbort(
      symbolTable, *this, "ProofOp::getTrait: couldn't find trait");
}

SmallVector<ClaimType> ProofOp::getPremises(SymbolTableCollection &symbolTable) {
  DeriveOp derive = getDerive(symbolTable);
  if (!derive)
    return {};
  return llvm::map_to_vector(derive.getAssumptions(), [](Value premise) {
    return cast<ClaimType>(premise.getType());
  });
}

SmallVector<ClaimType> ProofOp::getSubproofs(SymbolTableCollection &symbolTable) {
  SmallVector<ClaimType> subproofs;
  DeriveOp derive = getDerive(symbolTable);
  if (!derive)
    return subproofs;
  for (Value premise : derive.getAssumptions()) {
    if (cast<ClaimType>(premise.getType()).isEquality())
      continue;
    subproofs.push_back(cast<ClaimType>(throughCoercions(premise).getType()));
  }
  return subproofs;
}

/// The proof or impl `name` names in the symbol table `from` stands in, read
/// through `symbolTable`.
static FailureOr<Operation*> lookupProofSymbol(
    SymbolTableCollection &symbolTable,
    Operation *from,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  Operation* symOp = symbolTable.lookupNearestSymbolFrom(from, name);
  if (!symOp) {
    if (errFn) errFn() << "cannot find proof symbol '" << name << "'";
    return failure();
  }

  if (isa<ImplOp>(symOp) || isa<ProofOp>(symOp))
    return symOp;

  if (errFn) errFn() << "proof symbol '" << name << "' must refer to trait.proof or trait.impl";
  return failure();
}

FailureOr<Operation*> ProofOp::getProofOpOrUnconditionalImplOp(
    SymbolTableCollection &symbolTable,
    Operation *from,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto symOp = lookupProofSymbol(symbolTable, from, name, errFn);
  if (failed(symOp)) return failure();

  // if it's an ImplOp, it must be unconditional
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (failed(impl.verifyIsUnconditional(errFn))) return failure();
  }

  return *symOp;
}

FailureOr<Operation *> ProofOp::getRootOf(
    SymbolTableCollection &symbolTable, Operation *from, FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto cited = getProofOpOrUnconditionalImplOp(symbolTable, from, name, errFn);
  if (failed(cited))
    return failure();
  if (auto proof = dyn_cast<ProofOp>(*cited))
    if (Operation *source = proof.getCastSource(symbolTable))
      return source;
  return *cited;
}

//===----------------------------------------------------------------------===//
// WitnessOp
//===----------------------------------------------------------------------===//

// A spelled operand list: operands in parens, then their types in parens,
// `(%a, %b) : (T, U)`. SSA operands resolve against written types. The caller
// resolves the parsed operands once it knows where in the operand list they go.
static ParseResult parseTypedOperandList(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (p.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      p.parseColon() ||
      p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&] {
        Type ty;
        if (p.parseType(ty))
          return failure();
        types.push_back(ty);
        return success();
      }))
    return failure();
  return success();
}

// Prints the `(%a, %b) : (T, U)` form parseTypedOperandList reads. The caller
// prints the keyword that precedes it.
static void printTypedOperandList(OpAsmPrinter &p, ValueRange operands) {
  p << "(";
  llvm::interleaveComma(operands, p, [&](Value v) { p.printOperand(v); });
  p << ") : (";
  llvm::interleaveComma(operands.getTypes(), p, [&](Type t) { p.printType(t); });
  p << ")";
}

ParseResult WitnessOp::parse(OpAsmParser &p, OperationState& result) {
  MLIRContext *ctx = p.getContext();

  auto parseResultType = [&]() -> ParseResult {
    Type resultType;
    if (p.parseColon() || p.parseType(resultType)) return failure();
    result.addTypes(resultType);
    return success();
  };
  auto parsePremises = [&]() -> ParseResult {
    SmallVector<OpAsmParser::UnresolvedOperand> premises;
    SmallVector<Type> premiseTypes;
    if (parseTypedOperandList(p, premises, premiseTypes)) return failure();
    return p.resolveOperands(premises, premiseTypes, p.getCurrentLocation(),
                             result.operands);
  };

  // Equality proj-resolve arm: `proj_resolve !projection resolves !resolved
  // by @impl[args] [given(%premises...) : (types...)] : <result-type>`. The
  // projection and the resolved type are the result equality's two sides,
  // spelled ahead of the citation for the reader and stored once, in the
  // result type.
  if (succeeded(p.parseOptionalKeyword("proj_resolve"))) {
    Type projection, resolved;
    FlatSymbolRefAttr citedImpl;
    ImplArgumentsAttr citedArguments;
    if (p.parseType(projection) || p.parseKeyword("resolves") ||
        p.parseType(resolved) || p.parseKeyword("by") ||
        p.parseAttribute(citedImpl) ||
        parseImplArguments(p, citedArguments))
      return failure();
    result.addAttribute(getImplAttrName(result.name), citedImpl);
    result.addAttribute(getImplArgsAttrName(result.name), citedArguments);
    if (succeeded(p.parseOptionalKeyword("given")) && parsePremises())
      return failure();
    SMLoc resultLoc = p.getCurrentLocation();
    if (parseResultType())
      return failure();
    auto claim = dyn_cast<ClaimType>(result.types.front());
    TypeEqualityAttr equality = claim ? claim.getEqualityAttr() : TypeEqualityAttr();
    if (!equality || equality.getLhs() != projection ||
        equality.getRhs() != resolved)
      return p.emitError(resultLoc)
             << "expected the equality claim " << projection << " = "
             << resolved;
    return success();
  }

  // Equality refl arm: `refl : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("refl"))) {
    result.addAttribute(getReflAttrName(result.name), UnitAttr::get(ctx));
    return parseResultType();
  }

  // Application arm: `@Symbol for @Trait[Types...]`, which is the proven
  // claim the result type is.
  FlatSymbolRefAttr proof;
  if (p.parseAttribute(proof) || p.parseKeyword("for"))
    return failure();
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addTypes(ClaimType::get(p.getContext(), traitApp, proof));

  // parse additional attributes
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void WitnessOp::print(OpAsmPrinter &p) {
  if (isProjectionResolution()) {
    TypeEqualityAttr equality = getResultClaim().getEqualityAttr();
    p << " proj_resolve " << equality.getLhs() << " resolves "
      << equality.getRhs() << " by " << getImplAttr();
    printImplArguments(p, *this, getImplArgsAttr());
    if (!getPremises().empty()) {
      p << " given";
      printTypedOperandList(p, getPremises());
    }
    p << " : " << getResult().getType();
    return;
  }

  if (getRefl()) {
    p << " refl : " << getResult().getType();
    return;
  }

  // Application arm.
  p << " " << getProof() << " for ";
  getResultClaim().getTraitApplication().print(p);

  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elidedAttrs=*/{"impl", "impl_args", "refl"});
}

LogicalResult mlir::trait::verifyEqualitiesRelateTypes(Operation *op) {
  // Evidence is a value whose type is a claim, so an evidence op reads and
  // produces evidence at the root of its operand and result types; a claim
  // nested in another value's type is judged where an evidence op reads it
  // out, so every equality a coerce cites is judged at the coerce.
  auto err = [&] { return op->emitOpError(); };
  auto relatesTypes = [&](Type type) {
    auto claim = dyn_cast<ClaimType>(type);
    return !claim || succeeded(verifyEndpointsRelateTypes(claim, err));
  };
  for (Type type : op->getOperandTypes())
    if (!relatesTypes(type))
      return failure();
  for (Type type : op->getResultTypes())
    if (!relatesTypes(type))
      return failure();
  return success();
}

LogicalResult mlir::trait::verifyReadsUnderDominance(Operation *op) {
  // An op reading no value -- a witness citing a declaration, a refl witness
  // -- has nothing to compute from its own result.
  if (op->getNumOperands() == 0)
    return success();
  // A value `op` reads is defined in a region enclosing it, the outermost one
  // being its nearest isolated ancestor's own.
  for (Region *region = op->getParentRegion(); region;) {
    if (!mlir::mayHaveSSADominance(*region))
      return op->emitOpError()
             << "must be in a region that enforces SSA dominance, and so must "
                "every region around it up to its nearest isolated ancestor, "
                "so that no value it reads is computed from its own result";
    Operation *owner = region->getParentOp();
    if (!owner || owner->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
    region = owner->getParentRegion();
  }
  return success();
}

// The op's attributes must match the result claim's arm exactly. For the
// application arm the result claim names the proof it cites. For the equality
// arm, the result's left side is the projection a cited impl resolves
// (proj-resolve) or its endpoints are identical (refl); an equality no one leaf
// states as written is a trait.coerce of a refl witness, not a witness.
LogicalResult WitnessOp::verify() {
  ClaimType result = dyn_cast<ClaimType>(getResult().getType());
  if (!result)
    return emitOpError() << "result must be a !trait.claim";

  bool resolves = isProjectionResolution();
  bool hasRefl = getRefl();
  // A cited impl is cited at stated arguments, and nothing else states any.
  if (resolves != static_cast<bool>(getImplArgsAttr()))
    return emitOpError() << "states impl arguments exactly where it cites an "
                            "impl";

  // Equality arm.
  if (result.isEquality()) {
    if (resolves && hasRefl)
      return emitOpError() << "an equality witness carries at most one of a "
                              "cited impl or a refl marker";
    TypeEqualityAttr eq = result.getEqualityAttr();

    if (hasRefl) {
      if (!getPremises().empty())
        return emitOpError() << "a refl witness takes no premises";
      if (eq.getLhs() != eq.getRhs())
        return emitOpError() << "a refl witness requires identical endpoints, "
                             << "found " << eq.getLhs() << " and " << eq.getRhs();
      return success();
    }

    // An equality witness is a leaf: one with neither marker would state an
    // equality nothing it carries establishes.
    if (!resolves)
      return emitOpError() << "an equality witness cites an impl resolving a "
                              "projection or carries a refl marker; an "
                              "equality other equalities entail is a "
                              "trait.coerce of a refl witness citing them";

    // proj-resolve: the impl the witness cites resolves the projection its
    // result's left side spells, which its symbol uses verify.
    if (!isa<ProjectionType>(eq.getLhs()))
      return emitOpError() << "a proj-resolve witness resolves a projection, "
                           << "found " << eq.getLhs();
    return success();
  }


  // Application arm.
  if (resolves || hasRefl || !getPremises().empty())
    return emitOpError() << "an application witness carries neither a cited "
                            "impl, a refl marker, nor premises";
  if (!result.isProven())
    return emitOpError() << "an application witness's claim " << result
                         << " names the proof it cites";
  return success();
}

/// Refuses `supplied` unless it holds one claim per entry of `expected`, each
/// that entry, read modulo the evidence it names and, where `normalize` is
/// given, through it. `owner` names what states the entries and `supplier` the
/// op supplying them, as a refusal reads them.
static LogicalResult verifyPremisesSuppliedByPosition(
    ValueRange supplied, ArrayRef<ClaimType> expected, const Twine &owner,
    StringRef supplier, llvm::function_ref<InFlightDiagnostic()> errFn) {
  if (supplied.size() != expected.size())
    return errFn() << owner << " states " << expected.size()
                   << " premises, and the " << supplier << " supplies "
                   << supplied.size();
  for (auto [position, pair] : llvm::enumerate(llvm::zip(supplied, expected))) {
    auto [operand, premise] = pair;
    if (!equalModuloProofs(operand.getType(), premise))
      return errFn() << "premise " << position << " of " << owner << " is "
                     << premise << ", and the " << supplier << " supplies "
                     << stripClaimProofs(operand.getType());
  }
  return success();
}

/// Whether the citation a derive or a projection-resolution witness makes of
/// `impl` states one argument per parameter, `stated`, and, at those
/// `arguments`, the header `cited`, compared by identity, and the where entries
/// `premises` supply, positionally, each compared with the proof its claim
/// names stripped (`verifyPremisesSuppliedByPosition`).
static LogicalResult verifyCitationOf(
    ImplOp impl, ClaimType cited, ImplArgumentsAttr stated,
    const SpecializationMap &arguments, ValueRange premises,
    StringRef supplier, llvm::function_ref<InFlightDiagnostic()> errFn) {
  size_t parameterCount = impl.getNumTypeParams();
  if (stated.getTypes().size() != parameterCount)
    return errFn() << "impl '@" << impl.getSymName() << "' takes "
                   << parameterCount << " type arguments, and the " << supplier
                   << " states " << stated.getTypes().size();
  if (impl.getSelfApplicationAt(arguments) != cited.getTraitApplication())
    return errFn() << "impl '@" << impl.getSymName()
                   << "' at the arguments the citation gives it proves "
                   << ClaimType::get(cited.getContext(),
                                     impl.getSelfApplicationAt(arguments))
                   << ", not " << cited.asUnproven();
  return verifyPremisesSuppliedByPosition(
      premises, impl.getWhereClaimsAt(arguments),
      "impl '@" + impl.getSymName() + "'", supplier, errFn);
}

LogicalResult WitnessOp::verifyResolution(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };
  TypeEqualityAttr equality = getResultClaim().getEqualityAttr();
  auto projection = cast<ProjectionType>(equality.getLhs());
  auto impl = symbolTable.lookupNearestSymbolFrom<ImplOp>(*this, getImplAttr());
  if (!impl)
    return errFn() << "cannot find trait.impl '" << getImplAttr()
                   << "' cited by the witness";

  // The impl at the arguments the witness states proves the projection's
  // application over the premises, one per where entry, as a derive's does.
  ClaimType cited = projection.asClaim();
  SpecializationMap arguments = getImplArguments();
  if (failed(verifyCitationOf(impl, cited, getImplArgsAttr(), arguments,
                              getPremises(), "witness", errFn)))
    return failure();

  auto bound = impl.specializeAssociatedTypeBinding(
      projection.getAssocName().getValue(), projection.getAssocTypeArgs(),
      arguments, errFn);
  if (failed(bound))
    return failure();
  if (*bound != equality.getRhs())
    return errFn() << "impl '" << getImplAttr() << "' binds the projection to "
                   << *bound << ", not the certified resolution "
                   << equality.getRhs();
  return success();
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Equality proj-resolve arm: the cited impl, at the arguments the
  // projection's application and the premises give it, binds the projection
  // as stated.
  if (isProjectionResolution())
    return verifyResolution(symbolTable);

  // The refl arm cites nothing by symbol: its evidence is the spelling.
  if (getResultClaim().isEquality())
    return success();

  // Application arm: the declaration the witness names proves the application
  // its result claims, which the claim type holds of every proven claim
  // (`ClaimType::verifySymbolUses`).
  //
  // A witness in a proof's body names a root: a proof whose body holds its
  // derive, or an unconditional impl. A cast is then one hop from the derive
  // it records and a premise names the proof its evidence rests on, so a chain
  // of casts -- a proof casting itself, two casting each other -- is never
  // written: a respelling of a respelling is one coercion of the root citing
  // both steps' equalities.
  if (isa<ProofOp>(getOperation()->getParentOp()))
    if (auto cited =
            symbolTable.lookupNearestSymbolFrom<ProofOp>(*this, getProof());
        cited && cited.getCastSource(symbolTable))
      return emitOpError() << "names the cast " << getProof()
                           << ": a witness in a proof's body names a root, a "
                              "proof whose body holds its derive or an "
                              "unconditional impl";
  return success();
}


//===----------------------------------------------------------------------===//
// DeriveOp
//===----------------------------------------------------------------------===//

ImplOp DeriveOp::getImplOp(SymbolTableCollection &symbolTable) {
  return symbolTable.lookupNearestSymbolFrom<ImplOp>(*this, getImplAttr());
}

/// Verifies a derive: it states one argument per parameter of the impl, the
/// derived application is the impl's header at them, and each operand's claim,
/// with the proof it names stripped, is the impl's where-clause entry at them,
/// in order.
LogicalResult DeriveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };
  ImplOp impl = getImplOp(symbolTable);
  if (!impl)
    return errFn() << "cannot find trait.impl '" << getImplAttr() << "'";
  return verifyCitationOf(impl, getDerivedClaim(), getImplArgs(),
                          getImplArguments(), getAssumptions(), "derive",
                          errFn);
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // An impl returns evidence, which its own verifier holds against its
  // trait's requirements; a proof returns the one application it proves.
  if (!llvm::all_of(getOperandTypes(), llvm::IsaPred<ClaimType>))
    return emitOpError() << "returns claims only, from an impl or a proof";
  if (isa<ProofOp>((*this)->getParentOp()) &&
      (getNumOperands() != 1 ||
       !cast<ClaimType>(getOperand(0).getType()).isApplication()))
    return emitOpError() << "returns the one application claim its proof proves";
  return success();
}


//===----------------------------------------------------------------------===//
// CoerceOp
//===----------------------------------------------------------------------===//

LogicalResult CoerceOp::verify() {
  // A verdict that is a pure function of op, operands, and attributes.

  // A coerce reads its input where SSA dominance holds
  // (`ReadsUnderDominance`), so the chain of coercions defining a value is
  // acyclic and a reader of evidence passing through them reaches its end.

  // 1. A coerce respells one claim at its root or a type that holds no claim:
  // a claim nested in another type stands there as a value's own type, which a
  // coerce of the container never respells. So the root is the one position a
  // proof can sit.
  for (Type endpoint : {getInput().getType(), getResult().getType()})
    if (!isa<ClaimType>(endpoint) && containsClaim(endpoint))
      return emitOpError() << "may not respell the claim nested in " << endpoint
                           << ": a coerce respells a claim at its root or a "
                              "type that holds no claim";

  // 2. The result names no proof its input does not carry. A proven claim
  // names the proof of exactly its own application
  // (`ClaimType::verifySymbolUses`), so a proven result naming its input's
  // proof is the input's type unchanged; what this refuses is a proof minted
  // on an unproven input, or another proof exchanged for the input's (two
  // impls of one application, each with its proof). A result dropping its
  // input's proof is admitted, and the endpoints are compared without proofs.
  auto fromClaim = dyn_cast<ClaimType>(getInput().getType());
  auto toClaim = dyn_cast<ClaimType>(getResult().getType());
  if (toClaim && toClaim.isProven() &&
      (!fromClaim || fromClaim.getProof() != toClaim.getProof()))
    return emitOpError() << "names " << toClaim.getProof() << " for "
                         << toClaim.getTraitApplication()
                         << ", a proof its input does not carry: a coerce "
                            "neither mints a proof nor exchanges one";
  Type input = fromClaim ? Type(fromClaim.asUnproven()) : getInput().getType();
  Type result = toClaim ? Type(toClaim.asUnproven()) : getResult().getType();

  // 3. Collect the cited equalities; each operand must be an equality claim.
  SmallVector<TypeEqualityAttr> cited;
  for (Value e : getEqualities()) {
    auto claim = dyn_cast<ClaimType>(e.getType());
    if (!claim || !claim.isEquality())
      return emitOpError() << "coerce cites equality claims, but operand has "
                              "type " << e.getType();
    cited.push_back(claim.getEqualityAttr());
  }

  // 4. The two endpoints must fall in one class of the ground congruence
  // closure the cited equalities seed -- the shared entailment decision.
  if (!entailedByGroundCongruence(input, result, cited))
    return emitOpError() << "input type " << getInput().getType()
                         << " and result type " << getResult().getType()
                         << " are not equal under the cited equalities";

  return success();
}

OpFoldResult CoerceOp::fold(FoldAdaptor) {
  // The zero-evidence reflexive form is the discharged terminal state: it folds
  // to its operand, and any cited evidence then dies by ordinary DCE.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
}


//===----------------------------------------------------------------------===//
// FuncCallOp
//===----------------------------------------------------------------------===//

/// The type arguments a generic call supplies for the parameters `args` holds
/// slots for, read off the call's own types; a slot already filled is an
/// argument fixed before the call is read.
///
/// A call spells its operand, claim and result types and the callee's
/// declaration spells the same positions with its parameters standing in them,
/// so the pairing is a reading of one against the other -- the same one-way
/// reader impl selection runs (`extractTypeArguments`): every position outside
/// a projection first, then a projection the actual side spells the same way,
/// and a second differing reading keeps the first.
///
/// A parameter standing only inside a projection's associated-type arguments is
/// determined by a later round where the stage reads the call through impl
/// selection (`normalize`): the declaration is rebuilt at what has been read
/// and normalized, which reduces a projection whose head the reading has
/// grounded and exposes the positions its arguments stand in. Rounds stop when
/// one fills nothing new, and a parameter no position determines is refused,
/// named. Without selection a round would reread the positions the first one
/// read, so none runs.
///
/// Nothing here decides the verdict: filling a slot wrongly can only make the
/// rebuilt declaration differ from what the call spells, which
/// `verifyEqualAfterInstantiation` refuses.
static FailureOr<SpecializationMap> readTypeArguments(
    TypeArguments args, Type formal, Type actual,
    llvm::function_ref<Type(Type)> normalize, StringRef callee,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto filled = [&] {
    return llvm::count_if(llvm::seq(args.getBound()), [&](unsigned label) {
      return args.at(label).has_value();
    });
  };

  // An equality claim's endpoints are the one position an instance does not
  // resolve: stamping the callee rebinds the variables inside them and nothing
  // else, so its equality parameters take exactly the spelling a parameter is
  // read as there. Every other position is compared through normalization,
  // which reduces that spelling wherever the call spells it resolved. So a
  // parameter an equality operand spells is read there first.
  auto formalFn = dyn_cast<FunctionType>(formal);
  auto actualFn = dyn_cast<FunctionType>(actual);
  if (formalFn && actualFn && formalFn.getNumInputs() == actualFn.getNumInputs())
    for (auto [formalInput, actualInput] :
         llvm::zip(formalFn.getInputs(), actualFn.getInputs())) {
      auto formalClaim = dyn_cast<ClaimType>(formalInput);
      auto actualClaim = dyn_cast<ClaimType>(actualInput);
      if (formalClaim && actualClaim && formalClaim.isEquality() &&
          actualClaim.isEquality())
        extractTypeArguments(formalInput, actualInput, args);
    }
  extractTypeArguments(formal, actual, args);
  for (unsigned before = filled(); normalize && !args.complete(); ) {
    // The rebuilt declaration is read again as normalized: what normalization
    // could not reduce is the spelling already read, so a round learning
    // nothing stops the reading and the comparison downstream owns the verdict.
    Type exposed = normalize(instantiate(formal, args.toSpecialization()));
    extractTypeArguments(exposed, actual, args);
    unsigned after = filled();
    if (after == before)
      break;
    before = after;
  }

  if (!args.complete()) {
    if (err) {
      InFlightDiagnostic diagnostic = err();
      diagnostic << "call to @" << callee
                 << " determines no type argument for";
      for (unsigned label : llvm::seq(args.getBound()))
        if (!args.at(label))
          diagnostic << " " << Type(PolyType::get(formal.getContext(), label));
    }
    return failure();
  }

  return args.toSpecialization();
}


FailureOr<SpecializationMap> mlir::trait::readCallTypeArguments(
    Operation *call, FunctionType formal, unsigned bound, FunctionType actual,
    StringRef callee, llvm::function_ref<Type(Type)> selection,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return selection ? selection(ty) : ty;
  };

  // At pass time the comparison reads softly: a call whose evidence is not yet
  // proven waits for it. A call whose operands are already
  // monomorphic says everything it will ever say about the instance it wants,
  // so a parameter its types do not determine is refused here, named, rather
  // than surfacing later as a type variable nothing bound.
  auto reportHere = [&]() -> InFlightDiagnostic { return call->emitOpError(); };
  llvm::function_ref<InFlightDiagnostic()> refusal = err;
  if (!refusal && llvm::all_of(call->getOperandTypes(), isMonomorphicType))
    refusal = reportHere;

  auto arguments = readTypeArguments(TypeArguments(bound), Type(formal),
                                     Type(actual), selection, callee, refusal);
  if (failed(arguments)) return failure();

  // One identity: the callee's declaration instantiated at those arguments is
  // the signature spelled here.
  if (failed(verifyEqualAfterInstantiation(Type(formal), *arguments,
                                           Type(actual), normalize, err)))
    return failure();

  return arguments;
}

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };
  auto callee = getCallee(symbolTable, errFn);
  if (failed(callee))
    return failure();

  // A free function's labels are its parameters' positions, so a signature
  // skipping one binds a parameter no call determines: the callee is refused
  // where it is declared, naming the label it skips.
  FunctionType formal = callee->getFunctionType();
  BitVector labels = getSpelledLabels(Type(formal));
  if (int skipped = labels.find_first_unset(); skipped >= 0)
    return callee->emitOpError()
           << "labels its type parameters 0, 1, ... by position, and its "
              "signature skips "
           << Type(PolyType::get(getContext(), skipped));

  // The callee's declaration binds the parameters its signature spells, and
  // this call's own types determine each of them. A verifier holds no impl
  // selection, so the comparison reads both signatures as written.
  if (failed(readCallTypeArguments(getOperation(), formal, labels.size(),
                                   getActualFunctionType(), getCalleeName(),
                                   /*selection=*/nullptr, errFn)))
    return failure();
  return success();
}

FailureOr<MemberCall> mlir::trait::readMemberCall(
    FuncCallOp call, SymbolTableCollection &symbolTable) {
  // A member's leading parameter is its trait's own application, and the trait
  // lists it among its members; a member's position there is the only reading
  // of the callee's name.
  auto callee = call.getCallee(symbolTable);
  if (failed(callee) || callee->getNumArguments() == 0 ||
      call.getNumOperands() == 0)
    return MemberCall();
  auto self = dyn_cast<ClaimType>(callee->getArgumentTypes().front());
  if (!self || !self.isApplication() || self.isProven())
    return MemberCall();
  auto trait = self.getTraitApplication().getTrait(symbolTable, call);
  if (failed(trait) || !trait->getMemberPosition(call.getCalleeNameAttr()))
    return MemberCall();

  // The receiver's proof names the impl, through the root it rests on, and the
  // impl's pair for the member names the function implementing it; a member
  // the impl pairs with nothing is left to its trait's default, the member's
  // own function.
  ClaimType receiver = evidenceOf(call.getOperand(0));
  if (!receiver)
    return failure();
  auto root = ProofOp::getRootOf(symbolTable, call, receiver.getProof());
  if (failed(root))
    return failure();
  auto proof = dyn_cast<ProofOp>(*root);
  ImplOp impl = proof ? proof.getImpl(symbolTable) : cast<ImplOp>(*root);
  if (!impl)
    return failure();
  FlatSymbolRefAttr implemented = impl.getImplementationOf(call.getCalleeNameAttr());
  if (!implemented)
    implemented = call.getCalleeNameAttr();
  auto function =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(call, implemented);
  if (!function)
    return failure();

  MemberCall member;
  member.function = function;
  member.root = ClaimType::get(
      call.getContext(),
      proof ? proof.getTraitApplication() : impl.getSelfApplication(),
      FlatSymbolRefAttr::get(cast<SymbolOpInterface>(*root).getNameAttr()));
  if (implemented != call.getCalleeNameAttr()) {
    member.derive = proof ? proof.getDerive(symbolTable) : DeriveOp();
    member.whereCount = impl.getWhereClaims().size();
  }
  return member;
}

/// Maps in `mapping` each of `member.function`'s where parameters that its
/// body reads -- the impl's block arguments past the first, which follow the
/// receiver -- to a clone of the premise the root's derive is given at its
/// position, cloned at `rewriter`'s insertion point from the proof's body,
/// which is ground. A where parameter its body does not read is mapped to
/// nothing.
static void mapWhereParameters(RewriterBase &rewriter, MemberCall member,
                               Block &entry, IRMapping &mapping) {
  if (!member.derive)
    return;
  AttrTypeReplacer asWritten;
  IRMapping fromProof;
  for (auto [parameter, premise] :
       llvm::zip(entry.getArguments().slice(1, member.whereCount),
                 member.derive.getAssumptions()))
    if (!parameter.use_empty())
      mapping.map(parameter,
                  cloneDefiningTree(rewriter, premise, fromProof, asWritten));
}

func::FuncOp mlir::trait::cutMemberInstance(
    RewriterBase &rewriter, MemberCall member, StringRef instanceName,
    const SpecializationMap &variables, const ProjectionBindings &projections) {
  func::FuncOp instance = specializePolymorph(
      rewriter, member.function, instanceName, variables, projections);
  if (!instance || member.whereCount == 0)
    return instance;

  // The where parameters take what the root's derive is given, read in the
  // instance's own body, so the instance is named and called by its receiver
  // and its other operands alone.
  Block &entry = instance.getBody().front();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entry);
  IRMapping premises;
  mapWhereParameters(rewriter, member, entry, premises);
  BitVector dropped(entry.getNumArguments());
  for (BlockArgument parameter : entry.getArguments().slice(1, member.whereCount)) {
    if (Value premise = premises.lookupOrNull(parameter))
      rewriter.replaceAllUsesWith(parameter, premise);
    dropped.set(parameter.getArgNumber());
  }
  rewriter.modifyOpInPlace(instance, [&] {
    (void)instance.eraseArguments(dropped);
  });
  return instance;
}

LogicalResult mlir::trait::inlineMemberAt(PatternRewriter &rewriter,
                                          FuncCallOp call,
                                          MemberCall member,
                                          Value receiver,
                                          const CallSubstitution &subst) {
  Region &body = member.function.getBody();
  if (body.empty() || !body.hasOneBlock())
    return rewriter.notifyMatchFailure(call, "the member has no one-block body");

  // The body's receiver takes `receiver`, its where parameters the premises
  // the root's derive is given, and its inputs the call's other operands.
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(
      subst.getSpecialization(), CloneKind::Instance,
      subst.getProjectionBindings());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(call);
  Block &entry = body.front();
  IRMapping mapping;
  mapping.map(entry.getArgument(0), receiver);
  mapWhereParameters(rewriter, member, entry, mapping);
  for (auto [parameter, operand] :
       llvm::zip(entry.getArguments().drop_front(1 + member.whereCount),
                 call.getOperands().drop_front()))
    mapping.map(parameter, operand);
  // A where parameter the body does not read is no block argument of the
  // inlined clone either: the clone reads nothing through it.
  for (BlockArgument parameter :
       entry.getArguments().slice(1, member.whereCount))
    if (!mapping.contains(parameter))
      mapping.map(parameter, receiver);

  // The body is cloned and then stamped, as an instance's is.
  auto cloneBody = [&](OpBuilder &, Region *src, Block *, Block *postInsertBlock,
                       IRMapping &mapper, bool) {
    cloneRegionStampedBefore(rewriter, *src, *postInsertBlock->getParent(),
                             postInsertBlock->getIterator(), mapper, stamp);
  };
  // The inlined ops are located at the call, called from it.
  CalleeBodyInliner interface(call.getContext(), rewriter);
  if (failed(inlineRegion(interface, cloneBody, &body, call, mapping,
                          call->getResults(), call->getResultTypes(),
                          call.getLoc())))
    return rewriter.notifyMatchFailure(call, "the member's body is not inlined");
  rewriter.eraseOp(call);
  return success();
}

FailureOr<func::FuncOp> FuncCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter, SymbolTableCollection &symbolTable,
    const CallSubstitution &subst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  auto callee = getCallee(symbolTable);
  if (failed(callee)) return failure();

  // The instance is the one this call's type arguments and evidence name: a
  // template binding no type parameter is named by its evidence alone. The
  // specialization map is written when the substitution is built and is not
  // touched by closing it, so the arguments read here and the body cut below
  // are read off one object.
  SmallVector<Type> typeArguments;
  for (GenericTypeInterface parameter : getCalleeTypeParams(symbolTable))
    typeArguments.push_back(subst.getSpecialization().apply(parameter));
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(
      subst.getSpecialization(), CloneKind::Instance,
      subst.getProjectionBindings());
  // A position whose evidence is not yet proven names no instance yet: the
  // call waits for it, and one that never gets it is named by the stage's exit
  // walk.
  // Each operand is read at the evidence it carries (`evidenceTypeOf`).
  auto key = InstanceKey::get(getCalleeNameAttr(), typeArguments,
                              callee->getFunctionType().getInputs(),
                              llvm::map_to_vector(getOperands(), evidenceTypeOf),
                              stamp, respell);
  if (failed(key))
    return failure();

  func::FuncOp instance = getOrCutInstance(
      rewriter, symbolTable, *callee, *key, [&](StringRef instanceName) {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(*callee);
        // An external declaration has no body to clone; specialization refuses
        // it, so this call has no instance to name.
        return specializePolymorph(rewriter, *callee, instanceName,
                                   subst.getSpecialization(),
                                   subst.getProjectionBindings());
      });
  if (!instance)
    return failure();
  return instance;
}


//===----------------------------------------------------------------------===//
// ProjectOp
//===----------------------------------------------------------------------===//

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };
  auto requirement = getClaimRequirementAt(getSourceClaim(), symbolTable, *this,
                                           getIndex(), errFn);
  if (failed(requirement))
    return failure();

  // The result type is the selection: the index decides which claim this op
  // produces, so the written one must be that claim. A trait requirement is
  // the trait's at the source's arguments, which names no proof; its evidence
  // is the impl's return the stage inlines here (`inlineEvidence`), never a
  // proof the result names. A where entry is the claim the source's proof
  // gives its derive there, proven or not.
  if (*requirement != getResultClaim())
    return emitOpError() << "type mismatch: expected " << *requirement
                         << " but found " << getResultClaim();

  return success();
}
