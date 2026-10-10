// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
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
/// Parse and print the optional `private`/`nested` keyword a template op leads
/// with, so `trait.proof private @p ...` reads and prints as `func.func` does.
/// A public template elides the keyword.
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
  if (visibility && visibility.getValue() != "public")
    printer << visibility.getValue();
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
                                              ::mlir::ArrayAttr &arguments) {
  ::llvm::SmallVector<::mlir::Attribute> types;
  if (parser.parseCommaSeparatedList(
          ::mlir::OpAsmParser::Delimiter::OptionalSquare,
          [&]() -> ::mlir::ParseResult {
            ::mlir::Type type;
            if (parser.parseType(type))
              return ::mlir::failure();
            types.push_back(::mlir::TypeAttr::get(type));
            return ::mlir::success();
          }))
    return ::mlir::failure();
  arguments = ::mlir::ArrayAttr::get(parser.getContext(), types);
  return ::mlir::success();
}

static void printImplArguments(::mlir::OpAsmPrinter &printer,
                               ::mlir::Operation *, ::mlir::ArrayAttr arguments) {
  if (arguments.empty())
    return;
  printer << "[";
  ::llvm::interleaveComma(arguments.getAsValueRange<::mlir::TypeAttr>(),
                          printer);
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

/// A trait, impl or proof is a template: monomorphization cuts its instances
/// and the collector after erase takes what nothing names. Collection may only
/// take a symbol nothing outside its table may name, so a template is private
/// from birth at its birth site and a public one is refused where it is
/// written, not discovered at erase. The upstream twin is the rule that a
/// declaration cannot be public.
LogicalResult verifyTemplateIsNotPublic(Operation *op) {
  if (SymbolTable::getSymbolVisibility(op) != SymbolTable::Visibility::Public)
    return success();
  return op->emitOpError()
         << "must not be public: it is a template monomorphization "
            "instantiates and collection then takes, so it is private from "
            "birth";
}

/// Parses a declaration's function-like header after its visibility,
/// `@name(%a: T, ...) [-> results]`, into `result`'s symbol name, `arguments`
/// and `results`. A declaration's arguments and results carry no attributes.
ParseResult parseDeclarationHeader(OpAsmParser &parser, OperationState &result,
                                   SmallVectorImpl<OpAsmParser::Argument> &arguments,
                                   SmallVectorImpl<Type> &results) {
  StringAttr visibility;
  (void)parseVisibilityKeyword(parser, visibility);
  if (visibility)
    result.addAttribute(SymbolTable::getVisibilityAttrName(), visibility);
  StringAttr name;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  bool isVariadic = false;
  SmallVector<DictionaryAttr> resultAttrs;
  SMLoc signatureLoc = parser.getCurrentLocation();
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/false, arguments, isVariadic, results,
          resultAttrs))
    return failure();
  bool attributed = llvm::any_of(arguments, [](const OpAsmParser::Argument &arg) {
    return arg.attrs && !arg.attrs.empty();
  }) || llvm::any_of(resultAttrs, [](DictionaryAttr attrs) {
    return attrs && !attrs.empty();
  });
  if (attributed)
    return parser.emitError(signatureLoc)
           << "a declaration's arguments and results carry no attributes";
  return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}

/// Prints a declaration's header as `parseDeclarationHeader` reads it, the
/// arguments named by `op`'s body, followed by the attributes no header
/// position states.
void printDeclarationHeader(OpAsmPrinter &p, Operation *op, TypeRange results,
                            ArrayRef<StringRef> elided) {
  p << ' ';
  printVisibilityKeyword(p, op,
                         op->getAttrOfType<StringAttr>(
                             SymbolTable::getVisibilityAttrName()));
  if (op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()))
    p << ' ';
  p.printSymbolName(
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).getValue());
  Region &body = op->getRegion(0);
  call_interface_impl::printFunctionSignature(
      p, body.front().getArgumentTypes(), /*argAttrs=*/nullptr,
      /*isVariadic=*/false, results, /*resultAttrs=*/nullptr, &body,
      /*printEmptyResult=*/false);
  SmallVector<StringRef> omitted{SymbolTable::getSymbolAttrName(),
                                 SymbolTable::getVisibilityAttrName()};
  llvm::append_range(omitted, elided);
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), omitted);
}

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

/// The function type of a child method a parent's verifier is about to read.
///
/// A child's own invariants are verified after its parent's, so the type is read
/// through the attribute dictionary rather than through the getter that casts:
/// a malformed one is refused where it stands instead of aborting the cast.
/// `FunctionOpInterface` requires every implementer to hold its type in an
/// attribute named `function_type`.
static FailureOr<FunctionType> readChildFunctionType(FunctionOpInterface function) {
  StringLiteral attrName = "function_type";
  auto typeAttr = function->getAttrOfType<TypeAttr>(attrName);
  auto functionType =
      typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType();
  if (!functionType) {
    function.emitOpError()
        << "requires a function type in its '" << attrName << "' attribute";
    return failure();
  }
  return functionType;
}

/// Verifies that a function's result generics are determined by its inputs.
///
/// Generics supplied by the caller, such as trait-level parameters on
/// a trait method, are treated as already determined. Every other generic in a
/// result must also appear in an input type, including claim inputs that encode
/// ordinary where-clause evidence. Otherwise function monomorphization has no
/// source of evidence for choosing that result type. A claim result is the one
/// exception: a call of the function states its result types, and a claim it
/// returns is evidence the stage proves where the call is monomorphic rather
/// than an instance it cuts, so a generic only a claim result spells is
/// determined by the call that spells it. This is intentionally a syntactic
/// check: the verifier does not try to invert equality predicates or
/// associated-type bindings to recover missing result generics.
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
bool projectsOwnApplication(Value value, ImplOp impl) {
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
    auto proof = lookupSymbolFrom<ProofOp>(
        impl->getParentOfType<ModuleOp>(), witness.getProof());
    return proof && proof.getDerive().getImpl() == impl.getSymName() &&
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

/// The type variable entry `declared` of `assoc`'s type parameter list
/// declares, refused where it stands when the entry is not a type or not a type
/// variable. A child's own invariants are verified after its parent's, so the
/// entry is read as an attribute that may be anything rather than cast.
static FailureOr<Type> readAssociatedTypeParameter(AssocTypeOp assoc,
                                                   Attribute declared) {
  auto typeAttr = dyn_cast<TypeAttr>(declared);
  if (!typeAttr)
    return assoc.emitOpError()
           << "type parameter list holds " << declared << ", which is not a type";
  Type param = typeAttr.getValue();
  if (!isa<GenericTypeInterface>(param))
    return assoc.emitOpError()
           << "type parameter list holds " << param
           << ", which is not a type variable";
  return param;
}

ParseResult TraitOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> requirements;
  if (parseDeclarationHeader(parser, result, arguments, requirements))
    return failure();
  result.addAttribute(getRequirementsAttrName(result.name),
                      parser.getBuilder().getTypeArrayAttr(requirements));
  return parser.parseRegion(*result.addRegion(), arguments,
                            /*enableNameShadowing=*/false);
}

void TraitOp::print(OpAsmPrinter &p) {
  SmallVector<Type> requirements(getRequirements().getAsValueRange<TypeAttr>());
  printDeclarationHeader(p, *this, requirements,
                         {getRequirementsAttrName().getValue()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

void TraitOp::getAsmBlockArgumentNames(Region &region,
                                       OpAsmSetValueNameFn setNameFn) {
  nameDeclarationArguments(region, setNameFn);
}

LogicalResult TraitOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();

  // The one block argument is the trait's own application, whose arguments are
  // the trait's parameters, at least one: parameter `i` is an occurrence of
  // label `i`, its position.
  Block &body = getBody().front();
  auto self = body.getNumArguments() == 1
                  ? dyn_cast<ClaimType>(body.getArgument(0).getType())
                  : ClaimType();
  if (!self || !self.isApplication() || self.isProven() ||
      self.getTraitApplication().getTraitName().getValue() != getSymName())
    return emitOpError() << "takes one block argument, the unproven claim of "
                            "its own application @"
                         << getSymName() << "[...]";
  for (auto [position, ty] : llvm::enumerate(getTypeParams())) {
    auto label = dyn_cast_or_null<PolyType>(Type(getParameterOccurrence(ty)));
    if (!label || static_cast<size_t>(label.getLabel()) != position)
      return emitOpError() << "parameter " << position << " must be labelled "
                           << position << ", its position, found " << ty;
  }
  if (getTypeParams().empty())
    return emitOpError() << "requires at least one type parameter";
  DenseSet<Type> uniqueParams(getTypeParams().begin(), getTypeParams().end());

  // Collect the GAT parameters from the AssocTypeOp type_params, each of which
  // is a parameter of its own declaration: a projection through the associated
  // type supplies an argument for it, while the trait's own parameters come from
  // the application. A GAT that repeats one of the trait's parameters would have
  // the projection's argument overwrite the application's, so the two lists must
  // stand apart.
  //
  // A parameter is a type variable: it is the key a projection's argument is
  // substituted for, so a ground type standing in the list would carry every
  // occurrence of that same type in the binding away with it.
  DenseSet<Type> gatParams;
  for (Operation &op : body) {
    auto assoc = dyn_cast<AssocTypeOp>(op);
    if (!assoc)
      continue;
    ArrayAttr declaredParams = assoc.getTypeParamsAttr();
    if (!declaredParams)
      continue;
    for (Attribute declared : declaredParams) {
      FailureOr<Type> param = readAssociatedTypeParameter(assoc, declared);
      if (failed(param))
        return failure();
      if (uniqueParams.contains(*param))
        return assoc.emitOpError()
               << "type parameter " << *param << " is already a parameter of trait '@"
               << getSymName() << "'";
      gatParams.insert(*param);
    }
  }

  // An endpoint mentions a type parameter when any generic hiding inside it is
  // one of the trait's parameters or a GAT parameter; getGenericTypesIn descends
  // the attributes -- a trait application's arguments, an equality's endpoints --
  // that its own walk over immediate type sub-elements does not reach.
  auto mentionsParam = [&](Type ty) {
    for (GenericTypeInterface g : getGenericTypesIn(ty))
      if (uniqueParams.contains(Type(g)) || gatParams.contains(Type(g)))
        return true;
    return false;
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

  // check trait method result generics
  for (Operation &op : body) {
    if (auto method = dyn_cast<FunctionOpInterface>(op)) {
      auto methodType = readChildFunctionType(method);
      if (failed(methodType))
        return failure();
      if (failed(verifyFunctionResultGenericsAreDetermined(method, *methodType,
                                                           uniqueParams)))
        return failure();
    }
  }

  return success();
}

LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  // The requirements are types stored in an attribute, which the
  // symbol-use driver does not reach.
  for (Type requirement : getRequirements().getAsValueRange<TypeAttr>())
    if (failed(cast<ClaimType>(requirement).verifySymbolUses(getOperation(),
                                                             symbolTable)))
      return failure();
  return success();
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

SmallVector<ClaimType> TraitOp::getRequirementsAsClaims() {
  return llvm::map_to_vector(getRequirements().getAsValueRange<TypeAttr>(),
                             [](Type requirement) {
                               return cast<ClaimType>(requirement);
                             });
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
  auto module = getModule();
  if (failed(module)) return {};

  // Impls are top-level module children, so scan them directly and match this
  // trait's symbol name. This avoids a full-module symbol-use walk, which
  // materializes every operation's attribute dictionary.
  StringRef traitName = getSymName();
  SmallVector<ImplOp> result;
  for (Operation &op : *module->getBody()) {
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
  SmallVector<OpAsmParser::Argument> arguments;
  SmallVector<Type> results;
  SMLoc headerLoc = parser.getCurrentLocation();
  if (parseDeclarationHeader(parser, result, arguments, results))
    return failure();
  if (!results.empty())
    return parser.emitError(headerLoc)
           << "an impl states no results: it returns its trait's";
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, /*enableNameShadowing=*/false))
    return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void ImplOp::print(OpAsmPrinter &p) {
  printDeclarationHeader(p, *this, /*results=*/{}, /*elided=*/{});
  p << ' ';
  // An impl of a trait requiring nothing returns nothing, and its empty return
  // is the one the parser supplies.
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
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
/// the bindings does not settle.
static FailureOr<bool>
agreeInOwnContext(ImplOp impl, Type expected, Type actual,
                  llvm::function_ref<InFlightDiagnostic()> errFn) {
  // The impl's own bindings are spelled over its own parameters, so they are
  // read at the arguments its parameters take in its own body: themselves.
  SpecializationMap own;
  auto read = [&](Type ty) -> FailureOr<Type> {
    FailureOr<Type> out = impl.readOwnBindings(ty, own);
    if (failed(out) && errFn)
      errFn() << "projection normalization did not converge; check for "
                 "cyclic associated type bindings";
    return out;
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
  return entailedByGroundCongruence(*lhs, *rhs, hypotheses);
}

/// The labels a member's own type parameters take in a declaration binding
/// `declarationCount` of its own: `declarationCount` onward, up to the bound
/// the member's signature spells.
static SmallVector<GenericTypeInterface, 4>
getOwnTypeParameters(Type signature, unsigned declarationCount) {
  MLIRContext *ctx = signature.getContext();
  SmallVector<GenericTypeInterface, 4> own;
  for (unsigned label = declarationCount, bound = getLabelBound(signature);
       label < bound; ++label)
    own.push_back(cast<GenericTypeInterface>(Type(PolyType::get(ctx, label))));
  return own;
}

/// Verifies that `implMethod`, `impl`'s copy of a method of `traitOp`, is the
/// trait's declaration of it carried to the impl: rustc's
/// `compare_impl_method`, run once per impl method.
///
/// A method's declaration binds its declaration's parameters and then its own,
/// labelled after the declaration's without a gap, and the two declarations of
/// one method bind equally many of their own: the trait states how many a
/// caller supplies, and a copy with a different count is a different
/// declaration. So the carrying substitution is positional -- the trait's
/// parameters take the impl's self arguments, and the trait method's own
/// parameter `traitCount + j` becomes the impl method's `implCount + j`, the
/// rebase rustc's `rebase_onto` performs -- and the check is the one identity
/// it makes true, read through the impl's own bindings and where equalities.
static LogicalResult verifyImplMethodSignature(
    ImplOp impl, TraitOp traitOp, FunctionOpInterface implMethod,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  StringRef name = implMethod.getName();
  auto traitMethod = traitOp.getMethod(name, errFn);
  if (failed(traitMethod)) return failure();

  auto rebase =
      traitOp.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(rebase)) return failure();

  unsigned traitCount = traitOp.getTypeParams().size();
  unsigned implCount = impl.getTypeParams().size();
  Type traitMethodTy = Type(traitMethod->getFunctionType());
  Type implMethodTy = Type(implMethod.getFunctionType());
  FailureOr<unsigned> traitOwn = countDenseLabelsFrom(traitMethodTy, traitCount);
  FailureOr<unsigned> implOwn = countDenseLabelsFrom(implMethodTy, implCount);
  if (failed(traitOwn) || failed(implOwn)) {
    if (errFn)
      errFn() << "method '" << name << "' labels its own type parameters "
              << "after its declaration's, without a gap";
    return failure();
  }
  if (*traitOwn != *implOwn) {
    if (errFn)
      errFn() << "method '" << name << "' binds " << *implOwn
              << " type parameter(s) of its own, but trait '"
              << traitOp.getSymNameAttr() << "' declares it with "
              << *traitOwn;
    return failure();
  }

  // The copy may constrain the kind of an own parameter, so what the trait's
  // parameter becomes is the copy's spelling of its label: its first
  // occurrence in the copy's signature, a kind-constraining wrapper or the
  // label itself.
  MLIRContext *ctx = impl.getContext();
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
  FailureOr<bool> agree =
      agreeInOwnContext(impl, expected, implMethodTy, errFn);
  if (failed(agree))
    return failure();
  if (!*agree) {
    if (errFn)
      errFn() << "method '" << name << "' has incompatible signature: "
              << "expected " << expected << " but found " << implMethodTy;
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
    ImplOp impl, TraitOp traitOp,
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
  for (auto [index, operand, requirement] :
       llvm::enumerate(evidence.getOperands(), *requirements)) {
    auto supplied = dyn_cast<ClaimType>(operand.getType());
    if (!supplied)
      return errFn() << "returns " << operand.getType() << " for requirement "
                     << index << ", which is no claim";
    FailureOr<bool> agree =
        agreeInOwnContext(impl, Type(requirement), Type(supplied), errFn);
    if (failed(agree))
      return failure();
    if (!*agree)
      return errFn() << "returns " << supplied << " for requirement " << index
                     << ", which trait '@" << traitOp.getSymName()
                     << "' states as " << requirement;
    if (restsOn(operand, self))
      return errFn() << "returns evidence for requirement " << index
                     << " that rests on the impl's own application";
    if (projectsOwnApplication(operand, impl))
      return errFn() << "returns evidence for requirement " << index
                     << " that projects the impl's own application";
  }
  return success();
}

static LogicalResult verifyImplParametersAreConstrained(ImplOp impl);

/// Verifies each associated type binding against the two lists a use of it
/// supplies arguments for: the impl header's parameters, bound where the impl is
/// selected, and the binding's own parameters, bound by a projection's
/// associated type arguments. A binding whose own parameter repeats a header
/// parameter would take the header's argument in a position the projection
/// supplies, and a bound type mentioning a parameter from neither list has
/// nothing to supply it, so the resolved type would carry a parameter no
/// substitution reaches.
///
/// A binding's own parameter is a type variable: it is the key a projection's
/// argument is substituted for, so a ground type standing in the list would
/// carry every occurrence of that same type in the bound type away with it.
static LogicalResult verifyAssociatedTypeBindingScopes(ImplOp impl) {
  DenseSet<Type> headerParams;
  for (GenericTypeInterface parameter : impl.getTypeParams())
    headerParams.insert(Type(parameter));

  for (Operation &op : impl.getBody().front()) {
    auto assoc = dyn_cast<AssocTypeOp>(op);
    if (!assoc)
      continue;

    DenseSet<Type> ownParams;
    if (ArrayAttr declaredParams = assoc.getTypeParamsAttr()) {
      for (Attribute declared : declaredParams) {
        FailureOr<Type> param = readAssociatedTypeParameter(assoc, declared);
        if (failed(param))
          return failure();
        // A declared parameter may be a generic type another dialect wraps
        // around a label (a coordinate parameter carries the label it stands
        // for), and declaring it declares the label it carries.
        for (GenericTypeInterface inside : getTypeParametersIn(*param)) {
          if (headerParams.contains(Type(inside)))
            return assoc.emitOpError()
                   << "type parameter " << *param
                   << " is already a parameter of impl '@"
                   << impl.getSymName() << "'";
          ownParams.insert(Type(inside));
        }
      }
    }

    TypeAttr boundAttr = assoc.getBoundTypeAttr();
    if (!boundAttr)
      continue;
    for (GenericTypeInterface parameter :
         getTypeParametersIn(boundAttr.getValue()))
      if (!headerParams.contains(Type(parameter)) &&
          !ownParams.contains(Type(parameter)))
        return assoc.emitOpError()
               << "bound type mentions type parameter " << parameter
               << ", which neither impl '@" << impl.getSymName()
               << "' nor this associated type declares";
  }
  return success();
}

LogicalResult ImplOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();
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
  if (failed(verifyImplParametersAreConstrained(*this)))
    return failure();
  return verifyAssociatedTypeBindingScopes(*this);
}

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&]{ return emitOpError(); };

  // The self application and the where entries name traits by symbol; the
  // block arguments holding them are types the symbol-use driver reads no
  // further than their own attributes.
  for (BlockArgument argument : getBody().front().getArguments())
    if (failed(cast<ClaimType>(argument.getType())
                   .verifySymbolUses(getOperation(), symbolTable)))
      return failure();

  // Get the trait
  auto traitOp = getTrait();

  // Collect method names from the trait
  llvm::SmallSet<StringRef, 8> requiredMethodNames = traitOp.getRequiredMethodNames();
  std::vector<FunctionOpInterface> optionalMethods = traitOp.getOptionalMethods();
  llvm::SmallSet<StringRef, 8> optionalMethodNames;
  for (auto f : optionalMethods) {
    optionalMethodNames.insert(f.getName());
  }

  // Verify methods and associated type bindings
  llvm::SmallSet<StringRef, 8> definedMethods;
  llvm::SmallSet<StringRef, 8> definedAssocTypes;
  for (Operation &op : getBody().front()) {
    if (auto implMethod = dyn_cast<FunctionOpInterface>(op)) {
      StringRef name = implMethod.getName();
      if (!requiredMethodNames.contains(name) && !optionalMethodNames.contains(name)) {
        return emitOpError() << "implements unknown method '" << name
                             << "' (not found in trait '" << getTraitNameAttr() << "')";
      }
      if (implMethod.isExternal()) {
        return emitOpError() << "method '" << name << "' must have body";
      }
      if (!definedMethods.insert(name).second) {
        return emitOpError() << "implements method '" << name << "' multiple times";
      }

      // Verify that the impl method's declaration is the trait's declaration
      // of it, carried through the positional correspondence between them.
      if (failed(verifyImplMethodSignature(*this, traitOp, implMethod, errFn)))
        return failure();
    } else if (auto assocType = dyn_cast<AssocTypeOp>(op)) {
      StringRef name = assocType.getSymName();
      if (!definedAssocTypes.insert(name).second)
        return emitOpError() << "defines associated type '" << name << "' multiple times";

      // In an impl, the associated type must have a bound_type
      if (!assocType.getBoundType())
        return emitOpError() << "associated type '" << name << "' in impl must have a bound type";

      // Verify that the trait declares this associated type
      auto traitAssoc = traitOp.getAssociatedType(name);
      if (failed(traitAssoc))
        return emitOpError() << "associated type '" << name
                             << "' not found in trait '" << getTraitNameAttr() << "'";

      // Verify GAT type_params arity matches
      {
        unsigned traitArity = traitAssoc->getTypeParams() ? traitAssoc->getTypeParams()->size() : 0;
        unsigned implArity = assocType.getTypeParams() ? assocType.getTypeParams()->size() : 0;
        if (traitArity != implArity)
          return emitOpError() << "associated type '" << name
                               << "' has " << implArity << " type parameter(s) but trait declares "
                               << traitArity;
      }
    }
  }

  // Verify that all associated types in the trait have bindings in the impl
  for (auto traitAssoc : traitOp.getAssociatedTypes()) {
    if (!definedAssocTypes.contains(traitAssoc.getSymName()))
      return emitOpError() << "missing binding for associated type '"
                           << traitAssoc.getSymName()
                           << "' of trait '" << getTraitNameAttr() << "'";
  }

  // Verify that no required methods are missing
  for (StringRef name : requiredMethodNames) {
    if (!definedMethods.contains(name)) {
      return emitOpError() << "missing implementation for required method '" << name
                           << "' of trait '" << getTraitNameAttr() << "'";
    }
  }

  return verifyRequirementEvidence(*this, traitOp, errFn);
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
/// nothing constrains would leave the impl's methods and associated-type
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
  return getTypeParams().empty() && getWhereClaims().empty();
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

SmallVector<GenericTypeInterface, 4> ImplOp::getTypeParams() {
  // The parameters are labelled by position (`ImplOp::verify`), so they are
  // the labels below the bound the header and the where clause spell.
  unsigned count = getLabelBound(llvm::to_vector(getBody().front().getArgumentTypes()));
  SmallVector<GenericTypeInterface, 4> parameters;
  for (unsigned label = 0; label < count; ++label)
    parameters.push_back(
        cast<GenericTypeInterface>(Type(PolyType::get(getContext(), label))));
  return parameters;
}

bool mlir::trait::statesImplArguments(Operation *op, StringAttr name) {
  if (auto derive = dyn_cast<DeriveOp>(op))
    return name == derive.getImplArgsAttrName();
  if (auto witness = dyn_cast<WitnessOp>(op))
    return name == witness.getImplArgsAttrName();
  return false;
}

ArrayAttr ImplOp::stateArguments(const SpecializationMap &arguments) {
  SmallVector<Attribute> stated;
  for (GenericTypeInterface parameter : getTypeParams())
    stated.push_back(TypeAttr::get(instantiate(Type(parameter), arguments)));
  return ArrayAttr::get(getContext(), stated);
}

//===----------------------------------------------------------------------===//
// Method instances
//===----------------------------------------------------------------------===//

/// Clones at `rewriter`'s insertion point the ops of a declaration's or a
/// proof's body that compute `root`, operands before their users, each op once:
/// a value `mapping` already holds is read from it, so the caller maps the
/// body's block arguments before asking. `stamp` respells every type and
/// attribute the clones carry.
static Value cloneDefiningTree(RewriterBase &rewriter, Value root,
                               IRMapping &mapping, AttrTypeReplacer &stamp,
                               AttrTypeReplacer &spelling) {
  if (Value mapped = mapping.lookupOrNull(root))
    return mapped;
  Operation *producer = root.getDefiningOp();
  assert(producer && "a body's block arguments are mapped before it is read");
  for (Value operand : producer->getOperands())
    (void)cloneDefiningTree(rewriter, operand, mapping, stamp, spelling);
  Operation *clone = rewriter.clone(*producer, mapping);
  for (Value result : clone->getResults())
    result.setType(stamp.replace(result.getType()));
  for (NamedAttribute attr : clone->getAttrs()) {
    AttrTypeReplacer &replacer =
        statesImplArguments(clone, attr.getName()) ? spelling : stamp;
    clone->setAttr(attr.getName(), replacer.replace(attr.getValue()));
  }
  return mapping.lookup(root);
}

/// Maps in `mapping` each value of `reads` -- the values read from the
/// declaration `declaration`, by a method's body or by a projection of its
/// return -- to the evidence the receiver's proof `selfProof` supplies there,
/// cloning at `rewriter`'s insertion point what must be computed, each clone of
/// the declaration's ops stamped by `stamp`: argument 0 to `self`; an impl's
/// argument k to a clone of the claim the receiver's proof derives the impl
/// from at position k - 1, its defining ops copied out of the proof's body,
/// which is ground; a value the declaration's
/// body computes to a clone of its defining ops over those. A proof is closed,
/// so the claims its derive is given are computed by its own body alone. An
/// impl argument the receiver's proof supplies nothing for stays unmapped,
/// which the isolation of whatever the body is cut or inlined into refuses.
static void mapDeclarationReads(RewriterBase &rewriter, ModuleOp module,
                                Operation *declaration,
                                const llvm::SetVector<Value> &reads,
                                Value self, ClaimType selfProof,
                                AttrTypeReplacer &stamp,
                                AttrTypeReplacer &spelling,
                                IRMapping &mapping) {
  Block &declarationBody = declaration->getRegion(0).front();
  mapping.map(declarationBody.getArgument(0), self);
  if (declarationBody.getNumArguments() > 1)
    if (auto proof = lookupSymbolFrom<ProofOp>(module, selfProof.getProof())) {
      AttrTypeReplacer asWritten;
      IRMapping fromProof;
      for (auto [argument, premise] :
           llvm::zip(declarationBody.getArguments().drop_front(),
                     proof.getDerive().getAssumptions()))
        mapping.map(argument, cloneDefiningTree(rewriter, premise, fromProof,
                                                asWritten, asWritten));
    }
  for (Value read : reads)
    if (!isa<BlockArgument>(read) || mapping.contains(read))
      (void)cloneDefiningTree(rewriter, read, mapping, stamp, spelling);
}

/// The values `method`'s body reads from outside it: its declaration's block
/// arguments and the values the declaration's body computes. They are read off
/// the template before anything is cloned: once a clone stands elsewhere, no
/// query finds them again.
static llvm::SetVector<Value> readsFromDeclaration(FunctionOpInterface method) {
  llvm::SetVector<Value> reads;
  getUsedValuesDefinedAbove(method.getFunctionBody(), reads);
  return reads;
}

/// Cuts `method`, a method of the trait or impl `declaration`, into a
/// module-level function `functionName` for the receiver whose proven claim is
/// `selfProof`, its body stamped under `subst`.
///
/// The instance leads with the receiver's proof, and every value the method
/// reads from its declaration is replaced in it by the evidence that proof
/// supplies there (`mapDeclarationReads`). A value the replacement missed is
/// refused by the module-level function's isolation.
static func::FuncOp cutMethodInstance(PatternRewriter &rewriter, ModuleOp module,
                                      Operation *declaration,
                                      FunctionOpInterface method,
                                      StringRef functionName,
                                      ClaimType selfProof,
                                      const DenseMap<Type, Type> &subst) {
  llvm::SetVector<Value> reads = readsFromDeclaration(method);

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(declaration);

  // An external declaration has no body to clone; specialization has refused
  // it. Cut at module scope, the instance is a `func.func`.
  auto funcOp = cast_if_present<func::FuncOp>(
      specializePolymorph(rewriter, method, functionName, subst).getOperation());
  if (!funcOp)
    return nullptr;
  rewriter.modifyOpInPlace(funcOp, [&] {
    (void)funcOp.insertArgument(/*idx=*/0, selfProof,
                               /*argAttrs=*/mlir::DictionaryAttr(),
                               method->getLoc());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  });
  if (reads.empty())
    return funcOp;

  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  IRMapping replacements;
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(subst, CloneKind::Instance);
  AttrTypeReplacer spelling = makeSpellingReplacerFromSubstitution(subst);
  mapDeclarationReads(rewriter, module, declaration, reads,
                      funcOp.getArgument(0), selfProof, stamp, spelling,
                      replacements);
  for (Value read : reads)
    if (Value replacement = replacements.lookupOrNull(read))
      rewriter.replaceUsesWithIf(read, replacement, [&](OpOperand &use) {
        return funcOp->isProperAncestor(use.getOwner());
      });
  return funcOp;
}

bool mlir::trait::producesPositionalEvidence(Operation *op) {
  if (isa<ProjectOp, DeriveOp, CoerceOp>(op))
    return true;
  auto call = dyn_cast<MethodCallOp>(op);
  return call && call.computesEvidence();
}

namespace {

/// Inlines a method's body at a call: every op is legal to inline, since the
/// body's reads from outside it are mapped before it is cloned, and the call's
/// results take the method's `trait.return` operands, announced to `rewriter`
/// so their users are visited again. An operand spelled otherwise than the
/// call's result is bridged by a coercion, as a projection's inlined evidence
/// is (`ProjectOp::inlineEvidence`).
struct MethodBodyInliner : public InlinerInterface {
  MethodBodyInliner(MLIRContext *ctx, RewriterBase &rewriter)
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

/// Replaces `call`, a call of `method` of the trait or impl `declaration`
/// through the receiver whose proven claim is `selfProof`, by the method's body
/// stamped under `subst`: what the body reads from its declaration is mapped to
/// the evidence the receiver's proof supplies there (`mapDeclarationReads`),
/// its parameters to the call's arguments, and the call's result to the operand
/// of its `trait.return`. Fails, leaving the call standing, where the method
/// has no body or more than one block.
static LogicalResult inlineMethodAt(PatternRewriter &rewriter, ModuleOp module,
                                    Operation *declaration,
                                    FunctionOpInterface method,
                                    MethodCallOp call, ClaimType selfProof,
                                    const DenseMap<Type, Type> &subst) {
  Region &body = method.getFunctionBody();
  if (body.empty() || !body.hasOneBlock())
    return rewriter.notifyMatchFailure(call, "the method has no one-block body");

  llvm::SetVector<Value> reads = readsFromDeclaration(method);
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(subst, CloneKind::Instance);
  AttrTypeReplacer spelling = makeSpellingReplacerFromSubstitution(subst);
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(call);
  IRMapping mapping;
  mapDeclarationReads(rewriter, module, declaration, reads, call.getClaim(),
                      selfProof, stamp, spelling, mapping);
  for (auto [parameter, argument] :
       llvm::zip(body.front().getArguments(), call.getArguments()))
    mapping.map(parameter, argument);

  // The body is cloned and then stamped, as a cut instance's is: a call
  // computing evidence in it keeps its result spelling for its own inlining.
  auto cloneBody = [&](OpBuilder &, Region *src, Block *, Block *postInsertBlock,
                       IRMapping &mapper, bool) {
    cloneRegionStampedBefore(rewriter, *src, *postInsertBlock->getParent(),
                             postInsertBlock->getIterator(), mapper, stamp,
                             spelling);
  };
  // The inlined ops are located at the call, called from it.
  MethodBodyInliner interface(call.getContext(), rewriter);
  if (failed(inlineRegion(interface, cloneBody, &body, call, mapping,
                          call->getResults(), call->getResultTypes(),
                          call.getLoc())))
    return rewriter.notifyMatchFailure(call, "the method's body is not inlined");
  rewriter.eraseOp(call);
  return success();
}

/// The instance `method` of `declaration` names for a call through the proven
/// receiver `provenSelfClaim` supplying `actualArguments` for the method's own
/// parameters: the type arguments are `declarationArguments` for the
/// declaration's parameters, then every parameter the method's signature
/// spells, read through `subst`; the evidence the receiver's proof at the
/// leading position, then what the call supplies, position by position.
static FailureOr<func::FuncOp> getOrCutMethodInstance(
    PatternRewriter &rewriter, ModuleOp module, Operation *declaration,
    SymbolRefAttr templateRef, ArrayRef<Type> declarationArguments,
    FunctionOpInterface method, ClaimType formalSelf, ClaimType provenSelfClaim,
    TypeRange actualArguments, const DenseMap<Type, Type> &subst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  SmallVector<Type> typeArguments(declarationArguments);
  for (GenericTypeInterface parameter :
       getTypeParametersIn(method.getFunctionType()))
    typeArguments.push_back(applySubstitutionOnce(subst, parameter));
  SmallVector<Type> formalInputs{formalSelf};
  llvm::append_range(formalInputs, method.getArgumentTypes());
  SmallVector<Type> actualInputs{provenSelfClaim};
  llvm::append_range(actualInputs, actualArguments);
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(subst, CloneKind::Instance);
  auto key = InstanceKey::get(templateRef, typeArguments, formalInputs,
                              actualInputs, stamp, respell);
  if (failed(key))
    return method.emitOpError()
           << "is supplied a claim that names no proof, which identifies no "
              "instance";

  // The leading self proof is read as the instance spells it.
  auto selfProof = cast<ClaimType>(key->getEvidence().front());
  func::FuncOp instance = getOrCutInstance(
      rewriter, module, *key, [&](StringRef instanceName) {
        return cutMethodInstance(rewriter, module, declaration, method,
                                 instanceName, selfProof, subst);
      });
  if (!instance)
    return failure();
  return instance;
}

/// The substitution `method`, `impl`'s copy of a method of `trait`, is cut or
/// inlined under for a call through `provenSelfClaim` whose method-generic
/// bindings are `callSubst`'s: the arguments the impl's parameters take at the
/// receiver, then the call's bindings of the method's own parameters, which a
/// call names at the trait method's labels and the impl's copy binds at its
/// own: own parameter `j` is the trait's label `traitCount + j` and the copy's
/// `implCount + j` (`verifyImplMethodSignature`), so each binding moves by the
/// difference, rustc's `rebase_onto`. The call's projection bindings ride
/// along; its bindings of the trait's parameters are the impl's arguments
/// already. Answers the impl's arguments beside it.
static FailureOr<std::pair<SpecializationMap, DenseMap<Type, Type>>>
implMethodSubstitution(ImplOp impl, TraitOp trait, FunctionOpInterface method,
                       ClaimType provenSelfClaim,
                       const CallSubstitution &callSubst) {
  auto implArguments = impl.buildImplSpecialization(provenSelfClaim);
  if (failed(implArguments))
    return failure();
  DenseMap<Type, Type> subst = implArguments->toTypeMap();
  unsigned traitCount = trait.getTypeParams().size();
  unsigned implCount = impl.getTypeParams().size();
  MLIRContext *ctx = impl.getContext();
  for (const auto &[key, value] : callSubst.toTypeMap()) {
    if (!isa<GenericTypeInterface>(key)) {
      subst.try_emplace(key, value);
      continue;
    }
    auto label = dyn_cast_or_null<PolyType>(
        Type(getParameterOccurrence(key)));
    if (label && static_cast<unsigned>(label.getLabel()) >= traitCount)
      subst.try_emplace(
          PolyType::get(ctx, label.getLabel() - traitCount + implCount),
          value);
  }
  return std::make_pair(std::move(*implArguments), std::move(subst));
}

FailureOr<func::FuncOp> ImplOp::getOrSpecializeFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType provenSelfClaim,
    StringRef methodName,
    TypeRange actualArguments,
    const CallSubstitution &callSubst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  TraitOp trait = getTrait();
  if (!trait.hasMethod(methodName)) return failure();

  // A method the impl does not define is the trait's default, cut from the
  // trait with the receiver as its argument.
  auto method = getMethod(methodName);
  if (failed(method))
    return trait.getOrSpecializeFreeFunctionFromDefault(
        rewriter, provenSelfClaim, methodName, actualArguments, callSubst,
        respell);

  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  auto substitution = implMethodSubstitution(*this, trait, *method,
                                             provenSelfClaim, callSubst);
  if (failed(substitution)) return failure();
  auto &[implArguments, subst] = *substitution;

  SmallVector<Type> declarationArguments;
  for (GenericTypeInterface parameter : getTypeParams())
    declarationArguments.push_back(implArguments.apply(parameter));
  auto templateRef = SymbolRefAttr::get(
      getSymNameAttr(), {FlatSymbolRefAttr::get(getContext(), methodName)});
  return getOrCutMethodInstance(rewriter, module, *this, templateRef,
                                declarationArguments, *method, getSelfClaim(),
                                provenSelfClaim, actualArguments, subst, respell);
}

LogicalResult MethodCallOp::inlineEvidence(
    PatternRewriter &rewriter, const CallSubstitution &callSubst) {
  ImplOp impl = getProvenImpl();
  TraitOp trait = impl.getTrait();
  ModuleOp module = impl->getParentOfType<ModuleOp>();
  auto method = impl.getMethod(getMethodName());

  // A method the impl does not define is the trait's default, inlined with the
  // receiver as its declaration's argument, as it is cut
  // (`getOrSpecializeFreeFunctionFromDefault`).
  if (failed(method)) {
    auto byDefault = trait.getOptionalMethod(getMethodName());
    auto traitArguments = trait.buildSubstitutionForSelfClaim(getClaimType());
    if (failed(byDefault) || failed(traitArguments))
      return rewriter.notifyMatchFailure(*this, "no body defines the method");
    DenseMap<Type, Type> subst = traitArguments->toTypeMap();
    for (const auto &[k, v] : callSubst.toTypeMap())
      subst.try_emplace(k, v);
    return inlineMethodAt(rewriter, module, trait, *byDefault, *this,
                          getClaimType(), subst);
  }

  auto substitution = implMethodSubstitution(impl, trait, *method,
                                             getClaimType(), callSubst);
  if (failed(substitution))
    return failure();
  return inlineMethodAt(rewriter, module, impl, *method, *this, getClaimType(),
                        substitution->second);
}

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

/// The impl `value` commits to, read in `context`, the committed impl whose
/// body `value` stands in (none for a value outside every impl): a proven
/// claim's impl through its proof, a derived claim's through the derive, and
/// one of `context`'s where arguments through what its derive was given there.
/// None where the value commits to no impl yet.
static std::optional<CommittedImpl>
committedImplOf(Value value, std::shared_ptr<const CommittedImpl> context,
                ModuleOp module) {
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
                           context->enclosing, module);
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
    ImplOp impl = derive ? derive.getImplOp() : ImplOp();
    if (!impl)
      return std::nullopt;
    auto arguments = SpecializationMap::fromPositions(llvm::map_range(
        derive.getImplArgs().getAsValueRange<TypeAttr>(),
        [&](Type argument) { return instantiate(argument, outer); }));
    return CommittedImpl{impl, claim.getTraitApplication(),
                         std::move(arguments), derive, std::move(context)};
  }
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(
      module, claim.getProof(), /*errFn=*/nullptr);
  if (failed(cited))
    return std::nullopt;
  auto proof = dyn_cast<ProofOp>(*cited);
  ImplOp impl = proof ? proof.getImpl() : cast<ImplOp>(*cited);
  if (!impl)
    return std::nullopt;
  SpecializationMap arguments =
      proof ? proof.getImplArguments() : SpecializationMap();
  return CommittedImpl{impl, claim.getTraitApplication(), std::move(arguments),
                       proof ? proof.getDerive() : DeriveOp(), nullptr};
}

LogicalResult ProjectOp::inlineEvidence(RewriterBase &rewriter) {
  if (!getSourceClaim().isProven())
    return failure();
  ModuleOp module = getAnchorModule(getOperation());
  auto committed = committedImplOf(getSource(), nullptr, module);
  if (!committed)
    return failure();
  ImplOp impl = committed->impl;
  ClaimType source = getSourceClaim();

  // A trait requirement is the impl's return operand at its index; a where
  // entry, past them, is the impl's block argument there, which the source's
  // proof supplies.
  uint64_t traitCount = impl.getTrait().getRequirements().size();
  Block &body = impl.getBody().front();
  uint64_t index = getIndex();
  if (index >= traitCount && index - traitCount + 1 >= body.getNumArguments())
    return failure();
  Value read = index < traitCount ? impl.getReturn().getOperand(index)
                                  : Value(body.getArgument(index - traitCount + 1));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(*this);
  AttrTypeReplacer stamp =
      makeTypeReplacerFromSubstitution(committed->arguments.toTypeMap(), CloneKind::Instance);
  AttrTypeReplacer spelling =
      makeSpellingReplacerFromSubstitution(committed->arguments.toTypeMap());
  IRMapping mapping;
  llvm::SetVector<Value> reads;
  reads.insert(read);
  mapDeclarationReads(rewriter, module, impl, reads, getSource(), source, stamp,
                      spelling, mapping);
  Value inlined = mapping.lookupOrNull(read);
  if (!inlined)
    return failure();
  // The inlined ops are located where the impl wrote them, called from the
  // projection, as an inlined call's are.
  for (auto [original, clone] : mapping.getOperationMap())
    clone->setLoc(CallSiteLoc::get(clone->getLoc(), getLoc()));

  // The evidence is spelled as the impl wrote it, which the result may spell
  // otherwise: the two meet as their projections resolve, and a proof the
  // result spells must be the one the evidence carries, which the coercion
  // bridging them holds it to.
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

EvidenceReading ProjectOp::readEvidence() {
  ModuleOp module = getAnchorModule(getOperation());
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
      return committedImplOf(ending->first, ending->second, module);
    }
    return committedImplOf(value, context, module);
  };

  follow = [&](ProjectOp project, Context context) -> std::optional<Ending> {
    std::optional<CommittedImpl> current = commit(project.getSource(), context);
    uint64_t index = project.getIndex();
    // A where entry, past the trait's requirements, is evidence a proof or a
    // derive supplies, which is a base; only a requirement is read through a
    // return.
    while (current &&
           index < current->impl.getTrait().getRequirements().size()) {
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

FailureOr<func::FuncOp> TraitOp::getOrSpecializeFreeFunctionFromDefault(
    PatternRewriter &rewriter, ClaimType provenSelfClaim, StringRef methodName,
    TypeRange actualArguments, const CallSubstitution &callSubst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  auto method = getOptionalMethod(methodName);
  if (failed(method))
    return failure();
  auto traitArguments = buildSubstitutionForSelfClaim(provenSelfClaim);
  if (failed(traitArguments))
    return failure();

  // A call names its method-generic bindings under the trait method's own type
  // variables, which are the default's.
  DenseMap<Type, Type> subst = traitArguments->toTypeMap();
  for (const auto &[k, v] : callSubst.toTypeMap())
    subst.try_emplace(k, v);

  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  auto templateRef = SymbolRefAttr::get(
      getSymNameAttr(), {FlatSymbolRefAttr::get(getContext(), methodName)});
  return getOrCutMethodInstance(
      rewriter, module, *this, templateRef,
      provenSelfClaim.getTraitApplication().getTypeArgs(), *method,
      getSelfClaim(), provenSelfClaim, actualArguments, subst, respell);
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

TraitOp ImplOp::getTrait() {
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("ImplOp::getTrait: not inside of a module");
  return getSelfApplication().getTraitOrAbort(module, "ImplOp::getTrait: couldn't find trait");
}

TypeArguments ImplOp::readTypeArgumentsFor(ClaimType actualSelfClaim,
                                           Normalizer normalize) {
  TypeArguments args(getTypeParams());
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
    return llvm::count_if(args.getParameters(),
                          [&](GenericTypeInterface parameter) {
                            return args.lookup(parameter).has_value();
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
  // substitution: `verifyAssociatedTypeBindingScopes` refuses a binding whose
  // own parameter repeats a header parameter, so the union of the two lists is
  // a function, and one pass never revisits a term it stamped -- neither
  // argument list can be read as the parameters the other list answers for.
  SpecializationMap arguments = headerArguments;

  auto assoc = getAssociatedType(name);
  if (succeeded(assoc) && assoc->getTypeParams()) {
    auto typeParams = *assoc->getTypeParams();
    if (typeParams.size() != assocTypeArgs.size()) {
      if (err) err() << "GAT arity mismatch for '" << name
                     << "': expected " << typeParams.size()
                     << " type args but got " << assocTypeArgs.size();
      return failure();
    }
    for (auto [param, arg] : llvm::zip(typeParams, assocTypeArgs))
      arguments.bind(
          cast<GenericTypeInterface>(cast<TypeAttr>(param).getValue()), arg);
  }

  return instantiate(*binding, arguments);
}

FailureOr<Type> ImplOp::readOwnBindings(Type ty,
                                        const SpecializationMap &arguments) {
  TraitApplicationAttr own = getSelfApplicationAt(arguments);
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
    if (proj.getTraitApplication() != own)
      return std::nullopt;
    auto bound = specializeAssociatedTypeBinding(
        proj.getAssocName().getValue(), proj.getAssocTypeArgs(), arguments);
    if (failed(bound))
      return std::nullopt;
    return *bound;
  });
  Type out;
  if (failed(tryNormalizeProjectionsToFixedPoint(
          ty, [&](Type t) { return replacer.replace(t); }, out)))
    return failure();
  return out;
}

FailureOr<SpecializationMap> ImplOp::buildImplSpecialization(
    ClaimType provenSelfClaim, llvm::function_ref<InFlightDiagnostic()> err) {
  if (!provenSelfClaim.isProven()) {
    if (err) err() << "expected proven self claim for " << getSymName();
    return failure();
  }
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(
      module, provenSelfClaim.getProof(), err);
  if (failed(cited))
    return failure();
  if (auto proof = dyn_cast<ProofOp>(*cited))
    return proof.getImplArguments();
  return SpecializationMap();
}

//===----------------------------------------------------------------------===//
// ProofOp
//===----------------------------------------------------------------------===//

/// The derive `proven` is, or the derive a coercion `proven` is respells: a
/// proof returns its derive at the header's spelling or respelled once.
static DeriveOp derivedThroughRespelling(Value proven) {
  if (auto coerce = proven.getDefiningOp<CoerceOp>())
    proven = coerce.getInput();
  return proven.getDefiningOp<DeriveOp>();
}

LogicalResult ProofOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();

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
      !derivedThroughRespelling(returned.getOperand(0)))
    return emitOpError() << "returns the one claim a derive in its body "
                            "derives, or that claim respelled";
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

DeriveOp ProofOp::getDerive() {
  return derivedThroughRespelling(getProven());
}

TraitOp ProofOp::getTrait() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("ProofOp::getTrait: not inside a module");
  return getTraitApplication().getTraitOrAbort(module, "ProofOp::getTrait: couldn't find trait");
}

SmallVector<ClaimType> ProofOp::getPremises() {
  return llvm::map_to_vector(getDerive().getAssumptions(), [](Value premise) {
    return cast<ClaimType>(premise.getType());
  });
}

SmallVector<ClaimType> ProofOp::getSubproofs() {
  SmallVector<ClaimType> subproofs;
  for (Value premise : getDerive().getAssumptions()) {
    if (cast<ClaimType>(premise.getType()).isEquality())
      continue;
    subproofs.push_back(cast<ClaimType>(throughCoercions(premise).getType()));
  }
  return subproofs;
}

/// Look up a proof symbol and return the raw Operation* (ProofOp or ImplOp).
/// This is the shared lookup used by both getImplFromProof and
/// getProofOpOrUnconditionalImplOp.
static FailureOr<Operation*> lookupProofSymbol(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  Operation* symOp = lookupSymbolFrom(module, name);
  if (!symOp) {
    if (errFn) errFn() << "cannot find proof symbol '" << name << "'";
    return failure();
  }

  if (isa<ImplOp>(symOp) || isa<ProofOp>(symOp))
    return symOp;

  if (errFn) errFn() << "proof symbol '" << name << "' must refer to trait.proof or trait.impl";
  return failure();
}

FailureOr<ImplOp> ProofOp::getImplFromProof(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto symOp = lookupProofSymbol(module, name, errFn);
  if (failed(symOp)) return failure();

  if (auto implOp = dyn_cast<ImplOp>(*symOp))
    return implOp;

  auto proofOp = cast<ProofOp>(*symOp);
  ImplOp impl = proofOp.getImpl();
  if (!impl) {
    if (errFn) errFn() << "proof '" << name << "' does not resolve to an impl";
    return failure();
  }
  return impl;
}

FailureOr<Operation*> ProofOp::getProofOpOrUnconditionalImplOp(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto symOp = lookupProofSymbol(module, name, errFn);
  if (failed(symOp)) return failure();

  // if it's an ImplOp, it must be unconditional
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (failed(impl.verifyIsUnconditional(errFn))) return failure();
  }

  return *symOp;
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
    ArrayAttr citedArguments;
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

  // Equality composition arm: `compose(%premises...) : (types...) :
  // <result-type>`. The premise types are spelled -- SSA operands resolve
  // against written types -- and the result equality is spelled too, since it is
  // derived from the premises and not inferable from them.
  if (succeeded(p.parseOptionalKeyword("compose"))) {
    if (parsePremises()) return failure();
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

  // Composition arm: an equality result with neither an impl nor a refl
  // marker. Print the premises with their types and the spelled result equality.
  if (getResultClaim().isEquality()) {
    p << " compose";
    printTypedOperandList(p, getPremises());
    p << " : " << getResult().getType();
    return;
  }

  // Application arm.
  p << " " << getProof() << " for ";
  getResultClaim().getTraitApplication().print(p);

  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elidedAttrs=*/{"impl", "impl_args", "refl"});
}

// The op's attributes must match the result claim's arm exactly. For the
// application arm the result claim names the proof it cites. For the equality
// arm, the result's left side is the projection a cited impl resolves
// (proj-resolve), its endpoints are identical (refl), or the premises' ground
// congruence closure entails it (compose).
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

    // proj-resolve: the impl the witness cites resolves the projection its
    // result's left side spells, which its symbol uses verify.
    if (resolves) {
      if (!isa<ProjectionType>(eq.getLhs()))
        return emitOpError() << "a proj-resolve witness resolves a projection, "
                             << "found " << eq.getLhs();
      return success();
    }

    // Composition: neither a witness nor refl. The result equality is
    // derived from the leaf equality premises by replaying the ground congruence
    // closure -- the transitivity and congruence that carry the premises to the
    // result are never stored, only the leaves are, so only definitional leaves
    // are ever stored. An equality claim carries no proof by that rule, so there
    // is no proof-swap for this arm to police.
    //
    // The composition arm is the only equality leaf whose evidence is another
    // claim value rather than a witness or an identical-endpoint marker, so
    // it is the only one whose validity can rest on its operands. In a region
    // without SSA dominance (a graph region such as a module body) a premise may
    // be the op's own result, letting two composes justify each other in a cycle
    // that grounds a false equality on nothing. Requiring an SSA-dominance region
    // makes the induction bottom out at proj-resolve- or refl-anchored leaves: a
    // false composition would need a false premise, which needs a false leaf, and
    // the proj-resolve and refl leaves refuse those.
    if (Region *parent = getOperation()->getParentRegion();
        parent && !mlir::mayHaveSSADominance(*parent))
      return emitOpError() << "a composition witness must be in a region that "
                              "enforces SSA dominance, so its premises cannot be "
                              "justified by its own result";
    if (getPremises().empty())
      return emitOpError() << "a composition witness requires at least one "
                              "equality premise";
    SmallVector<TypeEqualityAttr> premiseEqualities;
    for (Value premise : getPremises()) {
      auto claim = dyn_cast<ClaimType>(premise.getType());
      if (!claim || !claim.isEquality())
        return emitOpError() << "a composition witness premise must be an "
                                "equality claim, but a premise has type "
                             << premise.getType();
      premiseEqualities.push_back(claim.getEqualityAttr());
    }
    if (!entailedByGroundCongruence(eq.getLhs(), eq.getRhs(), premiseEqualities))
      return emitOpError() << "the premises do not entail " << eq.getLhs()
                           << " = " << eq.getRhs();
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
    Type claim = stripClaimProofs(operand.getType());
    if (claim != Type(premise))
      return errFn() << "premise " << position << " of " << owner << " is "
                     << premise << ", and the " << supplier << " supplies "
                     << claim;
  }
  return success();
}

/// Whether the citation a derive or a projection-resolution witness makes of
/// `impl` states one argument per parameter, `stated`, and, at those
/// `arguments`, the header `cited` and the where entries `premises` supply,
/// positionally, each compared by identity.
static LogicalResult verifyCitationOf(
    ImplOp impl, ClaimType cited, ArrayAttr stated,
    const SpecializationMap &arguments, ValueRange premises,
    StringRef supplier, llvm::function_ref<InFlightDiagnostic()> errFn) {
  size_t parameterCount = impl.getTypeParams().size();
  if (stated.size() != parameterCount)
    return errFn() << "impl '@" << impl.getSymName() << "' takes "
                   << parameterCount << " type arguments, and the " << supplier
                   << " states " << stated.size();
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

LogicalResult WitnessOp::verifyResolution(ModuleOp module) {
  auto errFn = [&] { return emitOpError(); };
  TypeEqualityAttr equality = getResultClaim().getEqualityAttr();
  auto projection = cast<ProjectionType>(equality.getLhs());
  auto impl = lookupSymbolFrom<ImplOp>(module, getImplAttr());
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
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  // Equality proj-resolve arm: the cited impl, at the arguments the
  // projection's application and the premises give it, binds the projection
  // as stated.
  if (isProjectionResolution())
    return verifyResolution(module);

  // Refl and composition arms cite nothing by symbol: their evidence is the
  // spelling, or the premises, which are SSA values.
  if (getResultClaim().isEquality())
    return success();

  // Application arm: a witness carries the application the declaration it
  // names proves -- a proof's proven claim or an unconditional impl's header
  // (`verifyCitation`). Reading the impl's header alone would accept a witness
  // for an application the proof does not prove, because a blanket impl's
  // header carries to every application of its trait.
  return verifyCitation(getProvenClaim(), module, errFn);
}


//===----------------------------------------------------------------------===//
// DeriveOp
//===----------------------------------------------------------------------===//

ImplOp DeriveOp::getImplOp() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;
  return lookupSymbolFrom<ImplOp>(module, getImplAttr());
}

/// Verifies a derive: it states one argument per parameter of the impl, the
/// derived application is the impl's header at them, and each operand's claim
/// is the impl's where-clause entry at them, in order, each compared by
/// identity.
LogicalResult DeriveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&] { return emitOpError(); };
  ImplOp impl = getImplOp();
  if (!impl)
    return errFn() << "cannot find trait.impl '" << getImplAttr() << "'";
  return verifyCitationOf(impl, getDerivedClaim(), getImplArgs(),
                          getImplArguments(), getAssumptions(), "derive",
                          errFn);
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFunctionType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFunctionType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void MethodOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult MethodOp::verify() {
  // A method is a member of its declaration's symbol table and is collected
  // with it. A visibility would make a private one discardable on its own
  // (`SymbolOpInterface::canDiscardOnUseEmpty`), which no method is.
  if ((*this)->hasAttr(SymbolTable::getVisibilityAttrName()))
    return emitOpError() << "must carry no visibility: a method lives and dies "
                            "with its trait or impl";
  return success();
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // A method returns its results, by identity.
  if (auto method = dyn_cast<MethodOp>((*this)->getParentOp())) {
    ArrayRef<Type> results = method.getResultTypes();
    if (getNumOperands() != results.size())
      return emitOpError() << "has " << getNumOperands()
                           << " operands, but enclosing method (@"
                           << method.getName() << ") returns " << results.size();
    for (auto [index, operand, result] :
         llvm::enumerate(getOperandTypes(), results))
      if (operand != result)
        return emitOpError() << "type of return operand " << index << " ("
                             << operand << ") doesn't match method result type ("
                             << result << ") in method @" << method.getName();
    return success();
  }

  // An impl returns evidence, which its own verifier holds against its
  // trait's requirements; a proof returns the one application it proves.
  if (!llvm::all_of(getOperandTypes(), llvm::IsaPred<ClaimType>))
    return emitOpError() << "returns claims only from a declaration or a proof";
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

  // 1. Strip application-claim proofs from the input and result. Comparison
  // is modulo the proof, permanently.
  Type input = stripClaimProofs(getInput().getType());
  Type result = stripClaimProofs(getResult().getType());

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

  // 2. The no-proof-swap clause runs deep. The endpoints denote one claim once
  // the equalities reconcile them, so a proof present on the result and absent
  // or different on the input is a swap the coerce may not perform -- at every
  // position an application claim sits, not only the root. Positions are paired
  // by walking the two endpoint trees in lockstep off the same decomposition the
  // congruence closure keys on, over the unstripped types so the proofs are
  // still present.
  auto rejectProofSwap = [&](ClaimType fromClaim,
                             ClaimType toClaim) -> LogicalResult {
    if (!toClaim || !toClaim.isProven())
      return success();
    if (!fromClaim || !fromClaim.isProven() ||
        fromClaim.getProof() != toClaim.getProof())
      return emitOpError() << "may not swap the proof backing claim "
                           << toClaim.getTraitApplication()
                           << ": a coerce compares modulo a proof but may not "
                              "exchange it for another";
    return success();
  };
  // Does a proven application claim sit anywhere in this type?
  std::function<bool(Type)> carriesProvenClaim = [&](Type t) -> bool {
    if (auto c = dyn_cast<ClaimType>(t))
      if (c.isApplication() && c.isProven())
        return true;
    for (Type child : decomposeTerm(t).children)
      if (carriesProvenClaim(child))
        return true;
    return false;
  };
  std::function<LogicalResult(Type, Type)> checkNoSwap =
      [&](Type in, Type out) -> LogicalResult {
    if (failed(rejectProofSwap(dyn_cast<ClaimType>(in),
                               dyn_cast<ClaimType>(out))))
      return failure();
    TermShape di = decomposeTerm(in);
    TermShape dout = decomposeTerm(out);
    if (di.key == dout.key && di.children.size() == dout.children.size()) {
      for (auto [a, b] : llvm::zip(di.children, dout.children))
        if (failed(checkNoSwap(a, b)))
          return failure();
      return success();
    }
    // The two trees diverge in shape here, so no further positions pair. A proof
    // still standing on the result side has no input position to match and is a
    // swap; a proof-free divergence is the reconciliation the equalities
    // already licensed.
    if (carriesProvenClaim(out))
      return emitOpError() << "may not swap the proof backing a claim nested in "
                           << getResult().getType()
                           << ": a coerce compares modulo a proof but may not "
                              "exchange it for another";
    return success();
  };
  if (failed(checkNoSwap(getInput().getType(), getResult().getType())))
    return failure();

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
// MethodCallOp
//===----------------------------------------------------------------------===//

FailureOr<TraitOp> MethodCallOp::getTrait(llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();
  return getClaimType()
    .getTraitApplication()
    .getTrait(*module, err);
}

FailureOr<FunctionOpInterface> MethodCallOp::getMethod(llvm::function_ref<InFlightDiagnostic()> err) {
  auto maybeTrait = getTrait(err);
  if (failed(maybeTrait)) return failure();
  auto func = maybeTrait->getMethod(getMethodName(), err);
  if (failed(func)) {
    return failure();
  }
  return func;
}

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
    unsigned count = 0;
    for (GenericTypeInterface parameter : args.getParameters())
      if (args.lookup(parameter))
        ++count;
    return count;
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
      for (GenericTypeInterface parameter : args.getParameters())
        if (!args.lookup(parameter))
          diagnostic << " " << Type(parameter);
    }
    return failure();
  }

  return args.toSpecialization();
}


LogicalResult MethodCallOp::verify() {
  // the claim's type must be an ClaimType
  ClaimType claim = dyn_cast_or_null<ClaimType>(getClaim().getType());
  if (!claim)
    return emitOpError() << "expected !trait.claim type, found " << getClaim().getType();

  // A method call names its trait through the receiver claim's application, so
  // the receiver must be a trait-application claim. An equality claim names no
  // trait and is not a legal receiver.
  if (!claim.isApplication())
    return emitOpError() << "receiver (" << claim << ") must be a "
                            "trait-application claim; an equality claim names no "
                            "trait to call";

  // verify that the named trait matches the claim's trait
  auto expectedTraitAttr = getTraitAttr();
  auto foundTraitAttr = claim.getTraitApplication().getTraitName();
  if (expectedTraitAttr != foundTraitAttr)
    return emitOpError() << "expected claim for " << expectedTraitAttr << ", found " << foundTraitAttr;

  return success();
}

/// The type arguments a call supplies for the declaration it calls, once that
/// declaration instantiated at them is the signature the call spells.
///
/// `formal` is the callee's signature and `args` holds a slot for each of its
/// parameters, filled for the ones fixed before the call's own types are read
/// -- a method's trait arguments, which ride in its receiver claim. The rest
/// are read off `actual`, the signature the call spells, against `formal` as
/// declared: the declaration's parameters and the caller's labels are read in
/// one pass and never meet in one spelling.
///
/// `selection` resolves a type's projections through the stage's impl
/// selection, which at pass time is all both signatures are read through:
/// projections at instantiation are the stage's solver's to answer. A verifier
/// passes none and compares the two signatures as written: the call spells its
/// declaration's signature at its arguments, and a value spelled otherwise
/// reaches it through a coercion.
static FailureOr<SpecializationMap> readCallSpecialization(
    Operation *call, FunctionType formal, TypeArguments args,
    FunctionType actual, StringRef callee,
    llvm::function_ref<Type(Type)> selection,
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

  auto arguments = readTypeArguments(std::move(args), Type(formal),
                                     Type(actual), selection, callee, refusal);
  if (failed(arguments)) return failure();

  // One identity: the callee's declaration instantiated at those arguments is
  // the signature spelled here.
  if (failed(verifyEqualAfterInstantiation(Type(formal), *arguments,
                                           Type(actual), normalize, err)))
    return failure();

  return arguments;
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&]{ return emitOpError(); };

  // A verifier holds no impl selection, so the comparison reads both
  // signatures as written.
  return buildParameterSpecialization(/*selection=*/nullptr, errFn);
}

FailureOr<SpecializationMap> MethodCallOp::buildParameterSpecialization(
    llvm::function_ref<Type(Type)> selection,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  auto trait = getTrait(err);
  if (failed(trait)) return failure();

  auto methodFormalTy = getMethodFunctionType(err);
  if (failed(methodFormalTy)) return failure();

  // A method's declaration binds the trait's parameters, labels 0 to n - 1,
  // and then its own, labelled from n. The trait's come from the receiver
  // claim by position -- the trait's arguments ride in the application -- and
  // this call's types determine the method's own.
  if (failed(trait->buildSubstitutionForSelfClaim(getClaimType(), err)))
    return failure();
  ArrayRef<Type> traitArguments = getClaimType().getTraitApplication().getTypeArgs();
  SmallVector<GenericTypeInterface, 4> parameters;
  for (Type parameter : trait->getTypeParams())
    parameters.push_back(getParameterOccurrence(parameter));
  llvm::append_range(parameters,
                     getOwnTypeParameters(Type(*methodFormalTy),
                                          trait->getTypeParams().size()));
  TypeArguments args(parameters);
  for (auto [parameter, argument] : llvm::zip(parameters, traitArguments))
    (void)args.assign(parameter, argument, /*err=*/nullptr);

  return readCallSpecialization(getOperation(), *methodFormalTy,
                                std::move(args), getActualFunctionType(),
                                getMethodName(), selection, err);
}

ImplOp MethodCallOp::getProvenImpl() {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  assert(claimTy.isProven());

  // This reads a proven claim's impl during lowering, which runs on a verified
  // module: the op is nested in it (so `getModule` finds it), and the proof the
  // claim carries was checked by `ProofOp::verifySymbolUses` (so its impl
  // symbol resolves). Neither guard fires on a module that reached lowering; a
  // hostile blob is refused at the verify rung before any pass reads a proof.
  auto module = getModule();
  if (failed(module))
    llvm_unreachable("MethodCallOp::getProvenImpl: not in a module");

  auto impl = ProofOp::getImplFromProof(*module, claimTy.getProof());
  if (failed(impl))
    llvm_unreachable("MethodCallOp::getProvenImpl: getImplFromProof failed");

  return *impl;
}

FailureOr<func::FuncOp> MethodCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter, const CallSubstitution &subst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrSpecializeFreeFunctionFromMethod(rewriter, claimTy, getMethodName(),
                                           getArguments().getTypes(), subst,
                                           respell);
}

ParseResult MethodCallOp::parse(OpAsmParser& p, OperationState &st) {
  MLIRContext* ctx = p.getContext();

  // grammar:
  //
  // trait.method.call %claim @Trait[Types...]::@method(%arguments...)
  //   : (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // parse %claim
  OpAsmParser::UnresolvedOperand claim;
  if (p.parseOperand(claim)) return failure();

  // parse '@Trait[Types...]' as TraitApplicationAttr
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

  // parse '::'
  if (p.parseColon() || p.parseColon()) return failure();

  // parse '@method' as FlatSymbolRefAttr
  FlatSymbolRefAttr methodName;
  if (p.parseAttribute(methodName)) return failure();

  // add methodRef attribute
  auto traitName = traitApp.getTraitName().getValue();
  auto methodRef = SymbolRefAttr::get(ctx, traitName, methodName);
  st.addAttribute("method_ref", methodRef);

  // parse '(' %arguments... ')'
  SmallVector<OpAsmParser::UnresolvedOperand> arguments;
  if (p.parseOperandList(arguments, OpAsmParser::Delimiter::Paren)) return failure();

  // parse ':' methodFunctionType
  FunctionType argumentTypesAndResultType;
  if (p.parseColonType(argumentTypesAndResultType)) return failure();

  // add the result types
  st.addTypes(argumentTypesAndResultType.getResults());

  // parse optional 'by' @ProofSym
  FlatSymbolRefAttr proofSym;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proofSym)) return failure();
  }

  // build the type of %claim
  auto loc = p.getCurrentLocation();
  ClaimType claimTy = ClaimType::get(ctx, traitApp, proofSym);

  // resolve %claim
  if (p.resolveOperand(claim, claimTy, st.operands))
    return failure();

  // resolve arguments
  auto argumentTypes = argumentTypesAndResultType.getInputs();
  if (argumentTypes.size() != arguments.size())
    return p.emitError(loc, "argument count mismatch");

  if (p.resolveOperands(arguments, argumentTypes, loc, st.operands))
    return failure();

  // parse attributes
  if (p.parseOptionalAttrDictWithKeyword(st.attributes)) return failure();
  
  return success();
}

void MethodCallOp::print(OpAsmPrinter& p) {
  // grammar:
  //
  // trait.method.call %claim @Trait[Types...]::@method(%arguments...)
  //   : (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // print %claim
  p << " " << getClaim() << " ";

  // print '@Trait[Types...]'
  getTraitApplication().print(p);

  // '::@method(%arguments...)'
  p << "::" << getMethodAttr() << "(" << getArguments() << ")";

  // on a newline:
  // ': ' (argumentTypes) -> (resultTypes)`
  p.printNewline();
  p.getStream().indent(2);
  FunctionType actualFunctionType = FunctionType::get(
    getContext(),
    ValueRange(getArguments()).getTypes(),
    getResultTypes()
  );
  p << ": " << actualFunctionType;

  // on a newline:
  // (by @Proof)?
  if (getClaimType().isProven()) {
    p.printNewline();
    p.getStream().indent(2);
    p << "by " << getClaimType().getProof();
  }

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"method_ref"}
  );
}


//===----------------------------------------------------------------------===//
// FuncCallOp
//===----------------------------------------------------------------------===//

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto calleeName = getCalleeNameAttr();
  if (!calleeName)
    return emitOpError() << "requires a 'callee_name' symbol reference attribute";

  auto errFn = [&] { return emitOpError(); };

  // A verifier holds no impl selection, so the comparison reads both
  // signatures as written.
  return buildParameterSpecialization(/*selection=*/nullptr, errFn);
}

FailureOr<SpecializationMap> FuncCallOp::buildParameterSpecialization(
    llvm::function_ref<Type(Type)> selection,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  auto formal = getCalleeFunctionType(err);
  if (failed(formal)) return failure();

  // The callee's declaration binds the parameters its signature spells, and
  // this call's own types determine each of them.
  return readCallSpecialization(getOperation(), *formal,
                                TypeArguments(getCalleeTypeParams()),
                                getActualFunctionType(), getCalleeName(),
                                selection, err);
}

FailureOr<func::FuncOp> FuncCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter, const CallSubstitution &subst,
    llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  auto module = getModule();
  if (failed(module)) return failure();

  auto callee = getCallee();
  if (failed(callee)) return failure();

  SmallVector<GenericTypeInterface, 4> typeParams = getCalleeTypeParams();
  assert(!typeParams.empty() &&
         "a call of a callee binding no type parameter lowers as written");

  // The instance is the one this call's type arguments and evidence name. The
  // specialization map is written when the substitution is built and is not
  // touched by closing it, so the arguments read here and the body cut below
  // are read off one object.
  SmallVector<Type> typeArguments;
  for (GenericTypeInterface parameter : typeParams)
    typeArguments.push_back(subst.getSpecialization().apply(parameter));
  AttrTypeReplacer stamp =
      makeTypeReplacerFromSubstitution(subst.toTypeMap(), CloneKind::Instance);
  auto key = InstanceKey::get(getCalleeNameAttr(), typeArguments,
                              callee->getFunctionType().getInputs(),
                              getOperandTypes(), stamp, respell);
  if (failed(key))
    return emitOpError() << "supplies '@" << getCalleeName()
                         << "' a claim that names no proof, which identifies "
                            "no instance";

  func::FuncOp instance = getOrCutInstance(
      rewriter, *module, *key, [&](StringRef instanceName) {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(*callee);
        // An external polymorphic declaration has no body to clone;
        // specialization refuses it, so this call has no instance to name.
        // Cut at module scope, the instance is a `func.func`.
        return cast_if_present<func::FuncOp>(
            specializePolymorph(rewriter, *callee, instanceName,
                                subst.toTypeMap())
                .getOperation());
      });
  if (!instance)
    return failure();
  return instance;
}


//===----------------------------------------------------------------------===//
// ProjectOp
//===----------------------------------------------------------------------===//

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  auto errFn = [&] { return emitOpError(); };
  auto requirement =
      getClaimRequirementAt(getSourceClaim(), module, getIndex(), errFn);
  if (failed(requirement))
    return failure();

  // The result type is an annotation on the selection: the index decides which
  // claim this op produces, so the spelled one must be that claim. A where
  // entry carries the proof the source's proof gives it there; any other
  // requirement is compared modulo the proof, which the evidence the stage
  // inlines here decides (`inlineEvidence`).
  bool selected = requirement->isProven()
                      ? *requirement == getResultClaim()
                      : stripClaimProofs(Type(*requirement)) ==
                            stripClaimProofs(Type(getResultClaim()));
  if (!selected)
    return emitOpError() << "type mismatch: expected " << *requirement
                         << " but found " << getResultClaim();

  return success();
}
