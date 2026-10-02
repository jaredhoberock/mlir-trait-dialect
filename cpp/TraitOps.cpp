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
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/RegionKindInterface.h>
#include <optional>
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

/// The arguments a derive or proof states for its impl's parameters, in the
/// grammar every impl citation shares, `[!P = T, ...]`, which both always
/// state: the brackets are required, even for an impl binding no parameter.
static ::mlir::ParseResult
parseStatedImplArguments(::mlir::OpAsmParser &parser,
                         ::mlir::ArrayAttr &arguments) {
  ::llvm::SmallVector<TypeBindingAttr> bindings;
  ::mlir::FailureOr<bool> present = parseImplArguments(parser, bindings);
  if (::mlir::failed(present))
    return ::mlir::failure();
  if (!*present)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected the arguments the impl's parameters take, "
                            "`[!P = T, ...]`");
  arguments = parser.getBuilder().getArrayAttr(
      ::llvm::SmallVector<::mlir::Attribute>(bindings.begin(), bindings.end()));
  return ::mlir::success();
}

static void printStatedImplArguments(::mlir::OpAsmPrinter &printer,
                                     ::mlir::Operation *,
                                     ::mlir::ArrayAttr arguments) {
  printImplArguments(printer,
                     ::llvm::to_vector(arguments.getAsRange<TypeBindingAttr>()));
}

/// A declaration's where clause, `where [predicate, ...]`: read as the empty
/// clause where the keyword is absent, and printed, with the space before it,
/// only where it states something.
static ::mlir::ParseResult parseWhereClause(::mlir::OpAsmParser &parser,
                                            PredicateArrayAttr &clause) {
  if (::mlir::failed(parser.parseOptionalKeyword("where"))) {
    clause = PredicateArrayAttr::get(parser.getContext(),
                                     ::llvm::ArrayRef<::mlir::Attribute>());
    return ::mlir::success();
  }
  clause = ::llvm::dyn_cast_or_null<PredicateArrayAttr>(
      PredicateArrayAttr::parse(parser, {}));
  if (!clause)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected a predicate array");
  return ::mlir::success();
}

static void printWhereClause(::mlir::OpAsmPrinter &printer, ::mlir::Operation *,
                             PredicateArrayAttr clause) {
  if (clause.empty())
    return;
  printer << " where ";
  clause.print(printer);
}

/// An impl's header, `[@name] for @Trait[...] [where [...]]`. The name is read
/// as the one synthesized from the application and the where clause where it is
/// absent, and printed only where it is another.
static ::mlir::ParseResult
parseImplHeader(::mlir::OpAsmParser &parser, ::mlir::StringAttr &symName,
                TraitApplicationAttr &selfApplication,
                PredicateArrayAttr &assumptions) {
  (void)parser.parseOptionalSymbolName(symName);
  if (parser.parseKeyword("for"))
    return ::mlir::failure();
  selfApplication = ::llvm::dyn_cast_or_null<TraitApplicationAttr>(
      TraitApplicationAttr::parse(parser, {}));
  if (!selfApplication)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected a TraitApplicationAttr");
  if (parseWhereClause(parser, assumptions))
    return ::mlir::failure();
  if (!symName)
    symName = parser.getBuilder().getStringAttr(
        ImplOp::generateSymName(selfApplication, assumptions));
  return ::mlir::success();
}

static void printImplHeader(::mlir::OpAsmPrinter &printer, ::mlir::Operation *op,
                            ::mlir::StringAttr symName,
                            TraitApplicationAttr selfApplication,
                            PredicateArrayAttr assumptions) {
  if (symName.getValue() !=
      ImplOp::generateSymName(selfApplication, assumptions)) {
    printer.printSymbolName(symName.getValue());
    printer << ' ';
  }
  printer << "for ";
  selfApplication.print(printer);
  printWhereClause(printer, op, assumptions);
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
} // namespace mlir::trait

#define GET_OP_CLASSES
#include <TraitOps.cpp.inc>

using namespace mlir;
using namespace mlir::trait;

namespace mlir::trait { std::string hashToSuffix(StringRef input); }

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

/// A trait's or impl's body takes no block arguments, so a method, which is not
/// isolated from above, reads nothing from outside its own body: the
/// declaration is isolated from above and its child list admits no operation
/// with a result, so its block arguments are the only values a method could
/// read from around it.
///
/// XXX TODO the commit that gives trait.trait and trait.impl their self claim
/// and prerequisites as block arguments, read by their methods, deletes this
/// check.
LogicalResult verifyDeclarationBodyTakesNoArguments(Operation *op) {
  Region &body = op->getRegion(0);
  if (body.empty() || body.front().getNumArguments() == 0)
    return success();
  return op->emitOpError() << "body must take no block arguments: a method "
                              "reads nothing from outside its own body";
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
/// source of evidence for choosing that result type. This is intentionally a
/// syntactic check: the verifier does not try to invert equality predicates or
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

} // namespace

/// What `op` may read a spelling through: the hypotheses the scope it stands in
/// holds, and then the projections the evidence `values` carry justify reducing
/// -- for a proven claim the impls its proof tree names, by index; for a derived
/// claim the impl it commits to and whatever its given operands carry in turn.
static NormalizationContext buildLocalClaimNormalizationContext(
    Operation *op, ValueRange values, ModuleOp module);

/// Whether `ty` spells a projection anywhere.
///
/// A declaration that spells none is rebuilt by substitution alone, so there is
/// nothing for evidence to reduce in it and the evidence is not gathered.
static bool spellsAProjection(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) {
    if (isa<ProjectionType>(sub))
      found = true;
  });
  return found;
}

/// What a proven claim may be read through: the impl its proof names and, by
/// index, the impls the subproofs discharging that impl's obligations name.
static NormalizationContext buildProofNormalizationContext(ClaimType provenClaim,
                                                           ModuleOp module);

/// What the obligations of the impl `proof` stands on may be read through: the
/// impls the proofs discharging them name, by index, at the application `at`
/// carries those obligations to. A proof justifies nothing about itself, so its
/// own rule is not among these.
static NormalizationContext buildSubproofNormalizationContext(ProofOp proof,
                                                              ClaimType at,
                                                              ModuleOp module);


//===----------------------------------------------------------------------===//
// NormalizationContext
//===----------------------------------------------------------------------===//

FailureOr<Type> NormalizationContext::normalize(
    Type ty,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // The step resolves projections from this context's own rules first, so a
  // caller normalizes against exactly the evidence it holds. The shared
  // fallible driver spends the rewrite budget; on nonconvergence this reports
  // through the op-attached diagnostic rather than the driver's fatal
  // module-level reporter.
  // A lookup that will not ground refuses through the caller's own diagnostic,
  // which is the one report of it; the driver below then stops without adding a
  // second.
  bool lookupRefused = false;
  // The hypotheses read as classes, not as directed rules: every member of a
  // class rewrites to the one member the class stands for. Two hypotheses of
  // opposite orientation therefore settle, where a pair of directed rules would
  // trade the spelling back and forth until the budget below ran out.
  llvm::DenseMap<Type, Type> toCanonicalMembers =
      equalities.substitutionToCanonicalMembers();
  auto normalizeOnce = [&](Type root) {
    AttrTypeReplacer replacer = makeEndpointSealedReplacer();
    replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
      for (LocalProjectionRule &rule : localProjectionRules) {
        if (!rule.impl || proj.getTraitApplication() != rule.app)
          continue;

        auto resolved = rule.impl.specializeAssociatedTypeBinding(
            proj.getAssocName().getValue(), proj.getAssocTypeArgs(),
            rule.subst);
        if (failed(resolved))
          continue;
        return *resolved;
      }
      return std::nullopt;
    });
    root = replacer.replace(root);
    if (!toCanonicalMembers.empty())
      root = applySubstitutionOnce(toCanonicalMembers, root);
    // The recorded facts are read after the local rules, so a projection this
    // op's own evidence answers is answered from that evidence and only what it
    // leaves standing reaches the record. Each pass runs both, and the driver
    // below repeats them until the spelling settles.
    if (recordedFacts)
      root = recordedFacts->resolveProjectionsIn(root);
    if (moduleLookup && !lookupRefused) {
      FailureOr<Type> byLookup =
          resolveProjectionsByLookup(root, moduleLookup, moduleLookupOrigin,
                                     moduleLookupScope, err);
      if (failed(byLookup))
        lookupRefused = true;
      else
        root = *byLookup;
    }
    return root;
  };

  Type out;
  bool converged = succeeded(tryNormalizeProjectionsToFixedPoint(
      ty, normalizeOnce, out));
  if (lookupRefused)
    return failure();
  if (!converged) {
    if (err)
      err() << "projection normalization did not converge; check for cyclic "
               "associated type bindings";
    return failure();
  }
  return out;
}

FailureOr<FunctionType> NormalizationContext::normalize(
    FunctionType functionType,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto normalized = normalize(Type(functionType), err);
  if (failed(normalized))
    return failure();
  return cast<FunctionType>(*normalized);
}


//===----------------------------------------------------------------------===//
// TraitOp
//===----------------------------------------------------------------------===//

/// A bound requirement spells no parameter but the trait's own and its
/// binder's variables, so no premise a hop supplies is satisfied by an
/// unrelated same-labelled parameter of the caller. Its conclusion is a
/// requirement like any other, so it may not name the trait itself except
/// through a projection.
static LogicalResult verifyBoundRequirements(TraitOp trait,
                                             const DenseSet<Type> &traitParams) {
  for (auto [index, entry] : llvm::enumerate(trait.getRequirements())) {
    auto bound = dyn_cast<BoundPredicateAttr>(entry);
    if (!bound)
      continue;
    SmallVector<Attribute> predicates(bound.getPremises());
    predicates.push_back(bound.getConclusion());
    for (Attribute predicate : predicates)
      for (GenericTypeInterface parameter : getTypeParametersIn(
               Type(ClaimType::get(trait.getContext(), predicate, nullptr))))
        if (!isa<BoundVarType>(Type(parameter)) &&
            !traitParams.contains(Type(parameter)))
          return trait.emitOpError()
                 << "bound requirement " << index << " spells "
                 << Type(parameter) << ", which is neither a parameter of trait "
                 << "'@" << trait.getSymName() << "' nor a variable of its binder";
    if (auto app = dyn_cast<TraitApplicationAttr>(bound.getConclusion()))
      if (app.getTraitName().getValue() == trait.getSymName() &&
          !containsType<ProjectionType>(app.getTypeArgs().front()))
        return trait.emitOpError()
               << "bound requirement " << index << " concludes " << app
               << ", which must not reference the current trait";
  }
  return success();
}

/// A binder variable stands only inside the bound predicate that binds it and
/// the witness proving that predicate, never as a declaration's parameter:
/// the two kinds of type variable are then disjoint, so an evidence body that
/// compares a declaration's predicate with a binder's can never read one as
/// the other.
static LogicalResult verifyParameterIsNoBinderVariable(Operation *op,
                                                       Type parameter) {
  if (isa<BoundVarType>(parameter))
    return op->emitOpError()
           << "type parameter " << parameter
           << " is a binder variable, which stands only inside a bound "
              "predicate";
  return success();
}

/// The type variable entry `declared` of `assoc`'s type parameter list
/// declares, refused where it stands when the entry is not a type, not a type
/// variable, or carries a binder variable. A child's own invariants are
/// verified after its parent's, so the entry is read as an attribute that may
/// be anything rather than cast.
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
  for (GenericTypeInterface inside : getTypeParametersIn(param))
    if (failed(verifyParameterIsNoBinderVariable(assoc, Type(inside))))
      return failure();
  return param;
}

LogicalResult TraitOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();
  if (failed(verifyDeclarationBodyTakesNoArguments(getOperation())))
    return failure();

  auto typeParams = getTypeParams().getAsValueRange<TypeAttr>();

  // types must be unique GenericTypeParameters
  DenseSet<Type> uniqueParams;
  for (Type ty : typeParams) {
    if (!isa<GenericTypeInterface>(ty))
      return emitOpError() << "expected GenericTypeInterface (e.g., !trait.poly), found " << ty;
    if (failed(verifyParameterIsNoBinderVariable(getOperation(), ty)))
      return failure();
    if (!uniqueParams.insert(ty).second)
      return emitOpError() << "type parameters must be unique";
  }

  // there must be at least one type parameter
  if (uniqueParams.size() < 1)
    return emitOpError() << "requires at least one type parameter";

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
  for (Operation &op : getBody().front()) {
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
  auto endpointMentionsParam = [&](Type endpoint) {
    for (GenericTypeInterface g : getGenericTypesIn(endpoint))
      if (uniqueParams.contains(Type(g)) || gatParams.contains(Type(g)))
        return true;
    return false;
  };

  // check requirements
  for (Attribute pred : getRequirements()) {
    if (auto app = dyn_cast<TraitApplicationAttr>(pred)) {
      // each application requirement must use at least one of the trait's type
      // parameters OR at least one GAT type parameter
      bool mentionsTraitParam = llvm::any_of(uniqueParams, [&](Type param) {
        return app.mentionsType(param);
      });
      bool mentionsGatParam = llvm::any_of(gatParams, [&](Type param) {
        return app.mentionsType(param);
      });

      if (!mentionsTraitParam && !mentionsGatParam)
        return emitOpError() << "'where' clause requirement " << app
                             << " must mention at least one type parameter";

      // A direct self-reference like @Trait[!S] would create a circular
      // obligation that no impl can satisfy. However, a self-reference whose
      // self argument goes through a projection (e.g. @Trait[!trait.proj<...>])
      // is safe: the projection resolves to a concrete type during
      // monomorphization, so the obligation is discharged against a different
      // impl, not the one being defined.
      if (app.getTraitName().getValue() == getSymName()) {
        bool selfArgHasProjection = containsType<ProjectionType>(app.getTypeArgs().front());
        if (!selfArgHasProjection)
          return emitOpError() << "'where' clause requirement " << app
                               << " must not reference the current trait";
      }
    } else if (auto eq = dyn_cast<TypeEqualityAttr>(pred)) {
      // An equality requirement has no trait head, so there is no
      // self-reference to forbid; it must still relate the trait's parameters,
      // mentioning at least one through either endpoint.
      if (!endpointMentionsParam(eq.getLhs()) &&
          !endpointMentionsParam(eq.getRhs()))
        return emitOpError() << "'where' clause equality requirement "
                             << ClaimType::getEquality(getContext(), eq)
                             << " must mention at least one type parameter";
    }
  }

  if (failed(verifyBoundRequirements(*this, uniqueParams)))
    return failure();

  // check trait method result generics
  for (Operation &op : getBody().front()) {
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

  // verify obligations
  return getRequirements().verifySymbolUses(getOperation(), symbolTable);
}

FailureOr<SpecializationMap> TraitOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                      llvm::function_ref<InFlightDiagnostic()> errFn) {
  // A trait header's parameters are its type_params in array order, and its
  // self application spells exactly those, in that order. So an application of
  // this trait at the right arity determines every parameter by position:
  // nothing is read out of the arguments and nothing is compared afterwards.
  TraitApplicationAttr application = actualSelfClaim.getTraitApplication();
  if (application.getTraitName().getValue() != getSymName()) {
    if (errFn)
      errFn() << "trait mismatch: expected @" << getSymName() << ", but found "
              << application.getTraitName();
    return failure();
  }

  ArrayAttr parameters = getTypeParams();
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
    auto generic = dyn_cast<GenericTypeInterface>(
        cast<TypeAttr>(parameter).getValue());
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
  MLIRContext *ctx = getContext();
  // Each requirement becomes a claim of its arm: an application claim for an
  // application entry, an equality claim for an equality entry.
  SmallVector<ClaimType> result;
  for (Attribute pred : getRequirements()) {
    if (auto app = dyn_cast<TraitApplicationAttr>(pred))
      result.push_back(ClaimType::get(ctx, app));
    else if (auto eq = dyn_cast<TypeEqualityAttr>(pred))
      result.push_back(ClaimType::getEquality(ctx, eq));
  }
  return result;
}

FailureOr<SmallVector<ClaimType>> TraitOp::specializeRequirementsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // build a specialized substitution for actualSelfClaim
  auto spec = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(spec)) return failure();

  // apply the substitution to each requirement. A substitution rewrites the
  // type arguments a claim carries, never the claim wrapper itself: its keys
  // are this trait's type parameters, never a whole ClaimType, so the outer
  // constructor is preserved and the result is always a claim. This holds
  // structurally, independent of whether the module's symbols resolve, so it is
  // safe on unverified IR -- the cast never fails.
  return llvm::map_to_vector(getRequirementsAsClaims(), [&](ClaimType req) {
    ClaimType specializedReq = dyn_cast_or_null<ClaimType>(instantiate(req, *spec));
    if (!specializedReq)
      llvm_unreachable("TraitOp::specializeRequirementsAsClaimsFor: expected ClaimType");
    return specializedReq;
  });
}

SmallVector<ImplOp> TraitOp::getImpls() {
  auto module = getModule();
  if (failed(module)) return {};

  // Impls are top-level module children (ImplOp is HasParent<ModuleOp>), so scan
  // them directly and match this trait's symbol name. This avoids a full-module
  // symbol-use walk, which materializes every operation's attribute dictionary.
  StringRef traitName = getSymName();
  SmallVector<ImplOp> result;
  for (Operation &op : *module->getBody()) {
    auto impl = dyn_cast<ImplOp>(op);
    if (!impl)
      continue;
    TraitApplicationAttr selfApp = impl.getSelfApplication();
    if (selfApp && selfApp.getTraitName().getValue() == traitName)
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

/// Verifies an impl's equality-armed witnesses and returns each as a local
/// resolution rule. Such a witness certifies that a sibling impl binds the
/// witnessed projection to a resolved type; a witness citing a conditional
/// impl is legal exactly when the impl's own where clause covers the cited
/// impl's assumptions or an application-armed witness supplies them. Each
/// entry verifies with an EMPTY equality modulus: sibling witnesses never
/// serve as each other's modulus, because an attribute array has no dominance
/// and mutual justification could ground a false equality on nothing.
///
/// A witness projection carrying the impl's own parameters is verified like
/// any other: the head comparison is rigid, so a variable in the projection
/// equals only the same variable in the cited impl's head at the witness's
/// arguments. A witness citing a single-instance impl for a projection
/// quantified over the host impl's parameters fails that comparison -- it
/// would accept a generic impl on the strength of one instance -- while one
/// citing a blanket sibling at the host's parameters is the evidence a generic
/// impl's own header equalities need.
static FailureOr<SmallVector<LocalProjectionRule>> collectImplWitnessRules(
    ImplOp impl, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  SmallVector<LocalProjectionRule> rules;
  ArrayAttr witnesses = impl.getWitnessesAttr();
  if (!witnesses)
    return rules;

  SmallVector<TraitApplicationAttr> obligationPremises(
      impl.getAssumptions().getApplications());
  // The application-armed witnesses cover a cited conditional impl's standing
  // assumptions; gather them first so every equality-armed witness below
  // verifies against the whole discharge set regardless of array order.
  SmallVector<WitnessAttr> dischargeWitnesses;
  for (Attribute entry : witnesses) {
    auto witness = cast<WitnessAttr>(entry);
    if (isa<TraitApplicationAttr>(witness.getPredicate()))
      dischargeWitnesses.push_back(witness);
  }
  for (Attribute entry : witnesses) {
    auto witness = cast<WitnessAttr>(entry);
    if (!isa<TypeEqualityAttr>(witness.getPredicate()))
      continue;
    auto rule = verifyProjectionResolutionAtImpl(
        module, witness, /*premises=*/{}, obligationPremises,
        dischargeWitnesses, errFn);
    if (failed(rule))
      return failure();
    rules.push_back(std::move(*rule));
  }
  return rules;
}

/// The context an impl's own obligations are judged under.
///
/// Three rules and no more: the impl's own associated type bindings, for a
/// projection over its self application; the sibling bindings its declared
/// witnesses certify, verified above and passed in as `witnessRules`; and its
/// where clause's equalities as hypotheses, which are in scope wherever the
/// impl's own obligations are checked -- a trait-header equality requirement
/// and the impl's own method signature are both judged under the clause the
/// impl declares. The verifier enumerates no candidate impls, so a projection
/// none of these three reduces is equal to itself alone.
///
/// The impl's own bindings are spelled over the impl's own parameters, so the
/// substitution the first rule carries is what the impl's self claim says its
/// parameters take, which is those parameters themselves.
static FailureOr<NormalizationContext> buildImplOwnNormalizationContext(
    ImplOp impl, ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto ownArguments = impl.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(ownArguments))
    return failure();

  NormalizationContext ctx;
  ctx.addLocalProjectionRule(impl, impl.getSelfApplication(), *ownArguments);
  for (const LocalProjectionRule &rule : witnessRules)
    ctx.addLocalProjectionRule(rule.impl, rule.app, rule.subst);
  for (Attribute predicate : impl.getAssumptions())
    if (auto equality = dyn_cast<TypeEqualityAttr>(predicate))
      ctx.assumeEqual(equality.getLhs(), equality.getRhs());
  return ctx;
}

/// The pairing carrying a trait's declaration of a method onto the impl's copy
/// of it.
///
/// A method's declaration binds its header's parameters and then its own, in
/// that order, and the two declarations of one method must bind equally many of
/// their own: the trait states how many a caller supplies, and an impl copy
/// with a different count is a different declaration. So the correspondence is
/// positional -- the trait's j-th own variable is the impl's j-th -- and the
/// check is the one identity it makes true.
struct TraitMethodCorrespondence {
  /// The trait method's own type variables, in first-occurrence order. A call
  /// names its type arguments under these.
  SmallVector<GenericTypeInterface, 4> traitOwn;
  /// The impl method's own type variables, paired with `traitOwn` by position.
  /// The clone a call is lowered to binds these.
  SmallVector<GenericTypeInterface, 4> implOwn;
  /// The whole substitution carrying the trait's declaration to the impl's: the
  /// impl's self arguments for the trait's parameters, then the pairing above.
  SpecializationMap substitution;
};

/// The type parameters `signature` binds beyond `headerParams`, in
/// first-occurrence order: a method's own variables, as against the ones the
/// declaration it is written in supplies.
static SmallVector<GenericTypeInterface, 4> getOwnTypeParameters(
    Type signature, const DenseSet<Type> &headerParams) {
  SmallVector<GenericTypeInterface, 4> own;
  for (GenericTypeInterface parameter : getTypeParametersIn(signature))
    if (!headerParams.contains(Type(parameter)))
      own.push_back(parameter);
  return own;
}

/// The type parameters a trait header supplies to the methods written in it.
static DenseSet<Type> getTraitHeaderParameters(TraitOp traitOp) {
  DenseSet<Type> params;
  for (Attribute declared : traitOp.getTypeParams())
    if (auto typeAttr = dyn_cast<TypeAttr>(declared))
      for (GenericTypeInterface parameter :
           getTypeParametersIn(typeAttr.getValue()))
        params.insert(Type(parameter));
  return params;
}

/// Builds the correspondence above and checks it: this is the impl's signature
/// check and the rekeying a call lowering needs, which are one pairing. The
/// check is that the trait's declaration instantiated through it is the impl's,
/// and a binding a call names -- keyed by the trait's spelling, since a method
/// call reads the trait's declaration -- rekeys through the same pairing to the
/// copy of the method that is actually cloned.
static FailureOr<TraitMethodCorrespondence> buildTraitMethodCorrespondence(
    ImplOp impl, TraitOp traitOp, FunctionOpInterface implMethod,
    ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  StringRef name = implMethod.getName();
  auto traitMethod = traitOp.getMethod(name, errFn);
  if (failed(traitMethod)) return failure();

  // The prefix: the trait's parameters take this impl's self arguments, by
  // position.
  auto traitSubst =
      traitOp.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(traitSubst)) return failure();

  DenseSet<Type> implHeaderParams;
  for (GenericTypeInterface parameter : impl.getTypeParams())
    implHeaderParams.insert(Type(parameter));

  TraitMethodCorrespondence correspondence;
  Type traitMethodTy = Type(traitMethod->getFunctionType());
  Type implMethodTy = Type(implMethod.getFunctionType());
  correspondence.traitOwn =
      getOwnTypeParameters(traitMethodTy, getTraitHeaderParameters(traitOp));
  SmallVector<GenericTypeInterface, 4> implOwn =
      getOwnTypeParameters(implMethodTy, implHeaderParams);
  if (correspondence.traitOwn.size() != implOwn.size()) {
    if (errFn)
      errFn() << "method '" << name << "' binds " << implOwn.size()
              << " type parameter(s) of its own, but trait '"
              << traitOp.getSymNameAttr() << "' declares it with "
              << correspondence.traitOwn.size();
    return failure();
  }

  // What the impl's copy spells for each of the trait's own variables, read off
  // the position it stands in. The copy may rename a variable and may constrain
  // its kind -- so it is the occurrence and not the bare parameter that carries
  // over -- but it may not instantiate one: a caller supplies an argument for
  // every variable the trait declares, and a copy standing only at some of them
  // is a method no call to the trait's declaration can reach. Each spelling is
  // therefore one parameter occurrence, and distinct ones, which makes the
  // pairing a renaming the clone below can rekey a call's bindings through.
  correspondence.substitution = *traitSubst;
  TypeArguments ownArguments(correspondence.traitOwn);
  extractTypeArguments(instantiate(traitMethodTy, *traitSubst), implMethodTy,
                       ownArguments);
  DenseSet<Type> spelled;
  for (GenericTypeInterface traitVariable : correspondence.traitOwn) {
    std::optional<Type> spelling = ownArguments.lookup(traitVariable);
    if (!spelling) {
      if (errFn)
        errFn() << "method '" << name << "' does not say what its copy of "
                << Type(traitVariable) << " is";
      return failure();
    }
    GenericTypeInterface implVariable = getParameterOccurrence(*spelling);
    if (!implVariable || !spelled.insert(Type(implVariable)).second) {
      if (errFn)
        errFn() << "method '" << name << "' spells " << *spelling
                << " where trait '@" << traitOp.getSymName()
                << "' declares the type parameter " << Type(traitVariable)
                << ": an impl's copy of a method renames the trait's type"
                   " parameters, one for one";
      return failure();
    }
    correspondence.substitution.bind(traitVariable, *spelling);
    correspondence.implOwn.push_back(implVariable);
  }

  // Substituting this impl's self application into the trait's declaration can
  // mint a ground projection the impl's own bindings do not resolve -- a
  // sibling impl's application, e.g. Group[coop.block]::Shape. A declared
  // witness reduces exactly those, so both declarations reach the comparison at
  // the same grade.
  auto normalization = buildImplOwnNormalizationContext(impl, witnessRules, errFn);
  if (failed(normalization))
    return failure();
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return normalization->normalize(ty, errFn);
  };

  auto expected = normalize(instantiate(traitMethodTy,
                                        correspondence.substitution));
  if (failed(expected))
    return failure();
  auto actual = normalize(implMethodTy);
  if (failed(actual))
    return failure();
  if (stripClaimProofs(*expected) != stripClaimProofs(*actual)) {
    if (errFn)
      errFn() << "method '" << name << "' has incompatible signature: "
              << "expected " << *expected << " but found " << *actual;
    return failure();
  }
  return correspondence;
}

static LogicalResult verifyEqualityObligations(
    ImplOp impl, TraitOp traitOp, ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn);

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
  if (failed(verifyDeclarationBodyTakesNoArguments(getOperation())))
    return failure();
  // An impl's premises restrict the arguments it applies at; one quantified
  // over variables of its own would restrict nothing any argument supplies.
  for (auto [index, predicate] : llvm::enumerate(getAssumptions()))
    if (isa<BoundPredicateAttr>(predicate))
      return emitOpError() << "where-clause entry " << index
                           << " binds variables of its own; only a trait's "
                              "requirement is quantified";
  for (GenericTypeInterface parameter : getTypeParams())
    if (failed(verifyParameterIsNoBinderVariable(getOperation(),
                                                 Type(parameter))))
      return failure();
  if (failed(verifyImplParametersAreConstrained(*this)))
    return failure();
  return verifyAssociatedTypeBindingScopes(*this);
}

static LogicalResult verifyBoundRequirementEvidence(
    ImplOp impl, TraitOp traitOp, ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn);

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&]{ return emitOpError(); };

  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // Verify self application attribute exists
  auto selfApp = getSelfApplication();
  if (!selfApp)
    return emitOpError() << "requires a self application TraitApplicationAttr";

  // Verify the self application
  if (failed(selfApp.verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // Verify the where-clause predicates' symbol uses: application entries name a
  // valid trait at the right arity; equality entries carry their symbol users
  // nested in the endpoints. The automatic symbol-user driver skips inherent
  // attributes, so the owning op delegates here, exactly as trait.trait does for
  // its requirements.
  if (failed(getAssumptions().verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // The verified witnesses become local resolution rules the comparisons below
  // replay after their own-binding rule; the fixed-point walk applies them
  // innermost-first, so a nested projection reduces its inner application
  // before its outer one.
  auto premiseRules = collectImplWitnessRules(*this, *module, errFn);
  if (failed(premiseRules))
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
      if (failed(buildTraitMethodCorrespondence(
              *this, traitOp, implMethod, *premiseRules, errFn)))
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
    } else {
      return emitOpError() << "body may only contain 'trait.method' or "
                              "'trait.assoc_type' operations";
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

  if (failed(verifyEqualityObligations(*this, traitOp, *premiseRules, errFn)))
    return failure();

  if (failed(verifyBoundRequirementEvidence(*this, traitOp, *premiseRules,
                                            errFn)))
    return failure();

  return success();
}

/// What a bound requirement's evidence may cite, in the impl stating it: the
/// binder's premises and the impl's where clause by position, and every
/// predicate read through the impl's own bindings.
struct BoundEvidenceScope {
  ImplOp impl;
  ArrayRef<Attribute> premises;
  NormalizationContext &own;
  ModuleOp module;
  llvm::function_ref<InFlightDiagnostic()> errFn;

  /// `predicate` as a claim read through the impl's own bindings.
  FailureOr<Type> read(Attribute predicate) {
    return own.normalize(
        Type(ClaimType::get(impl.getContext(), predicate, nullptr)), errFn);
  }

  /// Whether `a` and `b` state one predicate once each is read.
  FailureOr<bool> same(Attribute a, Attribute b) {
    FailureOr<Type> readA = read(a);
    FailureOr<Type> readB = read(b);
    if (failed(readA) || failed(readB))
      return failure();
    return *readA == *readB;
  }
};

static LogicalResult verifyWitnessBody(BoundEvidenceScope &scope,
                                       Attribute body, Attribute predicate);

/// The predicate `body` proves in `scope`, read off the body itself: a binder
/// premise or where-clause entry by position; an impl citation's header at its
/// arguments, each of the cited impl's where-clause entries there discharged
/// in turn; a requirement hop's requirement of the application its body
/// proves, read by position as `trait.project` reads one, at its type
/// arguments, each premise there discharged in turn; an allegation's stated
/// application. Reflexivity states no predicate of its own -- it proves the
/// equality its position names, which `verifyWitnessBody` reads -- so it is
/// refused here. `refuse` reports under the predicate the outermost body is
/// verified against.
static FailureOr<Attribute>
readWitnessBody(BoundEvidenceScope &scope, Attribute body,
                llvm::function_ref<InFlightDiagnostic(const Twine &)> refuse) {
  if (auto premise = dyn_cast<BinderPremiseAttr>(body)) {
    if (premise.getPosition() >= scope.premises.size())
      return refuse(Twine("the binder states ") +
                    Twine(scope.premises.size()) + " premises");
    return scope.premises[premise.getPosition()];
  }
  if (auto premise = dyn_cast<ImplPremiseAttr>(body)) {
    PredicateArrayAttr where = scope.impl.getAssumptions();
    if (premise.getPosition() >= where.size())
      return refuse(Twine("the impl's where clause has ") +
                    Twine(where.size()) + " entries");
    return where.getPredicates()[premise.getPosition()];
  }
  if (isa<UnitAttr>(body))
    return refuse(Twine("reflexivity proves only the equality its position "
                        "names"));
  if (auto allegation = dyn_cast<AllegationAttr>(body))
    return Attribute(allegation.getApplication());
  if (auto hop = dyn_cast<RequirementHopAttr>(body)) {
    FailureOr<Attribute> of = readWitnessBody(scope, hop.getOf(), refuse);
    if (failed(of))
      return failure();
    auto application = dyn_cast<TraitApplicationAttr>(*of);
    if (!application)
      return refuse(Twine("a requirement is read off a trait application"));
    auto requirement = getClaimRequirementAt(
        ClaimType::get(scope.impl.getContext(), application), scope.module,
        hop.getPosition(), hop.getTypeArgs(), scope.errFn);
    if (failed(requirement))
      return failure();
    if (hop.getPremises().size() != requirement->premises.size())
      return refuse(Twine("requirement ") + Twine(hop.getPosition()) +
                    " states " + Twine(requirement->premises.size()) +
                    " premises, and the evidence discharges " +
                    Twine(hop.getPremises().size()));
    for (auto [premise, stated] :
         llvm::zip(hop.getPremises(), requirement->premises))
      if (failed(verifyWitnessBody(scope, premise, stated.getPredicate())))
        return failure();
    return requirement->conclusion.getPredicate();
  }

  auto citation = cast<ImplCitationAttr>(body);
  ImplOp cited = lookupSymbolFrom<ImplOp>(scope.module, citation.getImplRef());
  if (!cited)
    return refuse(Twine("it names no impl"));
  auto arguments = cited.substitutionFor(citation.getArguments(), scope.errFn);
  if (failed(arguments))
    return failure();
  SmallVector<ClaimType> where = cited.getWhereClauseAt(*arguments);
  if (citation.getDischarges().size() != where.size())
    return refuse(Twine("the cited impl's where clause has ") +
                  Twine(where.size()) + " entries, and the evidence discharges " +
                  Twine(citation.getDischarges().size()));
  for (auto [entry, discharge] : llvm::zip(where, citation.getDischarges()))
    if (failed(verifyWitnessBody(scope, discharge, entry.getPredicate())))
      return failure();
  return Attribute(cited.getSelfApplicationAt(*arguments));
}

/// Whether `body` proves `predicate` in `scope`: reflexivity when the
/// equality's two sides are one type read through the impl's own bindings;
/// any other body when the predicate it proves (`readWitnessBody`) is
/// `predicate` once both are read through those bindings.
static LogicalResult verifyWitnessBody(BoundEvidenceScope &scope,
                                       Attribute body, Attribute predicate) {
  auto refuse = [&](const Twine &why) {
    return scope.errFn() << "evidence does not prove " << predicate << ": "
                         << why;
  };

  if (isa<UnitAttr>(body)) {
    auto equality = dyn_cast<TypeEqualityAttr>(predicate);
    if (!equality)
      return refuse(Twine("reflexivity proves only an equality"));
    FailureOr<Type> lhs = scope.own.normalize(equality.getLhs(), scope.errFn);
    FailureOr<Type> rhs = scope.own.normalize(equality.getRhs(), scope.errFn);
    if (failed(lhs) || failed(rhs))
      return failure();
    if (*lhs != *rhs)
      return refuse(Twine("its two sides are two types"));
    return success();
  }
  if (isa<ImplCitationAttr>(body) && !isa<TraitApplicationAttr>(predicate))
    return refuse(Twine("an impl proves only a trait application"));

  FailureOr<Attribute> proved = readWitnessBody(scope, body, refuse);
  if (failed(proved))
    return failure();
  FailureOr<bool> same = scope.same(*proved, predicate);
  if (failed(same))
    return failure();
  if (!*same)
    return refuse(Twine("it states another predicate"));
  return success();
}

/// Verifies the witnesses this impl states for the bound requirements of its
/// trait: exactly one per such requirement, whose body proves the
/// requirement's conclusion at the impl's arguments, under its binder.
static LogicalResult verifyBoundRequirementEvidence(
    ImplOp impl, TraitOp traitOp, ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  PredicateArrayAttr requirements = traitOp.getRequirements();
  DenseMap<unsigned, WitnessAttr> byRequirement;
  if (ArrayAttr witnesses = impl.getWitnessesAttr()) {
    for (auto witness : witnesses.getAsRange<WitnessAttr>()) {
      std::optional<unsigned> position = witness.getRequirement();
      if (!position)
        continue;
      if (*position >= requirements.size() ||
          !isa<BoundPredicateAttr>(requirements.getPredicates()[*position]))
        return impl.emitOpError()
               << "states a witness for requirement " << *position
               << ", which is not a bound requirement of trait '@"
               << traitOp.getSymName() << "'";
      if (!byRequirement.try_emplace(*position, witness).second)
        return impl.emitOpError() << "states a witness for requirement "
                                  << *position << " twice";
    }
  }
  if (!requirements.hasBoundPredicates())
    return success();

  auto module = impl.getModule(errFn);
  if (failed(module))
    return failure();
  auto traitArguments = traitOp.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(traitArguments))
    return failure();
  auto own = buildImplOwnNormalizationContext(impl, witnessRules, errFn);
  if (failed(own))
    return failure();

  for (auto [position, requirement] : llvm::enumerate(requirements)) {
    auto bound = dyn_cast<BoundPredicateAttr>(requirement);
    if (!bound)
      continue;
    auto witness = byRequirement.find(position);
    if (witness == byRequirement.end())
      return impl.emitOpError()
             << "states no witness for bound requirement " << position
             << " of trait '@" << traitOp.getSymName() << "'";

    // The requirement at this impl's arguments: its binder's variables stay
    // free, since the binder is still quantified here, and no parameter of
    // this impl is one of them.
    SmallVector<Attribute> premises =
        llvm::map_to_vector(bound.getPremises(), [&](Attribute premise) {
          return instantiatePredicate(premise, *traitArguments).getPredicate();
        });
    BoundEvidenceScope scope{impl, premises, *own, *module, errFn};
    if (failed(verifyWitnessBody(
            scope, witness->second.getBody(),
            instantiatePredicate(bound.getConclusion(), *traitArguments)
                .getPredicate())))
      return failure();
  }
  return success();
}

/// Verifies the equality requirements the trait header states, specialized for
/// this impl's self arguments (e.g. Self::Output = Self).
///
/// A requirement is an obligation the impl owes, so the two endpoints must be
/// the same type: both are read through the impl's own bindings and its
/// declared witness rules, and whatever stays standing after that is equal to
/// itself alone. The impl's OWN where-clause equalities are not checked here --
/// they are premises restricting when the impl applies, read at every citation
/// that carries the impl to an application -- and application requirements are
/// proved at selection too.
static LogicalResult verifyEqualityObligations(
    ImplOp impl, TraitOp traitOp, ArrayRef<LocalProjectionRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // The guard keeps the self-claim specialization off an impl with nothing of
  // the kind to check.
  if (!traitOp.getRequirements().hasEqualities())
    return success();

  auto specReqs =
      traitOp.specializeRequirementsAsClaimsFor(impl.getSelfClaim(), errFn);
  if (failed(specReqs)) return failure();

  auto eqNorm = buildImplOwnNormalizationContext(impl, witnessRules, errFn);
  if (failed(eqNorm)) return failure();

  for (ClaimType req : *specReqs) {
    auto eq = req.getEqualityAttr();
    if (!eq) continue;
    auto lhsN = eqNorm->normalize(eq.getLhs(), errFn);
    if (failed(lhsN)) return failure();
    auto rhsN = eqNorm->normalize(eq.getRhs(), errFn);
    if (failed(rhsN)) return failure();
    if (*lhsN != *rhsN)
      return impl.emitOpError()
             << "does not satisfy trait-header equality requirement " << req
             << ": " << *lhsN << " and " << *rhsN << " are not the same type";
  }
  return success();
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
  for (TypeEqualityAttr equality : impl.getAssumptions().getEqualities()) {
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

bool ImplOp::isUnconditional() {
  // An impl is unconditional when it stands for nothing a subproof would have
  // to carry: it binds no type parameter, assumes no application, and its trait
  // requires none. A citation may then name it directly, because the given list
  // a proof would hold is empty.
  //
  // An equality predicate is not counted either way. A trait-HEADER equality is
  // an obligation this impl discharges at its own verification
  // (verifyEqualityObligations), the same for every application the impl
  // covers. This impl's OWN where-clause equality restricts where the impl
  // applies, and every citation naming it reads that premise at the application
  // it names -- a witness, a proof's or a call's citation, a derive, impl
  // selection -- so naming the impl directly leaves no premise unread.
  return getTypeParams().empty() &&
         !getAssumptions().hasApplications() &&
         !getTrait().getRequirements().hasApplications();
}

LogicalResult ImplOp::verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isUnconditional()) {
    if (err) err() << "impl '@" << getSymName()
                   << "' binds type parameters, assumes an application, or implements a trait requiring an application, so it must be cited through a trait.proof";
    return failure();
  }
  return success();
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

FailureOr<SpecializationMap> ImplOp::buildImplSpecialization(
    ClaimType provenSelfClaim,
    DemandOrigin origin,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!provenSelfClaim.isProven()) {
    if (err) err() << "expected proven self claim for " << getSymName();
    return failure();
  }

  auto module = getModule(err);
  if (failed(module)) return failure();

  // The self claim names the proof standing over this impl's obligations, so a
  // projection the header spells over one of them reduces through the impl that
  // obligation's subproof names -- the reading by index. Where the header spells
  // an application no subproof answers, the impls the module holds stand in.
  NormalizationContext throughProof;
  if (spellsAProjection(Type(getSelfClaim())))
    throughProof = buildProofNormalizationContext(provenSelfClaim, *module);
  throughProof.setModuleLookup(*module, LookupScope::Ground, origin);
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return throughProof.normalize(ty, err);
  };
  return buildSubstitutionForSelfClaim(provenSelfClaim, normalize, err);
}

SmallVector<GenericTypeInterface, 4> ImplOp::getTypeParams() {
  // collect all the types where a type variable could hide
  SmallVector<Type> allOurTypes;
  allOurTypes.push_back(getSelfClaim());
  for (ClaimType a : getAssumptionsAsClaims()) {
    allOurTypes.push_back(a);
  }
  // An assumed equality's endpoints are pushed directly, so a generic that
  // appears only there (e.g. the accumulator in `F::Output = Acc`) is one of
  // this impl's parameters and takes its position from where it is pushed.
  for (Attribute pred : getAssumptions()) {
    if (auto eq = dyn_cast<TypeEqualityAttr>(pred)) {
      allOurTypes.push_back(eq.getLhs());
      allOurTypes.push_back(eq.getRhs());
    }
  }

  // tuple the types
  TupleType tupled = TupleType::get(getContext(), allOurTypes);

  // The parameters those spellings bind, in first-occurrence order: a kind-
  // constraining wrapper is an occurrence of the parameter it wraps, not a
  // parameter of its own.
  return getTypeParametersIn(tupled);
}

FailureOr<SpecializationMap> ImplOp::substitutionFor(
    ArrayRef<TypeBindingAttr> arguments,
    llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<GenericTypeInterface, 4> params = getTypeParams();
  SpecializationMap substitution;
  for (TypeBindingAttr binding : arguments) {
    auto parameter = dyn_cast<GenericTypeInterface>(binding.getParameter());
    if (!parameter || !llvm::is_contained(params, parameter)) {
      if (err) err() << "the citation binds " << binding.getParameter()
                     << ", which is not a type parameter of impl '@"
                     << getSymName() << "'";
      return failure();
    }
    if (substitution.lookup(parameter)) {
      if (err) err() << "the citation binds type parameter " << Type(parameter)
                     << " of impl '@" << getSymName() << "' twice";
      return failure();
    }
    substitution.bind(parameter, binding.getArgument());
  }
  for (GenericTypeInterface parameter : params)
    if (!substitution.lookup(parameter)) {
      if (err) err() << "the citation binds no argument for type parameter "
                     << Type(parameter) << " of impl '@" << getSymName() << "'";
      return failure();
    }
  return substitution;
}

FailureOr<FunctionOpInterface> ImplOp::getOrSpecializeMethod(RewriterBase& rewriter, StringRef methodName) {
  auto trait = getTrait();

  // check that we've named a valid trait method
  if (!trait.hasMethod(methodName)) return failure();

  // check if the method already exists in the ImplOp
  auto method = getMethod(methodName);
  if (succeeded(method)) return method;

  // otherwise, we need to specialize the method from the default implementation in the trait
  auto traitMethod = trait.getOptionalMethod(methodName);
  if (failed(traitMethod)) return failure();

  // build a substitution that maps trait PolyType parameters to impl type arguments
  auto subst = trait.buildSubstitutionForSelfClaim(getSelfClaim());
  if (failed(subst)) return failure();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&getBody().front());
  auto specialized =
      specializePolymorph(rewriter, *traitMethod, methodName, subst->toTypeMap());
  // A default method with no body to clone is refused where the clone was
  // attempted; there is no method here to answer with.
  if (!specialized)
    return failure();

  // A positional assume in the trait's method cites the trait's where clause,
  // and in this impl a where-clause position names the impl's own entries. The
  // trait's requirement at that position is the one this impl's self claim
  // carries there, so the clone selects it off that claim: `self` still names
  // the declaration's own application, which in this impl is its own.
  SmallVector<AssumeOp> requirementAssumes;
  specialized->walk([&](AssumeOp assume) {
    if (assume.getWherePosition())
      requirementAssumes.push_back(assume);
  });
  for (AssumeOp assume : requirementAssumes) {
    OpBuilder::InsertionGuard assumeGuard(rewriter);
    rewriter.setInsertionPoint(assume);
    Value self = AssumeOp::create(rewriter, assume.getLoc(), getSelfClaim(),
                                  rewriter.getUnitAttr());
    Value requirement =
        ProjectOp::create(rewriter, assume.getLoc(), assume.getClaim(), self,
                          *assume.getWherePosition());
    assume.getResult().replaceAllUsesWith(requirement);
    assume.erase();
  }
  return specialized;
}

static func::FuncOp specializeMethodAsFreeFuncWithLeadingSelfProof(
    PatternRewriter& rewriter,
    ModuleOp module,
    FunctionOpInterface method,
    StringRef functionName,
    ClaimType selfProofTy,
    const DenseMap<Type,Type>& subst) {

  // specialize the method into the grandparent with a mangled name
  PatternRewriter::InsertionGuard guard(rewriter);

  // clone the method into the method's grandparent
  rewriter.setInsertionPointAfter(method->getParentOp());

  // An external declaration has no body to clone; specialization has refused
  // it. Cut at module scope, the instance is a `func.func`.
  auto funcOp = cast_if_present<func::FuncOp>(
      specializePolymorph(rewriter, method, functionName, subst).getOperation());
  if (!funcOp)
    return nullptr;

  // The clone leads with the proven self claim the call carries: the self is
  // ground and the impl's proof names it, in a template clone (a method with its
  // own free generic) as in a monomorphic one. Its citations of the impl's
  // where clause are read off that self proof by position below, and its
  // assumed equalities project to the impl's equality where-clauses, so no
  // assumption rides as a lifted claim parameter.
  rewriter.modifyOpInPlace(funcOp, [&] {
    (void)funcOp.insertArgument(/*idx=*/0, selfProofTy,
                               /*argAttrs=*/mlir::DictionaryAttr(),
                               method->getLoc());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  });
  BlockArgument selfProofArg = funcOp.getArgument(0);

  // Every citation the method makes of its declaration -- `self`, or entry N
  // of the impl's where clause, in whichever region or block of the body it
  // stands -- is replaced by the evidence the leading self proof supplies at
  // that position, read off the proof by index and never found by the entry's
  // spelling: two entries spelling one claim can be discharged by different
  // proofs, and only the position says which. The self proof's requirements are
  // its trait's, then the impl's where clause, so entry N is requirement
  // traitRequirementCount + N. A proven application there is the witness of the
  // subproof the self proof names at that index, spelled with its ground
  // projections resolved as the rest of the instance is stamped; an equality
  // carries no proof and is projected from the self proof. A citation the proof
  // cannot read is left standing, and the AssumeOp verifier refuses it once
  // this instance stands at module scope, where no declaration encloses it.
  uint64_t traitRequirementCount = 0;
  if (auto trait = selfProofTy.getTraitApplication().getTrait(module);
      succeeded(trait))
    traitRequirementCount = trait->getRequirements().size();

  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(a);

    Value replacement;
    if (a.citesSelf()) {
      replacement = selfProofArg;
    } else {
      uint64_t index = traitRequirementCount + *a.getWherePosition();
      auto requirement = getClaimRequirementAt(selfProofTy, module, index);
      if (failed(requirement))
        return;
      if (requirement->isProven()) {
        auto spelled = cast<ClaimType>(resolveProjectionsByLookup(
            *requirement, module, DemandOrigin::MonomorphStampOut,
            LookupScope::Ground));
        replacement = WitnessOp::create(rewriter, a.getLoc(),
                                        spelled.getProof(),
                                        spelled.getTraitApplication());
      } else {
        replacement = ProjectOp::create(rewriter, a.getLoc(), *requirement,
                                        selfProofArg, index);
      }
    }

    rewriter.replaceAllUsesWith(a.getResult(), replacement);
    toErase.push_back(a);
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

FailureOr<func::FuncOp> ImplOp::getOrSpecializeFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType provenSelfClaim,
    StringRef methodName,
    TypeRange actualArguments,
    const CallSubstitution &callSubst) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return failure();

  // The enclosing module: where a clone of the method is cut and where an
  // existing clone is looked up.
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();

  auto method = getOrSpecializeMethod(rewriter, methodName);
  if (failed(method)) return failure();

  auto implArguments =
      buildImplSpecialization(provenSelfClaim, DemandOrigin::ProofRecording);
  if (failed(implArguments)) return failure();

  // The substitution the method body is cut under: the arguments the impl's
  // parameters take at the receiver, then the method-generic bindings and the
  // evidence of this call. The call reads the receiver's proof beside every
  // claim argument's, so a claim two of them discharge by different proofs is
  // held for both and bound for neither: no claim's spelling picks one source's
  // proof over another's. Parameters and citations take their evidence by
  // position; a value no position decides is refused where the cut finds it.
  DenseMap<Type,Type> subst = implArguments->toTypeMap();

  // A call names its method-generic bindings under the trait method's own type
  // variables, while the method cloned below is the impl's copy, which carries
  // its own. Rekey each binding through the correspondence between the two, so
  // the clone is monomorphic in the method's variables as well as the impl's and
  // no partly substituted template stands between the call and its instance.
  auto errFn = [&] { return emitOpError(); };
  auto witnessRules = collectImplWitnessRules(*this, module, errFn);
  if (failed(witnessRules)) return failure();
  auto correspondence = buildTraitMethodCorrespondence(
      *this, getTrait(), *method, *witnessRules, errFn);
  if (failed(correspondence)) return failure();

  DenseMap<Type,Type> callBindings = callSubst.toTypeMap();
  for (auto [traitVariable, implVariable] :
       llvm::zip(correspondence->traitOwn, correspondence->implOwn)) {
    auto binding = callBindings.find(Type(traitVariable));
    if (binding != callBindings.end())
      subst.try_emplace(Type(implVariable), binding->second);
  }

  for (const auto &[k, v] : callBindings)
    subst.try_emplace(k, v);

  // The instance is the one the impl's method names at these type arguments
  // and this evidence: the receiver's proof at the leading position, then
  // whatever the call supplies for each of the method's own parameters. The
  // type arguments are the impl's, then every parameter the method's signature
  // spells, so different method-generic calls name different instances too.
  SmallVector<Type> typeArguments;
  for (GenericTypeInterface parameter : getTypeParams())
    typeArguments.push_back(implArguments->apply(parameter));
  for (GenericTypeInterface parameter :
       getTypeParametersIn((*method).getFunctionType()))
    typeArguments.push_back(applySubstitutionOnce(subst, parameter));
  SmallVector<Type> formalInputs{getSelfClaim()};
  llvm::append_range(formalInputs, (*method).getArgumentTypes());
  SmallVector<Type> actualInputs{provenSelfClaim};
  llvm::append_range(actualInputs, actualArguments);
  auto templateRef = SymbolRefAttr::get(
      getSymNameAttr(), {FlatSymbolRefAttr::get(getContext(), methodName)});
  AttrTypeReplacer stamp = makeTypeReplacerFromSubstitution(subst, module);
  auto key = InstanceKey::get(templateRef, typeArguments, formalInputs,
                              actualInputs, stamp);
  if (failed(key))
    return emitOpError() << "is supplied a claim that names no proof for '@"
                         << methodName << "', which identifies no instance";

  // The leading self proof is read as the instance spells it, so its
  // requirements are read at the arguments its proof states them for.
  auto selfProof = cast<ClaimType>(key->getEvidence().front());
  func::FuncOp instance = getOrCutInstance(
      rewriter, module, *key, [&](StringRef instanceName) {
        // A method with no body to clone is refused where the clone was
        // attempted; this call has no instance to name.
        return specializeMethodAsFreeFuncWithLeadingSelfProof(
            rewriter, module, *method, instanceName, selfProof, subst);
      },
      callSubst.getEvidence());
  if (!instance)
    return failure();
  return instance;
}

/// Generate a deterministic symbol name for an ImplOp.
/// 
/// The name has the form {TraitName}_impl_h{hash} where the hash is a
/// 64-bit xxHash of the full type argument and assumption signature. This
/// keeps symbols short and bounded in length.
std::string ImplOp::generateSymName(TraitApplicationAttr selfApp,
                                    PredicateArrayAttr assumptions) {
  // Build the full type-argument and where-clause signature for hashing. The
  // equality entries follow the application entries, so two impls that differ
  // only in an equality assumption synthesize distinct names.
  std::string signature;
  llvm::raw_string_ostream os(signature);
  for (auto ty : selfApp.getTypeArgs()) {
    os << "_" << ty;
  }
  SmallVector<TraitApplicationAttr> apps =
      assumptions ? assumptions.getApplications()
                  : SmallVector<TraitApplicationAttr>{};
  if (!apps.empty()) {
    os << "_where";
    for (auto app : apps) {
      os << "_" << app.getTraitName().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  os << "_eq";
  if (assumptions) {
    for (Attribute pred : assumptions) {
      auto eq = dyn_cast<TypeEqualityAttr>(pred);
      if (!eq) continue;
      os << "_" << eq.getLhs() << "_" << eq.getRhs();
    }
  }
  os.flush();

  return selfApp.getTraitName().getValue().str() + "_impl" + hashToSuffix(signature);
}

std::string ImplOp::generateMangledName(const SpecializationMap &arguments) {
  return getSymName().str() +
         applySubstitutionAndGenerateMangledNameSuffix(arguments,
                                                       getTypeParams());
}

SmallVector<ClaimType> ImplOp::getAssumptionsAsClaims() {
  MLIRContext *ctx = getContext();
  // The proof/derive/satisfiability streams read application-arm assumptions
  // only; an equality entry takes no subproof and is read at the application
  // the citation names (verifyEqualityPremisesHoldAt), so equality entries are
  // filtered out here at the one place every obligation consumer flows through.
  return llvm::map_to_vector(getAssumptions().getApplications(),
                             [ctx](TraitApplicationAttr app) {
    return ClaimType::get(ctx, app);
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeObligationsAt(
    ClaimType actualSelfClaim, const SpecializationMap &arguments,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto requirements =
      getTrait().specializeRequirementsAsClaimsFor(actualSelfClaim, errFn);
  if (failed(requirements))
    return failure();

  // The obligation stream is proved and derived through impl selection, an
  // application-arm operation. Trait-header equality requirements are checked
  // at impl verification against the impl's own bindings, never proved here, so
  // they do not enter the obligation stream (the proof/derive zips would have
  // no subproof for them).
  llvm::erase_if(*requirements, [](ClaimType c) { return c.isEquality(); });

  // Resolve projections in requirements using this impl's associated type
  // bindings (e.g., `Coord[Tensor[Self]::Shape]` becomes `Coord[tuple<i64,i64>]`
  // when the impl binds `Shape = S` and S is specialized to tuple<i64,i64>).
  // Only projections over this impl's own (actual) trait application resolve
  // through its bindings; a projection over a different trait application that
  // merely shares an associated-type name stays symbolic.
  NormalizationContext normalization;
  normalization.addLocalProjectionRule(
      *this, actualSelfClaim.getTraitApplication(), arguments);
  SmallVector<ClaimType> obligations;
  for (ClaimType requirement : *requirements) {
    auto resolved = normalization.normalize(requirement, errFn);
    if (failed(resolved))
      return failure();
    obligations.push_back(cast<ClaimType>(*resolved));
  }
  // obligations = requirements + the impl's application premises
  for (ClaimType assumption : getAssumptionsAsClaims())
    obligations.push_back(cast<ClaimType>(instantiate(Type(assumption), arguments)));
  return obligations;
}

/// Whether every outermost projection `side` still spells is over one of
/// `impl`'s where-clause applications at `arguments`, or a trait requirement
/// one of them carries, each read through `evidence`: a projection whose
/// evidence is the premise a citation of the impl supplies for that
/// application, or the requirement that premise stands over.
static bool projectionsStandOnPremises(Type side, ImplOp impl,
                                       const SpecializationMap &arguments,
                                       NormalizationContext &evidence) {
  ModuleOp module = impl->getParentOfType<ModuleOp>();
  SmallVector<TraitApplicationAttr> premises;
  SmallVector<std::pair<ClaimType, unsigned>> pending;
  for (TraitApplicationAttr app : impl.getAssumptions().getApplications())
    pending.push_back(
        {cast<ClaimType>(instantiate(Type(ClaimType::get(impl.getContext(), app)),
                                     arguments)),
         0});
  while (!pending.empty()) {
    auto [claim, depth] = pending.pop_back_val();
    auto read = evidence.normalize(Type(claim), /*err=*/nullptr);
    ClaimType premise = succeeded(read) ? cast<ClaimType>(*read) : claim;
    if (llvm::is_contained(premises, premise.getTraitApplication()))
      continue;
    premises.push_back(premise.getTraitApplication());
    if (depth == kInstantiationDepthLimit)
      continue;
    auto trait = premise.getTraitApplication().getTrait(module, /*err=*/nullptr);
    if (failed(trait))
      continue;
    auto requirements =
        trait->specializeRequirementsAsClaimsFor(premise, /*errFn=*/nullptr);
    if (succeeded(requirements))
      for (ClaimType requirement : *requirements)
        if (requirement.isApplication())
          pending.push_back({requirement, depth + 1});
  }
  bool standing = true;
  AttrTypeWalker walker;
  walker.addWalk([&](ProjectionType projection) {
    if (!llvm::is_contained(premises, projection.getTraitApplication()))
      standing = false;
    return WalkResult::skip();
  });
  walker.walk<WalkOrder::PreOrder>(side);
  return standing;
}

LogicalResult mlir::trait::verifyEqualityPremisesHoldAt(
    ImplOp impl, ClaimType cited, const SpecializationMap &arguments,
    NormalizationContext evidence, OpenPremise openPremise,
    StandingPremise standingPremise,
    llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<TypeEqualityAttr> equalities =
      impl.getAssumptions().getEqualities();
  if (equalities.empty())
    return success();

  evidence.addLocalProjectionRule(impl, cited.getTraitApplication(), arguments);
  for (TypeEqualityAttr equality : equalities) {
    auto reduce = [&](Type side) -> FailureOr<Type> {
      return evidence.normalize(instantiate(side, arguments), err);
    };
    FailureOr<Type> lhs = reduce(equality.getLhs());
    if (failed(lhs))
      return failure();
    FailureOr<Type> rhs = reduce(equality.getRhs());
    if (failed(rhs))
      return failure();
    if (premiseDefersToInstances(*lhs, *rhs)) {
      if (openPremise == OpenPremise::DecidedAtInstances)
        continue;
      if (err) err() << "a proof states its impl's premises at its own claim; "
                        "one the claim leaves open is stated at the instance "
                        "instead: "
                     << equality.getLhs() << " = " << equality.getRhs()
                     << " reads " << *lhs << " = " << *rhs << " at " << cited;
      return failure();
    }
    // Sides read as one type hold, whatever they spell.
    if (*lhs == *rhs)
      continue;
    // A side still spelling a projection after the reading is one this citation
    // cannot decide. The impls the reading saw bind that projection for nobody
    // or for two candidates at once; what it denotes is decided by the impl
    // selection chose for its application, which a reader holding no record may
    // not consult. So the premise is neither true nor false here, and
    // `standingPremise` says where it is decided.
    if (spellsAProjection(*lhs) || spellsAProjection(*rhs)) {
      if (standingPremise == StandingPremise::DecidedAtStageExit ||
          (projectionsStandOnPremises(*lhs, impl, arguments, evidence) &&
           projectionsStandOnPremises(*rhs, impl, arguments, evidence)))
        continue;
      if (err) err() << "impl '@" << impl.getSymName() << "' applies where "
                     << equality.getLhs() << " = " << equality.getRhs()
                     << ", and nothing here settles " << *lhs << " = " << *rhs
                     << " at " << cited;
      return failure();
    }
    if (*lhs != *rhs) {
      if (err) err() << "impl '@" << impl.getSymName() << "' applies where "
                     << equality.getLhs() << " = " << equality.getRhs()
                     << ", and nothing here makes " << *lhs << " and " << *rhs
                     << " one type at " << cited;
      return failure();
    }
  }
  return success();
}

/// Reads `impl`'s equality premises at `cited`, through `evidence` and then the
/// impls `module` holds under `origin`.
///
/// The arguments `cited` supplies for the impl's parameters are read through the
/// same context the premises are, so a parameter the header leaves open and the
/// where clause determines is read once, the way it is read.
static LogicalResult verifyEqualityPremisesOfImplAt(
    ImplOp impl, ClaimType cited, NormalizationContext evidence,
    ModuleOp module, DemandOrigin origin, OpenPremise openPremise,
    llvm::function_ref<InFlightDiagnostic()> err) {
  evidence.setModuleLookup(module, LookupScope::Ground, origin);
  auto throughEvidence = [&](Type ty) -> FailureOr<Type> {
    return evidence.normalize(ty, err);
  };
  auto arguments = impl.buildSubstitutionForSelfClaim(cited, throughEvidence, err);
  if (failed(arguments))
    return failure();

  return verifyEqualityPremisesHoldAt(impl, cited, *arguments, evidence,
                                      openPremise,
                                      StandingPremise::DecidedAtStageExit, err);
}

LogicalResult ImplOp::verifyEqualityPremisesAt(
    ClaimType cited, DemandOrigin origin,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!getAssumptions().hasEqualities())
    return success();

  auto module = getModule(err);
  if (failed(module))
    return failure();

  // A citation naming an impl carries no subproofs, so the impls the module
  // holds are the whole of what a premise endpoint reads through.
  return verifyEqualityPremisesOfImplAt(*this, cited, NormalizationContext(),
                                        *module, origin,
                                        OpenPremise::DecidedAtInstances, err);
}

//===----------------------------------------------------------------------===//
// ProofOp
//===----------------------------------------------------------------------===//

LogicalResult ProofOp::verifyEqualityPremisesAt(
    ClaimType cited, DemandOrigin origin, OpenPremise openPremise,
    llvm::function_ref<InFlightDiagnostic()> err) {
  ImplOp implOp = getImpl();
  if (!implOp) {
    if (err) err() << "cannot find impl '" << getImplNameAttr() << "'";
    return failure();
  }
  if (!implOp.getAssumptions().hasEqualities())
    return success();

  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module) {
    if (err) err() << "not inside a module";
    return failure();
  }

  return verifyEqualityPremisesOfImplAt(
      implOp, cited, buildSubproofNormalizationContext(*this, cited, module),
      module, origin, openPremise, err);
}

LogicalResult ProofOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();

  // Every entry names a symbol, or is `unit` where a requirement or premise is
  // decided without one; which entries those are is read against the impl
  // where its symbols are verified.
  for (Attribute name : getSubproofNames())
    if (!isa<FlatSymbolRefAttr, UnitAttr>(name))
      return emitOpError() << "'subproof_names' must contain only symbols and unit";
  return success();
}

FailureOr<SpecializationMap> ProofOp::getImplArgumentsAt(
    ClaimType at, llvm::function_ref<InFlightDiagnostic()> err) {
  ImplOp impl = getImpl();
  if (!impl) {
    if (err) err() << "cannot find impl '" << getImplNameAttr() << "'";
    return failure();
  }
  SmallVector<TypeBindingAttr> bindings;
  for (Attribute binding : getArgumentsAttr())
    bindings.push_back(cast<TypeBindingAttr>(binding));
  auto stated = impl.substitutionFor(bindings, err);
  if (failed(stated))
    return failure();

  // The stated arguments spell this proof's own variables; the claim it is
  // carried to is an instance of its own claim, which supplies them.
  Type own = Type(getProvenClaim().asUnproven());
  auto instance = matchDeclaration(getTypeParametersIn(own), own,
                                   Type(at.asUnproven()), Normalizer(), err);
  if (failed(instance))
    return failure();
  SpecializationMap atInstance;
  for (GenericTypeInterface parameter : impl.getTypeParams())
    atInstance.bind(parameter, instance->apply(*stated->lookup(parameter)));
  return atInstance;
}

/// Re-verifies `impl`'s projection-resolution witnesses at `arguments`, the
/// substitution a proof's claim makes for the impl's parameters. The impl
/// verified each witness at its own parameters, where a premise of the cited
/// impl that still spelled a type variable was left to the instances; the claim
/// a proof stands over is such an instance, and each witness, rebuilt there,
/// must hold -- its obligations covered by the impl's where clause at that
/// claim, which the proof's subproofs discharge, or by its discharge citations.
///
/// XXX TODO: deleted when a citation carries the evidence for its cited impl's
/// where-equalities by index (an application's premises admitting equalities,
/// the evidence-terms plan's C7), so the impl's own verification decides every
/// premise locally.
static LogicalResult verifyDeclaredWitnessesAt(
    ImplOp impl, const SpecializationMap &arguments, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  ArrayAttr declared = impl.getWitnessesAttr();
  if (!declared)
    return success();
  MLIRContext *ctx = impl.getContext();
  auto atClaim = [&](Type t) { return instantiate(t, arguments); };
  SmallVector<TraitApplicationAttr> obligationPremises;
  for (TraitApplicationAttr app : impl.getAssumptions().getApplications())
    obligationPremises.push_back(
        cast<ClaimType>(atClaim(Type(ClaimType::get(ctx, app))))
            .getTraitApplication());
  SmallVector<WitnessAttr> dischargeWitnesses;
  for (auto witness : declared.getAsRange<WitnessAttr>())
    if (isa<TraitApplicationAttr>(witness.getPredicate()))
      dischargeWitnesses.push_back(WitnessAttr::get(
          ctx,
          Attribute(cast<ClaimType>(atClaim(Type(ClaimType::get(
                                        ctx, witness.getApplication()))))
                        .getTraitApplication()),
          witness.getImplRef(), {}));
  for (auto witness : declared.getAsRange<WitnessAttr>()) {
    if (!isa<TypeEqualityAttr>(witness.getPredicate()))
      continue;
    auto rebuilt = respellWitness(witness, atClaim);
    if (!rebuilt) {
      if (err) err() << "the declaration witness of "
                     << witness.getProjection() << " = "
                     << witness.getResolved()
                     << " does not construct at this proof's claim";
      return failure();
    }
    if (failed(verifyProjectionResolutionAtImpl(
            module, cast<WitnessAttr>(rebuilt->first), /*premises=*/{},
            obligationPremises, dischargeWitnesses, err)))
      return failure();
  }
  return success();
}

LogicalResult ProofOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto module = (*this)->getParentOfType<ModuleOp>();
  auto errFn = [&] { return emitOpError(); };

  // The proven claim is synthesized from the inherent trait_application
  // attribute, so it is not a type on this op's surface and the module-wide
  // type walk never verifies it. Verify the trait application here.
  if (failed(getTraitApplication().verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // check that the named impl exists
  auto implOp = getImpl();
  if (!implOp)
    return emitOpError() << "cannot find impl '" << getImplNameAttr() << "'";

  // The evidence this proof holds: the proofs discharging the impl's own
  // obligations, by index, and the trees standing under them. The proof's own
  // rule is not among them -- nothing here is justified by what it is checking.
  // It is gathered on the first reading that meets a projection: a spelling
  // that has none is rebuilt by substitution alone.
  std::optional<NormalizationContext> subproofEvidence;
  auto evidence = [&]() -> NormalizationContext & {
    if (!subproofEvidence)
      subproofEvidence =
          buildSubproofNormalizationContext(*this, getProvenClaim(), module);
    return *subproofEvidence;
  };

  // The impl's header at the arguments this proof states must be the claim it
  // stands over. A header spelling a projection (`impl<T> Index<T::Shape,
  // T::Element> for T`) is read at them through that evidence and then the
  // impls the module holds, which is what carries it to a claim spelling the
  // resolution.
  auto arguments = getImplArgumentsAt(getProvenClaim(), errFn);
  if (failed(arguments))
    return failure();
  TraitApplicationAttr header = implOp.getSelfApplicationAt(*arguments);
  bool carries = header == getTraitApplication();
  if (spellsAProjection(Type(implOp.getSelfClaim()))) {
    NormalizationContext reading = evidence();
    reading.setModuleLookup(module, LookupScope::Ground,
                            DemandOrigin::ProofVerification);
    // A reading with no normal form is refused where it is read.
    auto read = [&](Type ty) {
      return reading.normalize(stripClaimProofs(ty), errFn);
    };
    FailureOr<Type> rebuilt = read(instantiate(Type(implOp.getSelfClaim()), *arguments));
    FailureOr<Type> wanted = read(Type(getProvenClaim()));
    if (failed(rebuilt) || failed(wanted))
      return failure();
    carries = *rebuilt == *wanted;
  }
  if (!carries)
    return emitOpError() << "impl '" << getImplNameAttr()
                         << "' at its stated arguments is an impl of " << header
                         << ", not of " << getTraitApplication();

  // The impl's equality premises stand over this claim, and this claim is where
  // they are decided: a citation of this proof reads nothing inside it, so a
  // premise this claim leaves open is one no later reading decides.
  if (failed(verifyEqualityPremisesAt(getProvenClaim(),
                                      DemandOrigin::ProofVerification,
                                      OpenPremise::RefusedHere, errFn)))
    return failure();

  if (failed(verifyDeclaredWitnessesAt(implOp, *arguments, module, errFn)))
    return failure();

  // One entry in the given list per obligation the impl states at this claim,
  // each naming evidence that discharges the obligation at its index. What that
  // evidence proves underneath is the business of its own verifier: a citation
  // is read at the top level and no deeper.
  auto subproofs = verifyAndGetSubproofClaims(getProvenClaim(), errFn);
  if (failed(subproofs))
    return failure();

  // A citation is read through the evidence above and not through the impls
  // standing around this proof. A projection an obligation spells over one of
  // the impl's where-clause or trait-requirement applications reduces through
  // the subproof at that application's index. One over an application no
  // subproof discharges -- one an impl proves, not this proof -- is left
  // standing, and the citation is declined for the stage to decide through what
  // selection settles.
  auto throughSubproofs = [&](Type ty) -> FailureOr<Type> {
    if (!spellsAProjection(ty))
      return ty;
    return evidence().normalize(ty, /*err=*/nullptr);
  };
  for (ClaimType subproof : *subproofs)
    // A citation nothing standing now decides leaves its obligation unproven,
    // which impl selection derives and the leftover walk refuses.
    if (verifyCitation(subproof.asUnproven(), subproof, module,
                       DemandOrigin::ProofVerification, throughSubproofs,
                       errFn) == Citation::Refused)
      return failure();

  return success();
}

TraitOp ProofOp::getTrait() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("ProofOp::getTrait: not inside a module");
  return getTraitApplication().getTraitOrAbort(module, "ProofOp::getTrait: couldn't find trait");
}

FailureOr<SmallVector<ClaimType>> ProofOp::verifyAndGetSubproofClaims(
    ClaimType at, llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<ClaimType> result;

  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  if (!module) {
    if (err) err() << "not in a module";
    return failure();
  }

  auto implOp = getImpl();
  if (!implOp) {
    if (err) err() << "cannot find impl '" << getImplNameAttr() << "'";
    return failure();
  }

  // The obligations at the application this proof is being carried to, at the
  // arguments it states carried there.
  auto arguments = getImplArgumentsAt(at, err);
  if (failed(arguments))
    return failure();
  auto obligations = implOp.specializeObligationsAt(at, *arguments, err);
  if (failed(obligations)) return failure();

  // The given list holds one entry per requirement and where-clause entry and
  // cites a symbol exactly at the application entries, which are the
  // obligations in order.
  SmallVector<Attribute> entries(implOp.getTrait().getRequirements().begin(),
                                 implOp.getTrait().getRequirements().end());
  llvm::append_range(entries, implOp.getAssumptions());
  ArrayAttr given = getSubproofNames();
  if (given.size() != entries.size()) {
    if (err) err() << "arity mismatch: impl '" << getImplNameAttr()
                   << "' and its trait state " << entries.size()
                   << " requirements and where-clause entries, but found "
                   << given.size() << " given entries";
    return failure();
  }
  SmallVector<FlatSymbolRefAttr> citations;
  for (auto [position, pair] : llvm::enumerate(llvm::zip(entries, given))) {
    auto [entry, name] = pair;
    bool application = isa<TraitApplicationAttr>(entry);
    if (application != isa<FlatSymbolRefAttr>(name)) {
      if (err) err() << "given entry " << position << " is " << name
                     << ", and entry " << position << " is "
                     << (application ? "an application a symbol discharges"
                                     : "decided without a symbol, so its "
                                       "given entry is unit");
      return failure();
    }
    if (application)
      citations.push_back(cast<FlatSymbolRefAttr>(name));
  }
  assert(citations.size() == obligations->size() &&
         "the obligations are the application entries in order");

  for (auto [obligation, subproofRef] : llvm::zip(*obligations, citations)) {
    // A coinductive self-citation needs no arm of its own: looking the name up
    // finds this proof, and whether its claim discharges the obligation is the
    // same comparison every other citation answers.
    if (failed(getProofOpOrUnconditionalImplOp(module, subproofRef, err)))
      return failure();

    // A subproof's claim is the obligation it discharges, spelled at `at`,
    // carrying the cited symbol: evidence built from the obligation by
    // position.
    result.push_back(ClaimType::get(getContext(),
                                    obligation.getTraitApplication(),
                                    subproofRef));
  }

  return result;
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

bool ProofOp::sameEvidence(ModuleOp module, FlatSymbolRefAttr a,
                           FlatSymbolRefAttr b) {
  // A proof may cite itself, directly or around a cycle, so two proofs are
  // compared coinductively: a pair already under comparison is assumed to
  // name one evidence, and the answer is no only where some path through the
  // two reaches a difference.
  llvm::DenseSet<std::pair<Attribute, Attribute>> assumed;
  SmallVector<std::pair<FlatSymbolRefAttr, FlatSymbolRefAttr>> pending{{a, b}};
  while (!pending.empty()) {
    auto [x, y] = pending.pop_back_val();
    if (x == y || !assumed.insert({x, y}).second)
      continue;
    auto citedX = getProofOpOrUnconditionalImplOp(module, x);
    auto citedY = getProofOpOrUnconditionalImplOp(module, y);
    if (failed(citedX) || failed(citedY))
      return false;
    auto proofX = dyn_cast<ProofOp>(*citedX);
    auto proofY = dyn_cast<ProofOp>(*citedY);
    if (!proofX || !proofY) {
      if (*citedX != *citedY)
        return false;
      continue;
    }
    if (proofX.getImplNameAttr() != proofY.getImplNameAttr() ||
        proofX.getArguments() != proofY.getArguments())
      return false;
    ArrayRef<Attribute> givenX = proofX.getSubproofNames().getValue();
    ArrayRef<Attribute> givenY = proofY.getSubproofNames().getValue();
    if (givenX.size() != givenY.size())
      return false;
    // A bound requirement's entry is unit in every proof; an application's is
    // the subproof discharging it.
    for (auto [entryX, entryY] : llvm::zip(givenX, givenY)) {
      auto subproofX = dyn_cast<FlatSymbolRefAttr>(entryX);
      auto subproofY = dyn_cast<FlatSymbolRefAttr>(entryY);
      if (!subproofX || !subproofY) {
        if (entryX != entryY)
          return false;
        continue;
      }
      pending.push_back({subproofX, subproofY});
    }
  }
  return true;
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
  // by @impl[!P = T, ...] [given(%premises...) : (types...)] : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("proj_resolve"))) {
    Type projection, resolved;
    FlatSymbolRefAttr citedImpl;
    SmallVector<TypeBindingAttr> arguments;
    if (p.parseType(projection) || p.parseKeyword("resolves") ||
        p.parseType(resolved) || p.parseKeyword("by") ||
        p.parseAttribute(citedImpl) ||
        failed(parseImplArguments(p, arguments)))
      return failure();
    auto err = [&] { return p.emitError(p.getCurrentLocation()); };
    auto equality = TypeEqualityAttr::getChecked(err, ctx, projection, resolved);
    if (!equality)
      return failure();
    auto witness = WitnessAttr::getChecked(err, ctx, Attribute(equality),
                                           citedImpl,
                                           ArrayRef<TypeBindingAttr>(arguments));
    if (!witness)
      return failure();
    result.addAttribute("witness", witness);

    if (succeeded(p.parseOptionalKeyword("given")) && parsePremises())
      return failure();

    return parseResultType();
  }

  // Equality refl arm: `refl : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("refl"))) {
    result.addAttribute("refl", UnitAttr::get(ctx));
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
  if (auto witness = getWitnessAttr()) {
    p << " proj_resolve " << witness.getProjection() << " resolves "
      << witness.getResolved() << " by " << witness.getImplRef();
    if (!witness.getArguments().empty())
      printImplArguments(p, witness.getArguments());
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

  // Composition arm: an equality result with neither a witness nor a refl
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
                                     /*elidedAttrs=*/{"witness", "refl"});
}

// The op's attributes must match the result claim's arm exactly. For the
// application arm the result claim names the proof it cites. For the equality
// arm, the result's equality must be the witness's own (proj-resolve), have
// identical endpoints (refl), or be entailed by the premises' ground congruence
// closure (compose).
LogicalResult WitnessOp::verify() {
  ClaimType result = dyn_cast<ClaimType>(getResult().getType());
  if (!result)
    return emitOpError() << "result must be a !trait.claim";

  bool hasWitness = static_cast<bool>(getWitnessAttr());
  bool hasRefl = getRefl();

  // Equality arm.
  if (result.isEquality()) {
    if (hasWitness && hasRefl)
      return emitOpError() << "an equality witness carries at most one of a "
                              "proj-resolve leaf or a refl marker";
    TypeEqualityAttr eq = result.getEqualityAttr();

    if (hasRefl) {
      if (!getPremises().empty())
        return emitOpError() << "a refl witness takes no premises";
      if (eq.getLhs() != eq.getRhs())
        return emitOpError() << "a refl witness requires identical endpoints, "
                             << "found " << eq.getLhs() << " and " << eq.getRhs();
      return success();
    }

    if (hasWitness) {
      // proj-resolve: the result is the witness's own equality. A clone
      // rebuilds the witness and respells the claim under one substitution
      // (`respellWitness`, `respellEqualityEndpoints`), so the two never part.
      WitnessAttr witness = getWitnessAttr();
      // The witness slot carries a proj-resolve leaf, so its predicate is an
      // equality; a coerce discharge's application-headed witness has no place
      // here. Guard before reading the endpoints off the equality.
      if (!isa<TypeEqualityAttr>(witness.getPredicate()))
        return emitOpError() << "a proj-resolve witness must carry an "
                                "equality";
      if (witness.getEquality() != eq)
        return emitOpError() << "result endpoints " << eq.getLhs() << " = "
                             << eq.getRhs() << " are not the witness's "
                             << witness.getProjection() << " = "
                             << witness.getResolved();
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
  if (hasWitness || hasRefl || !getPremises().empty())
    return emitOpError() << "an application witness carries neither a "
                            "proj-resolve leaf, a refl marker, nor premises";
  if (!result.isProven())
    return emitOpError() << "an application witness's claim " << result
                         << " names the proof it cites";
  return success();
}

LogicalResult WitnessOp::verifyResolution(ModuleOp module,
                                          const ReadOnlyImplResolver *settled) {
  SmallVector<TypeEqualityAttr> equalityPremises;
  SmallVector<TraitApplicationAttr> applicationPremises;
  for (Value premise : getPremises())
    if (auto claim = dyn_cast<ClaimType>(premise.getType())) {
      if (auto eq = claim.getEqualityAttr())
        equalityPremises.push_back(eq);
      else if (claim.isApplication())
        applicationPremises.push_back(claim.getTraitApplication());
    }
  auto siteEvidence = [&] {
    NormalizationContext evidence =
        buildLocalClaimNormalizationContext(getOperation(), getPremises(), module);
    if (settled)
      evidence.setRecordedFacts(settled);
    return evidence;
  };
  return verifyProjectionResolutionAtUse(module, getWitnessAttr(),
                                         equalityPremises, applicationPremises,
                                         siteEvidence,
                                         [&] { return emitOpError(); });
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  // Equality proj-resolve arm: verify the citation where its symbol uses are
  // checked. The cited impl, at the type arguments the witness carries, must
  // bind the associated type the witness's projection names to its resolved
  // type, its where-clause equalities must hold there, AND the witness's
  // premises must discharge the cited impl's own assumptions. The premises split
  // by arm: equality claims are the comparison modulus, application claims
  // discharge the assumptions. The where-clause equalities are read through
  // what this op may read any spelling through: the hypotheses of the scope it
  // stands in and the evidence its premises carry.
  if (getWitnessAttr())
    return verifyResolution(module, /*settled=*/nullptr);

  // Refl arm: nothing to verify here.
  if (getRefl())
    return success();

  // Composition arm: an equality result with neither a witness nor a refl
  // marker cites nothing by symbol -- its premises are SSA values -- so there is
  // no citation to verify here.
  if (getResultClaim().isEquality())
    return success();

  // Application arm: one lookup of the cited symbol; a directly-named impl must
  // be unconditional, and a proof names the impl it stands over.
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(module, getProof(),
                                                        errFn);
  if (failed(cited)) return failure();
  auto proof = dyn_cast<ProofOp>(*cited);
  ImplOp impl = proof ? proof.getImpl() : cast<ImplOp>(*cited);
  if (!impl)
    return emitOpError() << "proof '" << getProof()
                         << "' does not resolve to an impl";

  // As at a proof: a projection the impl's header spells reduces through the
  // evidence the witnessed claim names -- the proof tree it carries, by index
  // -- and then through the impls the module holds.
  NormalizationContext reading;
  if (spellsAProjection(Type(impl.getSelfClaim())) ||
      impl.getAssumptions().hasEqualities())
    reading = buildProofNormalizationContext(getProvenClaim(), module);
  reading.setModuleLookup(module, LookupScope::Ground,
                          DemandOrigin::ProofVerification);
  auto throughEvidence = [&](Type ty) -> FailureOr<Type> {
    return reading.normalize(ty, errFn);
  };
  // A witness carries the claim the evidence it names stands over. A proof
  // stands over one claim and its parameters take the arguments a use supplies,
  // so the proof's claim is the declaration and this one is the use -- the
  // comparison every citation of a proof is read by. Reading the impl's header
  // alone would accept a witness for an application the proof does not prove,
  // because a blanket impl's header carries to every application of its trait.
  if (proof) {
    auto citationErr = [&] {
      return errFn() << "the proof " << getProof()
                     << " this witness cites stands over another claim: ";
    };
    Type proofClaim = Type(proof.getProvenClaim());
    if (failed(matchDeclaration(getTypeParametersIn(proofClaim), proofClaim,
                                Type(getProvenClaim()), throughEvidence,
                                citationErr)))
      return failure();
  }

  auto subst = impl.buildSubstitutionForSelfClaim(getProvenClaim(),
                                                  throughEvidence, errFn);
  if (failed(subst))
    return failure();

  // A proof states its impl's equality premises at the claim it stands over,
  // and one that claim leaves open is refused there, so a witness of a proof
  // reads none. An impl named directly stands over no claim of its own, and it
  // takes no subproof, so its premises are read here or nowhere.
  if (proof)
    return success();
  return verifyEqualityPremisesHoldAt(impl, getProvenClaim(), *subst, reading,
                                      OpenPremise::DecidedAtInstances,
                                      StandingPremise::DecidedAtStageExit, errFn);
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

/// Refuses `supplied` unless it holds one claim per entry of `expected`, each
/// that entry, read modulo the evidence it names. `owner` names what states the
/// entries and `supplier` the op supplying them, as a refusal reads them.
static LogicalResult verifyPremisesSuppliedByPosition(
    ValueRange supplied, ArrayRef<ClaimType> expected, const Twine &owner,
    StringRef supplier, llvm::function_ref<InFlightDiagnostic()> errFn) {
  if (supplied.size() != expected.size())
    return errFn() << owner << " states " << expected.size()
                   << " premises, and the " << supplier << " supplies "
                   << supplied.size();
  for (auto [position, pair] : llvm::enumerate(llvm::zip(supplied, expected))) {
    auto [operand, premise] = pair;
    ClaimType claim = cast<ClaimType>(operand.getType()).asUnproven();
    if (claim != premise)
      return errFn() << "premise " << position << " of " << owner << " is "
                     << premise << ", and the " << supplier << " supplies "
                     << claim;
  }
  return success();
}

/// Verifies a derive: the derived application is the impl's header at the
/// arguments the derive states, and each operand's claim is the impl's
/// where-clause entry at them, in order. A substitution decides both, so no
/// spelling is read through anything.
LogicalResult DeriveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&] { return emitOpError(); };
  ImplOp impl = getImplOp();
  if (!impl)
    return errFn() << "cannot find trait.impl '" << getImplAttr() << "'";
  auto arguments = impl.substitutionFor(getImplArguments(), errFn);
  if (failed(arguments))
    return failure();

  TraitApplicationAttr header = impl.getSelfApplicationAt(*arguments);
  if (header != getTraitApplication())
    return errFn() << "impl '" << getImplAttr() << "' at its stated "
                   << "arguments is an impl of " << header << ", not of "
                   << getTraitApplication();

  return verifyPremisesSuppliedByPosition(
      getAssumptions(), impl.getWhereClauseAt(*arguments),
      "impl '@" + getImpl() + "'", "derive", errFn);
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
  auto method = cast<MethodOp>((*this)->getParentOp());
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


//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

ParseResult AssumeOp::parse(OpAsmParser &p, OperationState &st) {
  // `self` or an entry index, then the claim the entry states as the result
  // type.
  if (succeeded(p.parseOptionalKeyword("self"))) {
    st.addAttribute("entry", p.getBuilder().getUnitAttr());
  } else {
    uint64_t position;
    if (p.parseInteger(position))
      return failure();
    st.addAttribute("entry", p.getBuilder().getI64IntegerAttr(position));
  }
  Type claim;
  if (p.parseColonType(claim))
    return failure();
  st.addTypes(claim);
  return success();
}

void AssumeOp::print(OpAsmPrinter &p) {
  p << " ";
  if (citesSelf())
    p << "self";
  else
    p << *getWherePosition();
  p << " : " << Type(getClaim());
}

static Operation *getScopeOwner(Operation *op);

LogicalResult AssumeOp::verify() {
  // An assume cites an entry of the declaration whose method it stands in, so
  // the scope it stands in is that method's.
  //
  // XXX TODO the commit that gives trait.trait and trait.impl their self claim
  // and prerequisites as block arguments, read by their methods, deletes this op
  // and its verifier.
  Operation *scope = getScopeOwner(getOperation());
  if (!scope)
    return emitOpError("must be within a function");
  auto function = dyn_cast<FunctionOpInterface>(scope);
  if (!function)
    return emitOpError() << "must be within a function, found "
                         << scope->getName();

  // The entry it cites exists, is a claim the declaration holds as a
  // hypothesis, and is exactly the claim the result type spells.
  Operation *owner = function->getParentOp();
  TraitApplicationAttr selfApplication;
  PredicateArrayAttr where;
  if (auto trait = dyn_cast_or_null<TraitOp>(owner)) {
    selfApplication = trait.getSelfApplication();
    where = trait.getRequirements();
  } else if (auto impl = dyn_cast_or_null<ImplOp>(owner)) {
    selfApplication = impl.getSelfApplication();
    where = impl.getAssumptions();
  } else {
    return emitOpError()
           << "cites an entry of the declaration its function is a method of, "
              "but '@"
           << function.getName() << "' is a method of no trait or impl";
  }

  MLIRContext *ctx = getContext();
  ClaimType stated;
  if (citesSelf()) {
    stated = ClaimType::get(ctx, selfApplication);
  } else {
    uint64_t position = *getWherePosition();
    if (position >= where.size())
      return emitOpError()
             << "cites where-clause entry " << position << ", but the "
             << "enclosing declaration's where clause has " << where.size()
             << " entries";
    Attribute entry = where.getPredicates()[position];
    if (auto app = dyn_cast<TraitApplicationAttr>(entry))
      stated = ClaimType::get(ctx, app);
    else if (auto eq = dyn_cast<TypeEqualityAttr>(entry))
      stated = ClaimType::getEquality(ctx, eq);
    else
      return emitOpError()
             << "cites where-clause entry " << position
             << ", which binds variables of its own; select it with "
                "trait.project and its type arguments";
  }

  // The spelled claim is an annotation on the citation: the position decides
  // which claim this op produces.
  if (getClaim() != stated)
    return emitOpError() << "the cited entry states " << stated
                         << ", but the result type spells " << getClaim();
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

/// The type arguments a generic call supplies for `parameters`, read off the
/// call's own types.
///
/// A call spells its operand, claim and result types and the callee's
/// declaration spells the same positions with its parameters standing in them,
/// so the pairing is a reading of one against the other -- the same one-way
/// reader impl selection runs (`extractTypeArguments`): every position outside
/// a projection first, then a projection the actual side spells the same way,
/// and a second differing reading keeps the first.
///
/// A parameter standing only inside a projection's associated-type arguments is
/// determined by a later round: the declaration is rebuilt at what has been read
/// and normalized through the evidence this call holds, which reduces a
/// projection whose head the reading has grounded and exposes the positions its
/// arguments stand in. Rounds stop when one fills nothing new, and a parameter
/// no position determines is refused, named.
///
/// Nothing here decides the verdict: filling a slot wrongly can only make the
/// rebuilt declaration differ from what the call spells, which
/// `verifyEqualAfterInstantiation` refuses.
static FailureOr<SpecializationMap> readTypeArguments(
    ArrayRef<GenericTypeInterface> parameters, Type formal, Type actual,
    Normalizer normalize, StringRef callee,
    llvm::function_ref<InFlightDiagnostic()> err) {
  TypeArguments args(parameters);
  auto filled = [&] {
    unsigned count = 0;
    for (GenericTypeInterface parameter : parameters)
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
  for (unsigned before = filled(); !args.complete(); ) {
    // A round that will not normalize has learned nothing, so it stops the
    // reading rather than refusing the call: what it could not reduce is the
    // spelling already read, and the comparison downstream owns the verdict.
    FailureOr<Type> exposed =
        normalize(instantiate(formal, args.toSpecialization()));
    if (failed(exposed))
      break;
    extractTypeArguments(*exposed, actual, args);
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
      for (GenericTypeInterface parameter : parameters)
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

/// Whether the code `op` holds is judged on its own rather than in the scope
/// `op` stands in. An operation isolated from above sees nothing of that scope,
/// so a nested function, a trait, an impl and a proof each answer for what they
/// hold where they are declared; a region an op runs at run time -- a
/// conditional, a loop, a cooperative body -- is interior to the scope around
/// it. A method is not isolated from above but reads nothing of its
/// declaration (`verifyDeclarationBodyTakesNoArguments`), so it too answers for
/// what it holds.
///
/// XXX TODO the commit that gives trait.trait and trait.impl their self claim
/// and prerequisites as block arguments, read by their methods, extends a
/// method's scope to its declaration's and deletes the method arm.
static bool isJudgedOnItsOwn(Operation *op) {
  return op->hasTrait<OpTrait::IsIsolatedFromAbove>() || isa<MethodOp>(op);
}

/// The declaration `op` stands in: the innermost ancestor judged on its own.
///
/// A region an op runs at run time -- a conditional, a loop, a cooperative body
/// -- is interior to the scope around it, so the walk passes through it and
/// stops at the callable, trait, impl or proof that binds what `op` may name.
static Operation *getScopeOwner(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isJudgedOnItsOwn(parent))
      return parent;
  return nullptr;
}

/// Adds the rules one hypothesis in scope licenses.
///
/// An equality hypothesis says its two types are one wherever it stands. An
/// application hypothesis says its trait holds of those arguments, and a trait
/// holds only where its own requirements do, so each requirement instantiated at
/// that application is a hypothesis in turn -- the same reading `trait.project`
/// performs on a claim value. No impl is consulted: a hypothesis names none,
/// and what it licenses is what its trait declares.
///
/// Each application is read once. A requirement chain that reaches one trait at
/// ever larger arguments makes progress at every step, so it stops at the
/// instantiation limit, the bound every such chain meets.
static void addScopeHypothesis(NormalizationContext &ctx, ClaimType claim,
                               ModuleOp module,
                               DenseSet<TraitApplicationAttr> &visited,
                               unsigned depth) {
  if (auto equality = claim.getEqualityAttr()) {
    ctx.assumeEqual(equality.getLhs(), equality.getRhs());
    return;
  }

  TraitApplicationAttr application = claim.getTraitApplication();
  if (!application || depth == kInstantiationDepthLimit ||
      !visited.insert(application).second)
    return;

  auto trait = application.getTrait(module, /*err=*/nullptr);
  if (failed(trait))
    return;
  auto requirements = trait->specializeRequirementsAsClaimsFor(
      claim.asUnproven(), /*errFn=*/nullptr);
  if (failed(requirements))
    return;
  for (ClaimType requirement : *requirements)
    addScopeHypothesis(ctx, requirement, module, visited, depth + 1);
}

/// Adds the hypotheses the scope `op` stands in holds.
///
/// A declaration's claim parameters are its where clause, and a where clause is
/// the parameter environment of everything its body holds: the caller discharged
/// each one, so inside the body each is an axiom -- an equality parameter is a
/// rewrite rule there and an application parameter carries its trait's
/// requirements. They are read off the block arguments the scope owner binds,
/// which is a parent read and not a search.
static void addScopeHypotheses(NormalizationContext &ctx, Operation *op,
                               ModuleOp module) {
  Operation *scope = getScopeOwner(op);
  if (!scope || scope->getNumRegions() == 0)
    return;
  Region &body = scope->getRegion(0);
  if (body.empty())
    return;

  DenseSet<TraitApplicationAttr> visited;
  for (BlockArgument parameter : body.front().getArguments())
    if (auto claim = dyn_cast<ClaimType>(parameter.getType()))
      addScopeHypothesis(ctx, claim, module, visited, /*depth=*/0);
}

/// Adds the projection normalization rules a proven claim's proof tree
/// justifies: the rules of its subproofs first, then its own.
///
/// A proof stands over the obligations of the impl it names, each discharged by
/// the subproof at the same index, so the impls those subproofs name are
/// evidence at this site exactly as the impl the proof itself names is -- the
/// same reading by index a derive gets from its given operands. The children go
/// in first, and the head is read through the rules they contributed: an impl
/// header that spells a projection over one of its own obligations reduces it
/// through the rule that obligation's proof already contributed, and where a
/// trait has two impls whose headers could each bind that application, that rule
/// is the only thing that answers.
///
/// A claim whose own rule cannot be built contributes none: this reads the
/// evidence an op holds, and a proof that does not check is refused where it is
/// verified rather than reported from here. Contributing fewer rules leaves
/// more projections standing, which the comparison refuses on.
static void addLocalProjectionRulesFromProvenClaim(
    NormalizationContext &ctx, ClaimType claim, ModuleOp module,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  // The symbol this claim cites, read once: it is the proof whose subtree
  // contributes first and it names the impl whose bindings justify the rule
  // below, or it is that impl itself where the citation is a leaf.
  Operation *cited = lookupSymbolFrom(module, claim.getProof());
  auto proof = dyn_cast_or_null<ProofOp>(cited);
  ImplOp impl = proof ? proof.getImpl() : dyn_cast_or_null<ImplOp>(cited);
  if (!impl)
    return;

  // The proof's own subtree. A coinductive proof names itself among its
  // subproofs, so a proof already read contributes nothing a second time.
  if (proof)
    if (visited.insert(proof.getOperation()).second) {
      auto subproofs = proof.verifyAndGetSubproofClaims(claim, /*err=*/nullptr);
      if (succeeded(subproofs))
        for (ClaimType subproof : *subproofs)
          if (subproof.isProven())
            addLocalProjectionRulesFromProvenClaim(ctx, subproof, module,
                                                   visited);
    }

  // Store rules against the unproven application because projection heads do
  // not include proof symbols; proof only explains why the application holds.
  ClaimType unproven = claim.asUnproven();
  auto throughRulesSoFar = [&](Type ty) -> FailureOr<Type> {
    return ctx.normalize(ty, /*err=*/nullptr);
  };
  auto subst = impl.buildSubstitutionForSelfClaim(unproven, throughRulesSoFar,
                                                 /*errFn=*/nullptr);
  if (failed(subst))
    return;

  ctx.addLocalProjectionRule(impl, unproven.getTraitApplication(), *subst);
}

/// Adds projection normalization rules justified by one claim SSA value.
///
/// A rule records that projections for a specific trait application may use a
/// specific impl's associated type bindings while checking this operation.
///
/// Reading evidence never refuses: a claim whose rule cannot be built is
/// refused where it is verified, not at the op that holds it.
static void addLocalProjectionRulesFromClaim(
    NormalizationContext &ctx, Value claimValue, ModuleOp module,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  // Only claim-typed operands can carry trait evidence relevant to projection
  // normalization. Ordinary method arguments do not contribute rules.
  auto claim = dyn_cast<ClaimType>(claimValue.getType());
  if (!claim)
    return;

  // An equality claim IS evidence of its own equality, so an op holding one may
  // read either endpoint as the other.
  if (auto equality = claim.getEqualityAttr()) {
    ctx.assumeEqual(equality.getLhs(), equality.getRhs());
    return;
  }

  // A coerce respells the claim it forwards, and what licenses the respelling
  // are the equalities it cites -- checked at the coerce itself. An op holding
  // the result holds those equalities too, and the claim the coerce read them
  // onto in turn.
  if (auto coerce = claimValue.getDefiningOp<CoerceOp>()) {
    if (visited.insert(coerce.getOperation()).second) {
      for (Value equality : coerce.getEqualities())
        addLocalProjectionRulesFromClaim(ctx, equality, module, visited);
      addLocalProjectionRulesFromClaim(ctx, coerce.getInput(), module, visited);
    }
  }

  // A proven claim names a proof symbol. That proof identifies the impl whose
  // associated type bindings justify reducing projections with this exact
  // trait application, and stands over the subproofs discharging that impl's
  // obligations.
  if (claim.isProven()) {
    addLocalProjectionRulesFromProvenClaim(ctx, claim, module, visited);
    return;
  }

  // A derive op also commits to one impl, but the evidence may be nested in its
  // assumptions. For example, a FnUni derive can carry the Fn claim that
  // resolves a closure's Output projection.
  if (auto derive = claimValue.getDefiningOp<DeriveOp>()) {
    // Derived claims can refer to other derived claims through assumptions; the
    // visited set keeps malformed or cyclic IR from recursing forever.
    if (!visited.insert(derive.getOperation()).second)
      return;

    // The given operands are part of the local evidence package used to derive
    // this claim, and they go in first: an impl header that forwards an
    // associated type spells a projection over an application one of them
    // carries, so the header below is read at the grade their rules reach.
    for (Value assumption : derive.getAssumptions())
      addLocalProjectionRulesFromClaim(ctx, assumption, module, visited);

    ImplOp impl = derive.getImplOp();
    if (!impl)
      return;

    ClaimType derived = derive.getDerivedClaim();
    auto throughRulesSoFar = [&](Type ty) -> FailureOr<Type> {
      return ctx.normalize(ty, /*err=*/nullptr);
    };
    auto subst = impl.buildSubstitutionForSelfClaim(derived, throughRulesSoFar,
                                                    /*errFn=*/nullptr);
    if (failed(subst))
      return;

    ctx.addLocalProjectionRule(impl, derived.getTraitApplication(), *subst);
  }
}

NormalizationContext buildProofNormalizationContext(ClaimType provenClaim,
                                                    ModuleOp module) {
  NormalizationContext ctx;
  llvm::SmallPtrSet<Operation *, 8> visited;
  addLocalProjectionRulesFromProvenClaim(ctx, provenClaim, module, visited);
  return ctx;
}

NormalizationContext buildSubproofNormalizationContext(ProofOp proof,
                                                       ClaimType at,
                                                       ModuleOp module) {
  NormalizationContext ctx;
  llvm::SmallPtrSet<Operation *, 8> visited;
  // The proof itself is marked read before the walk starts, so the tree it
  // stands over contributes and it does not.
  visited.insert(proof.getOperation());
  auto subproofs = proof.verifyAndGetSubproofClaims(at, /*err=*/nullptr);
  if (succeeded(subproofs))
    for (ClaimType subproof : *subproofs)
      if (subproof.isProven())
        addLocalProjectionRulesFromProvenClaim(ctx, subproof, module, visited);
  return ctx;
}

NormalizationContext buildLocalClaimNormalizationContext(Operation *op,
                                                         ValueRange values,
                                                         ModuleOp module) {
  NormalizationContext ctx;
  // The hypotheses of the scope go in first: they hold throughout the body, so
  // the evidence read next is read at the grade they already reach.
  addScopeHypotheses(ctx, op, module);
  // The derives and proofs already read, so cyclic or self-referencing evidence
  // is read once.
  llvm::SmallPtrSet<Operation *, 8> visited;
  for (Value value : values)
    addLocalProjectionRulesFromClaim(ctx, value, module, visited);
  return ctx;
}

/// Checks every proof the claims a call carries name, each at its own claim:
/// the declaration the named evidence holds must carry to the application the
/// claim spells, and what that evidence proves underneath was decided at the
/// proof op holding it.
///
/// Each spelling is read through the call's own context first: a coerce
/// respells a claim through an equality it cites, and the proof standing on the
/// respelled claim is the proof of the spelling that equality carries it back
/// to.
static LogicalResult verifyProofsAtCall(Operation *call, ValueRange operands,
                                        Normalizer normalize, ModuleOp module,
                                        llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<Type> spellings(operands.getTypes());
  llvm::append_range(spellings, call->getResultTypes());

  for (Type spelling : spellings) {
    FailureOr<Type> read = normalize(spelling);
    if (failed(read))
      return failure();
    if (failed(verifyCitationsIn(*read, module,
                                 DemandOrigin::CallSignatureVerification,
                                 normalize, err)))
      return failure();
  }
  return success();
}

/// The type arguments a call supplies for the declaration it calls, once that
/// declaration instantiated at them is the signature the call spells.
///
/// `formal` is the callee's signature and `known` the arguments already fixed
/// before the call's own types are read -- a method's trait arguments, which
/// ride in its receiver claim; `parameters` are the ones this call's types
/// determine, read against `formal` instantiated at `known`. `actual` is the
/// signature the call spells, and `localClaims` the claims it holds, which are
/// its evidence, read by index. `commitsToEvidence` says whether one of those
/// claims commits to evidence -- a proven claim, or one a derive produced --
/// which is what licenses reading a ground projection through the module's
/// impls.
///
/// `reading` is the stage's record of what impl selection has settled, which
/// the comparison reads both signatures through on top of that evidence. A
/// verifier passes none and compares through the evidence alone, and then the
/// proofs the call's claims name are read at their own claims as well.
static FailureOr<SpecializationMap> readCallSpecialization(
    Operation *call, ModuleOp module, FunctionType formal,
    const SpecializationMap &known, ArrayRef<GenericTypeInterface> parameters,
    FunctionType actual, ValueRange localClaims, bool commitsToEvidence,
    StringRef callee, const ReadOnlyImplResolver *reading,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // Reading the evidence at this call resolves symbol names and writes nothing,
  // so what a name answers is held for the read. Under a stage already holding
  // answers this reads through those.
  SymbolLookupScope symbolAnswers;

  NormalizationContext normalization =
      buildLocalClaimNormalizationContext(call, localClaims, module);
  normalization.setRecordedFacts(reading);
  // XXX TODO A claim this call's own arguments spell can carry a ground
  // projection no evidence at this site reduces, because the impl serving it is
  // named nowhere the call can read. The module's impls stand in, and only
  // where the call commits to evidence -- an ordinary unproven claim grants
  // nothing. Deleted once the claim a call commits to carries the impls serving
  // the projections its arguments spell; see setModuleLookup.
  if (commitsToEvidence)
    normalization.setModuleLookup(module, LookupScope::Ground);
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return normalization.normalize(ty, err);
  };

  // The reading's own context, which differs from the comparison's in one way:
  // it reduces a projection whose own spelling determines the impl serving it
  // even where its associated-type arguments still hold a variable. A reading
  // compares a spelling and never serves one, which is what that scope licenses.
  NormalizationContext readingContext = normalization;
  readingContext.setModuleLookup(module, LookupScope::Determined);
  auto normalizeForReading = [&](Type ty) -> FailureOr<Type> {
    return readingContext.normalize(ty, err);
  };

  // At pass time the comparison reads softly: a call whose evidence is not yet
  // recorded waits for a later round. A call whose operands are already
  // monomorphic says everything it will ever say about the instance it wants,
  // so a parameter its types do not determine is refused here, named, rather
  // than surfacing later as a type variable nothing bound.
  auto reportHere = [&]() -> InFlightDiagnostic { return call->emitOpError(); };
  llvm::function_ref<InFlightDiagnostic()> refusal = err;
  if (!refusal && llvm::all_of(call->getOperandTypes(), isMonomorphicType))
    refusal = reportHere;

  auto read = readTypeArguments(parameters, instantiate(Type(formal), known),
                                Type(actual), normalizeForReading, callee,
                                refusal);
  if (failed(read)) return failure();

  SpecializationMap arguments = known;
  for (GenericTypeInterface parameter : parameters)
    if (auto argument = read->lookup(parameter))
      arguments.bind(parameter, *argument);

  // One identity: the callee's declaration instantiated at those arguments is
  // the signature spelled here.
  if (failed(verifyEqualAfterInstantiation(Type(formal), arguments,
                                           Type(actual), normalize, err)))
    return failure();

  // The proofs this call's claims name are read at their own claims: each
  // spells evidence for one application, and the declaration that evidence
  // holds must carry to it. What the evidence proves underneath was decided at
  // the proof op holding it. The factory that closes the substitution walks the
  // same spellings where the call is lowered; a verifier has no lowering behind
  // it, so it reads them here or nowhere.
  if (!reading &&
      failed(verifyProofsAtCall(call, call->getOperands(), normalize, module,
                                err)))
    return failure();

  return arguments;
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification writes nothing, so every name read under it resolves through
  // the symbol tables the walk this is one step of has already built.
  SymbolLookupScope symbolAnswers(getOperation(), symbolTable);

  auto errFn = [&]{ return emitOpError(); };

  // A verifier holds no record of what impl selection has settled, so the
  // comparison reads both signatures through the evidence this call carries and
  // nothing else.
  return buildParameterSpecialization(/*reading=*/nullptr, errFn);
}

FailureOr<SpecializationMap> MethodCallOp::buildParameterSpecialization(
    const ReadOnlyImplResolver *reading,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  auto trait = getTrait(err);
  if (failed(trait)) return failure();

  auto methodFormalTy = getMethodFunctionType(err);
  if (failed(methodFormalTy)) return failure();

  // A method's declaration binds the trait header's parameters and then its
  // own. The prefix comes from the receiver claim by position -- the trait's
  // arguments ride in the application -- and this call's types determine the
  // method's own variables, the ones its declaration binds beyond the header's.
  auto traitSubst = trait->buildSubstitutionForSelfClaim(getClaimType(), err);
  if (failed(traitSubst)) return failure();
  SmallVector<GenericTypeInterface, 4> ownParams = getOwnTypeParameters(
      Type(*methodFormalTy), getTraitHeaderParameters(*trait));

  // The evidence this call holds: the receiver claim's proof tree and the
  // claims its arguments carry. Only the receiver commits the call to evidence.
  SmallVector<Value> localClaims;
  localClaims.push_back(getClaim());
  for (Value argument : getArguments())
    if (isa<ClaimType>(argument.getType()))
      localClaims.push_back(argument);
  bool commitsToEvidence =
      getClaimType().isProven() || getClaim().getDefiningOp<DeriveOp>();

  return readCallSpecialization(getOperation(), *module, *methodFormalTy,
                                *traitSubst, ownParams, getActualFunctionType(),
                                localClaims, commitsToEvidence, getMethodName(),
                                reading, err);
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
    PatternRewriter &rewriter,
    const CallSubstitution &subst) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrSpecializeFreeFunctionFromMethod(rewriter, claimTy, getMethodName(),
                                           getArguments().getTypes(), subst);
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

  // A verifier holds no record of what impl selection has settled, so the
  // comparison reads both signatures through the evidence this call carries and
  // nothing else.
  return buildParameterSpecialization(/*reading=*/nullptr, errFn);
}

FailureOr<SpecializationMap> FuncCallOp::buildParameterSpecialization(
    const ReadOnlyImplResolver *reading,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  auto formal = getCalleeFunctionType(err);
  if (failed(formal)) return failure();

  // The evidence this call holds: the claims its operands carry, any of which
  // commits the call to evidence. The callee's declaration binds the parameters
  // its signature spells, and this call's own types determine each of them.
  SmallVector<Value> localClaims;
  for (Value operand : getOperands())
    if (isa<ClaimType>(operand.getType()))
      localClaims.push_back(operand);
  bool commitsToEvidence = llvm::any_of(localClaims, [](Value claim) {
    return cast<ClaimType>(claim.getType()).isProven() ||
           claim.getDefiningOp<DeriveOp>();
  });

  return readCallSpecialization(getOperation(), *module, *formal,
                                SpecializationMap(), getCalleeTypeParams(),
                                getActualFunctionType(), localClaims,
                                commitsToEvidence, getCalleeName(), reading, err);
}

FailureOr<func::FuncOp> FuncCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter,
    const CallSubstitution &subst) {
  auto module = getModule();
  if (failed(module)) return failure();

  auto callee = getCallee();
  if (failed(callee)) return failure();

  // A callee whose signature binds no type parameter is no template: the call
  // reaches it as written, so what the call supplies must be what it declares.
  SmallVector<GenericTypeInterface, 4> typeParams = getCalleeTypeParams();
  if (typeParams.empty()) {
    TypeRange parameters = callee->getFunctionType().getInputs();
    TypeRange operands = getOperandTypes();
    if (parameters.size() != operands.size())
      return emitOpError() << "passes " << operands.size()
                           << " operand(s) to '@" << getCalleeName()
                           << "', which takes " << parameters.size();
    for (auto [index, types] : llvm::enumerate(llvm::zip(parameters, operands))) {
      auto [parameter, operand] = types;
      if (parameter != operand)
        return emitOpError() << "passes " << operand << " as operand #"
                             << index << " to '@" << getCalleeName()
                             << "', which takes " << parameter;
    }
    return *callee;
  }

  // The instance is the one this call's type arguments and evidence name. The
  // specialization map is written when the substitution is built and is not
  // touched by closing it, so the arguments read here and the body cut below
  // are read off one object.
  SmallVector<Type> typeArguments;
  for (GenericTypeInterface parameter : typeParams)
    typeArguments.push_back(subst.getSpecialization().apply(parameter));
  AttrTypeReplacer stamp =
      makeTypeReplacerFromSubstitution(subst.toTypeMap(), *module);
  auto key = InstanceKey::get(getCalleeNameAttr(), typeArguments,
                              callee->getFunctionType().getInputs(),
                              getOperandTypes(), stamp);
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
      },
      subst.getEvidence());
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
  auto requirement = getClaimRequirementAt(
      getSourceClaim(), module, getIndex(), getBinderArguments(), errFn);
  if (failed(requirement))
    return failure();

  // A bound requirement holds where its premises do, so the hop carries one
  // claim per premise, each the premise at the arguments it supplies; a claim
  // is read modulo the evidence it names.
  if (failed(verifyPremisesSuppliedByPosition(
          getPremises(), requirement->premises,
          "requirement " + Twine(getIndex()), "hop", errFn)))
    return failure();

  // The result type is an annotation on the selection: the index decides which
  // claim this op produces, so the spelled one must be that claim.
  if (requirement->conclusion != getResultClaim())
    return emitOpError() << "type mismatch: expected "
                         << requirement->conclusion << " but found "
                         << getResultClaim();

  return success();
}
