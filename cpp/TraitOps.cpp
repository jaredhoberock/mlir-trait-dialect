// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/DenseMap.h>
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

/// The function type of a child `func.func` a parent's verifier is about to
/// read.
///
/// A child's own invariants are verified after its parent's, so the type is read
/// through the attribute dictionary rather than through the getter that casts:
/// a malformed one is refused where it stands instead of aborting the cast.
static FailureOr<FunctionType> readChildFunctionType(func::FuncOp function) {
  auto typeAttr =
      function->getAttrOfType<TypeAttr>(function.getFunctionTypeAttrName());
  auto functionType =
      typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType();
  if (!functionType) {
    function.emitOpError()
        << "requires a function type in its '"
        << function.getFunctionTypeAttrName().getValue() << "' attribute";
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
    func::FuncOp function, FunctionType functionType,
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
           << "function '" << function.getSymName()
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
/// impls the proofs discharging them name, by index. A proof justifies nothing
/// about itself, so its own rule is not among these.
static NormalizationContext buildSubproofNormalizationContext(ProofOp proof,
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
            proj.getAssocName().getValue(), proj.getAssocTypeArgs());
        if (failed(resolved))
          continue;
        return rule.subst.apply(*resolved);
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
// Type parameter scope
//===----------------------------------------------------------------------===//

namespace {

/// Whether the code `op` holds is judged on its own rather than in the scope
/// `op` stands in. An operation isolated from above sees nothing of that scope,
/// so a nested function, a trait, an impl and a proof each answer for what they
/// hold where they are declared; a region an op runs at run time -- a
/// conditional, a loop, a cooperative body -- is interior to the scope around
/// it.
bool isJudgedOnItsOwn(Operation *op) {
  return op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}

/// Collects into `parameters` every type parameter `root` binds, `root` being a
/// type or an attribute: one walk reads either, and each type the walk meets is
/// read with `getTypeParametersIn`, so a parameter standing in a claim's
/// application or an equality's endpoints is collected wherever a spelling
/// holds it.
template <typename RootT>
void collectTypeParametersIn(RootT root, SetVector<Type> &parameters) {
  root.walk([&](Type sub) {
    for (GenericTypeInterface parameter : getTypeParametersIn(sub))
      parameters.insert(Type(parameter));
  });
}

/// The type parameters `callable`'s signature binds: the parameters its
/// argument and result types spell, in first-occurrence order.
SetVector<Type> getSignatureTypeParams(CallableOpInterface callable) {
  SetVector<Type> params;
  for (Type type : callable.getArgumentTypes())
    collectTypeParametersIn(type, params);
  for (Type type : callable.getResultTypes())
    collectTypeParametersIn(type, params);
  return params;
}

/// The type parameters `function`'s declaration binds: the generics its own
/// signature spells, read as any callable's are, and, for a method, the generics
/// of the trait or impl header it is written in, which a use of that header
/// supplies.
SetVector<Type> getDeclaredTypeParams(FunctionOpInterface function) {
  SetVector<Type> declared =
      getSignatureTypeParams(cast<CallableOpInterface>(function.getOperation()));

  if (auto impl = function->getParentOfType<ImplOp>()) {
    for (GenericTypeInterface parameter : impl.getTypeParams())
      declared.insert(Type(parameter));
  } else if (auto trait = function->getParentOfType<TraitOp>()) {
    for (Attribute param : trait.getTypeParams())
      if (auto typeAttr = dyn_cast<TypeAttr>(param))
        collectTypeParametersIn(typeAttr.getValue(), declared);
  }
  return declared;
}

/// Whether `name` is the attribute a generic call spells its CALLEE's type
/// parameters in. Those labels stand in the callee's scope -- the call supplies
/// a type argument for each of them -- so a body spelling one there names none
/// of its own.
bool namesCalleeTypeParameters(Operation *op, StringAttr name) {
  if (auto call = dyn_cast<FuncCallOp>(op))
    return name == call.getTypeParamsAttrName();
  if (auto call = dyn_cast<MethodCallOp>(op))
    return name == call.getTypeParamsAttrName();
  return false;
}

/// The first mention of each type parameter no declaration in scope binds, in
/// the order a walk meets them, so each parameter is refused once and at a site.
struct StrayMentions {
  SmallVector<std::pair<Type, Operation *>> inOrder;
  DenseSet<Type> seen;

  /// Reads `root`, a type or an attribute, for the parameters `declared` does
  /// not bind, recording a first mention at `at`.
  template <typename RootT>
  void read(RootT root, const SetVector<Type> &declared, Operation *at) {
    root.walk([&](Type sub) {
      for (GenericTypeInterface parameter : getTypeParametersIn(sub)) {
        Type label(parameter);
        if (declared.contains(label))
          continue;
        if (seen.insert(label).second)
          inOrder.emplace_back(label, at);
      }
    });
  }
};

void judgeInterior(Operation *op, const SetVector<Type> &inside,
                   StrayMentions &mentions);

/// Judges `op`, and whatever it holds, against `declared`: the type parameters
/// the scope `op` stands in binds.
void judge(Operation *op, const SetVector<Type> &declared,
           StrayMentions &mentions) {
  // An operand and a result are values of the scope `op` stands in, whatever
  // `op` holds.
  for (Type type : op->getOperandTypes())
    mentions.read(type, declared, op);
  for (Type type : op->getResultTypes())
    mentions.read(type, declared, op);

  if (isJudgedOnItsOwn(op))
    return;

  // A callable the scope around it reaches into is a lambda a dialect
  // specializes per use -- a `tuple.map` body, once per element type -- so its
  // signature binds parameters of its own for what it holds, on top of the ones
  // already in scope. A callable holds one region, so that signature governs the
  // whole interior.
  if (auto callable = dyn_cast<CallableOpInterface>(op)) {
    SetVector<Type> inside = declared;
    inside.set_union(getSignatureTypeParams(callable));
    judgeInterior(op, inside, mentions);
    return;
  }
  judgeInterior(op, declared, mentions);
}

/// Reads the attributes, block arguments and operations `op` holds against
/// `inside`, the type parameters in scope within `op`.
void judgeInterior(Operation *op, const SetVector<Type> &inside,
                   StrayMentions &mentions) {
  for (NamedAttribute attribute : op->getAttrDictionary())
    if (!namesCalleeTypeParameters(op, attribute.getName()))
      mentions.read(attribute.getValue(), inside, op);

  for (Region &region : op->getRegions())
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments())
        mentions.read(argument.getType(), inside, op);
      for (Operation &inner : block)
        judge(&inner, inside, mentions);
    }
}

} // namespace

LogicalResult mlir::trait::verifyFunctionBodyIsWellScoped(
    FunctionOpInterface function) {
  SetVector<Type> declared = getDeclaredTypeParams(function);

  // The function's own attributes are its declaration, read above: the judgment
  // is about its body.
  StrayMentions mentions;
  for (Region &region : function->getRegions())
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments())
        mentions.read(argument.getType(), declared, function.getOperation());
      for (Operation &op : block)
        judge(&op, declared, mentions);
    }

  for (auto [generic, at] : mentions.inOrder) {
    InFlightDiagnostic diagnostic =
        function.emitError()
        << "type parameter " << generic
        << " is outside the signature scope of @"
        << SymbolTable::getSymbolName(function).getValue();
    diagnostic.attachNote(at->getLoc()) << "mentioned here";
  }
  return success(mentions.inOrder.empty());
}


//===----------------------------------------------------------------------===//
// TraitOp
//===----------------------------------------------------------------------===//

LogicalResult TraitOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();

  auto typeParams = getTypeParams().getAsValueRange<TypeAttr>();

  // types must be unique GenericTypeParameters
  DenseSet<Type> uniqueParams;
  for (Type ty : typeParams) {
    if (!isa<GenericTypeInterface>(ty))
      return emitOpError() << "expected GenericTypeInterface (e.g., !trait.poly), found " << ty;
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
  // A child's own invariants are verified after its parent's, so each entry is
  // read as an attribute that may be anything and refused where it stands rather
  // than cast.
  DenseSet<Type> gatParams;
  for (Operation &op : getBody().front()) {
    auto assoc = dyn_cast<AssocTypeOp>(op);
    if (!assoc)
      continue;
    ArrayAttr declaredParams = assoc.getTypeParamsAttr();
    if (!declaredParams)
      continue;
    for (Attribute tyAttr : declaredParams) {
      auto typeAttr = dyn_cast<TypeAttr>(tyAttr);
      if (!typeAttr)
        return assoc.emitOpError()
               << "type parameter list holds " << tyAttr << ", which is not a type";
      Type param = typeAttr.getValue();
      if (uniqueParams.contains(param))
        return assoc.emitOpError()
               << "type parameter " << param << " is already a parameter of trait '@"
               << getSymName() << "'";
      gatParams.insert(param);
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

  // check trait method result generics
  for (Operation &op : getBody().front()) {
    if (auto method = dyn_cast<func::FuncOp>(op)) {
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

// Parse the bracketed parameter list shared by trait and associated type
// declarations. Keep the explicit square delimiter, including the empty list.
static ParseResult parseTypeParameters(OpAsmParser &p, ArrayAttr &parameters) {
  SmallVector<Type> types;
  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&] {
        Type type;
        if (p.parseType(type)) return failure();
        types.push_back(type);
        return success();
      }))
    return failure();
  parameters = p.getBuilder().getTypeArrayAttr(types);
  return success();
}

ParseResult TraitOp::parse(OpAsmParser &p, OperationState &s) {
  MLIRContext *ctx = p.getContext();

  // optional `private`/`nested` keyword before the name, as func.func reads it;
  // a missing keyword is not an error (the op is public) and consumes nothing.
  (void)mlir::impl::parseOptionalVisibilityKeyword(p, s.attributes);

  // sym_name
  StringAttr symName;
  if (p.parseSymbolName(symName, "sym_name", s.attributes))
    return failure();

  // [ type_params ]
  ArrayAttr typeParams;
  if (parseTypeParameters(p, typeParams)) return failure();
  s.addAttribute("type_params", typeParams);

  // requirements
  auto requirements = PredicateArrayAttr::get(ctx, ArrayRef<Attribute>());
  if (succeeded(p.parseOptionalKeyword("where"))) {
    requirements = dyn_cast_or_null<PredicateArrayAttr>(PredicateArrayAttr::parse(p,{}));
    if (!requirements)
      return p.emitError(p.getCurrentLocation(), "expected a predicate array");
  }
  s.addAttribute("requirements", requirements);

  // attr-dict-with-keyword
  if (p.parseOptionalAttrDictWithKeyword(s.attributes))
    return failure();

  // region body
  Region *body = s.addRegion();
  if (p.parseRegion(*body, /*args=*/{}, /*types=*/{})) return failure();
  if (body->empty()) body->emplaceBlock();

  return success();
}

void TraitOp::print(OpAsmPrinter &p) {
  // optional `private`/`nested` keyword before the name; public is elided
  if (auto vis = getSymVisibilityAttr())
    if (vis.getValue() != "public")
      p << ' ' << vis.getValue();

  // `@sym_name`
  p << ' ';
  p.printSymbolName(getSymNameAttr());

  // `[ type_params ]`
  p << "[";
  llvm::interleaveComma(getTypeParams(), p, [&](Attribute tyAttr) {
    p.printType(cast<TypeAttr>(tyAttr).getValue());
  });
  p << ']';

  // print requirements if not empty
  if (hasRequirements()) {
    p << " where ";
    getRequirements().print(p);
  }

  // print any trailing attributes
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elided=*/{"sym_name","type_params","requirements","sym_visibility"});

  // region body
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}


//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

/// A verified equality-armed witness in replayable form: the cited impl, the
/// projection's trait application, and the head-match substitution, ready for
/// NormalizationContext::addLocalProjectionRule.
struct ImplWitnessRule {
  ImplOp impl;
  TraitApplicationAttr app;
  SpecializationMap subst;
};

/// Verifies an impl's equality-armed witnesses and returns each as a local
/// resolution rule. Such a witness certifies that a sibling impl binds a
/// ground projection to a resolved type; a witness citing a conditional impl
/// is legal exactly when the impl's own where clause covers the cited impl's
/// assumptions or an application-armed witness supplies them. Each entry
/// verifies with an EMPTY equality modulus: sibling witnesses never serve as
/// each other's modulus, because an attribute array has no dominance and
/// mutual justification could ground a false equality on nothing. A witness
/// must name a GROUND projection -- reading a poly-carrying projection's
/// variable off one cited impl's concrete head would accept a generic impl on
/// the strength of one instance.
static FailureOr<SmallVector<ImplWitnessRule>> collectImplWitnessRules(
    ImplOp impl, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  SmallVector<ImplWitnessRule> rules;
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
    auto projectionTy = dyn_cast<ProjectionType>(witness.getProjection());
    if (!projectionTy)
      return impl.emitOpError()
             << "a witness must name a projection, found "
             << witness.getProjection();
    if (isPolymorphicType(witness.getProjection()))
      return impl.emitOpError()
             << "witness projection " << witness.getProjection()
             << " is not ground; a witness resolves only a "
                "ground sibling projection";
    auto subst = verifyProjectionResolutionAtImpl(
        module, witness, /*premises=*/{}, obligationPremises,
        dischargeWitnesses, errFn);
    if (failed(subst))
      return failure();
    auto citedImpl = mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(
        module, witness.getImplRef());
    rules.push_back(
        {citedImpl, projectionTy.getTraitApplication(), std::move(*subst)});
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
    ImplOp impl, ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto ownArguments = impl.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
  if (failed(ownArguments))
    return failure();

  NormalizationContext ctx;
  ctx.addLocalProjectionRule(impl, impl.getSelfApplication(), *ownArguments);
  for (const ImplWitnessRule &rule : witnessRules)
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
    ImplOp impl, TraitOp traitOp, func::FuncOp implMethod,
    ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  StringRef name = implMethod.getSymName();
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
    ImplOp impl, TraitOp traitOp, ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn);

static LogicalResult verifyImplParametersAreConstrained(ImplOp impl);

/// Verifies each associated type binding against the two lists a use of it
/// supplies arguments for: the impl header's parameters, bound where the impl is
/// selected, and the binding's own parameters, bound by a projection's
/// associated type arguments. A binding whose own parameter repeats a header
/// parameter would have the projection's argument overwrite the header's, and a
/// bound type mentioning a parameter from neither list has nothing to supply it,
/// so the resolved type would carry a parameter no substitution reaches.
///
/// A child's own invariants are verified after its parent's, so each entry is
/// read as an attribute that may be anything and refused where it stands rather
/// than cast.
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
      for (Attribute tyAttr : declaredParams) {
        auto typeAttr = dyn_cast<TypeAttr>(tyAttr);
        if (!typeAttr)
          return assoc.emitOpError()
                 << "type parameter list holds " << tyAttr << ", which is not a type";
        Type param = typeAttr.getValue();
        // A declared parameter may be a generic type another dialect wraps
        // around a label (a coordinate parameter carries the label it stands
        // for), and declaring it declares the label it carries.
        for (GenericTypeInterface inside : getTypeParametersIn(param)) {
          if (headerParams.contains(Type(inside)))
            return assoc.emitOpError()
                   << "type parameter " << param
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
  if (failed(verifyImplParametersAreConstrained(*this)))
    return failure();
  return verifyAssociatedTypeBindingScopes(*this);
}

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
  std::vector<func::FuncOp> optionalMethods = traitOp.getOptionalMethods();
  llvm::SmallSet<StringRef, 8> optionalMethodNames;
  for (auto f : optionalMethods) {
    optionalMethodNames.insert(f.getSymName());
  }

  // Verify methods and associated type bindings
  llvm::SmallSet<StringRef, 8> definedMethods;
  llvm::SmallSet<StringRef, 8> definedAssocTypes;
  for (Operation &op : getBody().front()) {
    if (auto implMethod = dyn_cast<func::FuncOp>(op)) {
      StringRef name = implMethod.getSymName();
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
      return emitOpError() << "body may only contain 'func.func' or 'trait.assoc_type' operations";
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

  return success();
}

/// Verifies the equality requirements the trait header states, specialized for
/// this impl's self arguments (e.g. Self::Output = Self).
///
/// A requirement is an obligation the impl owes, so the two endpoints must be
/// the same type: both are read through the impl's own bindings and its
/// declared witness rules, and whatever stays standing after that is equal to
/// itself alone. The impl's OWN where-clause equalities are not checked here --
/// they are premises restricting when the impl applies, discharged where it is
/// selected -- and application requirements are proved at selection too.
static LogicalResult verifyEqualityObligations(
    ImplOp impl, TraitOp traitOp, ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // The guard keeps the self-claim specialization off an impl with nothing of
  // the kind to check.
  bool hasEqualityRequirement = llvm::any_of(
      traitOp.getRequirements(), [](Attribute pred) {
        return mlir::isa<TypeEqualityAttr>(pred);
      });
  if (!hasEqualityRequirement)
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

/// Rust's constrained-parameter rule (E0207): every parameter an impl binds
/// must be determined by the application the impl is selected for.
///
/// A parameter standing only inside a projection is not determined -- a
/// projection is not injective, so two arguments can reach one resolution --
/// and a parameter standing nowhere in the self application is determined only
/// by a where-clause equality that pins it: bare on one side, with the other
/// side's parameters determined in turn. A parameter nothing determines would
/// leave the impl's methods and associated-type bindings spelling a variable
/// selection never assigns.
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

  // Then close over the where clause's equalities: a bare parameter on one side
  // is determined once every parameter on the other side is.
  SmallVector<TypeEqualityAttr> premises;
  for (Attribute pred : impl.getAssumptions())
    if (auto eq = dyn_cast<TypeEqualityAttr>(pred))
      premises.push_back(eq);
  for (bool grew = true; grew;) {
    grew = false;
    auto determines = [&](Type bare, Type other) {
      GenericTypeInterface parameter = getParameterOccurrence(bare);
      if (!parameter || constrained.contains(Type(parameter)))
        return;
      for (GenericTypeInterface inside : getTypeParametersIn(other))
        if (!constrained.contains(Type(inside)))
          return;
      constrained.insert(Type(parameter));
      grew = true;
    };
    for (TypeEqualityAttr eq : premises) {
      determines(eq.getLhs(), eq.getRhs());
      determines(eq.getRhs(), eq.getLhs());
    }
  }

  for (GenericTypeInterface parameter : impl.getTypeParams())
    if (!constrained.contains(Type(parameter)))
      return impl.emitOpError()
             << "type parameter " << Type(parameter)
             << " is not constrained by the impl's trait application, so impl "
                "selection cannot determine it";
  return success();
}

bool ImplOp::isUnconditional() {
  // an ImplOp is unconditional if:
  // 1. it is monomorphic (no type parameters),
  // 2. its TraitOp has no application requirements, and
  // 3. it assumes no application predicates.
  // An equality predicate -- whether a trait-header requirement or one of this
  // impl's own assumptions -- does not make an impl conditional: it is settled at
  // impl verification when its endpoints reduce to ground (through the impl's own bindings or
  // a declared premise) and deferred to selection and use otherwise, never proved
  // through impl selection. So only application predicates count against
  // unconditionality.
  return getTypeParams().empty() &&
         !getAssumptions().hasApplications() &&
         !getTrait().getRequirements().hasApplications();
}

LogicalResult ImplOp::verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isUnconditional()) {
    if (err) err() << "impl '@" << getSymName()
                   << "' is polymorphic (has type parameters) or has obligations (trait requirements or impl assumptions) and must be proven with a trait.proof";
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

  // A parameter the header leaves open is one the where clause determines: an
  // equality with that parameter bare on one side says what it is, once the
  // other side is instantiated at what is known and read through `normalize`.
  // Determining one can determine another, so the reading runs until it stops
  // growing.
  for (bool grew = true; grew;) {
    grew = false;
    for (Attribute predicate : getAssumptions()) {
      auto equality = dyn_cast<TypeEqualityAttr>(predicate);
      if (!equality)
        continue;
      SpecializationMap known = args.toSpecialization();
      auto determines = [&](Type bare, Type other) {
        GenericTypeInterface parameter = getParameterOccurrence(bare);
        if (!parameter || !args.binds(parameter) || args.lookup(parameter))
          return false;
        Type value = instantiate(other, known);
        if (normalize) {
          FailureOr<Type> normalized = normalize(value);
          if (failed(normalized))
            return false;
          value = *normalized;
        }
        // A value still mentioning a parameter this reading has not settled
        // says nothing yet; the round that settles that one settles this.
        for (GenericTypeInterface inside : getTypeParametersIn(value))
          if (args.binds(inside) && !args.lookup(inside))
            return false;
        if (failed(args.assign(parameter, value, /*err=*/nullptr)))
          return false;
        grew = true;
        return true;
      };
      if (!determines(equality.getLhs(), equality.getRhs()))
        determines(equality.getRhs(), equality.getLhs());
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
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto binding = getAssociatedTypeBinding(name, err);
  if (failed(binding)) return failure();

  auto assoc = getAssociatedType(name);
  if (succeeded(assoc) && assoc->getTypeParams()) {
    auto typeParams = *assoc->getTypeParams();
    if (typeParams.size() != assocTypeArgs.size()) {
      if (err) err() << "GAT arity mismatch for '" << name
                     << "': expected " << typeParams.size()
                     << " type args but got " << assocTypeArgs.size();
      return failure();
    }
    *binding = applyGATSubstitution(typeParams, assocTypeArgs, *binding);
  }

  return *binding;
}

FailureOr<ImplSpecialization> ImplOp::buildImplSpecialization(
    ClaimType provenSelfClaim,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!provenSelfClaim.isProven()) {
    if (err) err() << "expected proven self claim for " << getSymName();
    return failure();
  }

  auto module = getModule(err);
  if (failed(module)) return failure();

  EvidenceBindings evidence;

  // Bind the same self claim without a proof to the proven self claim. This
  // recursively records claim -> proven-claim evidence bindings.
  ClaimType unprovenSelfClaim = provenSelfClaim.asUnproven();
  if (failed(verifyAndRecordProof(unprovenSelfClaim, provenSelfClaim, *module,
                                  evidence, origin, memo, err)))
    return failure();

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
  auto specialization =
      buildSubstitutionForSelfClaim(provenSelfClaim, normalize, err);
  if (failed(specialization)) return failure();

  return ImplSpecialization(*specialization, evidence);
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

FailureOr<func::FuncOp> ImplOp::getOrSpecializeMethod(OpBuilder& builder, StringRef methodName) {
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

  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&getBody().front());
  auto specialized =
      specializePolymorph(builder, *traitMethod, methodName, subst->toTypeMap());
  // A default method with no body to clone is refused where the clone was
  // attempted; there is no method here to answer with.
  if (!specialized)
    return failure();
  return specialized;
}

/// Specialize a polymorphic function and replace any AssumeOps whose
/// trait application matches a claim-typed function parameter.
///
/// Null when the callee has no body to clone -- an external declaration --
/// which specialization has already refused with a diagnostic. Every caller
/// turns that into a failure rather than reading the instance.
static func::FuncOp specializeAndReplaceAssumes(
    PatternRewriter &rewriter, func::FuncOp callee,
    StringRef name, const DenseMap<Type,Type> &subst) {
  auto funcOp = specializePolymorph(rewriter, callee, name, subst);
  if (!funcOp)
    return nullptr;

  // A trait.assume materializes a hypothesis whose evidence, once the function
  // is specialized, is carried by a claim-typed parameter of the same arm: an
  // application assume is satisfied by an application parameter naming the same
  // trait application, an equality assume by an equality parameter carrying the
  // same equality.
  DenseMap<TraitApplicationAttr, Value> applicationParams;
  DenseMap<TypeEqualityAttr, Value> equalityParams;
  for (auto arg : funcOp.getArguments())
    if (auto claimTy = dyn_cast<ClaimType>(arg.getType())) {
      if (claimTy.isApplication())
        applicationParams[claimTy.getTraitApplication()] = arg;
      else if (auto eq = claimTy.getEqualityAttr())
        equalityParams[eq] = arg;
    }

  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    ClaimType claim = a.getClaim();
    Value replacement;
    if (auto eq = claim.getEqualityAttr()) {
      auto it = equalityParams.find(eq);
      if (it != equalityParams.end())
        replacement = it->second;
    } else {
      auto it = applicationParams.find(claim.getTraitApplication());
      if (it != applicationParams.end())
        replacement = it->second;
    }
    if (replacement) {
      rewriter.replaceAllUsesWith(a.getResult(), replacement);
      toErase.push_back(a);
    }
  });
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

static func::FuncOp specializeMethodAsFreeFuncWithLeadingSelfProof(
    PatternRewriter& rewriter,
    ModuleOp module,
    func::FuncOp method,
    StringRef functionName,
    ClaimType selfProofTy,
    const DenseMap<Type,Type>& subst,
    const DenseMap<Type,Type>& implSubst) {

  // specialize the method into the grandparent with a mangled name
  PatternRewriter::InsertionGuard guard(rewriter);

  // clone the method into the method's grandparent
  rewriter.setInsertionPointAfter(method->getParentOp());

  // specialize the function and replace assumes matching claim-typed parameters
  auto funcOp = specializeAndReplaceAssumes(rewriter, method, functionName, subst);
  if (!funcOp)
    return nullptr;

  // The clone leads with the proven self claim the call carries: the self is
  // ground and the impl's proof names it, in a template clone (a method with its
  // own free generic) as in a monomorphic one. The clone's projections from that
  // self to the impl's own obligations are spelled proven by the impl's proof --
  // a fact of the impl fixed once the self is ground, not a call-site evidence
  // binding -- and its assumed equalities project to the impl's equality
  // where-clauses, so no assumption rides as a lifted claim parameter.
  rewriter.modifyOpInPlace(funcOp, [&] {
    (void)funcOp.insertArgument(/*idx=*/0, selfProofTy,
                               /*argAttrs=*/mlir::DictionaryAttr(),
                               method.getLoc());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  });
  BlockArgument selfProofArg = funcOp.getArgument(0);

  // Read a proven claim spelling from the IMPL's own specialization: its
  // evidence bindings map each of the impl's obligations (an unproven claim) to
  // the proven claim discharging it, a fact of the impl fixed once the self is
  // ground, so an application assume projects to the proven obligation. The call
  // site's evidence bindings are deliberately not read here: a template clone is
  // the template under variable bindings alone, and stamping one call's evidence
  // over it would bind the clone to that call. An equality assume and any
  // obligation the impl's specialization does not record fall back to the
  // assume's own claim -- an equality never carries a proof, and a monomorphic
  // clone's body already carries the proven spelling from full substitution.
  auto provenOrSame = [&](ClaimType claim) -> ClaimType {
    auto it = implSubst.find(claim);
    if (it != implSubst.end())
      if (auto proven = dyn_cast<ClaimType>(it->second))
        return proven;
    return claim;
  };

  // Replace every remaining AssumeOp with a projection from the self proof: an
  // application assume to the proven obligation, an equality assume to the
  // impl's equality where-clause. Both are candidate projections of the proven
  // self, so the clone holds no trait.assume an AssumeOp verifier would refuse
  // at module scope.
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(a);

    Value replacement = ProjectOp::create(
      rewriter,
      a.getLoc(),
      provenOrSame(a.getClaim()),
      selfProofArg
    );

    rewriter.replaceAllUsesWith(a.getResult(), replacement);
    toErase.push_back(a);
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  // A template clone (one whose method generics are still free) is stamped under
  // the variable bindings alone, so a ground claim its body derives from the
  // proven self -- one hop past the projections replaced above, e.g. a
  // requirement of an assumed application -- stays spelled unproven. A projection
  // from the now-proven source to that unproven claim fails proofness parity. The
  // impl's proof settles every such ground claim once the self is ground, so
  // respell them here from the IMPL's evidence bindings alone -- the same fact
  // of the impl, never a call site's -- over the whole body at once. A
  // monomorphic clone already carries the proven spelling from full
  // substitution, so this is an identity there.
  llvm::DenseMap<Type, Type> evidenceRespell;
  for (auto [key, value] : implSubst)
    if (isa<ClaimType>(key))
      if (auto proven = dyn_cast<ClaimType>(value))
        if (proven.isProven())
          evidenceRespell.try_emplace(key, value);
  if (!evidenceRespell.empty()) {
    AttrTypeReplacer replacer =
        makeTypeReplacerFromSubstitution(evidenceRespell, ModuleOp());
    replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }

  return funcOp;
}

FailureOr<func::FuncOp> ImplOp::getOrSpecializeFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType provenSelfClaim,
    StringRef methodName,
    const CallSubstitution &callSubst,
    ProofDerivationMemo *memo) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return failure();

  // The enclosing module: where a clone of the method is cut and where an
  // existing clone is looked up.
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();

  auto method = getOrSpecializeMethod(rewriter, methodName);
  if (failed(method)) return failure();

  auto implSpec =
      buildImplSpecialization(provenSelfClaim, DemandOrigin::ProofRecording,
                              memo);
  if (failed(implSpec)) return failure();

  // Build the same substitution that will be used to clone the method body:
  // first the enclosing impl substitution, then method-generic bindings from
  // this call site. The impl's own half is kept apart: the proven spellings a
  // clone receives are facts of the impl, so they are read from it and never
  // from the call.
  DenseMap<Type,Type> implSubst = implSpec->toTypeMap();
  DenseMap<Type,Type> subst = implSubst;

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

  // The extracted function name must include every substitution used to clone
  // the method body; otherwise different method-generic calls share a symbol.
  auto functionName = generateMangledName(implSpec->getSpecialization()) + "_" +
    methodName.str() +
    applySubstitutionAndGenerateMangledNameSuffix(subst, getTypeParametersIn((*method).getFunctionType()));

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
    module,
    FlatSymbolRefAttr::get(ctx, functionName)
  );

  if (!funcOp) {
    // specialize into grandparent with mangled name
    funcOp = specializeMethodAsFreeFuncWithLeadingSelfProof(
      rewriter,
      module,
      *method,
      functionName,
      provenSelfClaim,
      subst,
      implSubst
    );
    // A method with no body to clone is refused where the clone was attempted;
    // this call has no instance to name.
    if (!funcOp)
      return failure();
  }

  return funcOp;
}

/// Generate a deterministic symbol name for an ImplOp.
/// 
/// The name has the form {TraitName}_impl_h{hash} where the hash is a
/// 64-bit xxHash of the full type argument and assumption signature. This
/// keeps symbols short and bounded in length.
std::string ImplOp::generateSymName(TraitApplicationAttr selfApp,
                                    PredicateArrayAttr assumptions) {
  // Build the full type-argument and where-clause signature for hashing. The
  // application entries hash exactly as an application-only impl always has, so
  // generalizing the where clause to carry equalities never perturbs the
  // synthesized name of an impl that assumes only applications. Equality entries
  // then contribute their own entropy, so two impls that differ only in an
  // equality assumption synthesize distinct names.
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
  if (assumptions) {
    bool firstEquality = true;
    for (Attribute pred : assumptions) {
      auto eq = dyn_cast<TypeEqualityAttr>(pred);
      if (!eq) continue;
      if (firstEquality) {
        os << "_eq";
        firstEquality = false;
      }
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
  // only; equality entries are checked at impl verification against the impl's own bindings and
  // never proved through impl selection, so they are filtered out here at the
  // one place every obligation consumer flows through.
  return llvm::map_to_vector(getAssumptions().getApplications(),
                             [ctx](TraitApplicationAttr app) {
    return ClaimType::get(ctx, app);
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeAssumptionsAsClaimsFor(
    ClaimType actualSelfClaim,
    Normalizer normalize,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // The arguments actualSelfClaim supplies for this impl's parameters, read off
  // it by position; `normalize` is the caller's established context, which is
  // what a parameter the header leaves open and the where clause determines is
  // read through. Reading is all this does: whether the header carries to the
  // claim is settled where the impl was matched to it -- at selection, or at the
  // verifier of the proof or derive citing it -- and remaking that verdict here
  // would remake it under whatever context stands at the reading instead.
  SpecializationMap subst =
      readTypeArgumentsFor(actualSelfClaim, normalize).toSpecialization();

  // apply the substitution to each assumption. As with a trait's requirements,
  // a substitution rewrites the type arguments a claim carries and never the
  // claim wrapper (its keys are never a whole ClaimType), so the result is
  // always a claim; the cast holds structurally even on unverified IR.
  return llvm::map_to_vector(getAssumptionsAsClaims(), [&](ClaimType assumption) {
    ClaimType specializedAssumption = dyn_cast_or_null<ClaimType>(instantiate(assumption, subst));
    if (!specializedAssumption)
      llvm_unreachable("ImplOp::specializeAssumptionsAsClaimsFor: expected ClaimType");
    return specializedAssumption;
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeObligationsAsClaimsFor(
    ClaimType actualSelfClaim,
    DemandOrigin origin,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // A parameter the header leaves open and the where clause determines is read
  // through the impls the module holds; `origin` names that reading.
  // XXX TODO a projection a declaration spells must be over its own self
  // application, a where-clause application, a trait requirement or a declared
  // witness (Rust's projection well-formedness rule), so every projection has
  // evidence at a known index and this module read deletes with LookupScope and
  // the verifier DemandOrigins.
  GroundProjectionLookup byGroundLookup(*module, origin);

  // specialize requirements of the trait
  auto requirements = getTrait().specializeRequirementsAsClaimsFor(actualSelfClaim, errFn);
  if (failed(requirements)) return failure();

  // The obligation stream is proved and derived through impl selection, an
  // application-arm operation. Trait-header equality requirements are checked
  // at impl verification against the impl's own bindings, never proved here, so they
  // do not enter the obligation stream (the proof/derive zips would have no
  // subproof for them).
  llvm::erase_if(*requirements, [](ClaimType c) { return c.isEquality(); });

  // Resolve projections in requirements using this impl's associated type
  // bindings (e.g., `Coord[Tensor[Self]::Shape]` becomes `Coord[tuple<i64,i64>]`
  // when the impl binds `Shape = S` and S is specialized to tuple<i64,i64>).
  // Only projections over this impl's own (actual) trait application resolve
  // through its bindings; a projection over a different trait application that
  // merely shares an associated-type name stays symbolic.
  //
  // The arguments are read off the claim by position, as the assumptions'
  // are: whether the header carries to the claim is settled where the impl was
  // matched to it, not here.
  SpecializationMap subst =
      readTypeArgumentsFor(actualSelfClaim, byGroundLookup).toSpecialization();

  NormalizationContext normalization;
  normalization.addLocalProjectionRule(
      *this, actualSelfClaim.getTraitApplication(), subst);
  for (ClaimType &req : *requirements) {
    auto resolved = normalization.normalize(req, errFn);
    if (failed(resolved)) return failure();
    req = cast<ClaimType>(*resolved);
  }

  // specialize assumptions of the impl
  auto assumptions =
      specializeAssumptionsAsClaimsFor(actualSelfClaim, byGroundLookup, errFn);
  if (failed(assumptions)) return failure();

  // obligations = requirements + assumptions
  SmallVector<ClaimType> obligations = std::move(*requirements);
  obligations.append(std::move(*assumptions));

  return obligations;
}

ParseResult ImplOp::parse(OpAsmParser &p, OperationState &result) {
  // optional `private`/`nested` keyword before the name, as func.func reads it;
  // a missing keyword is not an error (the op is public) and consumes nothing.
  (void)mlir::impl::parseOptionalVisibilityKeyword(p, result.attributes);

  // parse optional symbolic name: trait.impl @Sym
  StringAttr parsedSymName;
  (void)p.parseOptionalSymbolName(parsedSymName);

  // parse mandatory for
  if (p.parseKeyword("for"))
    return failure();
  
  // parse @TraitName[Types...]
  TraitApplicationAttr selfApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!selfApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("self_application", selfApp);  
  
  // where clause: one mixed PredicateArrayAttr (application and equality arms in
  // declaration order), stored directly as $assumptions -- no second array and
  // no partition. An application-only clause parses as a plain array of trait
  // applications.
  auto assumptions = PredicateArrayAttr::get(p.getContext(), ArrayRef<Attribute>{});
  if (succeeded(p.parseOptionalKeyword("where"))) {
    assumptions = dyn_cast_or_null<PredicateArrayAttr>(PredicateArrayAttr::parse(p, {}));
    if (!assumptions)
      return p.emitError(p.getCurrentLocation(), "expected a PredicateArrayAttr");
  }
  result.addAttribute("assumptions", assumptions);

  // Optional witnesses: one array of #trait.witness entries, each an
  // equality-armed projection-resolution witness or an application-armed
  // obligation discharge. Absent, the printed and parsed form is byte-identical
  // to an impl without them; the synthesized sym_name reads only the self
  // application and assumptions, so witnesses never perturb it.
  if (succeeded(p.parseOptionalKeyword("witnesses"))) {
    ArrayAttr witnesses;
    if (p.parseAttribute(witnesses))
      return failure();
    result.addAttribute("witnesses", witnesses);
  }

  // sym_name: use parsed or synthesize from parameters
  StringAttr symNameAttr = parsedSymName
    ? parsedSymName
    : p.getBuilder().getStringAttr(generateSymName(selfApp, assumptions));
  result.addAttribute("sym_name", symNameAttr);
  
  // Parse attributes and body region
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
    
  Region* bodyRegion = result.addRegion();
  if (p.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  
  // Ensure the region has exactly one block (matching builder logic)
  if (bodyRegion->empty())
    bodyRegion->emplaceBlock();
    
  return success();
}

void ImplOp::print(OpAsmPrinter &printer) {
  // decide whether to print the symbolic name
  StringAttr symNameAttr = getSymNameAttr();
  std::string synthesized = generateSymName(getSelfApplication(), getAssumptions());
  bool printExplicitSymName = symNameAttr && symNameAttr.getValue() != synthesized;

  // Print: trait.impl [private] [@SymName] for @TraitName [types...] assumptions { ... }
  if (auto vis = getSymVisibilityAttr())
    if (vis.getValue() != "public")
      printer << ' ' << vis.getValue();
  printer << " ";
  if (printExplicitSymName) {
    printer.printSymbolName(symNameAttr);
    printer << " ";
  }

  printer << "for ";
  getSelfApplication().print(printer);

  // print assumptions if not empty
  if (!getAssumptions().empty()) {
    printer << "where ";
    getAssumptions().print(printer);
  }

  // print witnesses if present and non-empty
  if (ArrayAttr witnesses = getWitnessesAttr()) {
    if (!witnesses.empty()) {
      printer << "witnesses ";
      printer.printAttribute(witnesses);
    }
  }

  printer.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"sym_name", "self_application", "assumptions", "witnesses", "sym_visibility"}
  );
  printer << " ";
  printer.printRegion(getBody());
}


/// Refuses a citation of `impl` at `cited` whose equality premises do not hold
/// there.
///
/// An equality premise restricts where the impl applies, and only the
/// application being cited says whether it holds. Each side is read through the
/// arguments that application supplies, then through the impl's own
/// associated-type bindings for it -- a premise may project through the very
/// application being cited -- and then through `evidence`, the citation's own
/// context. Identity after that reading is the whole judgment, and it is the
/// one impl selection makes over a candidate: the impl's application-arm
/// premises travel as subproofs, its equality premises are decided here.
///
/// A reading carrying a type variable is a premise this citation cannot decide:
/// a template's variables stand for the instances made of it, and the instance
/// is where the premise is read. A symbolic equality defers here for the reason
/// it defers at the impl that states it, and what it defers to is the clone,
/// which reads it at the arguments the instance supplies.
static LogicalResult verifyEqualityPremisesHoldAt(
    ImplOp impl, ClaimType cited, const SpecializationMap &arguments,
    NormalizationContext evidence,
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
    if (premiseDefersToInstances(*lhs, *rhs))
      continue;
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

//===----------------------------------------------------------------------===//
// ProofOp
//===----------------------------------------------------------------------===//

LogicalResult ProofOp::verify() {
  if (failed(verifyTemplateIsNotPublic(getOperation())))
    return failure();

  // check that every name is a FlatSymbolRefAttr
  for (Attribute name : getSubproofNames()) {
    if (!isa<FlatSymbolRefAttr>(name)) {
      return emitOpError() << "'subproof_names' must contain only FlatSymbolRefAttr elements";
    }
  }
  return success();
}

LogicalResult ProofOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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

  // The impl's header must carry to the claim this proof stands over. What a
  // projection that header spells reduces through is the evidence this proof
  // holds: the proofs discharging the impl's own obligations, by index, and
  // then the impls the module holds. The proof's own rule is not among them --
  // nothing here is justified by what it is checking.
  NormalizationContext reading;
  if (spellsAProjection(Type(implOp.getSelfClaim())) ||
      !implOp.getAssumptions().getEqualities().empty())
    reading = buildSubproofNormalizationContext(*this, module);
  // XXX TODO a projection a declaration spells must be over its own self
  // application, a where-clause application, a trait requirement or a declared
  // witness (Rust's projection well-formedness rule), so every projection has
  // evidence at a known index and this module read deletes with LookupScope and
  // the verifier DemandOrigins.
  reading.setModuleLookup(module, LookupScope::Ground,
                          DemandOrigin::ProofVerification);
  auto throughEvidence = [&](Type ty) -> FailureOr<Type> {
    return reading.normalize(ty, errFn);
  };
  auto arguments = implOp.buildSubstitutionForSelfClaim(getProvenClaim(),
                                                        throughEvidence, errFn);
  if (failed(arguments))
    return failure();

  // The impl's equality premises stand over this claim too. They take no
  // subproof -- the given list is indexed by the impl's application-arm
  // obligations -- so a proof that did not read them stood over an impl that
  // does not apply, and only a use of the claim far downstream said so.
  if (failed(verifyEqualityPremisesHoldAt(implOp, getProvenClaim(), *arguments,
                                          reading, errFn)))
    return failure();

  // recursively verify proof structure and that proof bindings can be recorded.
  // A verifier runs on whatever thread the verification was handed to and holds
  // no memo, so it derives what it needs itself.
  EvidenceBindings evidence;
  if (failed(verifyAndRecordProof(getProvenClaim().asUnproven(),
                                  getProvenClaim(), module, evidence,
                                  DemandOrigin::ProofVerification,
                                  /*memo=*/nullptr, errFn)))
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
    DemandOrigin origin, llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<ClaimType> result;

  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  if (!module) {
    if (err) err() << "not in a module";
    return failure();
  }

  // Compute obligations so we can validate coinductive self-references
  // and check arity.
  auto implOp = getImpl();
  if (!implOp) {
    if (err) err() << "cannot find impl '" << getImplNameAttr() << "'";
    return failure();
  }

  auto obligations =
      implOp.specializeObligationsAsClaimsFor(getProvenClaim(), origin, err);
  if (failed(obligations)) return failure();

  ArrayAttr subproofNames = getSubproofNames();
  if (subproofNames.size() != obligations->size()) {
    if (err) err() << "arity mismatch: expected " << obligations->size()
                   << " subproofs, but found " << subproofNames.size();
    return failure();
  }

  for (Attribute name : subproofNames) {
    auto subproofRef = dyn_cast<FlatSymbolRefAttr>(name);
    if (!subproofRef) {
      if (err) err() << "expected FlatSymbolRefAttr";
      return failure();
    }

    // A coinductive self-citation needs no arm of its own: looking the name up
    // finds this proof, whose claim is the one a self-citation stands for, and
    // whether that claim discharges the obligation is the same comparison every
    // other citation answers.
    auto subproof = getProofOpOrUnconditionalImplOp(module, subproofRef, err);
    if (failed(subproof))
      return failure();

    TraitApplicationAttr subproofTraitApp;
    if (auto proofOp = dyn_cast<ProofOp>(*subproof))
      subproofTraitApp = proofOp.getTraitApplication();
    else
      subproofTraitApp = dyn_cast<ImplOp>(*subproof).getSelfApplication();

    result.push_back(ClaimType::get(getContext(), subproofTraitApp, subproofRef));
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
  Operation* symOp = SymbolTable::lookupNearestSymbolFrom(module, name);
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
  // by @impl [given(%premises...) : (types...)] : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("proj_resolve"))) {
    Type projection, resolved;
    FlatSymbolRefAttr citedImpl;
    if (p.parseType(projection) || p.parseKeyword("resolves") ||
        p.parseType(resolved) || p.parseKeyword("by") ||
        p.parseAttribute(citedImpl))
      return failure();
    auto err = [&] { return p.emitError(p.getCurrentLocation()); };
    auto equality = TypeEqualityAttr::getChecked(err, ctx, projection, resolved);
    if (!equality)
      return failure();
    auto witness = WitnessAttr::getChecked(err, ctx, Attribute(equality), citedImpl);
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

  // parse @Symbol
  FlatSymbolRefAttr proof;
  if (p.parseAttribute(proof, "proof", result.attributes))
    return failure();

  // parse `for`
  if (p.parseKeyword("for"))
    return failure();

  // parse @Trait[Types...]
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("trait_application", traitApp);

  // construct the result type
  ClaimType claimTy = ClaimType::get(p.getContext(), traitApp, proof);
  result.addTypes(claimTy);

  // parse additional attributes
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void WitnessOp::print(OpAsmPrinter &p) {
  if (auto witness = getWitnessAttr()) {
    p << " proj_resolve " << witness.getProjection() << " resolves "
      << witness.getResolved() << " by " << witness.getImplRef();
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
  p << " " << getProofAttr() << " for ";
  getTraitApplicationAttr().print(p);

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"proof", "trait_application", "witness", "refl"}
  );
}

// The op's attributes must match the result claim's arm exactly, and the result
// type must equal the claim reconstructed from those attributes. For the
// equality arm, the current endpoints must be a single-substitution structural
// instance of the witness's endpoints (proj-resolve), identical (refl), or
// entailed by the premises' ground congruence closure (compose).
LogicalResult WitnessOp::verify() {
  ClaimType result = dyn_cast<ClaimType>(getResult().getType());
  if (!result)
    return emitOpError() << "result must be a !trait.claim";

  bool hasProof = static_cast<bool>(getProofAttr());
  bool hasApp = static_cast<bool>(getTraitApplicationAttr());
  bool hasWitness = static_cast<bool>(getWitnessAttr());
  bool hasRefl = getRefl();

  // Equality arm.
  if (result.isEquality()) {
    if (hasProof || hasApp)
      return emitOpError() << "an equality witness carries no proof or trait "
                              "application";
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
      // proj-resolve: the current endpoints must be a single-substitution
      // structural instance of the witness's endpoints. The witness's
      // generic parameters are the variables; a single substitution must carry
      // the witness's projection and resolved type to the current pair. This
      // passes impl verification (identity), the clone-substituted state, and ground, and
      // rejects any non-substitution mangling. It is structural and local -- no
      // module lookup -- so the pair is matched with a null module.
      WitnessAttr witness = getWitnessAttr();
      // The witness slot carries a proj-resolve leaf, so its predicate is an
      // equality; a coerce discharge's application-headed witness has no place
      // here. Guard before reading the endpoints off the equality.
      if (!isa<TypeEqualityAttr>(witness.getPredicate()))
        return emitOpError() << "a proj-resolve witness must carry an "
                                "equality";
      MLIRContext *ctx = getContext();
      Type witnessPair = TupleType::get(ctx, {witness.getProjection(), witness.getResolved()});
      Type currentPair = TupleType::get(ctx, {eq.getLhs(), eq.getRhs()});
      if (failed(matchDeclaration(getTypeParametersIn(witnessPair), witnessPair,
                                  currentPair, /*normalize=*/Normalizer(),
                                  /*err=*/nullptr)))
        return emitOpError() << "result endpoints " << eq.getLhs() << " = "
                             << eq.getRhs()
                             << " are not an instance of the witness "
                             << witness.getProjection() << " = " << witness.getResolved();
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
  if (!hasProof || !hasApp)
    return emitOpError() << "an application witness carries a proof and a "
                            "trait application";
  if (result != getProvenClaim())
    return emitOpError() << "result " << result
                         << " does not match the witnessed claim "
                         << getProvenClaim();
  return success();
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  // Equality proj-resolve arm: verify the citation where its symbol uses are
  // checked. The cited impl must bind the associated type the witness's
  // projection names to its resolved type, once specialized for the projection's
  // trait application, AND the witness's premises must discharge the cited impl's
  // own assumptions. The premises split by arm: equality claims are the
  // comparison modulus, application claims discharge the assumptions. The module
  // read runs here, on every full module verification -- not per consumer --
  // through the same obligation-aware check the C-API projection-resolution query
  // runs in obligation mode, so a consumer classifying a witness cannot
  // disagree with this verdict.
  if (auto witness = getWitnessAttr()) {
    SmallVector<TypeEqualityAttr> equalityPremises;
    SmallVector<TraitApplicationAttr> applicationPremises;
    for (Value premise : getPremises())
      if (auto claim = dyn_cast<ClaimType>(premise.getType())) {
        if (auto eq = claim.getEqualityAttr())
          equalityPremises.push_back(eq);
        else if (claim.isApplication())
          applicationPremises.push_back(claim.getTraitApplication());
      }
    // The op's current result equality is the instance the discharge check
    // carries the stored evidence's assumptions to, so a clone's premises match
    // at the clone's spelling rather than the stored one.
    return verifyProjectionResolutionAtUse(module, witness, equalityPremises,
                                           applicationPremises, errFn,
                                           getResultClaim().getEqualityAttr());
  }

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
  auto cited = ProofOp::getProofOpOrUnconditionalImplOp(module, getProofAttr(),
                                                        errFn);
  if (failed(cited)) return failure();
  auto proof = dyn_cast<ProofOp>(*cited);
  ImplOp impl = proof ? proof.getImpl() : cast<ImplOp>(*cited);
  if (!impl)
    return emitOpError() << "proof '" << getProofAttr()
                         << "' does not resolve to an impl";

  // As at a proof: a projection the impl's header spells reduces through the
  // evidence the witnessed claim names -- the proof tree it carries, by index
  // -- and then through the impls the module holds.
  NormalizationContext reading;
  if (spellsAProjection(Type(impl.getSelfClaim())))
    reading = buildProofNormalizationContext(getProvenClaim(), module);
  // XXX TODO a projection a declaration spells must be over its own self
  // application, a where-clause application, a trait requirement or a declared
  // witness (Rust's projection well-formedness rule), so every projection has
  // evidence at a known index and this module read deletes with LookupScope and
  // the verifier DemandOrigins.
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
      return errFn() << "the proof " << getProofAttr()
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
  return failed(subst) ? failure() : success();
}


//===----------------------------------------------------------------------===//
// DeriveOp
//===----------------------------------------------------------------------===//

ParseResult DeriveOp::parse(OpAsmParser &p, OperationState &result) {
  // trait.derive @Trait[Types...] from @Impl given(%claims...)

  // parse @Trait[Types...]
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("trait_application", traitApp);

  // parse `from`
  if (p.parseKeyword("from"))
    return failure();

  // parse @Impl
  FlatSymbolRefAttr implRef;
  if (p.parseAttribute(implRef, "impl", result.attributes))
    return failure();

  // parse `given`
  if (p.parseKeyword("given"))
    return failure();

  // parse ( %claims... )
  SmallVector<OpAsmParser::UnresolvedOperand> assumptions;
  if (p.parseOperandList(assumptions, OpAsmParser::Delimiter::Paren))
    return failure();

  // parse `: (` type_list `)`
  SmallVector<Type> assumptionTypes;
  if (!assumptions.empty()) {
    if (p.parseColon())
      return failure();
    if (failed(p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&] {
          Type ty;
          if (p.parseType(ty)) return failure();
          assumptionTypes.push_back(ty);
          return success();
        })))
      return failure();

    if (assumptionTypes.size() != assumptions.size())
      return p.emitError(p.getCurrentLocation(), "assumption type count mismatch");

    auto loc = p.getCurrentLocation();
    if (p.resolveOperands(assumptions, assumptionTypes, loc, result.operands))
      return failure();
  }

  // construct the unproven result type
  ClaimType claimTy = ClaimType::get(p.getContext(), traitApp);
  result.addTypes(claimTy);

  // parse optional attributes
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void DeriveOp::print(OpAsmPrinter &p) {
  // trait.derive @Trait[Types...] from @Impl given(%claims...)

  p << " ";
  getTraitApplication().print(p);
  p << " from " << getImplAttr() << " given(";
  llvm::interleaveComma(getAssumptions(), p, [&](Value v) {
    p.printOperand(v);
  });
  p << ")";

  // print types if there are assumptions
  if (!getAssumptions().empty()) {
    p << " : (";
    llvm::interleaveComma(getAssumptions().getTypes(), p, [&](Type ty) {
      p.printType(ty);
    });
    p << ")";
  }

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"trait_application", "impl"}
  );
}

ImplOp DeriveOp::getImplOp() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;
  return mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, getImplAttr());
}

/// Verifies that a trait.derive op is well-formed with respect to its symbols:
///
///  1. The @impl symbol resolves to a trait.impl op.
///  2. The impl's self application can be specialized against the derived claim
///     (i.e., the impl's header structurally matches the claim we want to derive).
///  3. The number of assumption operands equals the impl's assumption count
///     after specialization.
///  4. Each operand's claim type matches the corresponding specialized
///     assumption (so the caller is providing exactly the evidence the impl
///     requires under this specialization).
LogicalResult DeriveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };

  // A trait.derive discharges the cited impl's application-arm assumptions, so
  // every assumption operand must be a trait-application claim. An equality
  // claim carries no application to match and is not a legal derive operand.
  for (auto [i, operand] : llvm::enumerate(getAssumptions())) {
    auto operandClaim = cast<ClaimType>(operand.getType());
    if (!operandClaim.isApplication())
      return emitOpError() << "assumption operand #" << i << " (" << operandClaim
                           << ") must be a trait-application claim; an equality "
                              "claim is not a legal trait.derive operand";
  }

  // look up impl by symbol
  auto implOp = getImplOp();
  if (!implOp)
    return emitOpError() << "cannot find trait.impl '" << getImplAttr() << "'";

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  // The evidence this derive holds: the claims its given operands carry, read
  // by index through their own impls and proof trees. An impl whose header
  // forwards an associated type spells a projection the derive's demand spells
  // through the base, and this is what reduces the two to one grade.
  NormalizationContext normalization =
      buildLocalClaimNormalizationContext(getOperation(), getAssumptions(),
                                          module);
  // XXX TODO A claim operand that is neither proven nor derived carries no
  // impl, so an impl whose header forwards through such an operand's own
  // application has nothing here to reduce it. The module's impls stand in
  // until such an operand carries the impl serving it; see setModuleLookup.
  normalization.setModuleLookup(module, LookupScope::Determined);
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return normalization.normalize(ty, errFn);
  };

  // build substitution: impl's self claim -> derived claim
  ClaimType derivedClaim = getDerivedClaim();
  auto subst =
      implOp.buildSubstitutionForSelfClaim(derivedClaim, normalize, errFn);
  if (failed(subst))
    return failure();

  // specialize impl's assumptions for the derived claim
  SmallVector<ClaimType> specializedAssumptions =
      llvm::map_to_vector(implOp.getAssumptionsAsClaims(), [&](ClaimType a) {
        return cast<ClaimType>(instantiate(Type(a), *subst));
      });

  // check operand count matches assumption count
  if (getAssumptions().size() != specializedAssumptions.size())
    return emitOpError() << "expected " << specializedAssumptions.size()
                         << " assumption operands, got " << getAssumptions().size();

  // check each operand's claim type matches the corresponding specialized assumption
  for (auto [i, pair] : llvm::enumerate(llvm::zip(getAssumptions(), specializedAssumptions))) {
    auto [operand, expected] = pair;
    ClaimType operandClaim = cast<ClaimType>(operand.getType());
    if (operandClaim.getTraitApplication() != expected.getTraitApplication())
      return emitOpError() << "assumption operand #" << i
                           << " has claim " << operandClaim
                           << " but expected " << expected;
  }

  // The impl's equality premises take no operand -- the operand list is indexed
  // by its application-arm assumptions -- so they are read here, through the
  // same context: the hypotheses the scope holds and the evidence the operands
  // carry. A premise neither settles is a premise this derive does not meet,
  // the judgment selection makes over the same impl.
  if (failed(verifyEqualityPremisesHoldAt(implOp, derivedClaim, *subst,
                                          normalization, errFn)))
    return failure();

  return success();
}


//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

ParseResult AssumeOp::parse(OpAsmParser &p, OperationState &st) {
  MLIRContext *ctx = p.getContext();

  // `@Trait[...]` is an application hypothesis; `!A = !B` is an equality
  // hypothesis. The claim's result type wraps whichever predicate is parsed.
  FailureOr<Attribute> pred = parseApplicationOrEqualityPredicate(p);
  if (failed(pred))
    return failure();
  if (auto app = dyn_cast<TraitApplicationAttr>(*pred))
    st.addTypes(ClaimType::get(ctx, app));
  else
    st.addTypes(ClaimType::getEquality(ctx, cast<TypeEqualityAttr>(*pred)));

  return success();
}

void AssumeOp::print(OpAsmPrinter &p) {
  p << " ";

  ClaimType claim = getClaim();
  if (auto eq = claim.getEqualityAttr()) {
    // equality arm: `!lhs = !rhs`
    p << eq.getLhs() << " = " << eq.getRhs();
    return;
  }

  // application arm: print the assumed trait application
  claim.getTraitApplication().print(p);
}

LogicalResult AssumeOp::verify() {
  // verify line-of-sight between trait.assume op its enclosing function-like op so
  // that we are able to replace uses of trait.assume with a function parameter
  Operation* isolatedAncestor = getOperation()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  if (!isolatedAncestor)
    return emitOpError("must be within an IsolatedFromAbove region");

  // the isolated ancestor must be a FuncOp
  auto funcOp = dyn_cast<func::FuncOp>(isolatedAncestor);
  if (!funcOp)
    return emitOpError() << "must be within a 'func.func', found "
                         << isolatedAncestor->getName();

  ClaimType claim = getClaim();
  TraitOp enclosingTrait = funcOp->getParentOfType<TraitOp>();
  ImplOp enclosingImpl = funcOp->getParentOfType<ImplOp>();

  // An assumed predicate is an axiom of the enclosing scope exactly when it
  // matches one by identity -- a method body shares the enclosing declaration's
  // polymorphic variables, so no weaker match is accepted. Application and
  // equality predicates are disjoint attribute kinds, so one set serves both
  // arms: an equality assume can match only an equality entry, an application
  // assume only an application entry. The sources are the enclosing function's
  // claim parameters, the enclosing impl's assumptions, the enclosing trait's
  // equality requirements, and -- anchoring an application assume as the impl's
  // assumption list anchors an equality one -- the enclosing trait's and impl's
  // own self-applications.
  DenseSet<Attribute> assumable;
  for (Type argType : funcOp.getArgumentTypes())
    if (auto c = dyn_cast<ClaimType>(argType)) {
      if (auto eq = c.getEqualityAttr())
        assumable.insert(eq);
      else if (c.isApplication())
        assumable.insert(c.getTraitApplication());
    }
  if (enclosingImpl) {
    assumable.insert(enclosingImpl.getSelfApplication());
    for (Attribute pred : enclosingImpl.getAssumptions())
      assumable.insert(pred);
  }
  if (enclosingTrait) {
    assumable.insert(enclosingTrait.getSelfApplication());
    for (Attribute pred : enclosingTrait.getRequirements())
      if (isa<TypeEqualityAttr>(pred))
        assumable.insert(pred);
  }

  if (auto assumedEq = claim.getEqualityAttr()) {
    if (!assumable.contains(assumedEq))
      return emitOpError() << "assumed equality " << assumedEq
                           << " is not assumable in this context";
    return success();
  }

  auto assumedApp = getTraitApplication();
  if (!assumable.contains(assumedApp))
    return emitOpError() << "assumed trait application " << assumedApp
                         << " is not assumable in this context";
  return success();
}

TraitOp AssumeOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("AssumeOp:getTrait: not inside of a module");
  return getTraitApplication().getTraitOrAbort(module, "AssumeOp::getTrait: couldn't find trait");
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

  if (getUnproven()) {
    // The marked form cites nothing: its reconciling equalities are minted only
    // at monomorphization, so the endpoints stand in the pending judgment.
    if (!getEqualities().empty())
      return emitOpError()
             << "an unproven coerce may not cite equalities; it stands in a "
                "pending judgment discharged at monomorphization";
    if (failed(verifyPendingCoerceEndpoints(
            input, result, [&]() -> InFlightDiagnostic { return emitOpError(); })))
      return failure();
  } else {
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
  }

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

FailureOr<func::FuncOp> MethodCallOp::getMethod(llvm::function_ref<InFlightDiagnostic()> err) {
  auto maybeTrait = getTrait(err);
  if (failed(maybeTrait)) return failure();
  auto func = maybeTrait->getMethod(getMethodName(), err);
  if (failed(func)) {
    return failure();
  }
  return func;
}

/// The type arguments a generic call supplies for `parameters`, read off its
/// parallel type_params/type_args arrays.
///
/// A declaration's parameters are what a use supplies arguments for, so the
/// call must name each of them exactly once: a spelling that names no parameter
/// of the callee is refused rather than stamped into a clone name, and a
/// generic callee called with no arguments at all is refused too -- the call
/// says nothing about the instance it wants, and re-inferring it from spellings
/// a sweep has normalized is guessing.
static FailureOr<SpecializationMap> readDeclaredTypeArguments(
    ArrayAttr typeParams, ArrayAttr typeArgs,
    ArrayRef<GenericTypeInterface> parameters, StringRef callee,
    llvm::function_ref<InFlightDiagnostic()> err) {
  TypeArguments args(parameters);
  if (typeParams) {
    if (!typeArgs || typeParams.size() != typeArgs.size()) {
      if (err) err() << "type_params and type_args must be parallel arrays";
      return failure();
    }
    for (auto [param, arg] : llvm::zip(typeParams.getAsValueRange<TypeAttr>(),
                                       typeArgs.getAsValueRange<TypeAttr>())) {
      GenericTypeInterface parameter = getParameterOccurrence(param);
      if (!parameter || !args.binds(parameter)) {
        if (err) err() << "type parameter " << param
                       << " is not a type variable of the callee";
        return failure();
      }
      if (failed(args.assign(parameter, arg, err)))
        return failure();
    }
  }
  if (!args.complete()) {
    if (err) {
      unsigned supplied = 0;
      for (GenericTypeInterface parameter : parameters)
        if (args.lookup(parameter))
          ++supplied;
      err() << "call to @" << callee << " supplies " << supplied << " of its "
            << parameters.size() << " type arguments";
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
  auto implOr = ProofOp::getImplFromProof(module, claim.getProof(),
                                          /*errFn=*/nullptr);
  if (failed(implOr))
    return;

  // The proof's own subtree. A coinductive proof names itself among its
  // subproofs, so a proof already read contributes nothing a second time.
  if (auto proof = SymbolTable::lookupNearestSymbolFrom<ProofOp>(
          module, claim.getProof()))
    if (visited.insert(proof.getOperation()).second) {
      auto subproofs = proof.verifyAndGetSubproofClaims(
          DemandOrigin::ProofVerification, /*err=*/nullptr);
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
  auto subst = implOr->buildSubstitutionForSelfClaim(unproven, throughRulesSoFar,
                                                     /*errFn=*/nullptr);
  if (failed(subst))
    return;

  ctx.addLocalProjectionRule(*implOr, unproven.getTraitApplication(), *subst);
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
                                                       ModuleOp module) {
  NormalizationContext ctx;
  llvm::SmallPtrSet<Operation *, 8> visited;
  // The proof itself is marked read before the walk starts, so the tree it
  // stands over contributes and it does not.
  visited.insert(proof.getOperation());
  auto subproofs = proof.verifyAndGetSubproofClaims(
      DemandOrigin::ProofVerification, /*err=*/nullptr);
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

/// Checks every proof the claims a call carries name.
///
/// Each spelling is read through the call's own context first: a coerce
/// respells a claim through an equality it cites, and the proof standing on the
/// respelled claim is the proof of the spelling that equality carries it back
/// to. A value a MARKED coerce produced is skipped: its reconciling equality is
/// minted only at monomorphization, so nothing here can carry its spelling
/// back, and the bonded erase pass judges it once every projection grounds.
static LogicalResult verifyProofsAtCall(Operation *call, ValueRange operands,
                                        Normalizer normalize, ModuleOp module,
                                        llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<Type> spellings;
  for (Value operand : operands) {
    auto coerce = operand.getDefiningOp<CoerceOp>();
    if (coerce && coerce.getUnproven())
      continue;
    spellings.push_back(operand.getType());
  }
  llvm::append_range(spellings, call->getResultTypes());

  // One binding set across every spelling: an obligation two of them prove by
  // different symbols is the incoherent proof mapping this reports.
  EvidenceBindings evidence;
  for (Type spelling : spellings) {
    FailureOr<Type> read = normalize(spelling);
    if (failed(read))
      return failure();
    if (failed(bindProofsIn(*read, module, evidence,
                            DemandOrigin::CallSignatureVerification,
                            /*memo=*/nullptr, err)))
      return failure();
  }
  return success();
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
  // arguments ride in the application -- and the suffix from this call's own
  // arrays, restricted to the method's own variables, so a trait parameter
  // named there is refused.
  auto traitSubst = trait->buildSubstitutionForSelfClaim(getClaimType(), err);
  if (failed(traitSubst)) return failure();

  DenseSet<Type> headerParams;
  for (Attribute declared : trait->getTypeParams())
    if (auto typeAttr = dyn_cast<TypeAttr>(declared))
      for (GenericTypeInterface parameter :
           getTypeParametersIn(typeAttr.getValue()))
        headerParams.insert(Type(parameter));
  SmallVector<GenericTypeInterface, 4> ownParams;
  for (GenericTypeInterface parameter :
       getTypeParametersIn(Type(*methodFormalTy)))
    if (!headerParams.contains(Type(parameter)))
      ownParams.push_back(parameter);

  auto ownArgs = readDeclaredTypeArguments(getTypeParamsAttr(),
                                           getTypeArgsAttr(), ownParams,
                                           getMethodName(), err);
  if (failed(ownArgs)) return failure();

  SpecializationMap arguments = *traitSubst;
  for (GenericTypeInterface parameter : ownParams)
    if (auto argument = ownArgs->lookup(parameter))
      arguments.bind(parameter, *argument);

  // The evidence this call holds: the receiver claim's proof tree and the
  // claims its arguments carry, read by index. At pass time the record of what
  // impl selection has settled is read on top of that; a verifier has none.
  SmallVector<Value> localClaims;
  localClaims.push_back(getClaim());
  for (Value argument : getArguments())
    if (isa<ClaimType>(argument.getType()))
      localClaims.push_back(argument);
  NormalizationContext normalization =
      buildLocalClaimNormalizationContext(getOperation(), localClaims, *module);
  normalization.setRecordedFacts(reading);
  // XXX TODO A claim this call's own arguments spell can carry a ground
  // projection no evidence at this site reduces, because the impl serving it is
  // named nowhere the call can read. The module's impls stand in, and only
  // where the receiver claim commits to evidence -- an ordinary unproven claim
  // grants nothing. Deleted once the claim a call commits to carries the impls
  // serving the projections its arguments spell; see setModuleLookup.
  if (getClaimType().isProven() || getClaim().getDefiningOp<DeriveOp>())
    normalization.setModuleLookup(*module, LookupScope::Ground);
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return normalization.normalize(ty, err);
  };

  // One identity: the method's declaration instantiated at the arguments above
  // is the signature spelled here. There is nothing left to infer -- every
  // argument was supplied -- so there is no second, input-only pass either.
  FunctionType actual = getActualFunctionType();
  if (failed(verifyEqualAfterInstantiation(Type(*methodFormalTy), arguments,
                                           Type(actual), normalize, err)))
    return failure();

  // The proofs this call's claims name are checked where the call is lowered:
  // the factory that closes the substitution walks the same spellings and reads
  // them off the record. A verifier has no lowering behind it, so it checks
  // them here or nowhere, and it runs on a worker thread with no memo of its
  // own to serve them from.
  if (!reading &&
      failed(verifyProofsAtCall(getOperation(), getOperands(), normalize,
                                *module, err)))
    return failure();

  return arguments;
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
    const CallSubstitution &subst,
    ProofDerivationMemo *memo) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrSpecializeFreeFunctionFromMethod(rewriter, claimTy, getMethodName(),
                                           subst, memo);
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

  auto maybeFormal = getCalleeFunctionType(err);
  if (failed(maybeFormal)) return failure();

  // The callee's declaration binds the parameters its signature spells, and
  // this call supplies an argument for each of them.
  auto arguments = readDeclaredTypeArguments(
      getTypeParamsAttr(), getTypeArgsAttr(), getCalleeTypeParams(),
      getCalleeName(), err);
  if (failed(arguments)) return failure();

  // The evidence this call holds: the claims its operands carry, read by index
  // through their proof trees. At pass time the record of what impl selection
  // has settled is read on top of that; a verifier has none.
  SmallVector<Value> localClaims;
  for (Value operand : getOperands())
    if (isa<ClaimType>(operand.getType()))
      localClaims.push_back(operand);
  NormalizationContext normalization =
      buildLocalClaimNormalizationContext(getOperation(), localClaims, *module);
  normalization.setRecordedFacts(reading);
  // XXX TODO As at a method call: a ground projection an operand claim spells
  // and no evidence here reduces is read through the module's impls, and only
  // where an operand claim commits to evidence. Deleted on the same trigger;
  // see setModuleLookup.
  if (llvm::any_of(localClaims, [](Value claim) {
        return cast<ClaimType>(claim.getType()).isProven() ||
               claim.getDefiningOp<DeriveOp>();
      }))
    normalization.setModuleLookup(*module, LookupScope::Ground);
  auto normalize = [&](Type ty) -> FailureOr<Type> {
    return normalization.normalize(ty, err);
  };

  // One identity: the callee's declaration instantiated at those arguments is
  // the signature spelled here.
  FunctionType actual = getActualFunctionType();
  if (failed(verifyEqualAfterInstantiation(Type(*maybeFormal), *arguments,
                                           Type(actual), normalize, err)))
    return failure();

  // The proofs this call's claims name are checked where the call is lowered:
  // the factory that closes the substitution walks the same spellings and reads
  // them off the record. A verifier has no lowering behind it, so it checks
  // them here or nowhere, and it runs on a worker thread with no memo of its
  // own to serve them from.
  if (!reading &&
      failed(verifyProofsAtCall(getOperation(), getOperands(), normalize,
                                *module, err)))
    return failure();

  return *arguments;
}

/// The name the instance of `op`'s callee carries, given the substitution that
/// specializes its body.
///
/// Mangling reads the specialization map alone, which is written when the
/// substitution is built and is not touched by closing it -- closing adds
/// projection and evidence bindings -- so the name a call is wired to and the
/// body it is wired to are read off one object.
static std::string calleeInstanceName(FuncCallOp op,
                                      const CallSubstitution &subst) {
  return op.getCalleeName().str() +
         applySubstitutionAndGenerateMangledNameSuffix(subst.getSpecialization(),
                                                       op.getCalleeTypeParams());
}

FailureOr<func::FuncOp> FuncCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter,
    const CallSubstitution &subst,
    ProofDerivationMemo *memo) {
  auto module = getModule();
  if (failed(module)) return failure();

  std::string instanceName = calleeInstanceName(*this, subst);
  auto *symOp = SymbolTable::lookupSymbolIn(*module, rewriter.getStringAttr(instanceName));
  func::FuncOp existing = dyn_cast_or_null<func::FuncOp>(symOp);
  if (existing) {
    return existing;
  }

  auto callee = getCallee();
  if (failed(callee)) return failure();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(*callee);
  auto instance =
      specializeAndReplaceAssumes(rewriter, *callee, instanceName, subst.toTypeMap());
  // An external polymorphic declaration has no body to clone; specialization
  // has refused it, so this call has no instance to name.
  if (!instance)
    return failure();
  return instance;
}


//===----------------------------------------------------------------------===//
// ProjectOp
//===----------------------------------------------------------------------===//

ParseResult ProjectOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `%src : @SrcTrait[Types...] (by @SrcProof)? to @DstTrait[Types...] (by @DstProof)?`

  // %src
  OpAsmParser::UnresolvedOperand src;
  if (p.parseOperand(src)) return failure();
  if (p.parseColon()) return failure();

  // @SrcTrait[...]
  TraitApplicationAttr srcApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!srcApp) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

  // (by @SrcProof)?
  FlatSymbolRefAttr srcProof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(srcProof))
      return failure();
  }

  // resolve %src with the appropriate claim type
  ClaimType srcTy = srcProof
    ? ClaimType::get(p.getContext(), srcApp, srcProof)
    : ClaimType::get(p.getContext(), srcApp);

  if (p.resolveOperand(src, srcTy, st.operands))
    return failure();

  // to
  if (p.parseKeyword("to"))
    return failure();

  // The result is either an application projection (@DstTrait[...] (by
  // @DstProof)?) or the equality hop to a trait's equality requirement
  // (!A = !B), disambiguated by the leading `@`. The equality arm never carries
  // a proof.
  ClaimType dstTy;
  FlatSymbolRefAttr dstTrait;
  OptionalParseResult dstSym = p.parseOptionalAttribute(dstTrait);
  if (dstSym.has_value()) {
    if (failed(*dstSym))
      return failure();
    if (p.parseLSquare())
      return failure();
    SmallVector<Type> dstArgs;
    do {
      Type ty;
      if (p.parseType(ty))
        return failure();
      dstArgs.push_back(ty);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRSquare())
      return failure();
    auto dstApp = TraitApplicationAttr::get(p.getContext(), dstTrait,
                                            ArrayRef<Type>(dstArgs));

    // (by @DstProof)?
    FlatSymbolRefAttr dstProof;
    if (succeeded(p.parseOptionalKeyword("by"))) {
      if (p.parseAttribute(dstProof))
        return failure();
    }
    dstTy = dstProof ? ClaimType::get(p.getContext(), dstApp, dstProof)
                     : ClaimType::get(p.getContext(), dstApp);
  } else {
    Type lhs, rhs;
    if (p.parseType(lhs) || p.parseEqual() || p.parseType(rhs))
      return failure();
    dstTy = ClaimType::getEquality(p.getContext(), lhs, rhs);
  }
  st.addTypes(dstTy);

  return success();
}

void ProjectOp::print(OpAsmPrinter& p) {
  // print `%src: %Trait1[Types...] to @Trait2[Types...]1

  p << " ";

  // Source: %src: @SrcTrait[...] (by @SrcProof)?
  p.printOperand(getSource());
  p << ": ";
  ClaimType srcTy = getSourceClaim();
  srcTy.getTraitApplication().print(p);

  if (srcTy.isProven())
    p << " by " << srcTy.getProof();

  // Destination: to @DstTrait[...] (by @DstProof)? or the equality hop to !A = !B
  p << " to ";
  ClaimType dstTy = getResultClaim();
  if (auto eq = dstTy.getEqualityAttr()) {
    eq.print(p);
  } else {
    dstTy.getTraitApplication().print(p);
    if (dstTy.isProven())
      p << " by " << dstTy.getProof();
  }
}

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &/*symbolTable*/) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  ClaimType src = getSourceClaim();
  ClaimType dst = getResultClaim();

  // Verify proofness parity for an application result: a proven source projects
  // to a proven result, an unproven to an unproven. An equality result is
  // exempt -- an equality claim is never proven, so projecting one from a proven
  // source does not force a proof it cannot carry.
  if (!dst.isEquality()) {
    bool srcProven = src.isProven();
    bool dstProven = dst.isProven();
    if (srcProven != dstProven) {
      if (!srcProven)
        return emitOpError() << "result cannot have 'by' when source has no 'by'";
      return emitOpError() << "result must have 'by' when source has 'by'";
    }
  }

  // The result must be one of the source's candidate projections.
  if (src.projectsTo(module, dst))
    return success();

  return emitOpError()
         << "projected claim " << dst
         << "is not a candidate projection of " << src;
}


//===----------------------------------------------------------------------===//
// AssocTypeOp
//===----------------------------------------------------------------------===//

ParseResult AssocTypeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse @Name
  StringAttr symName;
  if (p.parseSymbolName(symName, "sym_name", st.attributes))
    return failure();

  // parse optional <[type_params...]>
  if (succeeded(p.parseOptionalLess())) {
    ArrayAttr typeParams;
    if (parseTypeParameters(p, typeParams) || p.parseGreater())
      return failure();
    st.addAttribute("type_params", typeParams);
  }

  // parse optional = bound_type
  if (succeeded(p.parseOptionalEqual())) {
    TypeAttr boundType;
    if (p.parseAttribute(boundType, "bound_type", st.attributes))
      return failure();
  }

  // parse optional attr-dict
  if (p.parseOptionalAttrDict(st.attributes))
    return failure();

  return success();
}

void AssocTypeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymNameAttr());

  // print <[type_params...]> if present
  if (auto tp = getTypeParams(); tp && !tp->empty()) {
    p << "<[";
    llvm::interleaveComma(*tp, p, [&](Attribute tyAttr) {
      p.printType(cast<TypeAttr>(tyAttr).getValue());
    });
    p << "]>";
  }

  // print = bound_type if present
  if (auto bt = getBoundType()) {
    p << " = " << *bt;
  }

  // print any trailing attributes
  p.printOptionalAttrDict((*this)->getAttrs(),
                           /*elided=*/{"sym_name", "bound_type", "type_params"});
}


//===----------------------------------------------------------------------===//
// AllegeOp
//===----------------------------------------------------------------------===//

ParseResult AllegeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `@Trait[Types...]`
  TraitApplicationAttr app = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

  // parse optional `unsafe` keyword
  UnitAttr unsafeAttr;
  if (succeeded(p.parseOptionalKeyword("unsafe")))
    unsafeAttr = p.getBuilder().getUnitAttr();
  if (unsafeAttr)
    st.addAttribute("unsafe", unsafeAttr);

  // result type is the claim of the trait application
  auto claimTy = ClaimType::get(p.getContext(), app);
  st.addTypes(claimTy);

  return success();
}

void AllegeOp::print(OpAsmPrinter &p) {
  p << " ";

  // print the claimed trait application
  getClaim().getTraitApplication().print(p);

  // print optional unsafe
  if (getUnsafe())
    p << " unsafe";
}

LogicalResult AllegeOp::verify() {
  // claim must be monomorphic unless unsafe
  if (!getUnsafe() && !getClaim().isMonomorphic())
    return emitOpError() << "expected monomorphic claim, got "
                         << getClaim();
  return success();
}
