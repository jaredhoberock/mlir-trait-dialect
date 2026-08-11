// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir::trait::detail {

// Hand-written storage for TypeEqualityAttr. The uniquing key retains both
// endpoints, so distinct equalities are distinct attributes; getAsKey() returns
// no sub-elements, so MLIR's sub-element walkers and the generic
// AttrTypeReplacer never see or rewrite the endpoints. Endpoints move only
// through the sanctioned clone rule, and readers reach them through the
// dedicated endpoint accessors.
struct TypeEqualityAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, ::mlir::Type>;

  TypeEqualityAttrStorage(::mlir::Type lhs, ::mlir::Type rhs)
      : lhs(lhs), rhs(rhs) {}

  // Walk-opaque by construction: this storage defines no getAsKey(), so MLIR's
  // sub-element walkers and the generic AttrTypeReplacer never visit the
  // endpoints. Uniquing still distinguishes distinct equalities through the key
  // below. The endpoints move only through the sanctioned clone rule; readers
  // reach them through the dedicated accessors.
  bool operator==(const KeyTy &key) const {
    return lhs == std::get<0>(key) && rhs == std::get<1>(key);
  }

  static ::llvm::hash_code hashKey(const KeyTy &key) {
    return ::llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  static TypeEqualityAttrStorage *
  construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&key) {
    return new (allocator.allocate<TypeEqualityAttrStorage>())
        TypeEqualityAttrStorage(std::get<0>(key), std::get<1>(key));
  }

  ::mlir::Type lhs;
  ::mlir::Type rhs;
};

} // namespace mlir::trait::detail

#define GET_ATTRDEF_CLASSES
#include <TraitAttributes.cpp.inc>

namespace mlir::trait {

// Whether a type carries a proof receipt anywhere -- a proven claim spelled into
// a position that forbids one. The equality arm and the projection-resolution
// certificate both freeze receipt-free endpoints.
static bool carriesReceipt(Type type) {
  bool found = false;
  type.walk([&](Type sub) {
    if (auto claim = dyn_cast<ClaimType>(sub))
      if (claim.isProven())
        found = true;
  });
  return found;
}

// Structural well-formedness of an equality proposition. Endpoints must be
// receipt-free: a proven claim spelled into an endpoint would carry a proof
// receipt into the arm that forbids one, re-creating the asymmetric proof
// comparison the equality arm exists to avoid. Constructing the attribute does
// not assert the equality; only a value of the enclosing claim type is
// evidence.
LogicalResult TypeEqualityAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    Type lhs, Type rhs) {
  if (!lhs || !rhs)
    return emitError() << "type equality requires two endpoint types";

  if (carriesReceipt(lhs) || carriesReceipt(rhs))
    return emitError() << "type-equality endpoints must be receipt-free";

  return success();
}

// Endpoint accessors read the hand-written storage directly; the generated
// class declares them but leaves them to the custom storage owner.
Type TypeEqualityAttr::getLhs() const { return getImpl()->lhs; }
Type TypeEqualityAttr::getRhs() const { return getImpl()->rhs; }

Attribute TypeEqualityAttr::parse(AsmParser &parser, Type) {
  Type lhs, rhs;
  if (parser.parseType(lhs) || parser.parseEqual() || parser.parseType(rhs))
    return {};
  return TypeEqualityAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), lhs, rhs);
}

void TypeEqualityAttr::print(AsmPrinter &printer) const {
  printer << getLhs() << " = " << getRhs();
}

// A projection-resolution certificate cites an impl and freezes the equality it
// establishes. The frozen endpoints' receipt-freeness is the stored
// TypeEqualityAttr's own invariant, enforced when that equality is constructed,
// so this checks only that both fields are present.
LogicalResult WitnessCertificateAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    TypeEqualityAttr equality, FlatSymbolRefAttr citedImpl) {
  if (!equality)
    return emitError() << "a projection-resolution certificate freezes an "
                          "equality";
  if (!citedImpl)
    return emitError() << "a projection-resolution certificate must cite an impl";

  return success();
}

Attribute WitnessCertificateAttr::parse(AsmParser &parser, Type) {
  Type redex, contractum;
  FlatSymbolRefAttr citedImpl;
  if (parser.parseType(redex) || parser.parseKeyword("resolves") ||
      parser.parseType(contractum) || parser.parseKeyword("by") ||
      parser.parseAttribute(citedImpl))
    return {};
  auto err = [&]() { return parser.emitError(parser.getNameLoc()); };
  auto equality =
      TypeEqualityAttr::getChecked(err, parser.getContext(), redex, contractum);
  if (!equality)
    return {};
  return WitnessCertificateAttr::getChecked(err, parser.getContext(), equality,
                                            citedImpl);
}

void WitnessCertificateAttr::print(AsmPrinter &printer) const {
  printer << getRedex() << " resolves " << getContractum() << " by "
          << getCitedImpl();
}

Attribute DischargeCitationAttr::parse(AsmParser &parser, Type) {
  auto application =
      dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(parser, {}));
  FlatSymbolRefAttr dischargingImpl;
  if (!application || parser.parseKeyword("by") ||
      parser.parseAttribute(dischargingImpl))
    return {};
  return DischargeCitationAttr::get(parser.getContext(), application,
                                    dischargingImpl);
}

void DischargeCitationAttr::print(AsmPrinter &printer) const {
  getApplication().print(printer);
  printer << " by " << getDischargingImpl();
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
  TraitOp traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTraitName());
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

// Recover the module that anchors symbol lookups: the operation verification
// reached, or that operation itself when it is the anchoring symbol table.
static ModuleOp getAnchorModule(Operation *anchor) {
  if (!anchor)
    return {};
  if (auto module = dyn_cast<ModuleOp>(anchor))
    return module;
  return anchor->getParentOfType<ModuleOp>();
}

// A trait application references a trait symbol applied to a fixed number of
// type arguments; verify the trait exists and its arity is respected. These
// attributes appear as inherent operation arguments, which the automatic
// symbol-user driver never walks -- it visits only discardable attributes -- so
// their owning ops verify them by delegating to this entry point. The interface
// is adopted for uniformity with symbol-using types; the same method would also
// verify a trait application encountered in a discardable position.
LogicalResult TraitApplicationAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &) const {
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

Attribute TraitApplicationAttr::parse(AsmParser &parser, Type type) {
  // Expect: @TraitName[!T1, !T2, ...]
  FlatSymbolRefAttr traitName;
  if (parser.parseAttribute(traitName))
    return {};

  // Parse required type arguments in brackets
  if (parser.parseLSquare())
    return {};

  SmallVector<Type> typeArgs;
  do {
    Type ty;
    if (parser.parseType(ty))
      return {};
    typeArgs.push_back(ty);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare())
    return {};

  return TraitApplicationAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), traitName, typeArgs);
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

// Verify each predicate: application entries name a trait symbol, equality
// entries carry their symbol users nested (opaque) in the endpoints. The
// equality arm defers to the claim verifier, which reaches the endpoints
// through the accessor.
LogicalResult PredicateArrayAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  for (Attribute p : getPredicates()) {
    if (auto app = mlir::dyn_cast<TraitApplicationAttr>(p)) {
      if (failed(app.verifySymbolUses(op, symbolTable)))
        return failure();
    } else if (auto eq = mlir::dyn_cast<TypeEqualityAttr>(p)) {
      auto claim = ClaimType::getEquality(op->getContext(), eq);
      if (failed(claim.verifySymbolUses(op, symbolTable)))
        return failure();
    }
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
