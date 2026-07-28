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

// Verify each application in turn; see TraitApplicationAttr above.
LogicalResult TraitApplicationArrayAttr::verifySymbolUses(
    Operation *op, SymbolTableCollection &symbolTable) const {
  for (TraitApplicationAttr app : getApplications())
    if (failed(app.verifySymbolUses(op, symbolTable)))
      return failure();
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

Attribute TraitApplicationArrayAttr::parse(AsmParser &p, Type type) {
  MLIRContext *ctx = p.getContext();
  auto errFn = [&]{ return p.emitError(p.getCurrentLocation()); };

  SmallVector<TraitApplicationAttr> apps;

  // expect `[ ... ]`
  if (p.parseLSquare())
    return {};

  // handle empty list early
  if (succeeded(p.parseOptionalRSquare()))
    return TraitApplicationArrayAttr::getChecked(errFn, ctx, apps);

  // parse at least one TraitApplicationAttr, then optional `,`-separated rest
  do {
    Attribute raw = TraitApplicationAttr::parse(p, {});
    if (!raw) return {}; // parse already emitted a diagnostic

    auto app = mlir::dyn_cast<TraitApplicationAttr>(raw);
    if (!app) {
      errFn() << "expected trait application like @Trait[Types...]";
      return {};
    }
    apps.push_back(app);
  } while(succeeded(p.parseOptionalComma()));

  if (p.parseRSquare())
    return {};

  return TraitApplicationArrayAttr::getChecked(errFn, ctx, apps);
}

void TraitApplicationArrayAttr::print(mlir::AsmPrinter &printer) const {
  printer << "[";
  llvm::interleaveComma(getApplications(), printer,
                        [&](TraitApplicationAttr a) {
                          a.print(printer);
                        });
  printer << ']';
}

} // end mlir::trait
