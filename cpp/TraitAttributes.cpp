#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_ATTRDEF_CLASSES
#include "TraitAttributes.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TraitAttributes.cpp.inc"
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

LogicalResult TraitApplicationAttr::verifyTraitApplication(
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError
) const {
  // Get the trait
  auto maybeTrait = getTrait(module, emitError);
  if (failed(maybeTrait))
    return failure();
  TraitOp trait = *maybeTrait;

  // Check the trait's expected arity against typeArgs
  auto expectedArity = trait.getTypeParams().size();
  if (getTypeArgs().size() != expectedArity)
    return emitError() << "trait '" << getTraitName() << "' expects " << expectedArity
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
