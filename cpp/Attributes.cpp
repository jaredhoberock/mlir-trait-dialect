#include "Dialect.hpp"
#include "Attributes.hpp"
#include "Ops.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_ATTRDEF_CLASSES
#include "Attributes.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Attributes.cpp.inc"
  >();
}

LogicalResult TraitApplicationAttr::verifyTraitApplication(
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError
) const {
  // Get the trait
  TraitOp traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTrait());
  if (!traitOp)
    return emitError() << "cannot find trait '" << getTrait() << "'";

  // Check the trait's expected arity against typeArgs
  auto expectedArity = traitOp.getTypeParams().size();
  if (getTypeArgs().size() != expectedArity)
    return emitError() << "trait '" << getTrait() << "' expects " << expectedArity
                       << " type arguments, found " << getTypeArgs().size();

  return success();
}

Attribute TraitApplicationAttr::parse(AsmParser &parser, Type type) {
  // Expect: @TraitName[!T1, !T2, ...]
  FlatSymbolRefAttr traitRef;
  if (parser.parseAttribute(traitRef))
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
      parser.getContext(), traitRef, typeArgs);
}

void TraitApplicationAttr::print(mlir::AsmPrinter &printer) const {
  printer << '@' << getTrait().getValue(); // print the trait symbol name

  printer << '[';
  llvm::interleaveComma(getTypeArgs(), printer);
  printer << ']';
}

Attribute ConstraintsAttr::parse(AsmParser &p, Type type) {
  SmallVector<TraitApplicationAttr> applications;

  if (succeeded(p.parseOptionalKeyword("where"))) {
    if (p.parseLSquare())
      return {};

    do {
      TraitApplicationAttr app = mlir::dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
      if (!app)
        return {};
      applications.push_back(app);
    } while (succeeded(p.parseOptionalComma()));

    if (p.parseRSquare())
      return {};
  }

  return ConstraintsAttr::getChecked(
      [&]() { return p.emitError(p.getNameLoc()); },
      p.getContext(), applications);
}

void ConstraintsAttr::print(mlir::AsmPrinter &printer) const {
  printer << "where [";
  llvm::interleaveComma(getApplications(), printer,
                        [&](TraitApplicationAttr a) {
                          a.print(printer);
                        });
  printer << ']';
}

} // end mlir::trait
