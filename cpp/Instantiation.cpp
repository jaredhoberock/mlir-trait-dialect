#include "Instantiation.hpp"
#include "Types.hpp"
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::trait {

static bool containsSymbolicType(Type ty) {
  // XXX TODO we actually need to traverse subelements of ty
  //          for this check to be correct
  return isa<SymbolicTypeInterface>(ty);
}

static bool containsSymbolicType(TypeRange types) {
  return llvm::any_of(types, [](Type t) {
    return containsSymbolicType(t);
  });
}

static bool containsSymbolicType(ArrayRef<NamedAttribute> attrs) {
  for (const auto &attr : attrs) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      if (containsSymbolicType(typeAttr.getValue())) {
        return true;
      }
    }
  }
  return false;
}

static bool functionTypeContainsSymbolicType(FunctionType ty) {
  return containsSymbolicType(ty.getInputs()) || containsSymbolicType(ty.getResults());
}

bool isPolymorph(func::FuncOp fn) {
  return functionTypeContainsSymbolicType(fn.getFunctionType());
}

std::string mangleFunctionName(StringRef name,
                               const DenseMap<Type, Type> &substitution) {
  std::string result = name.str();

  // append substituted types to the name
  for (auto [_, substitutedTy] : substitution) {
    llvm::raw_string_ostream os(result);
    os << "_";
    substitutedTy.print(os);
    os.flush();
  }

  return result;
}
                                                      
std::string mangleMethodName(
    StringRef traitName,
    Type receiverType,
    StringRef methodName) 
{
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "__trait_" << traitName;
  os << "_impl_";

  receiverType.print(os);

  os << "_" << methodName; // e.g., "eq"

  return os.str();
}

static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer);

static Operation *cloneOpWithTypeReplacement(
    OpBuilder &builder,
    Operation &oldOp,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer) {
  OperationState state(oldOp.getLoc(), oldOp.getName());

  // remap operands
  for (Value operand : oldOp.getOperands())
    state.addOperands(mapping.lookupOrDefault(operand));

  // replace result types
  for (Type t : oldOp.getResultTypes())
    state.addTypes(typeReplacer.replace(t));

  // replace attributes
  for (NamedAttribute attr : oldOp.getAttrs()) {
    Attribute rewritten = typeReplacer.replace(attr.getValue());
    state.addAttribute(attr.getName(), rewritten);
  }

  // recursively clone regions
  for (Region &oldRegion : oldOp.getRegions()) {
    Region *newRegion = state.addRegion();
    cloneRegionWithTypeReplacement(builder, oldRegion, *newRegion,
                                   mapping, typeReplacer);
  }

  // remap results
  Operation *newOp = builder.create(state);
  for (auto [oldRes, newRes] : llvm::zip(oldOp.getResults(), newOp->getResults()))
    mapping.map(oldRes, newRes);

  return newOp;
}

static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer) {
  // create blocks with replaced argument types
  for (Block &oldBlock : oldRegion.getBlocks()) {
    Block *newBlock = builder.createBlock(&newRegion);
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      Type newType = typeReplacer.replace(oldArg.getType());
      BlockArgument newArg = newBlock->addArgument(newType, oldArg.getLoc());
      mapping.map(oldArg, newArg);
    }
  }

  // clone each operation in each new block
  auto& oldBlocks = oldRegion.getBlocks();
  auto& newBlocks = newRegion.getBlocks();
  for (auto [oldBlock, newBlock] : llvm::zip(oldBlocks, newBlocks)) {
    PatternRewriter::InsertionGuard guard(builder);

    builder.setInsertionPointToEnd(&newBlock);
    for (Operation &op : oldBlock)
      cloneOpWithTypeReplacement(builder, op, mapping, typeReplacer);
  }
}

func::FuncOp instantiatePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot instantiate external function");
    return nullptr;
  }

  if (!isPolymorph(polymorph)) {
    polymorph.emitError("cannot instantiate function that is not polymorphic");
    return nullptr;
  }

  Location loc = polymorph.getLoc();
  MLIRContext *ctx = builder.getContext();

  // set up type replacer
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<Type> {
    auto it = substitution.find(t);
    return (it != substitution.end()) ? std::optional<Type>(it->second) : std::nullopt;
  });

  // replace the polymorphic function type
  auto oldFunctionType = polymorph.getFunctionType();
  auto newFunctionType = llvm::cast<FunctionType>(replacer.replace(oldFunctionType));

  // create the instance with the new type and instance name
  func::FuncOp instance = builder.create<func::FuncOp>(loc, instanceName, newFunctionType);

  // clone the polymorph's attributes with type replacement
  for (NamedAttribute attr : polymorph->getAttrs()) {
    StringRef n = attr.getName();

    // don't copy the polymorph's name or function type
    if (n == polymorph.getSymNameAttrName() ||
        n == polymorph.getFunctionTypeAttrName()) {
      continue;
    }

    instance->setAttr(attr.getName(), replacer.replace(attr.getValue()));
  }

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph.getBody(),
                                 instance.getBody(),
                                 mapping,
                                 replacer);

  return instance;
}

}
