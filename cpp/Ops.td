#ifndef TRAIT_OPS_TD
#define TRAIT_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"

def TraitOp : TraitDialectOp<"trait", [
  Symbol, 
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Declares a trait interface";
  let description = [{
    Declares a trait interface, similar to a Rust trait. A `trait.trait` operation
    defines a symbol that can be referenced by `trait.param` types and `trait.impl`.
    It contains a region which holds `func.func` operations that define the
    interface requirements for implementors. If any of these `func.func` operations have
    a body, then this interface is optional for implementors.

    Example:
      trait.trait @Add {
        func.func @add(%self: !trait.self, %rhs: i32) -> i32
      }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);  // No results
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    func::FuncOp getMethod(StringRef methodName) {
      return dyn_cast_or_null<func::FuncOp>(
        mlir::SymbolTable::lookupSymbolIn(*this, methodName));
    }

    bool hasMethod(StringRef methodName) {
      return getMethod(methodName) != nullptr;
    }

    llvm::SmallSet<llvm::StringRef,8> getRequiredMethodNames() {
      llvm::SmallSet<llvm::StringRef,8> result;
      for (Operation& op : getBody().front()) {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
          if (func.isExternal()) {
            result.insert(func.getSymName());
          }
        }
      }
      return result;
    }

    std::vector<func::FuncOp> getOptionalMethods() {
      std::vector<func::FuncOp> result;
      for (Operation& op : getBody().front()) {
        if (auto func = dyn_cast<func::FuncOp>(op)) {
          if (!func.isExternal()) {
            result.push_back(func);
          }
        }
      }
      return result;
    }

    func::FuncOp getOptionalMethod(StringRef methodName) {
      for (auto method : getOptionalMethods()) {
        if (method.getSymName() == methodName)
          return method;
      }
      return nullptr;
    }

    ImplOp getImpl(Type receiverType);

    ImplOp getOrInstantiateImpl(OpBuilder& builder, Type receiverType);
  }];

  // builder creates one empty block in the body region
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$sym_name), [{
      $_state.addAttribute("sym_name", sym_name);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";

  let hasVerifier = 1;
}

def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Implements a trait for a receiver type";
  let description = [{
    Declares an implementation of a trait for a specific receiver type.

    The `trait` attribute is a FlatSymbolRefAttr pointing to a `trait.trait`
    declaration (e.g., `@PartialEq`), and `receiver_type` is a TypeAttr representing
    the type that implements the trait (e.g., `i32` or a custom type).

    This operation contains a region with `func.func` operations, each
    implementing one of the trait's required methods.

    Example:
      trait.impl @PartialEq for i32 {
        func.func @eq(%self: i32, %other: i32) -> i1 {
          %same = arith.cmpi eq, %self, %other : i32
          trait.return %same : i1
        }
      }
  }];

  let arguments = (ins FlatSymbolRefAttr:$trait_name, TypeAttr:$receiver_type);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    TraitOp getTrait();

    std::vector<func::FuncOp> getMethods() {
      std::vector<func::FuncOp> result;
      for (auto& op : getBody().front()) {
        if (auto method = dyn_cast<func::FuncOp>(op)) {
          result.push_back(method);
        }
      }
      return result;
    }

    func::FuncOp getMethod(StringRef name) {
      return lookupSymbol<func::FuncOp>(name);
    }

    bool hasMethod(StringRef name) {
      return getMethod(name) != nullptr;
    }

    func::FuncOp getOrInstantiateMethod(OpBuilder& builder, StringRef methodName);

    func::FuncOp getOrInstantiateFunctionFromMethod(OpBuilder& builder, StringRef methodName);
  }];

  // builder creates one empty block in the body region
  let builders = [
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$trait_name,
                   "::mlir::TypeAttr":$receiver_type), [{
      $_state.addAttribute("trait_name", trait_name);
      $_state.addAttribute("receiver_type", receiver_type);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "$trait_name `for` $receiver_type attr-dict-with-keyword $body";
}

def MethodCallOp : TraitDialectOp<"method.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Call a trait method";
  let description = [{
    Calls a method from a trait using fully-qualified syntax like `@Trait::@method<type>`.

    This can appear in either polymorphic or monomorphic code.
    Resolution to an implementation is performed during lowering.

    Example:
      %r = trait.method.call @PartialEq::@eq<i32>(%x, %y) : (!trait.self, !trait.self) to (i32, i32) -> i1
  }];

  let arguments = (ins 
      SymbolRefAttr:$method_ref,
      TypeAttrOf<FunctionType>:$method_function_type,
      TypeAttr:$receiver_type,
      Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  // custom builders
  let builders = [
    // builder that takes trait and method as FlatSymbolRefAttrs
    OpBuilder<(ins
      "TypeRange":$results,
      "FlatSymbolRefAttr":$traitRef,
      "FlatSymbolRefAttr":$methodRef,
      "TypeAttr":$methodFunctionType,
      "TypeAttr":$receiverType,
      "ValueRange":$operands), [{
        MLIRContext *ctx = $_builder.getContext();

        // create the nested SymbolRefAttr
        SymbolRefAttr qualifiedRef = SymbolRefAttr::get(
          ctx,
          traitRef.getValue(),
          methodRef
        );

        // build the operation with the created nested reference
        return build($_builder, $_state, results, qualifiedRef, methodFunctionType, receiverType, operands);
    }]>,

    // builder that takes trait and method names separately
    OpBuilder<(ins
      "TypeRange":$results,
      "StringRef":$traitName,
      "StringRef":$methodName,
      "TypeAttr":$methodFunctionType,
      "TypeAttr":$receiverType,
      "ValueRange":$operands), [{
        // convert strings to FlatSymbolRefAttrs and call the other builder
        MLIRContext *ctx = $_builder.getContext();
        FlatSymbolRefAttr traitRef = FlatSymbolRefAttr::get(ctx, traitName);
        FlatSymbolRefAttr methodRef = FlatSymbolRefAttr::get(ctx, methodName);

        // call the other builder
        return build($_builder, $_state, results, traitRef, methodRef, methodFunctionType, receiverType, operands);
    }]>,
  ];

  let extraClassDeclaration = [{
    func::FuncOp getOrInstantiateCallee(OpBuilder& builder);

    TraitOp getTrait();

    StringRef getTraitName() {
      return getMethodRef().getRootReference().getValue();
    }

    FlatSymbolRefAttr getTraitAttr() {
      return FlatSymbolRefAttr::get(
        getContext(),
        getMethodRef().getRootReference().getValue()
      );
    }

    FlatSymbolRefAttr getMethodAttr() {
      auto methodRef = getMethodRef();
      if (methodRef.getNestedReferences().empty()) {
        emitOpError() << "expected nested symbol reference with @trait::@method format";
        return nullptr;
      }
      return cast<FlatSymbolRefAttr>(getMethodRef().getNestedReferences().front());
    }

    StringRef getMethodName() {
      assert(!getMethodRef().getNestedReferences().empty() && 
             "Expected nested method reference");
      return cast<FlatSymbolRefAttr>(getMethodRef().getNestedReferences().front()).getValue();
    }

    std::string getNameOfCalleeInstance();

    // CallOpInterface methods
    operand_range getArgOperands() {
      return getOperands();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("method");
    }

    void setCalleeFromCallable(CallInterfaceCallable method) {
      (*this)->setAttr("method", cast<SymbolRefAttr>(method));
    }
  }];

  let assemblyFormat = [{
    $method_ref `<` $receiver_type `>` `(` $operands `)` attr-dict `:` $method_function_type `to` functional-type($operands, results)
  }];
}

def FuncCallOp : TraitDialectOp<"func.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Call a possibly polymorphic function defined with func.func and !trait.poly";
  let description = [{
    Calls a function that may be polymorphic over one or more `!trait.poly` type parameters.
    These functions are typically declared using `func.func` and their signatures may involve
    polymorphic type variables.

    Example:
      %r = trait.func.call @foo(%x, %y) : (i32, i32) -> i1
  }];

  let arguments = (ins 
      FlatSymbolRefAttr:$callee,
      TypeAttrOf<FunctionType>:$callee_function_type,
      Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    func::FuncOp getOrInstantiateCallee(OpBuilder& builder);

    func::FuncOp instantiateCalleeAtInsertionPoint(OpBuilder& builder);

    // for each PolyType in the callee's parameter types,
    // creates the mapping poly type -> argument type
    llvm::DenseMap<Type, Type> buildSubstitution();

    // returns the name of the callee after instantiation
    std::string getNameOfCalleeInstance();

    operand_range getArgOperands() {
      return getOperands();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` $callee_function_type `to` functional-type($operands, results)";
}

#endif // TRAIT_OPS_TD
