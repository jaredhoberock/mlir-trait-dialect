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
include "Types.td"


def TraitOp : TraitDialectOp<"trait", [
  IsolatedFromAbove,
  Symbol,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Declares a trait interface";
  let description = [{
    Declares a trait interface, similar to a Rust or Scala trait. A `trait.trait` operation
    defines a symbol that can be referenced by `!trait.witness` types, `trait.impl`, and `trait.method.call`.
    It contains a region which holds `func.func` operations that define the
    method requirements for implementors. If any of these `func.func` operations have
    a body, then this method is optional for implementors.

    Example:
      !S = !trait.poly<0>
      !O = !trait.poly<1>

      trait.trait @PartialEq[!S,!O] {
        // a required method
        func.func @eq(%self: !S, %other: !O) -> i1
      }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeArrayAttr:$type_params);
  let results = (outs); // No results
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    MethodOp getMethod(StringRef methodName) {
      return dyn_cast_or_null<MethodOp>(
        mlir::SymbolTable::lookupSymbolIn(*this, methodName));
    }

    bool hasMethod(StringRef methodName) {
      return getMethod(methodName) != nullptr;
    }

    llvm::SmallSet<llvm::StringRef,8> getRequiredMethodNames() {
      llvm::SmallSet<llvm::StringRef,8> result;
      for (Operation& op : getBody().front()) {
        if (auto method = dyn_cast<MethodOp>(op)) {
          if (method.isRequired()) {
            result.insert(method.getSymName());
          }
        }
      }
      return result;
    }

    std::vector<MethodOp> getOptionalMethods() {
      std::vector<MethodOp> result;
      for (Operation& op : getBody().front()) {
        if (auto method = dyn_cast<MethodOp>(op)) {
          if (method.isOptional()) {
            result.push_back(method);
          }
        }
      }
      return result;
    }

    MethodOp getOptionalMethod(StringRef methodName) {
      for (auto method : getOptionalMethods()) {
        if (method.getSymName() == methodName)
          return method;
      }
      return nullptr;
    }

    ImplOp getImpl(ArrayRef<Type> typeArgs);
    ImplOp getOrInstantiateImpl(OpBuilder& builder, ArrayRef<Type> typeArgs);

    DenseMap<Type,Type> buildSubstitutionFor(TypeRange typeArgs);
  }];

  // builder creates one empty block in the body region
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayAttr":$type_params), [{
      $_state.addAttribute("sym_name", sym_name);
      $_state.addAttribute("type_params", type_params);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "$sym_name $type_params attr-dict-with-keyword $body";

  let hasVerifier = 1;
}


def MethodOp : TraitDialectOp<"method", [
  Symbol,
  FunctionOpInterface,
  HasParent<"TraitOp, ImplOp">,
  IsolatedFromAbove,
  AutomaticAllocationScope,
  SingleBlockImplicitTerminator<"ReturnOp">,
  OpAsmOpInterface
]> {
  let summary = "Declares a trait method";
  let description = [{
    Declares a method within a trait or implementation. 

    In TraitOp context:
    - Required methods have no body (declaration only)
    - Optional methods have a body (default implementation)

    In ImplOp context:
    - Implementation methods must have a body

    Example:
      trait.trait @PartialEq (%witness: !trait.witness<@PartialEq[!S,!O]>) {
        // required method
        trait.method @eq(%self: !S, %other: !O) -> i1
        
        // optional method with default implementation
        trait.method @neq(%self: !S, %other: !O) -> i1 {
          %eq = trait.method.call @PartialEq::@eq<%witness>(%self, %other)
            : (!S, !O) -> i1
          %true = arith.constant 1 : i1
          %result = arith.xori %eq, %true : i1
          trait.return %result : i1
        }
      }
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "mlir::StringRef":$name, "mlir::FunctionType":$type,
    CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs,
    CArg<"mlir::ArrayRef<mlir::DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    // FunctionOpInterface methods
    Region *getCallableRegion() { return hasBody() ? &getBody() : nullptr; }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    // OpAsmOpInterface methods
    static StringRef getDefaultDialect() { return "trait"; }

    // Method-specific functionality
    bool hasBody() { return !getBody().empty(); }
    
    // Context-aware semantics (TraitOp only)
    bool isRequired() { 
      assert(isa<TraitOp>((*this)->getParentOp()) && "isRequired() only valid in TraitOp context");
      return !hasBody(); 
    }

    bool isOptional() { 
      assert(isa<TraitOp>((*this)->getParentOp()) && "isOptional() only valid in TraitOp context");
      return hasBody(); 
    }
    
    StringRef getName() {
      return getSymName();
    }
    
    // Clone this method as a func::FuncOp and hoist it into the grandparent operation (module level)
    func::FuncOp cloneAsFuncOpAndHoistIntoGrandparent(OpBuilder& builder, StringRef funcName);
  }];

  let hasCustomAssemblyFormat = 1;
}


def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  IsolatedFromAbove,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Implements a trait for a list of types";
  let description = [{
    Declares an implementation of a trait for a specific list of types.

    The `trait_name` attribute is a FlatSymbolRefAttr pointing to a `trait.trait`
    declaration (e.g., `@PartialEq`), and `type_args` is an ArrayRefParameter of Types
    representing the list of types that implement the trait (e.g., [`i32`, `i32`]).

    This operation contains a region with `func.func` operations, each
    implementing one of the trait's methods.

    Example:
        trait.impl @PartialEq for [i32,i32] {
          func.func @eq(%self: i32, %other: i32) -> i1 {
            %res = arith.cmpi eq, %self, %other : i32
            return %res : i1
          }
        }
  }];

  let arguments = (ins FlatSymbolRefAttr:$trait_name, TypeArrayAttr:$type_args);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    TraitOp getTrait();

    std::vector<MethodOp> getMethods();

    MethodOp getMethod(StringRef name);

    bool hasMethod(StringRef name);

    MethodOp getOrInstantiateMethod(OpBuilder& builder, StringRef methodName);

    func::FuncOp getOrInstantiateFunctionFromMethod(OpBuilder& builder, StringRef methodName);

    // Builds a substitution that maps the TraitOp's PolyType parameters to this ImplOp's type arguments
    DenseMap<Type,Type> buildSubstitution();
  }];

  // builders create one empty block in the body region
  let builders = [
    // lower-level builder
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$trait_name,
                   "::mlir::ArrayAttr":$type_args), [{
      $_state.addAttribute("trait_name", trait_name);
      $_state.addAttribute("type_args", type_args);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "$trait_name $type_args attr-dict-with-keyword $body";
}


def Witness : TypeConstraint<
  CPred<"::mlir::isa<WitnessType>($_self)">,
  "any '!trait.witness' type"
>;


def WitnessOp : TraitDialectOp<"witness", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Materializes a witness value for a known trait implementation";
  let description = [{
    Creates a zero-sized SSA value of type `!trait.witness<@Trait, [types]>` that
    acts as proof that the trait is satisfied. This operation has no operands.

    The type arguments must be concrete; symbolic types are illegal. Symbol use
    verification checks that a matching `trait.impl` exists for the named trait and type list.
  }];

  let results = (outs Witness:$result);

  let extraClassDeclaration = [{
    TraitOp getTrait();

    FlatSymbolRefAttr getTraitAttr();
    ArrayRef<Type> getTypeArgs();
  }];

  let assemblyFormat = "`:` type($result) attr-dict";
}

def MethodCallOp : TraitDialectOp<"method.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]>{
  let summary = "Call a trait method";
  let description = [{
    Calls a method from a trait. This can appear in either polymorphic or monomorphic code.
    Resolution to an implementation is performed during monomorphization.

    Example:
      %witness = trait.witness : !trait.witness<@PartialEq, [i32,i32]>
      %x = ... : i32
      %y = ... : i32
      %res = trait.method.call @PartialEq::@eq<%witness>(%x, %y) : (!trait.poly<0>, !trait.poly<1>) to (i32, i32) -> i1
  }];

  let arguments = (ins
    SymbolRefAttr:$method_ref,
    TypeAttrOf<FunctionType>:$method_function_type,
    WitnessType:$witness,
    Variadic<AnyType>:$arguments
  );

  // custom builders
  let builders = [
    // builder that takes trait and method as FlatSymbolRefAttrs
    OpBuilder<(ins
      "TypeRange":$results,
      "FlatSymbolRefAttr":$traitRef,
      "FlatSymbolRefAttr":$methodRef,
      "TypeAttr":$methodFunctionType,
      "Value":$witness,
      "ValueRange":$arguments), [{
        MLIRContext *ctx = $_builder.getContext();

        // create the nested SymbolRefAttr
        SymbolRefAttr qualifiedRef = SymbolRefAttr::get(
          ctx,
          traitRef.getValue(),
          methodRef
        );

        // build the operation with the created nested reference
        return build($_builder, $_state, results, qualifiedRef, methodFunctionType, witness, arguments);
    }]>,

    // builder that takes trait and method names separately
    OpBuilder<(ins
      "TypeRange":$results,
      "StringRef":$traitName,
      "StringRef":$methodName,
      "TypeAttr":$methodFunctionType,
      "Value":$witness,
      "ValueRange":$arguments), [{
        // convert strings to FlatSymbolRefAttrs and call the other builder
        MLIRContext *ctx = $_builder.getContext();
        FlatSymbolRefAttr traitRef = FlatSymbolRefAttr::get(ctx, traitName);
        FlatSymbolRefAttr methodRef = FlatSymbolRefAttr::get(ctx, methodName);

        // call the other builder
        return build($_builder, $_state, results, traitRef, methodRef, methodFunctionType, witness, arguments);
    }]>,
  ];

  let results = (outs Variadic<AnyType>);

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

    // CallOpInterface methods
    operand_range getArgOperands() {
      return getArguments();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getArgumentsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("method");
    }

    void setCalleeFromCallable(CallInterfaceCallable method) {
      (*this)->setAttr("method", cast<SymbolRefAttr>(method));
    }
  }];

  let assemblyFormat = [{
    $method_ref `<` $witness `>` `(` $arguments `)` attr-dict `:` $method_function_type `as` type($witness) functional-type($arguments, results)
  }];

  let hasVerifier = 1;
}

def FuncCallOp : TraitDialectOp<"func.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Call a possibly polymorphic function defined with func.func and !trait.poly";
  let description = [{
    Calls a function that may be polymorphic over one of more `!trait.poly` type parameters.
    These functions are typically declared using `func.func` and their signatures may involve
    polymorphic type variables.

    Example:
      %x = ... : i32
      %y = ... : i32
      %w = trait.witness : !trait.witness<@PartialEq[i32,i32]>
      %r = trait.func.call @foo(%w, %x, %y)
        : (!trait.witness<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>, !trait.poly<0>) -> i1
        to (i32, i32) -> i1
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

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` $callee_function_type `as` functional-type($operands, results)";
}

def ReturnOp : TraitDialectOp<"return", [
  Pure,
  HasParent<"MethodOp">,
  MemRefsNormalizable,
  ReturnLike,
  Terminator
]> {
  let summary = "Trait method return operation";
  let description = [{
    The `trait.return` operation represents a return operation within a trait method.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the method
    that contains the operation.

    Example:

    ```mlir
    trait.method @eq(%self: !S, %other: !O) -> i1 {
      %result = arith.cmpi eq, %self, %other : i32
      trait.return %result : i1
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // TRAIT_OPS_TD
