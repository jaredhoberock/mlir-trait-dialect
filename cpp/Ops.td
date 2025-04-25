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
    It contains a region which holds `trait.method` declarations that define the
    interface requirements for implementors.

    Example:
      trait.trait @Add {
        trait.method @add(%self: !trait.self, %rhs: i32) -> i32
      }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);  // No results
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";

  let hasVerifier = 1;
}

def MethodOp : TraitDialectOp<"method", [
  Symbol, 
  HasParent<"TraitOp">
]> {
  let summary = "Declares a required method signature within a trait";
  let description = [{
    Declares a required method inside a `trait.trait` declaration. This is similar to
    a method signature in a Rust trait. It defines a symbol (`sym_name`) that must be
    implemented in trait impls via `trait.impl.method`.

    The method specifies input and result types, and may refer to `!trait.self` to
    denote the implementing type.

    Example:
      trait.method @add(!trait.self, i32) -> i32
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );

  let results = (outs);

  let hasCustomAssemblyFormat = 1;
}

def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Implements a trait for a concrete type";
  let description = [{
    Declares an implementation of a trait for a specific concrete type.

    The `trait` attribute is a FlatSymbolRefAttr pointing to a `trait.trait`
    declaration (e.g., `@PartialEq`), and `concrete_type` is a TypeAttr representing
    the type that implements the trait (e.g., `i32` or a custom type).

    This operation contains a region with `trait.impl.method` definitions, each
    implementing one of the trait's required methods.

    Example:
      trait.impl @PartialEq<i32> {
        trait.impl.method @eq(%self: i32, %other: i32) -> i1 {
          %same = arith.cmpi eq, %self, %other : i32
          trait.return %same : i1
        }
      }
  }];

  let arguments = (ins FlatSymbolRefAttr:$trait, TypeAttr:$concrete_type);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$trait`<` $concrete_type `>` attr-dict-with-keyword $body";
}

def MethodCallOp : TraitDialectOp<"method.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Call a trait method";
  let description = [{
    Calls a method from a trait using fully-qualified syntax like `@Trait<type>::@method`.

    This can appear in either polymorphic or monomorphic code.
    Resolution to an implementation is performed during lowering.

    Example:
      %r = trait.method.call @PartialEq<i32>::@eq(%x, %y) : (i32, i32) -> i1
  }];

  let arguments = (ins 
      FlatSymbolRefAttr:$trait,
      TypeAttr:$self_type,
      FlatSymbolRefAttr:$method,
      Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
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
    $trait `<` $self_type `>` `:` `:` $method `(` $operands `)` attr-dict `:` functional-type($operands, results)
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
      Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
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

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

#endif // TRAIT_OPS_TD
