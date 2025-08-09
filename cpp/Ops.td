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


// Op's regions have only this type of child operation.
class HasChild<string op>
    : ParamNativeOpTrait<"HasOnlyChildOp", op>, StructuralOpTrait;


def TraitOp : TraitDialectOp<"trait", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChild<"func::FuncOp">,
  HasParent<"ModuleOp">,
  IsolatedFromAbove,
  SingleBlock,
  Symbol,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Declares a trait interface";
  let description = [{
    Declares a trait interface, similar to a Rust or Scala trait. A `trait.trait` operation
    defines a symbol that can be referenced by `!trait.claim` types, `trait.impl`, and `trait.method.call`.
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

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeArrayAttr:$type_params,
    ConstraintsAttr:$requirements
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    TraitApplicationAttr getSelfApplication() {
      MLIRContext* ctx = getContext();
      FlatSymbolRefAttr selfRef = FlatSymbolRefAttr::get(ctx, getSymName());
      return TraitApplicationAttr::get(ctx, selfRef, getTypeParams());
    }

    ClaimType getSelfClaim() {
      return ClaimType::get(getContext(), getSelfApplication());
    }

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

    DenseMap<Type,Type> buildSubstitutionFor(ClaimType claimTy);

    SmallVector<TraitOp,4> getRequiredTraits();

    LogicalResult verifyRequirements(ImplOp impl, ArrayRef<ImplOp> subproofs, llvm::function_ref<InFlightDiagnostic()> errFn);

    SmallVector<ClaimType> getRequirementsAsClaims();
  }];

  // builder creates one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayAttr":$type_params,
                   "ConstraintsAttr":$requirements), [{
      $_state.addAttribute("sym_name", sym_name);
      $_state.addAttribute("type_params", type_params);
      $_state.addAttribute("requirements", requirements);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "$sym_name $type_params $requirements attr-dict-with-keyword $body";

  let hasVerifier = 1;
}


def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChild<"func::FuncOp">,
  HasParent<"ModuleOp">,
  IsolatedFromAbove,
  SingleBlock,
  Symbol,
  SymbolTable
] # GraphRegionNoTerminator.traits
> {
  let summary = "Implements a trait for a list of types";
  let description = [{
    Declares an implementation of a trait for a specific list of types.

    The `sym_name` attribute is a SymbolNameAttr allowing `trait.witness` operations
    to refer to this implementation by name. If omitted from the textual IR, a unique
    `sym_name` will be automatically synthesized from the other arguments.

    The `trait_name` attribute is a FlatSymbolRefAttr pointing to a `trait.trait`
    declaration (e.g., `@PartialEq`), and `type_args` is an ArrayRefParameter of Types
    representing the list of types that implement the trait (e.g., [`i32`, `i32`]).

    This operation contains a region with `func.func` operations, each
    implementing one of the trait's methods.

    Example:
        trait.impl @PartialEq_impl_i32 @PartialEq for [i32,i32] {
          func.func @eq(%self: i32, %other: i32) -> i1 {
            %res = arith.cmpi eq, %self, %other : i32
            return %res : i1
          }
        }
  }];

  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TraitApplicationAttr:$self_application,
    ConstraintsAttr:$assumptions
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    TraitApplicationAttr getSelfApplication() const {
      return const_cast<ImplOp*>(this)->getSelfApplication();
    }

    ClaimType getSelfClaim() {
      return ClaimType::get(getContext(), getSelfApplication());
    }

    FlatSymbolRefAttr getTraitNameAttr() const {
      return getSelfApplication().getTrait();
    }

    TraitOp getTrait();

    ArrayRef<Type> getTypeArgs() {
      return getSelfApplication().getTypeArgs();
    }

    func::FuncOp getMethod(StringRef name) {
      return lookupSymbol<func::FuncOp>(name);
    }

    bool hasMethod(StringRef name) {
      return getMethod(name) != nullptr;
    }

    func::FuncOp getOrInstantiateMethod(OpBuilder& builder, StringRef methodName);

    func::FuncOp getOrInstantiateFreeFunctionFromMethod(PatternRewriter& rewriter, ClaimType proof, StringRef methodName);

    DenseMap<Type,Type> buildSubstitutionFor(ClaimType claim);

    // Helper to generate unique symbol name
    static std::string generateSymName(TraitApplicationAttr selfApp, ConstraintsAttr assumptions);

    LogicalResult verifyAssumptions(ArrayRef<ImplOp> subproofs, llvm::function_ref<InFlightDiagnostic()> errFn);

    bool isSelfProof();

    LogicalResult verifyIsSelfProof(llvm::function_ref<InFlightDiagnostic()> errFn);

    SmallVector<ClaimType> getRequirementsAsClaims();
  }];

  // builders create one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    // override the default low-level builder to ensure that a block always exists
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "ConstraintsAttr":$assumptions), [{
      $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
      $_state.addAttribute("self_application", self_application);
      $_state.addAttribute("assumptions", assumptions);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>,

    // this builder automatically generates $sym_name if it is not given
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "ConstraintsAttr":$assumptions), [{
      // Auto-generate unique symbol name
      std::string symName = generateSymName(self_application, assumptions);
      build($_builder, $_state, symName, self_application, assumptions);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}


def Claim : TypeConstraint<
  CPred<"::mlir::isa<ClaimType>($_self)">,
  "any '!trait.claim' type"
>;


def AssumeOp : TraitDialectOp<"assume", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Materializes a claim value for a symbolic trait application";
  let description = [{
    Creates a zero-sized SSA value of type `!trait.claim<@Trait[types]>` that
    acts as an assumed claim within trait or impl method bodies. This operation
    has no operands.

    At least one type argument must be symbolic (!trait.poly); concrete types
    should use `trait.witness` instead. This operation must appear within a
    `trait.trait` or `trait.impl` region.

    The assumed trait application must exactly match one of:
      1. The `trait.trait` or `trait.impl`'s self application, or
      2. When used inside a `trait.impl`, an assumed trait application

    Example:
      trait.trait @PartialEq[!S,!O] {
        func.func @neq(%self: !S, %other: !O) -> i1 {
          %e = trait.assume @PartialEq[!S,!O]
          // ... use %e for method calls
        }
      }
  }];

  let results = (outs Claim:$result);

  let extraClassDeclaration = [{
    ClaimType getClaim() {
      return cast<ClaimType>(getResult().getType());
    }

    TraitApplicationAttr getTraitApplication() {
      return getClaim().getTraitApplication();
    }

    TraitOp getTrait();
  }];

  let hasCustomAssemblyFormat = 1;
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
      %p = trait.witness : @PartialEq[i32,i32]
      %x = ... : i32
      %y = ... : i32
      %res = trait.method.call @PartialEq::@eq<%p>(%x, %y) : (!trait.poly<0>, !trait.poly<1>) to (i32, i32) -> i1
  }];

  let arguments = (ins
    SymbolRefAttr:$method_ref,
    TypeAttrOf<FunctionType>:$method_function_type,
    Claim:$claim,
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
      "Value":$claim,
      "ValueRange":$arguments), [{
        MLIRContext *ctx = $_builder.getContext();

        // create the nested SymbolRefAttr
        SymbolRefAttr qualifiedRef = SymbolRefAttr::get(
          ctx,
          traitRef.getValue(),
          methodRef
        );

        // build the operation with the created nested reference
        return build($_builder, $_state, results, qualifiedRef, methodFunctionType, claim, arguments);
    }]>,

    // builder that takes trait and method names separately
    OpBuilder<(ins
      "TypeRange":$results,
      "StringRef":$traitName,
      "StringRef":$methodName,
      "TypeAttr":$methodFunctionType,
      "Value":$claim,
      "ValueRange":$arguments), [{
        // convert strings to FlatSymbolRefAttrs and call the other builder
        MLIRContext *ctx = $_builder.getContext();
        FlatSymbolRefAttr traitRef = FlatSymbolRefAttr::get(ctx, traitName);
        FlatSymbolRefAttr methodRef = FlatSymbolRefAttr::get(ctx, methodName);

        // call the other builder
        return build($_builder, $_state, results, traitRef, methodRef, methodFunctionType, claim, arguments);
    }]>,
  ];

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    func::FuncOp getOrInstantiateCallee(PatternRewriter& rewriter);

    TraitApplicationAttr getTraitApplication() {
      return cast<ClaimType>(getClaim().getType()).getTraitApplication();
    }

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

    ImplOp getProvenImpl();

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
    $method_ref `<` $claim `>` `(` $arguments `)` attr-dict `:` $method_function_type `as` type($claim) functional-type($arguments, results)
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
      %p = trait.witness @PartialEq[i32,i32]
      %r = trait.func.call @foo(%p, %x, %y)
        : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<0>]>, !trait.poly<0>, !trait.poly<0>) -> i1
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

    // returns the name of the (possibly instantiated) callee after instantiation
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

def ProjectOp : TraitDialectOp<"project", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]> {
  let summary = "Project a prerequisite claim from a claim";
  let description = [{
    Given a claim of `@Trait[Args...]`, produce a claim of one prerequisite
    `@Prereq[Args...]` of `@Trait` as declared in the `given` clause of
    `trait.trait @Trait`, after substituting `Args...` for `@Trait`'s type
    parameters.

    Notes:
    * Only a **single hop** is performed; chain `trait.project` for transitive projections.
    * If `@Trait` has no `given` clause, no projections from it are valid.

    Example:
      // @Ord[!S] given [ @Eq[!S], @PartialOrd[!S,!S] ]
      %p = trait.assume @Ord[!T]
      %eq = trait.project %p @Ord[!T] to @Eq[!T]
  }];

  let arguments = (ins Claim:$source);
  let results   = (outs Claim:$result);

  let extraClassDeclaration = [{
    TraitOp getSourceTrait();
    TraitOp getProjectedTrait();
  }];

  let hasCustomAssemblyFormat = 1;
}

def AllegeOp : TraitDialectOp<"allege", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]> {
  let summary = "Unverified claim that a trait implementation exists";
  let description = [{
    Alleges an unproven claim of a trait application (e.g., `@Trait[Types...]`)
    by producing a value of type `!trait.claim<@Trait[Types...]>`.

    The type arguments must be concrete; symbolic types are illegal.

    This operation does not require a matching `trait.impl` satisfaction.
    It is intended to be elaborated into verified `trait.witness` operations prior to
    monomorphization via a claim proving pass.

    Example:
      %p = trait.allege @Ord[i32]
  }];

  let results = (outs Claim:$result);

  let builders = [
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$app), [{
      mlir::MLIRContext* ctx = $_builder.getContext();
      auto claimTy = ClaimType::get(ctx, app);
      return build($_builder, $_state, claimTy);
    }]>
  ];

  let extraClassDeclaration = [{
    TraitOp getTrait();
    ArrayRef<Type> getTypeArgs();
    TraitApplicationAttr getTraitApplication() {
      return cast<ClaimType>(getResult().getType()).getTraitApplication();
    }
    FlatSymbolRefAttr getTraitAttr();

    SmallVector<TraitApplicationAttr> getRequiredTraitApplications();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def ProofOp : TraitDialectOp<"proof", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasParent<"ModuleOp">,
  IsolatedFromAbove,
  Symbol,
  Pure
]>{
  let summary = "Canonical proof tree node";
  let description = [{
    Declares a named, canonical proof that a `trait.impl` satisfies a concrete trait application,
    and that all one-hop trait requirements required by the trait under that application and 
    assumptions of the target impl are already proven by other symbols (either other `trait.proof`s or `trait.impl`s).

    Leaf `trait.impl`s have nothing to discharge and may not be wrapped in a `trait.proof`; they are self-proofs.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,                // the name of this proof
    FlatSymbolRefAttr:$impl_name,            // the name of the impl being proven
    TraitApplicationAttr:$trait_application, // must match the impl's trait application
    ArrayAttr:$subproof_names                // the names of prerequisite proofs
  );

  let extraClassDeclaration = [{
    ImplOp getImpl() {
      ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
      if (!module)
        return nullptr;
      return mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, getImplNameAttr());
    }

    TraitOp getTrait();

    SmallVector<ClaimType> getSubproofClaims();
  }];

  let assemblyFormat = "$sym_name `proves` $impl_name `for` $trait_application `given` $subproof_names attr-dict";
  let hasVerifier = 1;
}


def WitnessOp : TraitDialectOp<"witness", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Materializes a proven claim value for a concrete trait implementation";
  let description = [{
    Introduces a value of type `!trait.claim<@Trait[Types...], @Symbol>` that
    represents a proof that the trait application is satisfied.

    The symbol may refer to either
      * a `trait.proof` that transitively discharges the trait's requirements and the impl's assumptions
      * a `trait.impl` that is self-proving

    Verification ensures that the symbol proves the trait application as claimed.

    Example:
      // @PartialEq has no obligations, so we refer directly to a trait.impl
      %partial_eq_w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

      // @Ord has obligations, so we refer to a trait.proof
      %ord_w = trait.witness @Ord_i32_p for @Ord[i32]
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$proof, // refers to trait.proof or trait.impl
    TraitApplicationAttr:$trait_application
  );

  let builders = [
    // builder infers the result type
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$proof,
                   "::mlir::trait::TraitApplicationAttr":$trait_application), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::get(ctx, trait_application, proof);
      build($_builder, $_state, claimTy, proof, trait_application);
    }]>
  ];

  let results = (outs Claim:$result);

  let hasCustomAssemblyFormat = 1;
}


#endif // TRAIT_OPS_TD
