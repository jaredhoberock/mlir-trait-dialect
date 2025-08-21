#ifndef TRAIT_OPS_TD
#define TRAIT_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "Trait.td"
include "TraitTypes.td"


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
    Declares a trait interface, similar to a Rust or Scala trait, or a Haskell typeclass.
    A `trait.trait` operation defines a symbol that can be referenced by `!trait.claim`
    types, `trait.impl`, and `trait.method.call`. It contains a region which holds
    `func.func` operations that define the method requirements for implementors. If any of
    these `func.func` operations have a body, then the method is optional for implementors.

    Example:

    ```mlir
    !S = !trait.poly<0>
    !O = !trait.poly<1>

    trait.trait @PartialEq[!S,!O] {
      // a required method
      func.func @eq(!S, !O) -> i1

      // an optional method
      func.func @ne(%self: !S, %other: !O) -> i1 {
        %a = trait.assume @PartialEq[!S,!O]
        %eq = trait.method.call @PartialEq::@eq<%a>(%self, %other)
          : (!S, !O) -> i1
          as !trait.claim<@PartialEq[!S,!O]> (!S,!O) -> i1
        %true = arith.constant 1 : i1
        %res = arith.xori %eq, %true : i1
        return %res : i1
      }
    }
    ```

    The `where` clause lists the one-hop prerequisite trait applications that must hold
    whenever the trait holds. Each requirement is written using the trait's type
    parameters and is stored in the `requirements` (`ConstraintsAttr`) argument.

    Concretely, if you declare:

    ```mlir
    !S = !trait.poly<0>
    !O = !trait.poly<1>

    trait.trait @PartialOrd[!S,!O] where [@PartialOrd[!S,!O]] {
      func.func @partial_cmp(!S, !O) -> !opt_ord

      func.func @lt(%self: !S, %other: !O) -> !opt_ord {
        ...
      }
    }
    ```

    Then any proof of `@PartialOrd[T,U]` must also discharge `@PartialEq[T,U]`.
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

    SmallVector<ClaimType> getRequirementsAsClaimsWith(const DenseMap<Type,Type>& subst);

    SmallVector<ImplOp> getImpls();
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

    ```mlir
    trait.impl @PartialEq_impl_i32 @PartialEq for [i32,i32] {
      func.func @eq(%self: i32, %other: i32) -> i1 {
        %res = arith.cmpi eq, %self, %other : i32
        return %res : i1
      }
    }
    ```

    Implementations may also declare **assumptions** that must hold for the impl
    to be valid under its self application. These are listed in the `where [...]`
    clause (stored in the `assumptions` `ConstraintsAttr` argument). Assumptions
    are one-hop claims, analogous to trait-level `where` requirements, and are
    materializable inside method bodies via `trait.assume`.

    Concretely, given a couple of traits:

    ```mlir
    !T0 = !trait.poly<0>
    trait.trait @A[!T0] {
      func.func @method_a(!T0) -> i32
    }

    !T1 = !trait.poly<1>
    trait.trait @B[!T1] {
      func.func @method_B(!T1) -> i32
    }
    ```

    you can write a polymorphic impl that depends on other claims:

    ```mlir
    // a concrete impl: @B[i32]
    trait.impl for @B[i32] {
      func.func @method_b(%self: i32) -> i32 {
        %res = arith.constant 42 : i32
        return %res : i32
      }
    }

    // this polymorphic impl matches any type T2 with an impl for @B[T2]
    !T2 = !trait.poly<2>
    trait.impl for @A[!T2] where [
      @B[!T2]
    ] {
      func.func @method_a(%self: !T2) -> i32 {
        %a = trait.assume @B[!T2]
        %res = trait.method.call @B::@method_b<%a>(%self)
          : !(T1) -> i32
          as !trait.claim<@B[!T2]> (!T2) -> i32
        return %res : i32
      }
    }
    ```
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
      return getSelfApplication().getTraitName();
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

    // returns all distinct PolyTypes that occur in this ImplOp's self application and assumptions
    SmallVector<PolyType, 4> getTypeParams();

    func::FuncOp getOrInstantiateMethod(OpBuilder& builder, StringRef methodName);

    func::FuncOp getOrInstantiateFreeFunctionFromMethod(PatternRewriter& rewriter, ClaimType proof, StringRef methodName);

    DenseMap<Type,Type> buildSubstitutionFor(ClaimType claim);

    // Helper to generate unique symbol name for an ImplOp
    static std::string generateSymName(TraitApplicationAttr selfApp, ConstraintsAttr assumptions);

    // Helper to generate a unique symbol prefix for polymophic ImplOps
    // getSelfClaim() will be substituted with claim
    std::string generateMangledName(ClaimType claim);

    bool isSelfProof();

    LogicalResult verifyIsSelfProof(llvm::function_ref<InFlightDiagnostic()> errFn);

    SmallVector<ClaimType> getAssumptionsAsClaimsWith(const DenseMap<Type,Type>& subst);

    // impl obligations = trait requirements concatentated with impl assumptions
    SmallVector<ClaimType> getObligationsAsClaimsWith(const DenseMap<Type,Type>& subst);
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
        func.func @eq(!S, !O) -> i1

        func.func @ne(%self: !S, %other: !O) -> i1 {
          // trait.assume materializes a claim value
          // to use inside a method body
          %a = trait.assume @PartialEq[!S,!O]

          %eq = trait.method.call @PartialEq::@eq<%a>(%self, %other)
            : (!S, !O) -> i1
            as !trait.claim<@PartialEq[!S,!O]> (!S,!O) -> i1
          %true = arith.constant 1 : i1
          %res = arith.xori %eq, %true : i1
          return %res : i1
        }
      }

    Note that `trait.assume` is the *only* context-sensitive operation in this dialect.
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
      %x = ... : i32
      %y = ... : i32

      // get an unproven claim value for @PartialEq[i32,i32]
      %partial_eq = trait.allege : @PartialEq[i32,i32]

      // call @eq using that unproven claim
      %res0 = trait.method.call %partial_eq @PartialEq[i32,i32]::@eq(%x, %y)
        :  (!trait.poly<0>, !trait.poly<1>)
        as (i32, i32) -> i1

      // get a proven claim value for @PartialEq[i32,i32]
      %partial_eq_p = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

      // call @eq using that proven claim
      %res1 = trait.method.call %partial_eq_p @PartialEq[i32,i32]::@eq(%x, %y)
        :  (!trait.poly<0>, !trait.poly<1>)
        as (i32, i32) -> i1
        by @PartialEq_impl_i32_i32
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

    ClaimType getClaimType() {
      return cast<ClaimType>(getClaim().getType());
    }

    TraitApplicationAttr getTraitApplication() {
      return getClaimType().getTraitApplication();
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

//  let assemblyFormat = [{
//    $method_ref `<` $claim `>` `(` $arguments `)` attr-dict `:` $method_function_type `as` type($claim) functional-type($arguments, results)
//  }];

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;
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
      // a polymorphic function @foo
      !T = !trait.poly<2>
      func.func @foo(
        %c: !trait.claim<@PartialEq[!T,!T]>,
        %x: !T,
        %y: !T
      ) -> i1 {
        ...
      }

      ...

      %x = ... : i32
      %y = ... : i32
      %a = trait.allege @PartialEq[i32,i32]
      %r = trait.func.call @foo(%a, %x, %y)
        : (!trait.claim<@PartialEq[!S,!O]>, !trait.poly<0>, !trait.poly<0>) -> i1
        to (!trait.claim<@PartialEq[i32,i32]>, i32, i32) -> i1
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
  let summary = "Project a subclaim from a claim";
  let description = [{
    Given a claim of `@Trait[Args...]`, produce a claim of either:
      1. the **identity** projection `@Trait[Args...]` itself, or
      2. a one-hop **trait requirement** `@Req[Args...]` declared in the
         `where` clause of `trait.trait @Trait`, or
      3. (if the input claim is **proven**) an **impl assumption** declared in the
         `where` clause of the referenced `trait.impl` (or discharged via
         its `trait.proof`).

    Substitution of `Args...` for `@Trait`'s type parameters is performed prior
    to matching the requested projection target.

    Notes:
    * Only a **single hop** is performed; chain `trait.project` for transitive projections.
    * Identity projection is always valid.
    * If `@Trait` has no `where` requirements and any proven impl has no assumptions,
      only the identity projection is possible.

    Examples:

    ```mlir
    // identity projection
    %id = trait.project %p : @A[i32] to @A[i32] // same claim as %p

    // trait-level requirement (works for unproven or proven source claims)
    // @Ord[!S] where [ @Eq[!S], @PartialOrd[!S,!S] ]
    %p = trait.assume @Ord[!T]
    %eq = trait.project %p : @Ord[!T] to @Eq[!T]

    // impl assumption (requires a proven source claim)
    %wA = trait.witness @A_impl_i32 for @A[i32]
    %wB = trait.project %wA : @A[i32] by @A_impl_i32 to @B[i32] by @B_impl_i32
    ```

    Verification:
    * For an **unproven** source claim (e.g., from `trait.assume` or `trait.allege`),
      the target must match one of `@Trait`'s `where` requirements after substitution,
      or be the identity. The target of an unproven source claim must also be unproven.
    * For a **proven** source claim (e.g., from `trait.witness`), the target may match
      the identity, a trait-level requirement, or an assumption of the referenced proof,
      after substitution. The target of a proven source claim must also be proven.
  }];

  let arguments = (ins Claim:$source);
  let results   = (outs Claim:$result);

  let extraClassDeclaration = [{
    ClaimType getSourceClaim() {
      return cast<ClaimType>(getSource().getType());
    }

    ClaimType getResultClaim() {
      return cast<ClaimType>(getResult().getType());
    }

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
    ClaimType getClaim() {
      return cast<ClaimType>(getResult().getType());
    }
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
    ClaimType getProvenClaim() {
      MLIRContext* ctx = getContext();
      FlatSymbolRefAttr symRef = FlatSymbolRefAttr::get(ctx, getSymNameAttr());
      return ClaimType::get(ctx, getTraitApplication(), symRef); 
    }

    ImplOp getImpl() {
      ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
      if (!module)
        return nullptr;
      return mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, getImplNameAttr());
    }

    TraitOp getTrait();

    FailureOr<SmallVector<ClaimType>> verifyAndGetSubproofClaims(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    SmallVector<ClaimType> getSubproofClaims() {
      auto maybeResult = verifyAndGetSubproofClaims();
      if (failed(maybeResult))
        llvm_unreachable("ProofOp::getSubproofClaims: verifyAndGetSubproofClaims failed");
      return *maybeResult;
    }

    SmallVector<ClaimType> getImplAssumptionClaims();

    // look up a proof by name. It must be either a ProofOp or a self-proving impl
    static FailureOr<Operation*> getProofOpOrSelfProofImplOp(
      ModuleOp module,
      FlatSymbolRefAttr name,
      llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);
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
      %ord_w = trait.witness @Ord_impl_i32_p for @Ord[i32]
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

  let extraClassDeclaration = [{
    ClaimType getProvenClaim() {
      return ClaimType::get(getContext(), getTraitApplication(), getProofAttr());
    }
  }];

  let hasCustomAssemblyFormat = 1;
}


#endif // TRAIT_OPS_TD
