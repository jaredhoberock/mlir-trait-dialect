// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
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


// Adds a `getModule()` convenience method to the op.
def HasGetModule : NativeOpTrait<"HasGetModule"> {
  let cppNamespace = "::mlir::trait";
}

// Op's regions have only these types of child operations.
class HasChildren<list<string> ops>
    : ParamNativeOpTrait<"HasOnlyChildOps", !interleave(ops, ", ")>, StructuralOpTrait;


def TraitOp : TraitDialectOp<"trait", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChildren<["func::FuncOp", "trait::AssocTypeOp"]>,
  HasGetModule,
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
    parameters and is stored in the `requirements` argument.

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
    TraitApplicationArrayAttr:$requirements
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

    FailureOr<func::FuncOp> getMethod(StringRef methodName, llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto func = dyn_cast_or_null<func::FuncOp>(mlir::SymbolTable::lookupSymbolIn(*this, methodName));
      if (!func) {
        if (err) err() << "cannot find method '@" << methodName
                       << "' in trait '" << getSymNameAttr() << "'";
        return failure();
      }
      return func;
    }

    bool hasMethod(StringRef methodName) {
      return succeeded(getMethod(methodName));
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

    FailureOr<func::FuncOp> getOptionalMethod(StringRef methodName) {
      for (auto method : getOptionalMethods()) {
        if (method.getSymName() == methodName)
          return method;
      }
      return failure();
    }

    /// Builds a substitution mapping type parameters in this TraitOp to types found in the given claim
    FailureOr<DenseMap<Type,Type>> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                 llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    bool hasRequirements() {
      return !getRequirements().empty();
    }

    SmallVector<TraitOp,4> getRequiredTraits();

    SmallVector<ClaimType> getRequirementsAsClaims();

    FailureOr<SmallVector<ClaimType>> specializeRequirementsAsClaimsFor(
        ClaimType actualSelfClaim,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// Return all AssocTypeOps declared in this trait
    SmallVector<AssocTypeOp> getAssociatedTypes() {
      SmallVector<AssocTypeOp> result;
      for (Operation& op : getBody().front())
        if (auto assoc = dyn_cast<AssocTypeOp>(op))
          result.push_back(assoc);
      return result;
    }

    /// Look up a specific associated type by name
    FailureOr<AssocTypeOp> getAssociatedType(StringRef name) {
      for (auto assoc : getAssociatedTypes())
        if (assoc.getSymName() == name)
          return assoc;
      return failure();
    }

    /// Return all the ImplOps for this TraitOp
    SmallVector<ImplOp> getImpls();

    /// Return a list of ImplOps of this TraitOp whose self claim types
    /// can be specialized against wanted
    SmallVector<ImplOp> getCandidateImplsFor(ClaimType wanted);
  }];

  // builders create one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    // low-level builder
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayAttr":$type_params,
                   "TraitApplicationArrayAttr":$requirements), [{
      $_state.addAttribute("sym_name", sym_name);
      $_state.addAttribute("type_params", type_params);
      $_state.addAttribute("requirements", requirements);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>,

    // convenience accepts type_params as ArrayRef<Type>
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayRef<Type>":$type_params,
                   "TraitApplicationArrayAttr":$requirements), [{
      MLIRContext* ctx = $_builder.getContext();
      SmallVector<Attribute,8> typeAttrs;
      typeAttrs.reserve(type_params.size());
      for (Type ty : type_params) {
        typeAttrs.push_back(TypeAttr::get(ty));
      }
      ArrayAttr typeParamsAttr = ArrayAttr::get(ctx, typeAttrs);
      build($_builder, $_state, sym_name, typeParamsAttr, requirements);
    }]>,

    // convenience accepts requirements as an array of TraitApplicationAttr
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayRef<Type>":$type_params,
                   "::mlir::ArrayRef<TraitApplicationAttr>":$requirements), [{
      MLIRContext* ctx = $_builder.getContext();
      auto reqsAttr = TraitApplicationArrayAttr::get(ctx, requirements);
      build($_builder, $_state, sym_name, type_params, reqsAttr);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChildren<["func::FuncOp", "trait::AssocTypeOp"]>,
  HasGetModule,
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
    clause (stored in the `assumptions` argument). Assumptions are one-hop claims,
    analogous to trait-level `where` requirements, and are materializable inside
    method bodies via `trait.assume`.

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
    TraitApplicationArrayAttr:$assumptions
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

    FailureOr<func::FuncOp> getMethod(StringRef methodName, llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto func = dyn_cast_or_null<func::FuncOp>(mlir::SymbolTable::lookupSymbolIn(*this, methodName));
      if (!func) {
        if (err) err() << "cannot find method '@" << methodName
                       << "' in impl '" << getSymNameAttr() << "'";
        return failure();
      }
      return func;
    }

    bool hasMethod(StringRef name) {
      return succeeded(getMethod(name));
    }

    // returns all distinct generic types that occur in this ImplOp's self application and assumptions
    SmallVector<GenericTypeInterface, 4> getTypeParams();

    FailureOr<func::FuncOp> getOrInstantiateMethod(OpBuilder& builder, StringRef methodName);

    /// Extracts a method from this impl as a free function with a leading self parameter.
    ///
    /// Builds a poly-to-concrete substitution from `provenSelfClaim`, then merges
    /// `extraSubst` (e.g., projection-to-concrete bindings from `addProjectionBindings`)
    /// before monomorphizing the clone.
    FailureOr<func::FuncOp> getOrInstantiateFreeFunctionFromMethod(PatternRewriter& rewriter, ClaimType provenSelfClaim, StringRef methodName,
                                                                    const DenseMap<Type,Type> &extraSubst);

    /// Builds a substitution that maps generic types appearing in the ImplOp's type parameters
    /// to Types in the given actualSelfClaim
    FailureOr<DenseMap<Type,Type>> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                 llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// Build the substitution needed to monomorphize this impl for a **given proven self claim**.
    ///
    /// Returns a `DenseMap<Type,Type>` that:
    /// - Replaces each generic type in the impl’s self claim with the concrete type from `proof`.
    /// - Adds **Claim → proven Claim** bindings for the impl’s self claim and all specialized
    ///   obligations so downstream rewriting preserves proof annotations.
    /// The result has no inference variables and is safe to pass to instantiation utilities.
    FailureOr<DenseMap<Type,Type>> buildMonomorphizationSubstitutionFor(ClaimType provenSelfClaim,
                                                                        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // Helper to generate unique symbol name for an ImplOp
    static std::string generateSymName(TraitApplicationAttr selfApp, TraitApplicationArrayAttr assumptions);

    // Helper to generate a unique symbol prefix for polymophic ImplOps
    // getSelfClaim() will be substituted with claim
    std::string generateMangledName(ClaimType claim);

    bool isUnconditional();

    LogicalResult verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> errFn);

    SmallVector<ClaimType> getAssumptionsAsClaims();

    FailureOr<SmallVector<ClaimType>> specializeAssumptionsAsClaimsFor(
        ClaimType actualSelfClaim,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    FailureOr<SmallVector<ClaimType>> specializeObligationsAsClaimsFor(
        ClaimType actualSelfClaim,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// Look up an associated type op by name
    FailureOr<AssocTypeOp> getAssociatedType(StringRef name) {
      for (Operation& op : getBody().front())
        if (auto assoc = dyn_cast<AssocTypeOp>(op))
          if (assoc.getSymName() == name)
            return assoc;
      return failure();
    }

    /// Look up an associated type binding by name, without GAT substitution.
    /// For GAT types, the result may reference the impl's own poly IDs
    /// instead of concrete types. Use `specializeAssociatedTypeBinding`
    /// to get a fully-substituted result.
    FailureOr<Type> getAssociatedTypeBinding(
        StringRef name,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      for (Operation& op : getBody().front()) {
        if (auto assoc = dyn_cast<AssocTypeOp>(op)) {
          if (assoc.getSymName() == name) {
            if (auto boundType = assoc.getBoundType())
              return *boundType;
            if (err) err() << "associated type '" << name
                           << "' has no binding in impl '" << getSymName() << "'";
            return failure();
          }
        }
      }
      if (err) err() << "no associated type '" << name
                     << "' in impl '" << getSymName() << "'";
      return failure();
    }

    /// Look up an associated type binding and apply GAT substitution.
    ///
    /// Given an associated type `name` (e.g., "Output") and concrete type
    /// arguments `assocTypeArgs` for a GAT, returns the fully-resolved
    /// concrete type that the impl binds to that associated type.
    ///
    /// For non-GAT associated types, `assocTypeArgs` should be empty and
    /// the raw binding is returned directly.
    ///
    /// For GATs, the raw binding from the impl may reference the impl's own
    /// poly IDs for the GAT's type parameters (which may differ from the
    /// trait's). This method substitutes `assocTypeArgs` for those parameters
    /// to produce the final resolved type.
    ///
    /// Returns failure if:
    ///   - No associated type named `name` exists in this impl
    ///   - The associated type has no bound type
    ///   - GAT arity mismatch (`assocTypeArgs` size != type parameter count)
    FailureOr<Type> specializeAssociatedTypeBinding(
        StringRef name,
        ArrayRef<Type> assocTypeArgs,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);
  }];

  // builders create one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    // override the default low-level builder to ensure that a block always exists
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "TraitApplicationArrayAttr":$assumptions), [{
      $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
      $_state.addAttribute("self_application", self_application);
      $_state.addAttribute("assumptions", assumptions);
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>,

    // this builder automatically generates $sym_name if it is not given
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "TraitApplicationArrayAttr":$assumptions), [{
      // Auto-generate unique symbol name
      std::string symName = generateSymName(self_application, assumptions);
      build($_builder, $_state, symName, self_application, assumptions);
    }]>,

    // convenience wraps the assumptions in a TraitApplicationArrayAttr and defers to the lower-level builder
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<TraitApplicationAttr>":$assumptions), [{
      MLIRContext *ctx = $_builder.getContext();
      auto appArrayAttr = TraitApplicationArrayAttr::get(ctx, assumptions);
      build($_builder, $_state, self_application, appArrayAttr);
    }]>,

    // convenience accepts sym_name as StringRef and assumptions as ArrayRef
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<TraitApplicationAttr>":$assumptions), [{
      MLIRContext *ctx = $_builder.getContext();
      auto appArrayAttr = TraitApplicationArrayAttr::get(ctx, assumptions);
      build($_builder, $_state, sym_name, self_application, appArrayAttr);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}


def AssocTypeOp : TraitDialectOp<"assoc_type", [
  ParentOneOf<["trait::TraitOp", "trait::ImplOp"]>,
  Symbol
]> {
  let summary = "Declares or binds an associated type";
  let description = [{
    Inside a `trait.trait`, declares an associated type by name:
    ```mlir
    trait.trait @Iterator[!S] {
      trait.assoc_type @Item
      func.func private @next(!S) -> !trait.proj<@Iterator[!S], "Item">
    }
    ```

    Inside a `trait.impl`, binds the associated type to a concrete type:
    ```mlir
    trait.impl for @Iterator[i32] {
      trait.assoc_type @Item = i64
      func.func @next(%self: i32) -> i64 { ... }
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttr>:$bound_type,
    OptionalAttr<TypeArrayAttr>:$type_params
  );

  let hasCustomAssemblyFormat = 1;
}


def Claim : Type<
  CPred<"::mlir::isa<ClaimType>($_self)">,
  "any '!trait.claim' type",
  "ClaimType"
>;

def Projection : Type<
  CPred<"::mlir::isa<ProjectionType>($_self)">,
  "any '!trait.proj' type",
  "ProjectionType"
>;


//===----------------------------------------------------------------------===//
// Claim-producing ops
//===----------------------------------------------------------------------===//
//
// Five operations produce `!trait.claim` values. Together they cover every way
// a claim can originate or be transformed:
//
//   Op              Role                                         Analogy
//   ----------      ------------------------------------------   -----------
//   trait.allege    Fabricate a monomorphic claim at a call       Assertion
//                   site, to be proven later by impl resolution
//   trait.assume    Materialize the enclosing trait/impl's        Hypothesis
//                   self claim or a declared assumption
//   trait.derive    Construct a new claim from a conditional      Introduction
//                   impl plus existing claim operands
//   trait.project   Extract a sub-claim from an existing claim    Elimination
//                   via a trait requirement or impl assumption
//   trait.witness   Materialize a proven claim backed by a        Evidence
//                   trait.proof or unconditional trait.impl
//
// Claims also flow through function parameters as `!trait.claim` SSA values,
// but that is plain SSA plumbing rather than a dedicated op.
//===----------------------------------------------------------------------===//


def AllegeOp : TraitDialectOp<"allege", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]> {
  let summary = "Unverified, monomorphic claim that a trait implementation exists";
  let description = [{
    Produces a `!trait.claim<@Trait[Types...]>` value asserting that a trait
    application is satisfied. The type arguments must be concrete (monomorphic);
    polymorphic types are illegal.

    This restriction enforces a key invariant of the trait system: polymorphic
    code never fabricates claims. Claims originate at monomorphic call sites
    via `trait.allege` and flow into polymorphic code through `!trait.claim`
    parameters. A polymorphic function that needs a trait satisfaction must
    declare it as a bound and receive a claim value from its caller. This
    ensures that polymorphic code is correct by construction because it cannot
    assume a trait is implemented without the caller proving it.

    `trait.allege` does not verify that a matching `trait.impl` exists at the
    point of emission. It is intended to be elaborated into verified
    `trait.witness` operations by a claim-proving pass prior to
    monomorphization. Unresolved monomorphic alleges after that pass are
    errors.

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


def AssumeOp : TraitDialectOp<"assume", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Materializes a claim value for a symbolic trait application";
  let description = [{
    Creates a zero-sized SSA value of type `!trait.claim<@Trait[types]>` that
    acts as an assumed claim within polymorphic code. This operation has no
    operands.

    The assumed trait application must match one of (checked in order):
      1. A `!trait.claim`-typed parameter of the enclosing function
      2. The enclosing `trait.trait`'s self application
      3. The enclosing `trait.impl`'s self application or a declared assumption

    Example inside a trait method body:
      trait.trait @PartialEq[!S,!O] {
        func.func @eq(!S, !O) -> i1

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

    Example inside a standalone polymorphic function:
      func.func @call_foo(%c: !trait.claim<@Foo[!T]>, %x: !T) -> !T {
        %a = trait.assume @Foo[!T]
        ...
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

  let builders = [
    // convenience builds from a TraitApplicationAttr instead of a ClaimType
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$assumption), [{
      auto ctx = $_builder.getContext();
      build($_builder, $_state, ClaimType::get(ctx, assumption));
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}


def DeriveOp : TraitDialectOp<"derive", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Derive a claim from a conditional impl and assumption claims";
  let description = [{
    Produces a new claim by combining a conditional `trait.impl` with claim
    operands that satisfy its assumptions. None of the other claim-producing
    ops can construct a claim from a conditional impl; consider
    `impl<U: Trait> Trait for Struct<U> where [@Trait[U]]`: given an
    in-scope claim for `Trait[!T]`, there is no way to produce
    `Trait[Struct<!T>]` without `trait.derive`.

    It names the conditional impl and takes claim operands satisfying its
    assumptions, producing the impl's derived claim. During
    `instantiateMonomorphs`, once types are concrete, it is rewritten into
    a `trait.witness` backed by a minted `trait.proof`.

    Example:

    ```mlir
    // Given: impl<U: Trait> Trait for Struct<U> where [@Trait[U]]
    // and an in-scope claim %t for Trait[!T]:
    %d = trait.derive @Trait[Struct<!T>] from @the_impl given(%t)
    ```

    Verification ensures:
    * The `@impl` symbol refers to a `trait.impl`.
    * The impl can be specialized for the derived claim.
    * The number of assumption operands matches the impl's assumption count.
    * Each operand's claim type matches the corresponding specialized assumption.
  }];

  let arguments = (ins
    TraitApplicationAttr:$trait_application,
    FlatSymbolRefAttr:$impl,
    Variadic<Claim>:$assumptions
  );

  let results = (outs Claim:$result);

  let builders = [
    OpBuilder<(ins "TraitApplicationAttr":$traitApp,
                   "FlatSymbolRefAttr":$impl,
                   "ValueRange":$assumptions), [{
      ClaimType claimTy = ClaimType::get($_builder.getContext(), traitApp);
      build($_builder, $_state, claimTy, traitApp, impl, assumptions);
    }]>
  ];

  let extraClassDeclaration = [{
    ClaimType getDerivedClaim() {
      return cast<ClaimType>(getResult().getType());
    }

    ImplOp getImplOp();
  }];

  let hasCustomAssemblyFormat = 1;
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
      * a `trait.impl` that is unconditional

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


def ProjWitnessOp : TraitDialectOp<"proj.witness", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]> {
  let summary = "Wrap a concrete value with projection evidence";
  let description = [{
    Wraps a concrete-typed value into a proven `!trait.proj` type, providing
    evidence that the projection resolves to the concrete type.

    Example:
      %val_proj = trait.proj.witness %val by @Base_proof
        : i1 -> !trait.proj<@Base[i64], "Assoc" by @Base_proof>
  }];

  let arguments = (ins
    AnyType:$input,
    FlatSymbolRefAttr:$proof
  );
  let results = (outs Projection:$result);

  let assemblyFormat = "$input `by` $proof attr-dict `:` type($input) `->` type($result)";
}


//===----------------------------------------------------------------------===//
// Call ops
//===----------------------------------------------------------------------===//


def MethodCallOp : TraitDialectOp<"method.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasGetModule
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
        : (i32, i32) -> i1

      // get a proven claim value for @PartialEq[i32,i32]
      %partial_eq_p = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

      // call @eq using that proven claim
      %res1 = trait.method.call %partial_eq_p @PartialEq[i32,i32]::@eq(%x, %y)
        : (i32, i32) -> i1
        by @PartialEq_impl_i32_i32
  }];

  let arguments = (ins
    SymbolRefAttr:$method_ref,
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
        return build($_builder, $_state, results, qualifiedRef, claim, arguments);
    }]>,

    // builder that takes trait and method names separately
    OpBuilder<(ins
      "TypeRange":$results,
      "StringRef":$traitName,
      "StringRef":$methodName,
      "Value":$claim,
      "ValueRange":$arguments), [{
        // convert strings to FlatSymbolRefAttrs and call the other builder
        MLIRContext *ctx = $_builder.getContext();
        FlatSymbolRefAttr traitRef = FlatSymbolRefAttr::get(ctx, traitName);
        FlatSymbolRefAttr methodRef = FlatSymbolRefAttr::get(ctx, methodName);

        // call the other builder
        return build($_builder, $_state, results, traitRef, methodRef, claim, arguments);
    }]>,
  ];

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    FailureOr<TraitOp> getTrait(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    FailureOr<func::FuncOp> getMethod(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    FailureOr<FunctionType> getMethodFunctionType(llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto maybeMethod = getMethod(err);
      if (failed(maybeMethod)) return failure();
      return maybeMethod->getFunctionType();
    }

    /// Get a FunctionType corresponding to the actual argument and result types of this MethodCallOp
    /// Note that the claim type is not part of the function type
    FunctionType getActualFunctionType() {
      return FunctionType::get(getContext(), getArguments().getTypes(), getResultTypes());
    }

    /// Build a substitution by unifying the callee's formal parameter types
    /// with the actual argument types at this call site.
    ///
    /// The resulting substitution maps generic types to concrete types and
    /// claims to proven claims. It does NOT resolve projection types in the
    /// callee's return type; use `addProjectionBindings` for that.
    FailureOr<DenseMap<Type,Type>> buildParameterSpecialization(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Instantiates the callee as a free function using a pre-built substitution.
    FailureOr<func::FuncOp> getOrInstantiateCallee(PatternRewriter &rewriter,
                                                   const DenseMap<Type,Type> &subst);

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

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}


def FuncCallOp : TraitDialectOp<"func.call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasGetModule
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
        : (!trait.claim<@PartialEq[i32,i32]>, i32, i32) -> i1
  }];

  let arguments = (ins
      FlatSymbolRefAttr:$callee_name,
      Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    FailureOr<func::FuncOp> getCallee(llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto name = getCalleeNameAttr();
      auto module = getModule(err);
      if (failed(module)) return failure();

      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*module, name);
      if (!callee) {
        if (err) err() << "'" << name.getValue()
                       << "' does not refer to a valid func.func";
        return failure();
      }
      return callee;
    }

    FailureOr<FunctionType> getCalleeFunctionType(llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto callee = getCallee(err);
      if (failed(callee)) return failure();
      return callee->getFunctionType();
    }

    /// Get a FunctionType corresponding to the actual operand and result types of this FuncCallOp
    FunctionType getActualFunctionType() {
      return FunctionType::get(getContext(), getOperands().getTypes(), getResultTypes());
    }

    /// Looks up or instantiates the monomorphized callee using a pre-built substitution.
    FailureOr<func::FuncOp> getOrInstantiateCallee(OpBuilder &builder,
                                                   const DenseMap<Type,Type> &subst);

    /// Build a substitution by unifying the callee's formal parameter types
    /// with the actual argument types at this call site.
    ///
    /// The resulting substitution maps generic types to concrete types and
    /// claims to proven claims. It does NOT resolve projection types in the
    /// callee's return type; use `addProjectionBindings` for that.
    FailureOr<DenseMap<Type, Type>> buildParameterSpecialization(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // returns all distinct generic types that occur in the FunctionType of this FuncOp's callee
    SmallVector<GenericTypeInterface, 4> getCalleeTypeParams() {
      return getGenericTypesIn(*getCalleeFunctionType());
    }

    // returns the name of the (possibly instantiated) callee after monomorphization
    std::string getNameOfCalleeInstance();

    operand_range getArgOperands() {
      return getOperands();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee_name");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee_name", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = "$callee_name `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}


//===----------------------------------------------------------------------===//
// Proof infrastructure
//===----------------------------------------------------------------------===//


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

    Leaf `trait.impl`s have nothing to discharge and may not be wrapped in a `trait.proof`; they are unconditional self-proofs.
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

    /// Look up a proof symbol and return the underlying ImplOp.
    /// The symbol may name a ProofOp (which forwards to its impl) or
    /// an unconditional ImplOp directly.
    static FailureOr<ImplOp> getImplFromProof(
      ModuleOp module,
      FlatSymbolRefAttr name,
      llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    // look up a proof by name. It must be either a ProofOp or an unconditional impl
    static FailureOr<Operation*> getProofOpOrUnconditionalImplOp(
      ModuleOp module,
      FlatSymbolRefAttr name,
      llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);
  }];

  let builders = [
    // convenience: StringRef + ArrayRef<FlatSymbolRefAttr>
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::FlatSymbolRefAttr":$impl_name,
                   "::mlir::trait::TraitApplicationAttr":$trait_application,
                   "::mlir::ArrayRef<FlatSymbolRefAttr>":$subproof_names), [{
      SmallVector<Attribute> attrs(subproof_names.begin(), subproof_names.end());
      build($_builder, $_state,
            $_builder.getStringAttr(sym_name),
            impl_name,
            trait_application,
            $_builder.getArrayAttr(attrs));
    }]>
  ];

  let assemblyFormat = "$sym_name `proves` $impl_name `for` $trait_application `given` $subproof_names attr-dict";
  let hasVerifier = 1;
}


#endif // TRAIT_OPS_TD
