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


//===----------------------------------------------------------------------===//
// Design principles
//===----------------------------------------------------------------------===//
//
// Goal: polymorphic IR is correct by construction.  A well-formed module
// that passes verification cannot monomorphize into ill-typed code.  The
// principles below exist to make that goal achievable and maintainable.
//
// 1. Claims are first-class SSA values.  A claim is an assertion that a
//    trait is satisfied for specific types (e.g., @PartialEq[i32,i32]).
//    Claims are represented as explicit !trait.claim SSA values flowing
//    through SSA edges.  Every op that needs a trait guarantee takes a
//    claim operand.  This makes proof obligations visible and trackable
//    in the IR rather than implicit in the context.
//
// 2. Verification is always local.  Because claims are SSA values,
//    verifiers inspect operands directly — no searching op-def chains,
//    no impl resolution, no global analysis.  This keeps verifiers simple
//    and free of surprising dependencies on distant IR.  The one exception
//    is trait.assume, which is semi-local (inspects the enclosing scope).
//    Impl resolution is a separate global pass that runs before
//    monomorphization.
//
// 3. Polymorphic code never fabricates claims — with one exception.
//    Claims normally originate at monomorphic call sites (trait.allege)
//    and flow into polymorphic code through !trait.claim parameters.
//    This ensures that polymorphic code cannot assume a trait is
//    implemented without the caller proving it — the core of "correct
//    by construction."
//
//    The exception is `trait.allege ... unsafe`, which allows polymorphic
//    code to fabricate a claim whose types are not yet concrete.  These
//    claims are skipped by resolve-impls and resolved post-monomorphization
//    once types are concrete.  This exists for ImplGenerator-provided
//    evidence where the obligation can only be checked after type
//    parameters are instantiated.
//
// 4. The IR is the sole authority on claims.  All proof structure lives
//    in the IR itself — no side tables, no external state.  Analysis
//    passes read the IR; transformation passes update it.  This keeps
//    the IR self-contained and makes correctness checkable by inspection.
//
//===----------------------------------------------------------------------===//


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

    The `where` clause is one ordered `PredicateArrayAttr` whose entries are of two
    arms: a one-hop trait application that must hold whenever the trait holds, or a
    type equality two spellings must satisfy (checked at an impl's verification against the
    impl's bindings, and projectable through `trait.project`'s equality hop). Each
    requirement is written using the trait's type parameters and is stored in the
    `requirements` argument.

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

    A `where` clause may mix both arms; an equality entry names a type equality the
    trait's associated types satisfy:

    ```mlir
    // @Callable[!S] must hold, and @FoldFn's "Output" projects to !S.
    trait.trait @FoldFn[!S] where [
      @Callable[!S],
      !trait.proj<@FoldFn[!S], "Output"> = !S
    ] { ... }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeArrayAttr:$type_params,
    PredicateArrayAttr:$requirements,
    OptionalAttr<StrAttr>:$sym_visibility
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

    /// Builds the specialization that maps this trait's generic parameters to
    /// the corresponding types in `actualSelfClaim`.
    FailureOr<SpecializationMap> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                           llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

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
    /// The impls whose headers match `wanted`, read through the established
    /// context `normalize`: a header spelling a projection reproduces a demand
    /// spelling its resolution only where the caller's context can reduce it.
    SmallVector<ImplOp> getCandidateImplsFor(ClaimType wanted,
                                             Normalizer normalize = Normalizer());
  }];

  // builders create one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    // low-level builder
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayAttr":$type_params,
                   "PredicateArrayAttr":$requirements), [{
      $_state.addAttribute("sym_name", sym_name);
      $_state.addAttribute("type_params", type_params);
      $_state.addAttribute("requirements", requirements);
      // A trait is a template that dies with monomorphization, so it is private
      // from birth: erase collects it by liveness, never by kind.
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>,

    // convenience accepts type_params as ArrayRef<Type>
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::ArrayRef<Type>":$type_params,
                   "PredicateArrayAttr":$requirements), [{
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
      auto reqsAttr = PredicateArrayAttr::get(ctx, requirements);
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

    The `where` clause is one ordered `PredicateArrayAttr`, the same mixed shape
    a `trait.trait` carries: each entry is either a trait application the impl
    assumes as a proof obligation, or a type equality the impl asserts about its
    own bindings. The two arms share one list in declaration order; there is no
    second parallel array. Application entries are discharged when the impl is
    selected; equality entries are checked at impl verification against the impl's own bindings
    and drive the equality-arm `trait.assume` inside the impl's method bodies.

    The optional `witnesses` array carries the impl's declaration-level
    witnesses, each a `#trait.witness` read by its predicate arm. An
    equality-armed witness is a projection-resolution witness: it names a
    ground sibling projection this impl's own bindings cannot resolve, the type
    it resolves to, and the impl that binds it. The verifier checks each against
    its cited impl and replays it as a local resolution rule while comparing this
    impl's method signatures and its equality endpoints, so a projection a sibling
    impl resolves is reduced without the verifier enumerating candidate impls; the
    replayed rules ride the same associated-type replacer the own bindings use,
    which holds an equality's endpoints as a leaf, so they are blind inside one
    and cure no equality-claim-parameter mismatch. When such a witness cites a
    CONDITIONAL sibling impl, its own assumptions must be discharged for the
    citation to be legal: an application-armed witness names, per obligation,
    the impl that supplies it, and the discharge recursion resolves every
    discharger by symbol without scanning the module.

    Example of the mixed `where` clause and the `witnesses` array:

    ```mlir
    trait.impl @T2_i64 for @T2[i64] where [!trait.proj<@Sib[i64], "Elem"> = i64]
        witnesses [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i64 by @Sib_cond>,
                   #trait<witness @X[i64] by @X_i64>] {
      // the equality-armed witness resolves the sibling projection to i64;
      // the application-armed witness discharges the cited conditional impl's
      // @X[i64] assumption.
      ...
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TraitApplicationAttr:$self_application,
    PredicateArrayAttr:$assumptions,
    OptionalAttr<TypedArrayAttrBase<WitnessAttr,
        "array of impl witnesses">>:$witnesses,
    OptionalAttr<StrAttr>:$sym_visibility
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

    /// Returns the impl-local definition of `methodName`, synthesizing it from
    /// the trait default method when the impl does not provide its own body.
    ///
    /// The synthesized method is specialized for this impl's self application
    /// and inserted into the impl body so future lookups reuse the same symbol.
    FailureOr<func::FuncOp> getOrSpecializeMethod(OpBuilder& builder, StringRef methodName);

    /// Extracts a method from this impl as a free function with a leading self parameter.
    ///
    /// Builds the impl specialization from the proven self claim, merges the
    /// call-site substitution, and monomorphizes the cloned method.
    FailureOr<func::FuncOp> getOrSpecializeFreeFunctionFromMethod(PatternRewriter& rewriter, ClaimType provenSelfClaim, StringRef methodName,
                                                                   const CallSubstitution &callSubst,
                                                                   ProofDerivationMemo *memo);

    /// The arguments carrying this impl's header to `actualSelfClaim`: read
    /// them off the demanded application by position, rebuild the header at
    /// them, and require the rebuilt header to be the demand after both are
    /// read through `normalize`.
    ///
    /// `normalize` is the established context the caller holds -- a verifier's
    /// evidence, the stage's recorded facts -- and is what lets a header
    /// spelling a projection (`impl<T> Index<T::Shape, T::Element> for T`)
    /// reproduce a demand spelling the resolution. A caller with no context
    /// passes none, and then a projection is equal to itself alone.
    FailureOr<SpecializationMap> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                            Normalizer normalize,
                                                            llvm::function_ref<InFlightDiagnostic()> errFn);

    /// As above, with no established context.
    FailureOr<SpecializationMap> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                            llvm::function_ref<InFlightDiagnostic()> errFn = nullptr) {
      return buildSubstitutionForSelfClaim(actualSelfClaim, Normalizer(), errFn);
    }

    /// The arguments this impl takes for `actualSelfClaim`, without judging
    /// them: read off the demanded application by position, then filled out
    /// from the where clause's equalities, each of which says what the
    /// parameter standing bare on one of its sides is once the other side is
    /// read through `normalize`. A parameter neither settles is left open, and
    /// what that means is the caller's: at impl selection an open parameter is
    /// an impl that does not apply, at a comparison it stands in the rebuilt
    /// header.
    TypeArguments readTypeArgumentsFor(ClaimType actualSelfClaim,
                                       Normalizer normalize);

    /// Builds the impl specialization needed for a proven self claim.
    ///
    /// The result contains the impl's generic type specialization and the
    /// evidence bindings that preserve proof annotations when specializing the
    /// impl method body.
    ///
    /// `origin` names the caller: recording the self claim's proof normalizes
    /// both claims through the ground-projection lookup, so this raises demand.
    FailureOr<ImplSpecialization> buildImplSpecialization(ClaimType provenSelfClaim,
                                                                  DemandOrigin origin,
                                                                  ProofDerivationMemo *memo,
                                                                  llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // Helper to generate unique symbol name for an ImplOp
    static std::string generateSymName(TraitApplicationAttr selfApp, PredicateArrayAttr assumptions);

    // The name of the instance this impl serves one claim with: its own symbol
    // plus the arguments that claim supplied for its parameters. The caller
    // hands the arguments it already built matching this impl to that claim,
    // since remaking the match here would remake it under whatever context
    // stands at the naming rather than the one the match was made under.
    std::string generateMangledName(const SpecializationMap &arguments);

    bool isUnconditional();

    LogicalResult verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> errFn);

    /// Refuses this impl at `cited` when one of its own where-clause equalities
    /// does not hold there.
    ///
    /// `cited` is the application the citation names. A premise takes no
    /// subproof -- the given list is indexed by the impl's application-arm
    /// obligations -- so it is read at each citation or nowhere, and a premise
    /// mentioning a variable the citation still spells is left standing for the
    /// instances made of it.
    ///
    /// A citation naming an impl carries no evidence of its own, so what a
    /// projection an endpoint spells reduces through is the impls the module
    /// holds under `origin`.
    LogicalResult verifyEqualityPremisesAt(ClaimType cited, DemandOrigin origin,
                                           llvm::function_ref<InFlightDiagnostic()> err);

    SmallVector<ClaimType> getAssumptionsAsClaims();

    FailureOr<SmallVector<ClaimType>> specializeAssumptionsAsClaimsFor(
        ClaimType actualSelfClaim,
        Normalizer normalize,
        llvm::function_ref<InFlightDiagnostic()> errFn);

    FailureOr<SmallVector<ClaimType>> specializeAssumptionsAsClaimsFor(
        ClaimType actualSelfClaim,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr) {
      return specializeAssumptionsAsClaimsFor(actualSelfClaim, Normalizer(),
                                              errFn);
    }

    /// Specializes this impl's requirements and assumptions for `actualSelfClaim`.
    ///
    /// `origin` names the caller: normalization reads only this impl's own
    /// bindings, so a projection over a sibling application survives it and is
    /// recorded as demand this normalization did not serve. Both the stage's
    /// resolver and a proof op's verifier reach here.
    FailureOr<SmallVector<ClaimType>> specializeObligationsAsClaimsFor(
        ClaimType actualSelfClaim,
        DemandOrigin origin,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// Look up an associated type op by name
    FailureOr<AssocTypeOp> getAssociatedType(StringRef name) {
      for (Operation& op : getBody().front())
        if (auto assoc = dyn_cast<AssocTypeOp>(op))
          if (assoc.getSymName() == name)
            return assoc;
      return failure();
    }

    /// Look up an associated type binding by name, with neither this impl's
    /// header arguments nor a projection's associated-type arguments applied:
    /// the result spells the impl's own parameters. Use
    /// `specializeAssociatedTypeBinding` to read it at both argument lists.
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

    /// The type this impl binds for associated type `name`, read at the two
    /// argument lists a use of it supplies: `headerArguments`, the arguments
    /// this impl's header parameters take at the application selecting it chose
    /// it for, and `assocTypeArgs`, the arguments a projection supplies for the
    /// binding's own parameters. For a binding with no parameters of its own
    /// `assocTypeArgs` is empty.
    ///
    /// The header's parameters take their arguments first and the projection's
    /// arguments are stamped into that result, because an argument is a term of
    /// whoever spelled the projection: a label it shares with one of this impl's
    /// header parameters names that caller's variable, and a stamped term is
    /// never revisited.
    ///
    /// Returns failure if:
    ///   - No associated type named `name` exists in this impl
    ///   - The associated type has no bound type
    ///   - GAT arity mismatch (`assocTypeArgs` size != type parameter count)
    FailureOr<Type> specializeAssociatedTypeBinding(
        StringRef name,
        ArrayRef<Type> assocTypeArgs,
        const SpecializationMap &headerArguments,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

  }];

  // builders create one empty block in the body region
  let skipDefaultBuilders = 1;
  let builders = [
    // override the default low-level builder to ensure that a block always exists.
    // The where-clause is one mixed PredicateArrayAttr (application and equality
    // arms in declaration order).
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "PredicateArrayAttr":$assumptions), [{
      $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
      $_state.addAttribute("self_application", self_application);
      $_state.addAttribute("assumptions", assumptions);
      // An impl is a template that dies with monomorphization, so it is private
      // from birth: erase collects it by liveness, never by kind.
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock();
    }]>,

    // this builder automatically generates $sym_name if it is not given
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "PredicateArrayAttr":$assumptions), [{
      // Auto-generate unique symbol name
      std::string symName = generateSymName(self_application, assumptions);
      build($_builder, $_state, symName, self_application, assumptions);
    }]>,

    // convenience wraps application-only assumptions into the mixed
    // PredicateArrayAttr and defers to the auto-naming builder
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<TraitApplicationAttr>":$assumptions), [{
      MLIRContext *ctx = $_builder.getContext();
      auto preds = PredicateArrayAttr::get(ctx, assumptions);
      build($_builder, $_state, self_application, preds);
    }]>,

    // convenience accepts sym_name as StringRef and application-only assumptions
    // as ArrayRef
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<TraitApplicationAttr>":$assumptions), [{
      MLIRContext *ctx = $_builder.getContext();
      auto preds = PredicateArrayAttr::get(ctx, assumptions);
      build($_builder, $_state, sym_name, self_application, preds);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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
  CPred<"::mlir::isa<::mlir::trait::ClaimType>($_self)">,
  "any '!trait.claim' type",
  "::mlir::trait::ClaimType"
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
  Pure
]> {
  let summary = "Unverified claim that a trait implementation exists";
  let description = [{
    Produces a `!trait.claim<@Trait[Types...]>` value asserting that a trait
    application is satisfied.

    By default, type arguments must be concrete (monomorphic). This enforces
    a key invariant: polymorphic code never fabricates claims. Claims
    originate at monomorphic call sites and flow into polymorphic code
    through `!trait.claim` parameters.

    The optional `unsafe` keyword relaxes this restriction, allowing
    polymorphic type arguments. This is used for ImplGenerator-provided
    evidence where the obligation cannot be checked until types are
    instantiated.

    `trait.allege` does not verify that a matching `trait.impl` exists at
    the point of emission. It is intended to be elaborated into verified
    `trait.witness` operations by a claim-proving pass. The resolve-impls
    pass resolves monomorphic alleges and skips polymorphic ones. The
    instantiate-monomorphs pass resolves the remaining (now-concrete)
    alleges. Unresolved monomorphic alleges after resolve-impls are errors.

    Examples:
      // monomorphic: resolved by resolve-impls
      %p = trait.allege @Ord[i32]

      // polymorphic: deferred to instantiate-monomorphs
      %q = trait.allege @Printable[!trait.poly<0>] unsafe
  }];

  let arguments = (ins OptionalAttr<UnitAttr>:$unsafe);

  let results = (outs Claim:$result);

  let builders = [
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$app), [{
      mlir::MLIRContext* ctx = $_builder.getContext();
      auto claimTy = ClaimType::get(ctx, app);
      return build($_builder, $_state, claimTy, /*unsafe=*/nullptr);
    }]>,
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$app,
                   "bool":$isUnsafe), [{
      mlir::MLIRContext* ctx = $_builder.getContext();
      auto claimTy = ClaimType::get(ctx, app);
      auto unsafeAttr = isUnsafe
        ? $_builder.getUnitAttr()
        : mlir::UnitAttr();
      return build($_builder, $_state, claimTy, unsafeAttr);
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
  Pure
]>{
  let summary = "Introduce an axiomatic claim from the enclosing scope";
  let description = [{
    Creates a zero-sized SSA value of type `!trait.claim<...>` that introduces a
    **hypothesis** holding axiomatically in the current scope. The claim is
    either arm: an application `@Trait[types]`, or a type equality `!A = !B`.

    Use `trait.project` to derive *consequences* of an application axiom (e.g.,
    trait requirements or impl assumptions). Use `trait.witness` for claims
    backed by concrete impl evidence.

    An **application** assume must name an axiom of the enclosing scope, checked
    in order:
      1. A `!trait.claim`-typed parameter of the enclosing function
      2. The enclosing `trait.trait`'s self application
      3. The enclosing `trait.impl`'s self application or an assumed application

    An **equality** assume is legal exactly when its equality matches, by
    identity, one equality axiom of the enclosing scope:
      1. An equality `!trait.claim`-typed parameter of the enclosing function
      2. An equality entry of the enclosing declaration's `where` clause -- the
         enclosing `trait.impl`'s assumptions inside an impl method, or the
         enclosing `trait.trait`'s requirements inside a trait method.
    A trait's requirement list anchors an equality assume exactly as an impl's
    assumption list anchors an application assume; a method body shares the
    enclosing declaration's polymorphic variables, so identity of the equality
    predicate is the check and no weaker match is accepted.

    Example of an equality assume anchored by the enclosing trait's requirement:
      trait.trait @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
        trait.assoc_type @Output
        func.func @fold(%x: !S) -> !S {
          %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !S
          ...
        }
      }

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
  let hasVerifier = 1;
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
    // Each assumption operand must be an application claim; the verifier checks
    // the arm so an equality claim is diagnosed at its operand.
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
  let summary = "Select one of a claim's requirements by position";
  let description = [{
    Given a claim, select one of the requirements it carries -- a trait
    application or a type equality that follows from the claim -- by its
    position in the claim's requirement list. While `trait.assume` introduces
    axioms, `trait.project` reads a consequence off one.

    A claim's requirements are, in this order:

      1. the `where` predicates of the `trait.trait` the claim applies, in
         declaration order. Supertraits, where-clause applications,
         associated-type bounds and header equalities all stand in that one
         list, so all of them are reachable by position;
      2. for a **proven** claim, the `where` assumptions of the `trait.impl`
         its proof cites, in declaration order -- the same order that
         `trait.proof` names its subproofs in.

    The result claim is derived, never spelled: requirement `index`
    instantiated at the source claim's arguments. An application requirement of
    a proven source carries the provider of the subproof discharging it, read
    out of the proof by position; every other requirement is unproven, and an
    equality requirement never carries a provider at all (the equality arm
    seals its endpoints). So a proven source projects to a proven application
    and an unproven one to an unproven application, without a parity rule
    saying so.

    The printed result type is an annotation on that derivation: verification
    compares it with the requirement the index selects and refuses a
    disagreement.

    Notes:
    * Only a **single hop** is performed; chain `trait.project` for transitive
      consequences.
    * A claim whose trait declares no `where` clause, and whose proof cites an
      impl that assumes nothing, reaches no index at all.

    Examples:

    ```mlir
    // a trait-level requirement (an unproven or a proven source claim)
    // trait.trait @Ord[!S] where [@Eq[!S], @PartialOrd[!S,!S]]
    %p = trait.assume @Ord[!T]
    %eq = trait.project %p[0] : !trait.claim<@Ord[!T]> -> !trait.claim<@Eq[!T]>

    // an impl assumption: the requirement list continues past the trait's two
    // into the impl's own, and the result carries the subproof that discharged it
    %wB = trait.project %wA[2]
      : !trait.claim<@A[i32] by @A_proof> -> !trait.claim<@B[i32] by @B_impl>

    // an equality requirement (an unproven or a proven source claim)
    // trait.trait @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64]
    %he = trait.project %p[0]
      : !trait.claim<@Has[i32] by @Has_i32>
      -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
    ```
  }];

  let arguments = (ins Claim:$source,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
  let results   = (outs Claim:$result);

  let extraClassDeclaration = [{
    ClaimType getSourceClaim() {
      return cast<ClaimType>(getSource().getType());
    }

    ClaimType getResultClaim() {
      return cast<ClaimType>(getResult().getType());
    }
  }];

  let assemblyFormat =
    "$source `[` $index `]` attr-dict `:` type($source) `->` type($result)";
}


def WitnessOp : TraitDialectOp<"witness", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Introduces verified evidence for a claim";
  let description = [{
    The dialect's sole verified-evidence introduction. It has two arms, matched
    to the arm of the result claim.

    This op is the value-producing carrier of a witness: it wraps the same
    content a `#trait.witness` attribute holds and hands code a typed claim value
    wherever evidence must be consumed as one -- call arguments, coercion
    citations. The same content stands alone as a `#trait.witness` attribute at
    declaration level -- the entries an impl's `witnesses` array carries
    (detailed under `trait.impl`). The equality arm's compose leaf is
    the one form that is irreducibly op-shaped, because it builds evidence from
    other claim values.

    Application arm. Introduces a value of type
    `!trait.claim<@Trait[Types...] by @Symbol>` that represents a proof that the
    trait application is satisfied. The symbol may refer to either
      * a `trait.proof` that transitively discharges the trait's requirements
        and the impl's assumptions
      * a `trait.impl` that is unconditional

    Example:
      // @PartialEq has no obligations, so we refer directly to a trait.impl
      %partial_eq_w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]

      // @Ord has obligations, so we refer to a trait.proof
      %ord_w = trait.witness @Ord_impl_i32_p for @Ord[i32]

    Equality arm. Introduces a value of type `!trait.claim<!A = !B>` that is
    evidence the two spellings name one type. There are three leaves:

      * proj-resolve: cites an impl by symbol and carries an immutable
        witness. The equality holds because the cited impl binds the
        associated type the projection names to the resolved type. When the
        cited impl's own Self application is itself spelled with projections,
        structural matching cannot align it with the projection being resolved,
        so verification reads the explicit equality `premises` and matches
        modulo those cited equalities. This is the sole leaf form of equality
        evidence: the refl and compose arms only combine equalities that
        proj-resolve leaves establish.

        // @A[i64]::Item resolves to i64 through impl @A_impl's binding.
        %item_eq = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl

        Verifying the witness resolves the projection through the cited
        impl's binding and additionally discharges the cited impl's own
        assumptions; `verifyProjectionResolutionAtUse` (TraitOps.hpp) states the
        full contract.

      * refl: proof-free, cites nothing; the endpoints are identical at
        construction. It exists because a concrete instantiation of an equality
        predicate can specialize to `A = A`, and a caller must still supply a
        claim value.

        %refl = trait.witness refl : !trait.claim<i64 = i64>

      * compose: cites nothing by symbol; consumes variadic equality-claim
        `premises` and introduces the equality their ground congruence closure
        entails. It builds a multi-hop equality without alleging it: the result
        is derived, not asserted.

        // From @A[i64]::Item = i64 and @B[i64]::Item = i64, derive
        // @A[i64]::Item = @B[i64]::Item without asserting it.
        %a_eq = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl
        %b_eq = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl
        %item_eq = trait.witness compose(%a_eq, %b_eq)

        Only the leaf premises are stored; the transitivity and congruence
        carrying them to the result are re-derived at verify by replaying the
        same ground congruence closure `trait.coerce` uses to decide equational
        entailment. Storing only definitional leaves keeps every stored equality
        leaf definitional while letting a caller name a derived equality.

    Admissible leaves are only definitional equalities -- one type, two
    spellings. The three leaves above are exhaustive; none is representational (a
    layout cast, a zero-cost view), so such a leaf is unconstructable. The
    dialect's nominal types are generative and congruence is exercised through
    projection heads -- the combination that made GHC's unscoped newtype coercions
    unsound -- so admitting a representational leaf requires role machinery first.

    Verification (verify() and verifySymbolUses) ensures the present attributes
    match the result claim's arm, that the result type equals the claim
    reconstructed from those attributes, and, for the equality arm's proj-resolve
    leaf, that the cited impl actually binds as stated and that the current
    endpoints are a single-substitution instance of the witness's
    endpoints. For
    the composition leaf verify() replays the ground congruence closure over the
    premises and requires it to entail the result equality; the leaf cites no
    symbol, so verifySymbolUses is a no-op for it.
  }];

  let arguments = (ins
    // Application arm.
    OptionalAttr<FlatSymbolRefAttr>:$proof, // refers to trait.proof or trait.impl
    OptionalAttr<TraitApplicationAttr>:$trait_application,
    // Equality arm.
    OptionalAttr<WitnessAttr>:$witness,               // proj-resolve leaf
    UnitAttr:$refl,                                     // refl leaf
    Variadic<Claim>:$premises
  );

  let builders = [
    // Application arm: builder infers the result type.
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$proof,
                   "::mlir::trait::TraitApplicationAttr":$trait_application), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::get(ctx, trait_application, proof);
      build($_builder, $_state, claimTy, proof, trait_application,
            /*witness=*/nullptr, /*refl=*/nullptr, /*premises=*/{});
    }]>,
    // Equality proj-resolve arm.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality,
                   "::mlir::trait::WitnessAttr":$witness,
                   "::mlir::ValueRange":$premises), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, /*proof=*/nullptr,
            /*trait_application=*/nullptr, witness, /*refl=*/nullptr,
            premises);
    }]>,
    // Equality refl arm.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, /*proof=*/nullptr,
            /*trait_application=*/nullptr, /*witness=*/nullptr,
            $_builder.getUnitAttr(), /*premises=*/{});
    }]>,
    // Equality composition arm. Distinct in arity from the proj-resolve builder
    // (that one takes a witness too); it carries only the premises whose
    // ground congruence closure entails the result equality.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality,
                   "::mlir::ValueRange":$premises), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, /*proof=*/nullptr,
            /*trait_application=*/nullptr, /*witness=*/nullptr,
            /*refl=*/nullptr, premises);
    }]>
  ];

  let results = (outs Claim:$result);

  let extraClassDeclaration = [{
    ClaimType getResultClaim() {
      return cast<ClaimType>(getResult().getType());
    }

    // Valid only on the application arm.
    ClaimType getProvenClaim() {
      return ClaimType::get(getContext(), getTraitApplicationAttr(), getProofAttr());
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def CoerceOp : TraitDialectOp<"coerce", [
  Pure
]> {
  let summary = "Change a value's written type by citing type equalities";
  let description = [{
    The only op that changes a value's written type; it consumes equality
    evidence to justify the change from the input spelling to the result
    spelling.

    Operands: the value, plus N equality-claim operands, N >= 0. There is no
    application-claim operand; a cited equality's endpoints are leaves of the
    claim type holding them, so an ordinary sub-element walk reaches them and
    walks that classify projections and claims see them.

    The verdict is a pure function of op, operands, and attributes:
      1. application-claim proofs are stripped uniformly from the input and
         result types -- comparison is modulo the label, permanently;
      2. a coerce may not relate two application spellings whose proofs name
         different proofs (the no-proof-swap clause): stripping is symmetric,
         but exchanging which impl proves a claim is refused;
      3. ground congruence closure over the subterm DAG of the two endpoints
         plus every cited equality's current endpoints, seeded with the
         equalities and closed under congruence -- union only, never
         decomposition at projection heads or anywhere else;
      4. the two endpoint classes must be equal.

    Closure decides ground equational entailment, so every derivation the
    frontend's smart constructors build replays, and nothing beyond what the
    cited leaves entail is accepted. The endpoints are assumed ground and
    binder-free.

    Example, citing evidence:
      %c = trait.coerce %v : !trait.proj<@Has[i32], "Out"> to i64 via (%eq)
        : (!trait.claim<!trait.proj<@Has[i32], "Out"> = i64>)

    The zero-evidence reflexive form (input and result identical, no
    outstanding equality) is legal and is the discharged terminal state; it
    folds to its operand, and cited evidence then dies by ordinary DCE.

    Example, reflexive:
      %r = trait.coerce %v : i64 to i64  // discharged terminal state; folds to %v

    The `unproven` marker names a coerce standing in a projection-convergence
    obligation whose evidence is not available at emission: the reconciling
    equalities cannot be cited yet because the impl that supplies them is an
    ImplGenerator's, minted only at monomorphization. A marked coerce cites
    nothing -- `equalities` must be empty -- and stands in a pending judgment:
    with the proofs stripped and the no-proof-swap clause enforced exactly as
    above, the endpoints stand under the rule `verifyPendingCoerceEndpoints`
    owns, which lets an endpoint still spelling a projection or a type variable
    stand and refuses two ground endpoints that differ. That rule, not this
    description, is the authoritative statement of what stands; what an open
    endpoint denotes is settled by instantiation and by the impls
    monomorphization mints. The marked form mints nothing consumable: monomorphization
    respells its projections to ground, discharging the obligation to the reflexive
    form the folder collapses -- or, if the two lookups ground to different types,
    failing the build. A marked coerce surviving past monomorphization is a hard
    error (the erase barrier refuses it), never a tolerated residual; the erase
    barrier is where ground truth is arbitrated.

    Example, marked:
      %u = trait.coerce %x : !trait.proj<@Fold[i64], "Item"> to i64 unproven
  }];

  let arguments = (ins AnyType:$input, Variadic<Claim>:$equalities,
                   UnitAttr:$unproven);
  let results = (outs AnyType:$result);

  // Prints `%v : In to Out`, then an optional `via (%eqs) : (types)` citing the
  // equalities, then the optional `unproven` marker. The explicit space literal
  // before `(` restores the space the declarative printer drops ahead of an
  // opening paren, preserving the `via (` spelling.
  let assemblyFormat = "$input `:` type($input) `to` type($result) (`via` ` ` `(` $equalities^ `)` `:` `(` type($equalities) `)`)? (`unproven` $unproven^)? attr-dict";
  let hasVerifier = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// Call ops
//===----------------------------------------------------------------------===//


def MethodCallOp : TraitDialectOp<"method.call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasGetModule
]>{
  let summary = "Call a trait method";
  let description = [{
    Calls a method from a trait. This can appear in either polymorphic or monomorphic code.
    Resolution to an implementation is performed during monomorphization.

    The call spells its method, its receiver claim, its arguments and its
    results, and says nothing about the method's type variables: the instance it
    wants is read off those types against the method's declaration. The trait's
    arguments ride in the receiver claim's application, the method's own
    variables stand in the argument and result positions, and a variable no
    position determines is refused.

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
    // The receiver must be an application claim -- a method call names its trait
    // through it -- and the verifier checks the arm so an equality receiver is
    // diagnosed at its operand.
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

    /// The type arguments this call supplies for its callee's parameters, once
    /// the callee's signature instantiated at them is the signature spelled
    /// here. The projection and proof bindings that close it are read off the
    /// record when the call is lowered.
    ///
    /// `reading` is the stage's record of what impl selection has settled,
    /// which the comparison reads both signatures through on top of the
    /// evidence this call holds. A verifier passes none and compares through
    /// that evidence alone.
    FailureOr<SpecializationMap> buildParameterSpecialization(const ReadOnlyImplResolver *reading, llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Specializes the callee as a free function using a pre-built substitution.
    FailureOr<func::FuncOp> getOrSpecializeCallee(PatternRewriter &rewriter,
                                                  const CallSubstitution &subst,
                                                  ProofDerivationMemo *memo);

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

  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}


def FuncCallOp : TraitDialectOp<"func.call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasGetModule
]> {
  let summary = "Call a possibly polymorphic function defined with func.func and !trait.poly";
  let description = [{
    Calls a function that may be polymorphic over one of more `!trait.poly` type parameters.
    These functions are typically declared using `func.func` and their signatures may involve
    polymorphic type variables.

    The call spells its callee, its operands and its results, and says nothing
    about the callee's type variables: the instance it wants is read off those
    types against the callee's declaration, and a variable no position
    determines is refused.

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

      auto callee = lookupSymbolFrom<func::FuncOp>(*module, name);
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

    /// Looks up or specializes the monomorphized callee using a pre-built substitution.
    FailureOr<func::FuncOp> getOrSpecializeCallee(PatternRewriter &rewriter,
                                                  const CallSubstitution &subst,
                                                  ProofDerivationMemo *memo);

    /// The type arguments this call supplies for its callee's parameters, once
    /// the callee's signature instantiated at them is the signature spelled
    /// here. The projection and proof bindings that close it are read off the
    /// record when the call is lowered.
    ///
    /// `reading` is the stage's record of what impl selection has settled,
    /// which the comparison reads both signatures through on top of the
    /// evidence this call holds. A verifier passes none and compares through
    /// that evidence alone.
    FailureOr<SpecializationMap> buildParameterSpecialization(const ReadOnlyImplResolver *reading, llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // The type parameters the callee's signature binds, in first-occurrence
    // order.
    SmallVector<GenericTypeInterface, 4> getCalleeTypeParams() {
      return getTypeParametersIn(*getCalleeFunctionType());
    }

  }];

  let assemblyFormat = "$callee_name `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}


//===----------------------------------------------------------------------===//
// Proof infrastructure
//===----------------------------------------------------------------------===//


def ProofOp : TraitDialectOp<"proof", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
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
    ArrayAttr:$subproof_names,               // the names of prerequisite proofs
    OptionalAttr<StrAttr>:$sym_visibility
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
      return lookupSymbolFrom<ImplOp>(module, getImplNameAttr());
    }

    TraitOp getTrait();

    /// Refuses this proof's impl at `cited` when one of the impl's own
    /// where-clause equalities does not hold there.
    ///
    /// `cited` is the application this proof is being carried to: its own claim
    /// where the proof is verified, and the instance a clone is cut at where
    /// the stage carries it there. A premise takes no subproof -- the given
    /// list is indexed by the impl's application-arm obligations -- so it is
    /// read here or nowhere, and the claim this proof stands over is where it
    /// is decided.
    ///
    /// What a projection an endpoint spells reduces through is the evidence
    /// this proof holds: the proofs discharging the impl's obligations, by
    /// index, and then the impls the module holds under `origin`. The proof's
    /// own rule is not among them -- nothing here is justified by what it is
    /// checking.
    ///
    /// `openPremise` says where a reading that still carries a type variable is
    /// decided: this proof's own claim refuses one, since a citation of this
    /// proof reads nothing inside it, and a citation at an instance leaves it
    /// to the instances made of that one.
    LogicalResult verifyEqualityPremisesAt(ClaimType cited, DemandOrigin origin,
                                           OpenPremise openPremise,
                                           llvm::function_ref<InFlightDiagnostic()> err);

    /// The claims this proof's subproofs stand over, at `at`.
    ///
    /// A subproof's claim is the obligation it discharges, spelled at the
    /// application the citation names, carrying the cited symbol. The given
    /// list is indexed by the impl's obligations, so the obligation at each
    /// index is what the name at that index answers for, and `at` is where
    /// those obligations are read: a proof written over type variables states
    /// them over those same variables, and a citation supplies the instance.
    ///
    /// `origin` names the caller, which the obligation specialization this
    /// performs raises its demand under.
    FailureOr<SmallVector<ClaimType>> verifyAndGetSubproofClaims(ClaimType at,
                                                                 DemandOrigin origin,
                                                                 llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    SmallVector<ClaimType> getSubproofClaims() {
      auto maybeResult = verifyAndGetSubproofClaims(getProvenClaim(),
                                                    DemandOrigin::ProofVerification);
      if (failed(maybeResult))
        llvm_unreachable("ProofOp::getSubproofClaims: verifyAndGetSubproofClaims failed");
      return *maybeResult;
    }

    /// Resolve a proof symbol to the impl it is backed by: a trait.proof yields
    /// its impl, a directly-named trait.impl yields itself. Whether the symbol
    /// is one this citation may name is the citing op's own check.
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

  let skipDefaultBuilders = 1;
  let builders = [
    // low-level builder mirroring the generated default's signature, so every
    // mint site (ImplResolution.cpp, the convenience builder below) reaches it.
    // A proof is a template that dies with monomorphization, so it is private
    // from birth: erase collects it by liveness, never by kind.
    OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                   "::mlir::FlatSymbolRefAttr":$impl_name,
                   "::mlir::trait::TraitApplicationAttr":$trait_application,
                   "::mlir::ArrayAttr":$subproof_names), [{
      $_state.addAttribute("sym_name", sym_name);
      $_state.addAttribute("impl_name", impl_name);
      $_state.addAttribute("trait_application", trait_application);
      $_state.addAttribute("subproof_names", subproof_names);
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
    }]>,

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

  let assemblyFormat = "custom<VisibilityKeyword>($sym_visibility) $sym_name `proves` $impl_name `for` $trait_application `given` $subproof_names attr-dict";
  let hasVerifier = 1;
}


#endif // TRAIT_OPS_TD
