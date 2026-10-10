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
//    and free of surprising dependencies on distant IR.  Impl resolution
//    is the work of the stage described under "How claims are proven".
//
// 3. Claims flow into polymorphic code through !trait.claim parameters --
//    a function's, and a trait's or impl's block arguments, which its
//    methods read -- so polymorphic code relies on a trait only where its
//    caller proves it.  A claim code fabricates is an allegation
//    (trait.allege), a trusted assertion that impl selection proves where
//    its types are concrete: instantiate-monomorphs proves a monomorphic one
//    where it stands and each instance of a polymorphic one, and one nothing
//    proves is refused where it stands.
//
// 4. The IR is the sole authority on claims.  All proof structure lives
//    in the IR itself — no side tables, no external state.  Analysis
//    passes read the IR; transformation passes update it.  This keeps
//    the IR self-contained and makes correctness checkable by inspection.
//
//===----------------------------------------------------------------------===//
// How claims are proven
//===----------------------------------------------------------------------===//
//
// A claim's evidence is a value: a block argument of the function, trait or
// impl that assumes it, or the result of an evidence op.  trait.witness cites
// a proof, or an impl with no parameters and no where entries; for an
// equality it cites a projection's resolution by an impl
// (`proj_resolve P resolves T by @impl`), a composition of such steps
// (`compose(...)`), or reflexivity (`refl`).  trait.derive applies an impl
// to evidence for its where entries.  trait.project reads a claim's
// requirement by position.  trait.coerce respells a value's type, and
// accepts the respelling where the ground congruence closure of the
// equalities it cites joins the two spellings.  trait.allege asserts a claim
// for impl selection to prove.  A proven claim's type names its proof
// (`!trait.claim<@T[i32] by @p>`), and a method call prints its receiver's
// proof apart from the receiver's type, after its signature (`by @p`).
//
// The stage proving claims is the monomorphize-trait pipeline:
// instantiate-monomorphs-trait, then erase-polymorphs-trait.  Instantiation
// is one greedy driver, which re-walks every function a rewrite landed in
// until none moves.  Each op asks impl selection for what it needs where it
// needs it, as Rust's monomorphization collector asks `Instance::resolve`:
// an allegation its proof, an op spelling an unproven monomorphic claim or a
// ground projection the proof or the type it resolves to.  Selection is
// memoized in the module the op stands in -- a refusal included, entered once
// nothing still part-way through could answer it otherwise -- asks the
// dialects' impl generators for an impl where none binds an application, and
// writes a proof as a trait.proof at the end of that module.  The driver cuts
// one instance per template, type arguments and evidence at each call whose
// operands are proven, lowering the call to a func.call of the instance.
// What selection refuses stays spelled where it stands, and the stage's exit
// walk names it there.  An instance's evidence parameters
// take its caller's proofs, so a body reads the evidence it was given by
// position.  A chain of instances holding 128 instances of one template is
// refused, as Rust's monomorphization collector counts its limit per function;
// a proof whose derivation stands 128 obligations deep is refused before any
// instance is cut, as Rust's trait solver refuses an overflowing obligation,
// which is what stops a proof citing itself, or a ring of proofs citing one
// another, at ever larger arguments.
//
// The evidence an impl returns for its trait's requirements is read by
// position: a projection of a proven claim is replaced by the ops the impl's
// return computes there, cloned at the projection with the impl's where
// arguments taken from the claim's proof, and a call of a method computing
// evidence is replaced by the method's body, so the stage's patterns are the
// only reading of either -- a derive there is transcribed, an allegation
// decided where it stands, a coercion settled.  An instance is cut with the
// projections off its proven parameters inlined, and those the inlined
// returns compute in turn, so the claims it returns are proven in its
// signature. Evidence whose returns project one another back to the
// requirement they stand for has no base and is never inlined
// (`ProjectOp::readEvidence`): the stage names the projection
// reading it.
// An allegation and a projection each state an obligation the stage has yet
// to discharge -- an allegation until it is decided, a projection until it is
// inlined, however its result is spelled -- and say so as an effect on
// `ObligationResource`, as `cf.assert` does, so no rewrite erases one as dead:
// a coercion that leaves its input in its place leaves the evidence it cited
// standing for the stage to decide, and an allegation nothing reads is decided
// where it stands.  Evidence
// may cite its own impl through a derive or a witness, which builds evidence,
// but an impl may not return a projection of its own application, which
// would read the very return it stands in (GHC's rule for instance
// superclasses); and an op in an impl's body outside its methods holds only
// the impl's where arguments as hypotheses, never its own application.  A
// proof is identified by its impl, its application and the proof each premise
// names: one standing with that body answers wherever the stage would write
// it.
//
//===----------------------------------------------------------------------===//


// Adds a `getModule()` convenience method to the op.
def HasGetModule : NativeOpTrait<"HasGetModule"> {
  let cppNamespace = "::mlir::trait";
}

// Op's regions have only these types of child operations.
class HasChildren<list<string> ops>
    : ParamNativeOpTrait<"HasOnlyChildOps", !interleave(ops, ", ")>, StructuralOpTrait;


// The operations a trait's or an impl's body may hold beside its methods and
// associated types: the evidence ops, which compute evidence over the
// declaration's block arguments for its methods to share and, in an impl, for
// its return.
defvar DeclarationEvidenceOps = ["trait::AllegeOp", "trait::CoerceOp",
                                 "trait::DeriveOp", "trait::ProjectOp",
                                 "trait::WitnessOp"];

def TraitOp : TraitDialectOp<"trait", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChildren<!listconcat(["trait::MethodOp", "trait::AssocTypeOp"],
                          DeclarationEvidenceOps)>,
  HasGetModule,
  IsolatedFromAbove,
  NoTerminator,
  SingleBlock,
  Symbol,
  SymbolTable
]> {
  let summary = "Declares a trait interface";
  let description = [{
    Declares a trait interface, similar to a Rust or Scala trait, or a Haskell
    typeclass, printed as a function signature. Its one block argument is the
    trait's own application, the hypothesis every default method runs under;
    the application's type arguments, distinct type variables, are the trait's
    parameters in order. Its result signature states the trait's requirements:
    one claim per fact every proof of the trait carries, in the one order
    `trait.project` indexes them by and every impl returns their evidence in.

    ```mlir
    !S = !trait.poly<0>
    !O = !trait.poly<1>

    trait.trait private @PartialOrd(%self: !trait.claim<@PartialOrd[!S, !O]>)
        -> !trait.claim<@PartialEq[!S, !O]> {
      // a required method
      trait.method @partial_cmp(!S, !O) -> !opt_ord

      // a default method, which reads the trait's argument
      trait.method @lt(%a: !S, %b: !O) -> i1 {
        %eq = trait.project %self[0] : !trait.claim<@PartialOrd[!S, !O]>
                                    -> !trait.claim<@PartialEq[!S, !O]>
        ...
        trait.return %res : i1
      }
    }
    ```

    A requirement is a trait application or a type equality the trait's
    associated types satisfy (`!trait.proj<@FoldFn[!S], "Output"> = !S`). A
    requirement quantified over variables of its own -- a generic associated
    type's bound -- is a function from its premises to its conclusion, and is
    declared as one: a required method of the trait whose type variables are the
    requirement's, whose claim parameters are its premises and whose result is
    its conclusion. Every impl defines it, and a use calls it.

    The body holds methods, associated-type declarations, and evidence ops over
    `%self` that default methods may share. It is a function body, so a value is
    defined before the ops that read it, and it ends without a terminator. A
    trait is a template that dies with monomorphization, so it is private.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeArrayAttr:$requirements,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// The trait's own application, the claim its one block argument holds.
    ClaimType getSelfClaim() {
      return ::llvm::cast<ClaimType>(getBody().front().getArgument(0).getType());
    }

    TraitApplicationAttr getSelfApplication() {
      return getSelfClaim().getTraitApplication();
    }

    /// The trait's parameters, its self application's arguments in order.
    ArrayRef<Type> getTypeParams() {
      return getSelfApplication().getTypeArgs();
    }

    FailureOr<FunctionOpInterface> getMethod(StringRef methodName, llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto func = dyn_cast_or_null<FunctionOpInterface>(mlir::SymbolTable::lookupSymbolIn(*this, methodName));
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
        if (auto func = dyn_cast<FunctionOpInterface>(op)) {
          if (func.isExternal()) {
            result.insert(func.getName());
          }
        }
      }
      return result;
    }

    std::vector<FunctionOpInterface> getOptionalMethods() {
      std::vector<FunctionOpInterface> result;
      for (Operation& op : getBody().front()) {
        if (auto func = dyn_cast<FunctionOpInterface>(op)) {
          if (!func.isExternal()) {
            result.push_back(func);
          }
        }
      }
      return result;
    }

    FailureOr<FunctionOpInterface> getOptionalMethod(StringRef methodName) {
      for (auto method : getOptionalMethods()) {
        if (method.getName() == methodName)
          return method;
      }
      return failure();
    }

    /// Builds the specialization that maps this trait's generic parameters to
    /// the corresponding types in `actualSelfClaim`.
    FailureOr<SpecializationMap> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                           llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// The trait's requirements, its result signature, in order.
    SmallVector<ClaimType> getRequirementsAsClaims();

    /// The trait's requirements, in order, instantiated at the application
    /// of `actualSelfClaim`: the substitution its arguments give the trait's
    /// parameters, built once and applied to each.
    FailureOr<SmallVector<ClaimType>> specializeRequirementsAsClaimsFor(
        ClaimType actualSelfClaim,
        llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

    /// Requirement `index` of the trait instantiated at the application of
    /// `actualSelfClaim`, the one requirement a reader by position needs.
    /// `index` is below the trait's requirement count.
    FailureOr<ClaimType> specializeRequirementAsClaimFor(
        ClaimType actualSelfClaim, unsigned index,
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

    /// The impls whose headers match `wanted`, read through the established
    /// context `normalize`: a header spelling a projection reproduces a demand
    /// spelling its resolution only where the caller's context can reduce it.
    SmallVector<ImplOp> getCandidateImplsFor(ClaimType wanted,
                                             Normalizer normalize = Normalizer());

    /// Cuts the default method `methodName` of this trait into a module-level
    /// function for the receiver `provenSelfClaim`, as an impl method is cut
    /// (`ImplOp::getOrSpecializeFreeFunctionFromMethod`): the trait's argument
    /// becomes the instance's leading self proof.
    FailureOr<func::FuncOp> getOrSpecializeFreeFunctionFromDefault(
        PatternRewriter &rewriter, ClaimType provenSelfClaim,
        StringRef methodName, TypeRange actualArguments,
        const CallSubstitution &callSubst,
        llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell);
  }];

  // The body's one block takes the self claim; a trait is a template that dies
  // with monomorphization, so it is private from birth: erase collects it by
  // liveness, never by kind.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::mlir::ArrayRef<Type>":$typeParams,
                   "::mlir::ArrayRef<Type>":$requirements), [{
      ::mlir::MLIRContext *ctx = $_builder.getContext();
      $_state.addAttribute("sym_name", $_builder.getStringAttr(name));
      $_state.addAttribute("requirements", $_builder.getTypeArrayAttr(requirements));
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
      ClaimType self = ClaimType::get(ctx, TraitApplicationAttr::get(ctx, name, typeParams));
      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock().addArgument(self, $_state.location);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ImplOp : TraitDialectOp<"impl", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasChildren<!listconcat(["trait::MethodOp", "trait::AssocTypeOp",
                           "trait::ReturnOp"],
                          DeclarationEvidenceOps)>,
  HasGetModule,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"trait::ReturnOp">,
  Symbol,
  SymbolTable
]> {
  let summary = "Implements a trait for a list of types";
  let description = [{
    Implements a trait, printed as a function. Its block arguments are its
    self claim, the trait application it implements, whose arguments name the
    types it implements the trait for, and then one claim per entry of its
    where clause, in order: an application the impl assumes, or a type equality
    it assumes about its own bindings. A citation of the impl -- `trait.derive`,
    a proof's derive, a projection-resolution `trait.witness` -- supplies one
    claim per where entry in that order, so a where entry exists once, as its
    block argument, and its position is the argument's.

    The body holds the associated-type bindings, the methods, and the evidence
    ops that compute, from the arguments, the evidence for each of the trait's
    requirements, which `trait.return` returns in the trait's order. The
    returned evidence is the trait's requirements at the self claim's
    arguments, read through the impl's own bindings and its where equalities,
    and none of it may rest on the impl's own application, argument 0. An impl
    of a trait requiring nothing returns nothing, and its return is not printed.

    ```mlir
    trait.impl private @Into_impl(%self: !trait.claim<@Into[!T, !U]>,
                                  %from: !trait.claim<@From[!U, !T]>) {
      trait.method @into(%x: !T) -> !U {
        %0 = trait.method.call %from @From[!U, !T]::@from(%x) : (!T) -> !U
        trait.return %0 : !U
      }
    }

    trait.impl private @Dense_array(%self: !trait.claim<@Dense[!array.array<!S, !E>]>,
                                    %coord: !trait.claim<@Coord[!S]>) {
      %tensor = trait.derive @Tensor[!array.array<!S, !E>] from @Tensor_array given(%coord)
        : (!trait.claim<@Coord[!S]>)
      trait.return %tensor : !trait.claim<@Tensor[!array.array<!S, !E>]>
    }
    ```

    A method of the impl reads the arguments and the evidence values the body
    defines before it as ordinary SSA uses. An impl is a template that dies
    with monomorphization, so it is private.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// The impl's own application, the claim its first block argument holds.
    ClaimType getSelfClaim() {
      return ::llvm::cast<ClaimType>(getBody().front().getArgument(0).getType());
    }

    TraitApplicationAttr getSelfApplication() {
      return getSelfClaim().getTraitApplication();
    }

    /// The claims the impl's where clause states, its block arguments after
    /// the first, in order.
    SmallVector<ClaimType> getWhereClaims() {
      return llvm::map_to_vector(getBody().front().getArguments().drop_front(),
                                 [](BlockArgument argument) {
                                   return ::llvm::cast<ClaimType>(argument.getType());
                                 });
    }

    /// The application this impl's header states at `arguments`, a
    /// substitution for its parameters.
    TraitApplicationAttr getSelfApplicationAt(const SpecializationMap &arguments) {
      return ::llvm::cast<ClaimType>(instantiate(Type(getSelfClaim()), arguments))
          .getTraitApplication();
    }

    /// Each entry of this impl's where clause at `arguments`, a substitution
    /// for its parameters, in order.
    SmallVector<ClaimType> getWhereClaimsAt(const SpecializationMap &arguments) {
      return llvm::map_to_vector(getWhereClaims(), [&](ClaimType entry) {
        return ::llvm::cast<ClaimType>(instantiate(Type(entry), arguments));
      });
    }

    /// The where clause's application entries, in order: the claims a
    /// proof's subproofs discharge.
    SmallVector<ClaimType> getApplicationPremises() {
      SmallVector<ClaimType> premises = getWhereClaims();
      llvm::erase_if(premises, [](ClaimType claim) { return claim.isEquality(); });
      return premises;
    }

    /// The where clause's equality entries, in order.
    SmallVector<TypeEqualityAttr> getEqualityPremises() {
      SmallVector<TypeEqualityAttr> equalities;
      for (ClaimType claim : getWhereClaims())
        if (TypeEqualityAttr equality = claim.getEqualityAttr())
          equalities.push_back(equality);
      return equalities;
    }

    /// The `trait.return` ending the body.
    ReturnOp getReturn();

    FlatSymbolRefAttr getTraitNameAttr() {
      return getSelfApplication().getTraitName();
    }

    TraitOp getTrait();

    ArrayRef<Type> getTypeArgs() {
      return getSelfApplication().getTypeArgs();
    }

    FailureOr<FunctionOpInterface> getMethod(StringRef methodName, llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto func = dyn_cast_or_null<FunctionOpInterface>(mlir::SymbolTable::lookupSymbolIn(*this, methodName));
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

    /// The impl's type parameters, labels 0 to n - 1: a parameter's label is
    /// its position, and the header and where clause spell each.
    SmallVector<GenericTypeInterface, 4> getTypeParams();

    /// The argument list a citation of this impl states for `arguments`: the
    /// argument of each parameter, by position.
    ImplArgumentsAttr stateArguments(const SpecializationMap &arguments);

    /// Extracts a method from this impl as a free function with a leading self parameter.
    ///
    /// Reads the impl's arguments off the proven self claim, adds the
    /// call-site substitution, and monomorphizes the cloned method. The
    /// instance is named by the evidence it is made with: the proven self
    /// claim at the leading position, then `actualArguments`, what the call
    /// supplies for the method's own parameters, position by position. A
    /// method this impl does not define is the trait's default, cut from the
    /// trait. `respell` carries a proven claim the call supplies to the
    /// spelling the instance's parameter has (`InstanceKey::get`).
    FailureOr<func::FuncOp> getOrSpecializeFreeFunctionFromMethod(PatternRewriter& rewriter, ClaimType provenSelfClaim, StringRef methodName,
                                                                   TypeRange actualArguments,
                                                                   const CallSubstitution &callSubst,
                                                                   llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell);

    /// The arguments carrying this impl's header to `actualSelfClaim`: read
    /// them off the demanded application by position, rebuild the header at
    /// them, and require the rebuilt header to be the demand after both are
    /// read through `normalize`.
    ///
    /// `normalize` is the established context the caller holds -- a verifier's
    /// evidence, the stage's impl selection -- and is what lets a header
    /// spelling a projection (`impl<T> Index<T::Shape, T::Element> for T`)
    /// reproduce a demand spelling the resolution. A caller with no context
    /// passes none, and then a projection is equal to itself alone.
    FailureOr<SpecializationMap> buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                            Normalizer normalize,
                                                            llvm::function_ref<InFlightDiagnostic()> errFn);

    /// The arguments this impl takes for `actualSelfClaim`, without judging
    /// them: read off the demanded application by position, then filled out
    /// from the where clause's equalities, one side read against the type the
    /// other side names once that side is read through `normalize` -- never
    /// a projection of this impl's own trait application, which names a type
    /// only through the parameters being read. A parameter neither settles
    /// is left open, and what that means is the caller's: at impl selection
    /// an open parameter is an impl that does not apply, at a comparison it
    /// stands in the rebuilt header.
    TypeArguments readTypeArgumentsFor(ClaimType actualSelfClaim,
                                       Normalizer normalize);

    /// The arguments this impl's parameters take at the proven self claim
    /// `provenSelfClaim`: the ones its proof's derive states this impl's
    /// header at, read by position off the proof, whatever spelling the claim
    /// citing it has. An impl named as its own proof binds no parameter.
    FailureOr<SpecializationMap> buildImplSpecialization(ClaimType provenSelfClaim,
                                                         llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // Helper to generate unique symbol name for an ImplOp
    static std::string generateSymName(TraitApplicationAttr selfApp, ArrayRef<ClaimType> where);

    // The name this impl takes at one claim, which the proof selection mints
    // for it there is named after: its own symbol plus the arguments that
    // claim supplied for its parameters. The caller hands the arguments it
    // already built matching this impl to that claim, since remaking the match
    // here would remake it under whatever context stands at the naming rather
    // than the one the match was made under.
    std::string generateMangledName(const SpecializationMap &arguments);

    /// Whether a citation may name this impl directly: it binds no parameter
    /// and has no where clause, so a citation supplies it nothing.
    bool isUnconditional();

    LogicalResult verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> errFn);

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
    /// Both lists stand in one substitution, applied in a single pass. An
    /// argument is a term of whoever spelled the use, so a label it shares with
    /// a parameter of either list names that caller's variable and not this
    /// declaration's: one pass never revisits a term it stamped, so no argument
    /// is read as a parameter the other list answers for.
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

    /// `ty` with every projection over this impl's header at `arguments`
    /// resolved through this impl's associated-type bindings at those
    /// arguments, to a fixed point: the facts the impl states of its own
    /// application, which hold anywhere it is read at those arguments. A
    /// projection over any other application, and an equality's endpoints,
    /// stand as spelled. Fails where the bindings do not settle within the
    /// projection-step bound.
    FailureOr<Type> readOwnBindings(Type ty,
                                    const SpecializationMap &arguments);

  }];

  // The body's one block takes the self claim and then one claim per where
  // entry, and ends with an empty `trait.return` the caller fills or replaces.
  // An impl is a template that dies with monomorphization, so it is private
  // from birth: erase collects it by liveness, never by kind.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::StringRef":$sym_name,
                   "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<::mlir::Type>":$where), [{
      $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
      ::mlir::Region *region = $_state.addRegion();
      ::mlir::Block &body = region->emplaceBlock();
      body.addArgument(ClaimType::get($_builder.getContext(), self_application),
                       $_state.location);
      for (::mlir::Type entry : where)
        body.addArgument(entry, $_state.location);
      ensureTerminator(*region, $_builder, $_state.location);
    }]>,

    // this builder automatically generates $sym_name
    OpBuilder<(ins "::mlir::trait::TraitApplicationAttr":$self_application,
                   "::mlir::ArrayRef<ClaimType>":$where), [{
      std::string symName = generateSymName(self_application, where);
      build($_builder, $_state, symName, self_application,
            ::llvm::SmallVector<::mlir::Type>(where.begin(), where.end()));
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
    trait.trait private @Iterator(%self: !trait.claim<@Iterator[!S]>) {
      trait.assoc_type @Item
      trait.method @next(!S) -> !trait.proj<@Iterator[!S], "Item">
    }
    ```

    Inside a `trait.impl`, binds the associated type to a concrete type:
    ```mlir
    trait.impl private @Iterator_i32(%self: !trait.claim<@Iterator[i32]>) {
      trait.assoc_type @Item = i64
      trait.method @next(%x: i32) -> i64 { ... }
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<TypeAttr>:$bound_type,
    OptionalAttr<TypeArrayAttr>:$type_params
  );

  let assemblyFormat = [{
    $sym_name (`<` $type_params^ `>`)? (`=` $bound_type^)? attr-dict
  }];
}


def MethodOp : TraitDialectOp<"method", [
  AutomaticAllocationScope,
  FunctionOpInterface,
  ParentOneOf<["trait::TraitOp", "trait::ImplOp"]>
]> {
  let summary = "A method a trait declares or an impl defines";
  let description = [{
    A function-like member of a `trait.trait` or `trait.impl`, printed as a
    `func.func` is and ended by `trait.return`. Its function type is the method's
    interface: its value parameters, then its own claim parameters. In a trait, a
    method without a body is required of every impl and a method with one is a
    default; in an impl, every method has a body and defines the trait's method
    of its name.

    ```mlir
    trait.trait private @PartialEq(%self: !trait.claim<@PartialEq[!S, !O]>) {
      trait.method @eq(!S, !O) -> i1
      trait.method @ne(%a: !S, %b: !O) -> i1 {
        %eq = trait.method.call %self @PartialEq[!S, !O]::@eq(%a, %b)
          : (!S, !O) -> i1
        %true = arith.constant 1 : i1
        %res = arith.xori %eq, %true : i1
        trait.return %res : i1
      }
    }
    ```

    A method's parent is a trait or an impl (`ParentOneOf`), and a method lives
    and dies with it: it carries no visibility (the verifier refuses one), so it
    is never discardable on its own, and a method without a body is a requirement
    rather than a declaration of something defined elsewhere (`isDeclaration` is
    the interface's default, false).

    A method is not isolated from above: it reads its declaration's block
    arguments and the evidence values the declaration's body defines before it
    as ordinary SSA uses, and nothing else from outside itself, since its
    declaration is isolated from above and its child list admits evidence ops
    alone beside methods and associated types. A method that returns claims
    only computes evidence -- a quantified requirement of its trait -- and is
    called for that evidence rather than cut. The constants the folder
    materializes while simplifying a method's body are placed in that body, as
    in an isolated function's: the dialect's `DialectFoldInterface` claims a
    method's body as the region to materialize into.

    A method is instantiated by cutting it into a module-level `func.func`, its
    `trait.return`s becoming `func.return`s and every value it reads from its
    declaration replaced by the evidence the receiver's proof supplies there.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    // A method of `name` and `type` with no body; a bodied method's caller adds
    // the entry block.
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::FunctionType":$type), [{
      $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
      $_state.addAttribute(getFunctionTypeAttrName($_state.name),
                           ::mlir::TypeAttr::get(type));
      $_state.addRegion();
    }]>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// The body a call runs, null for a required method, which has none.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def ReturnOp : TraitDialectOp<"return", [
  ParentOneOf<["trait::MethodOp", "trait::ImplOp", "trait::ProofOp"]>,
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Returns from a method, an impl, or a proof";
  let description = [{
    The one terminator of the dialect's three function-like bodies. It ends a
    block of a `trait.method`'s body, returning the method's results, whose
    number and types are the method's result types, by identity; it ends a
    `trait.impl`'s body, returning the evidence for each of the impl's trait's
    requirements, in the trait's order; and it ends a `trait.proof`'s body,
    returning the one claim the proof proves.

    ```mlir
    trait.method @into(%x: !T) -> !U {
      ...
      trait.return %r : !U
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ::mlir::ValueRange());
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}


def Claim : Type<
  CPred<"::mlir::isa<::mlir::trait::ClaimType>($_self)">,
  "any '!trait.claim' type",
  "::mlir::trait::ClaimType"
>;

/// A claim of a trait application, the only claim impl selection can prove.
def ApplicationClaim : Type<
  And<[CPred<"::mlir::isa<::mlir::trait::ClaimType>($_self)">,
       CPred<"::mlir::cast<::mlir::trait::ClaimType>($_self).isApplication()">]>,
  "a '!trait.claim' of a trait application",
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
// Four operations produce `!trait.claim` values, and one source holds them
// without an op. Together they cover every way a claim can originate or be
// transformed:
//
//   Op              Role                                         Analogy
//   ----------      ------------------------------------------   -----------
//   trait.allege    Fabricate a claim, to be proven later by      Assertion
//                   impl resolution where its types are concrete
//   trait.derive    Construct a new claim from an impl plus       Introduction
//                   one claim operand per where entry
//   trait.project   Extract a sub-claim from an existing claim    Elimination
//                   via a trait requirement or impl premise
//   trait.witness   Materialize a proven claim backed by a        Evidence
//                   trait.proof or unconditional trait.impl
//   (argument)      A function's, trait's or impl's claim         Hypothesis
//                   block argument, which its caller supplies
//
// `trait.coerce` respells a claim it is given under cited equalities, and a
// claim-returning method call computes one from its claim operands.
//===----------------------------------------------------------------------===//


def AllegeOp : TraitDialectOp<"allege", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Unverified claim that a trait application or a type equality holds";
  let description = [{
    Produces a `!trait.claim` value asserting that a trait application is
    satisfied, `@Trait[Types...]`, or that two spellings name one type,
    `!A = !B`.

    `trait.allege` does not verify that a matching `trait.impl` exists at
    the point of emission. It is intended to be elaborated into verified
    `trait.witness` operations by the instantiate-monomorphs pass, which
    proves a monomorphic allegation where it stands and a polymorphic one in
    each instance that grounds it. A monomorphic allegation nothing proves is
    an error.
    An equality is proved by resolving each projection its sides spell
    through the impl selection selects for the projection's application: the
    projection-resolution witness citing that impl at its arguments, or the
    composition of one such witness per projection resolved.

    Examples:
      // monomorphic: proven where it stands
      %p = trait.allege @Ord[i32]

      // polymorphic: deferred to instantiate-monomorphs
      %q = trait.allege @Printable[!trait.poly<0>]

      // the type the impl serving @Iterator[i32] binds `Item` to
      %e = trait.allege !trait.proj<@Iterator[i32], "Item"> = i64

    An allegation is a trusted assertion and carries no justification: impl
    selection proves its claim by whichever impl it selects, and a claim
    nothing proves -- an application no impl serves, or an equality whose
    sides resolve to two types -- is refused here.
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

  let assemblyFormat = [{
    custom<ClaimPredicate>(type($result)) attr-dict
  }];
}


def DeriveOp : TraitDialectOp<"derive", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Derive a claim from an impl and one claim per where entry";
  let description = [{
    Produces a new claim by applying a `trait.impl` to claim operands that
    satisfy its where clause; it is a call of the impl with one operand per
    block argument after the first. None of the other claim-producing ops can
    construct a claim from a conditional impl; consider `impl<U: Trait> Trait
    for Struct<U>`: given an in-scope claim for `Trait[!T]`, there is no way to
    produce `Trait[Struct<!T>]` without `trait.derive`.

    It names the impl and states the arguments its parameters take, the
    argument of `!trait.poly<i>` at position `i` (rustc's `ImplSource` carries
    an impl's arguments beside it, GHC's `evDFunApp` applies a dictionary
    function to its type arguments); `given` holds one claim per entry of the
    impl's where clause in its order -- an application claim for an application
    entry and an equality claim for an equality entry. The derived application
    is the impl's header at the arguments and each operand's claim is its entry
    at them, compared by identity.

    ```mlir
    // Given: impl<U> Trait for Struct<U> where Trait[U], Trait[U]::Out = i64
    %d = trait.derive @Trait[Struct<!T>] from @the_impl[!T] given(%t, %e)
      : (!trait.claim<@Trait[!T]>, !trait.claim<!trait.proj<@Trait[!T], "Out"> = i64>)
    ```

    Once monomorphic, a derive becomes the witness of the proof whose body it
    is, at the application selection resolves its claim to, coerced to the
    derive's own spelling.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$impl,
    // The argument of each of the impl's parameters, by position.
    ImplArgumentsAttr:$impl_args,
    // One operand per where-clause entry; the verifier checks each against its
    // entry so a mismatched claim is diagnosed at its operand.
    Variadic<Claim>:$assumptions
  );

  let results = (outs ApplicationClaim:$result);

  let extraClassDeclaration = [{
    ClaimType getDerivedClaim() {
      return cast<ClaimType>(getResult().getType());
    }

    /// The arguments this derive states for its impl's parameters.
    SpecializationMap getImplArguments() {
      return SpecializationMap::fromPositions(getImplArgs().getTypes());
    }

    /// The application this derive derives, which its result claim states.
    TraitApplicationAttr getTraitApplication() {
      return getDerivedClaim().getTraitApplication();
    }

    ImplOp getImplOp();
  }];

  // `@Trait[...] from @impl[args] given(%claims) : (types)`; the derived claim
  // is read off the application, the argument list is printed only when the
  // impl takes arguments, and the operand types only when there are operands.
  let assemblyFormat = [{
    custom<ClaimPredicate>(type($result)) `from` $impl
    `` custom<ImplArguments>($impl_args)
    `given` `(` $assumptions `)`
    (`:` `(` type($assumptions)^ `)`)? attr-dict-with-keyword
  }];
}

def ProjectOp : TraitDialectOp<"project", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Select one of a claim's requirements by position";
  let description = [{
    Given a claim, select one of the requirements it carries -- a trait
    application or a type equality that follows from the claim -- by its
    position in the claim's requirement list. A claim's requirements are, in
    this order:

      1. the requirements of the `trait.trait` the claim applies, its result
         signature, in order;
      2. for a **proven** claim whose proof stands over an impl, that impl's
         where entries, its block arguments after the first, in order -- the
         operands its proof's derive supplies.

    The result claim is derived, never spelled: requirement `index`
    instantiated at the source claim's arguments. A where entry of a proven
    source is the claim its proof's derive is given at that position. A
    trait requirement of a proven source is the evidence the impl returns at
    that position, which the stage puts in the projection's place
    (`inlineEvidence`), where the ops computing it are proven as anywhere else;
    a proof the result spells is held to that evidence by the coercion bridging
    the two. The printed result type is an annotation on that derivation:
    verification compares it with the requirement the index selects, modulo the
    proof where no where entry fixes one, and refuses a disagreement.

    A requirement quantified over variables of its own is no result of the trait
    but a method of it, and is called rather than projected.

    ```mlir
    // trait.trait @Ord(%self: !trait.claim<@Ord[!S]>)
    //     -> (!trait.claim<@Eq[!S]>, !trait.claim<@PartialOrd[!S, !S]>)
    %eq = trait.project %p[0] : !trait.claim<@Ord[!T]> -> !trait.claim<@Eq[!T]>

    // an impl's where entry: the requirement list continues past the trait's
    // two into the impl's own, and the result carries the subproof the proof's
    // derive was given there
    %wB = trait.project %wA[2]
      : !trait.claim<@A[i32] by @A_proof> -> !trait.claim<@B[i32] by @B_impl>
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

    /// Replaces this projection of a proven source by the evidence the
    /// source's impl returns at its index, its defining ops cloned here with
    /// the impl's arguments read through the source's proof (or, past the
    /// trait's requirements, by the where-entry evidence that proof supplies),
    /// bridged to the result's spelling by a coercion where the two differ.
    /// Fails, leaving the projection standing, where the source names no
    /// proof.
    LogicalResult inlineEvidence(RewriterBase &rewriter);

    /// What this projection's evidence stands on. The evidence is the
    /// source's impl's return at this index, and where that is a projection in
    /// turn, the return of the impl its source commits to, at the arguments
    /// each gives the next; a source that is one of an impl's where arguments
    /// commits to what the derive or proof committing that impl was given
    /// there, and a source an unproven projection produces commits to what
    /// that projection's own evidence does. A reading that reaches a
    /// requirement of an impl at an application twice has no base, and one
    /// still projecting past the instantiation depth limit overflows:
    /// inlining either would never finish (GHC's rule for instance
    /// superclasses, read across impls).
    EvidenceReading readEvidence();
  }];

  let assemblyFormat = [{
    $source `[` $index `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def WitnessOp : TraitDialectOp<"witness", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure
]>{
  let summary = "Introduces verified evidence for a claim";
  let description = [{
    The dialect's sole verified-evidence introduction. It has two arms, matched
    to the arm of the result claim.

    Application arm. Introduces a value of type
    `!trait.claim<@Trait[Types...] by @Symbol>` that represents a proof that the
    trait application is satisfied. The symbol may refer to either
      * a `trait.proof`, whose body derives the claim
      * a `trait.impl` that is unconditional: it binds no parameter and has no
        where clause, so a citation supplies it nothing

    Example:
      %partial_eq_w = trait.witness @PartialEq_impl_i32_i32 for @PartialEq[i32,i32]
      %ord_w = trait.witness @Ord_impl_i32_p for @Ord[i32]

    Equality arm. Introduces a value of type `!trait.claim<!A = !B>` that is
    evidence the two spellings name one type. There are three leaves:

      * proj-resolve: cites an impl by symbol at stated arguments, with one
        claim per entry of its where clause, in order, as a derive cites one.
        The equality holds because the cited impl, at those arguments, binds
        the associated type the projection names to the resolved type.

        %item_eq = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl
          : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>

      * refl: proof-free, cites nothing; the endpoints are identical at
        construction. It exists because a concrete instantiation of an equality
        predicate can specialize to `A = A`, and a caller must still supply a
        claim value.

        %refl = trait.witness refl : !trait.claim<i64 = i64>

      * compose: cites nothing by symbol; consumes variadic equality-claim
        `premises` and introduces the equality their ground congruence closure
        entails. It builds a multi-hop equality without alleging it: the result
        is derived, not asserted.

        %item_eq = trait.witness compose(%a_eq, %b_eq) : (...) : !trait.claim<...>

        Only the leaf premises are stored; the transitivity and congruence
        carrying them to the result are re-derived at verify by replaying the
        same ground congruence closure `trait.coerce` uses to decide equational
        entailment.

    Admissible leaves are only definitional equalities -- one type, two
    spellings. The three leaves above are exhaustive; none is representational (a
    layout cast, a zero-cost view), so such a leaf is unconstructable. The
    dialect's nominal types are generative and congruence is exercised through
    projection heads -- the combination that made GHC's unscoped newtype coercions
    unsound -- so admitting a representational leaf requires role machinery first.

    The application arm is its result claim alone, and a proj-resolve leaf is
    its result equality, the impl it cites and the arguments it cites it at.
  }];

  // The application arm is its result claim alone, which names the proof or
  // impl it cites; the equality arm carries its leaf.
  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$impl,               // proj-resolve leaf
    OptionalAttr<ImplArgumentsAttr>:$impl_args,          // proj-resolve leaf
    UnitAttr:$refl,                                      // refl leaf
    Variadic<Claim>:$premises
  );

  let builders = [
    // Application arm: builder infers the result type.
    OpBuilder<(ins "::mlir::FlatSymbolRefAttr":$proof,
                   "::mlir::trait::TraitApplicationAttr":$trait_application), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::get(ctx, trait_application, proof);
      build($_builder, $_state, claimTy, /*impl=*/nullptr, /*impl_args=*/nullptr,
            /*refl=*/nullptr, /*premises=*/{});
    }]>,
    // Equality proj-resolve arm.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality,
                   "::mlir::FlatSymbolRefAttr":$impl,
                   "::mlir::trait::ImplArgumentsAttr":$impl_args,
                   "::mlir::ValueRange":$premises), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, impl, impl_args, /*refl=*/nullptr,
            premises);
    }]>,
    // Equality refl arm.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, /*impl=*/nullptr, /*impl_args=*/nullptr,
            $_builder.getUnitAttr(), /*premises=*/{});
    }]>,
    // Equality composition arm. Distinct in arity from the proj-resolve builder
    // (that one takes an impl too); it carries only the premises whose ground
    // congruence closure entails the result equality.
    OpBuilder<(ins "::mlir::trait::TypeEqualityAttr":$equality,
                   "::mlir::ValueRange":$premises), [{
      MLIRContext* ctx = $_builder.getContext();
      ClaimType claimTy = ClaimType::getEquality(ctx, equality);
      build($_builder, $_state, claimTy, /*impl=*/nullptr, /*impl_args=*/nullptr,
            /*refl=*/nullptr, premises);
    }]>
  ];

  let results = (outs Claim:$result);

  let extraClassDeclaration = [{
    /// Verifies this projection-resolution witness: the cited impl, at the
    /// arguments the witness states, proves the projection's application over
    /// the premises and binds the projection as stated, each compared by
    /// identity.
    LogicalResult verifyResolution(ModuleOp module);

    /// The arguments a projection-resolution witness states for its impl's
    /// parameters.
    SpecializationMap getImplArguments() {
      return SpecializationMap::fromPositions(getImplArgsAttr().getTypes());
    }

    ClaimType getResultClaim() {
      return cast<ClaimType>(getResult().getType());
    }

    /// Whether this witness resolves a projection through an impl's binding.
    bool isProjectionResolution() { return static_cast<bool>(getImplAttr()); }

    // The claim an application witness proves, and the proof or impl it
    // cites; valid only on the application arm.
    ClaimType getProvenClaim() { return getResultClaim(); }
    FlatSymbolRefAttr getProof() { return getResultClaim().getProof(); }
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
  }];

  let arguments = (ins AnyType:$input, Variadic<Claim>:$equalities);
  let results = (outs AnyType:$result);

  // Prints `%v : In to Out`, then an optional `via (%eqs) : (types)` citing the
  // equalities. The explicit space literal before `(` restores the space the
  // declarative printer drops ahead of an opening paren, preserving the `via (`
  // spelling.
  let assemblyFormat = "$input `:` type($input) `to` type($result) (`via` ` ` `(` $equalities^ `)` `:` `(` type($equalities) `)`)? attr-dict";
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

    FailureOr<FunctionOpInterface> getMethod(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    FailureOr<FunctionType> getMethodFunctionType(llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
      auto maybeMethod = getMethod(err);
      if (failed(maybeMethod)) return failure();
      return cast<FunctionType>(maybeMethod->getFunctionType());
    }

    /// Get a FunctionType corresponding to the actual argument and result types of this MethodCallOp
    /// Note that the claim type is not part of the function type
    FunctionType getActualFunctionType() {
      return FunctionType::get(getContext(), getArguments().getTypes(), getResultTypes());
    }

    /// The type arguments this call supplies for its callee's parameters, once
    /// the callee's signature instantiated at them is the signature spelled
    /// here. The projection bindings that close it are asked of impl selection
    /// when the call is lowered.
    ///
    /// `selection` resolves a type's projections through the stage's impl
    /// selection, which is all the stage reads both signatures through. A
    /// verifier passes none and compares the two signatures as written.
    FailureOr<SpecializationMap> buildParameterSpecialization(llvm::function_ref<Type(Type)> selection, llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Specializes the callee as a free function using a pre-built
    /// substitution; `respell` carries a proven claim this call supplies to the
    /// spelling the instance's parameter has (`InstanceKey::get`).
    FailureOr<func::FuncOp> getOrSpecializeCallee(PatternRewriter &rewriter,
                                                  const CallSubstitution &subst,
                                                  llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell);

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

    /// Whether this call computes evidence: its one result is a claim, so its
    /// method is a quantified requirement of the trait, and the call is
    /// replaced by the method's body where it is monomorphic
    /// (`inlineEvidence`) rather than cut into an instance.
    bool computesEvidence() {
      return getNumResults() == 1 && ::llvm::isa<ClaimType>(getResult(0).getType());
    }

    /// Replaces this call, one that computes evidence, by the body of the
    /// method the receiver's proven impl defines, stamped under `subst`: the
    /// body's reads from its impl take the evidence the receiver's proof
    /// supplies there, its parameters this call's arguments, and this call's
    /// result the operand the body returns. The claim it leaves is read by
    /// position, never selected again. The receiver impl's arguments are the
    /// ones its proof states (`ImplOp::buildImplSpecialization`).
    LogicalResult inlineEvidence(PatternRewriter &rewriter,
                                 const CallSubstitution &subst);
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

    /// Looks up or specializes the monomorphized callee, a template binding at
    /// least one type parameter, using a pre-built substitution; `respell`
    /// carries a proven claim this call supplies to the spelling the instance's
    /// parameter has (`InstanceKey::get`).
    FailureOr<func::FuncOp> getOrSpecializeCallee(PatternRewriter &rewriter,
                                                  const CallSubstitution &subst,
                                                  llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell);

    /// The type arguments this call supplies for its callee's parameters, once
    /// the callee's signature instantiated at them is the signature spelled
    /// here. The projection bindings that close it are asked of impl selection
    /// when the call is lowered.
    ///
    /// `selection` resolves a type's projections through the stage's impl
    /// selection, which is all the stage reads both signatures through. A
    /// verifier passes none and compares the two signatures as written.
    FailureOr<SpecializationMap> buildParameterSpecialization(llvm::function_ref<Type(Type)> selection, llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    // The type parameters the callee's signature binds: the labels below the
    // bound it spells, a label being a position in the declaration.
    SmallVector<GenericTypeInterface, 4> getCalleeTypeParams() {
      SmallVector<GenericTypeInterface, 4> parameters;
      for (unsigned label = 0,
                    bound = getLabelBound(Type(*getCalleeFunctionType()));
           label < bound; ++label)
        parameters.push_back(cast<GenericTypeInterface>(
            Type(PolyType::get(getContext(), label))));
      return parameters;
    }

  }];

  let assemblyFormat = "$callee_name `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}


//===----------------------------------------------------------------------===//
// Proof infrastructure
//===----------------------------------------------------------------------===//


def ProofOp : TraitDialectOp<"proof", [
  HasChildren<["trait::CoerceOp", "trait::DeriveOp", "trait::ReturnOp",
               "trait::WitnessOp"]>,
  IsolatedFromAbove,
  SingleBlock,
  Symbol
]>{
  let summary = "A named, closed proof of one claim";
  let description = [{
    A closed evidence function that takes nothing and returns one claim: its
    body derives the impl's header from an impl over evidence for each of the
    impl's where entries, and `trait.return` returns the derived claim, or its
    coercion to the application with the projections the header spells
    resolved. A proof of another spelling of an application a proof already
    proves is a cast of that proof (GHC's `$dX' = $dX |> co`): its body
    returns the witness of the proof it casts, coerced to its own spelling.
    Every witness in a proof's body names a root, a proof whose body holds its
    derive or an unconditional impl, never a cast: a respelling of a
    respelling is one coercion of the root.
    The proven claim is the returned application, carrying this proof's name;
    the impl, its arguments and its premises are read off the derive, a cast's
    off the derive of the proof it casts. The body
    holds decided evidence only -- witnesses of other proofs and unconditional
    impls, equality leaves, coercions, derives -- and never an allegation: a
    proof records a decision already made. Nor does it hold a projection, whose
    evidence is whatever the projected impl returns there, an allegation
    included; a premise a projection would read is named by its own proof.

    The trait's requirements of the proven claim are not in the proof: the
    impl's own return holds their evidence, read at this proof's derive.

    ```mlir
    trait.proof private @Into_f32_f64_p {
      %from = trait.witness @From_f64_f32 for @From[f64, f32]
      %p = trait.derive @Into[f32, f64] from @Into_impl given(%from)
        : (!trait.claim<@From[f64, f32]>)
      trait.return %p : !trait.claim<@Into[f32, f64]>
    }
    ```

    A proof is a template that dies with monomorphization, so it is private.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// The claim the body returns: the derive, or its respelling.
    Value getProven();

    /// The derive the body returns, read through its respelling, or the derive
    /// of the proof this one casts: the decision this proof records. Null for
    /// a cast of an unconditional impl, which derives from nothing.
    DeriveOp getDerive();

    /// The proof or unconditional impl this proof casts to its own spelling, a
    /// root whose body holds its derive where it is a proof; null for a proof
    /// whose body holds its derive.
    Operation *getCastSource();

    /// The impl the derive this proof records cites, or the unconditional impl
    /// it casts.
    ImplOp getImpl();

    TraitApplicationAttr getTraitApplication() {
      return cast<ClaimType>(getProven().getType()).getTraitApplication();
    }

    ClaimType getProvenClaim() {
      MLIRContext* ctx = getContext();
      FlatSymbolRefAttr symRef = FlatSymbolRefAttr::get(ctx, getSymNameAttr());
      return ClaimType::get(ctx, getTraitApplication(), symRef);
    }

    TraitOp getTrait();

    /// The claims the derive is given, one per where entry of the impl.
    SmallVector<ClaimType> getPremises();

    /// The proof each application premise of the derive rests on, in order:
    /// the witness beneath the premise's respelling to its where entry, at the
    /// application that proof proves. An equality premise rests on none.
    SmallVector<ClaimType> getSubproofs();

    /// The arguments this proof's impl takes, as its derive states them; none
    /// for a cast of an unconditional impl.
    SpecializationMap getImplArguments() {
      DeriveOp derive = getDerive();
      return derive ? derive.getImplArguments() : SpecializationMap();
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

    /// The root the proof or unconditional impl `name` names rests on: that
    /// declaration, or the one it casts where it is a cast.
    static FailureOr<Operation*> getRootOf(
      ModuleOp module,
      FlatSymbolRefAttr name,
      llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);
  }];

  // A proof is a template that dies with monomorphization, so it is private
  // from birth: erase collects it by liveness, never by kind. Its body is one
  // empty block the caller fills and ends with `trait.return`.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$sym_name), [{
      $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
      $_state.addAttribute("sym_visibility", $_builder.getStringAttr("private"));
      $_state.addRegion()->emplaceBlock();
    }]>
  ];

  let assemblyFormat = "custom<VisibilityKeyword>($sym_visibility) $sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}


#endif // TRAIT_OPS_TD
