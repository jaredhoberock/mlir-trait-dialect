// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <memory>
#include <variant>

namespace mlir::trait {

// Interface to generate a new ImplOp for the wanted claim, or fail
struct ImplGenerator {
  virtual ~ImplGenerator() = default;

  // Creates exactly one new ImplOp for the wanted claim, or fails.
  // Upon success, returns the newly created ImplOp whose self claim, read for
  // the arguments its own parameters take, must rebuild wanted. A declaration
  // the impl names that the module lacks -- a trait its where entries apply --
  // is the generator's to declare beside it.
  //
  // A generator only builds IR, so an OpBuilder suffices. The caller places
  // the builder where a generated impl belongs before calling: a generator
  // that creates its impl at the ambient insertion point puts it wherever the
  // caller left the builder. On the way out a generator may leave the
  // insertion point parked inside the IR it just generated, and restoring it
  // is again the caller's responsibility.
  //
  // A generated impl is IR nothing revisits unless the caller hears about it,
  // so the builder must carry a listener and the caller must act on what it
  // hears. When generation runs underneath a greedy pattern driver, that means
  // passing the driver's active PatternRewriter itself rather than a builder of
  // its own: the driver's listener is what places generated ops on its
  // worklist, and a builder without it silently changes which ops the driver
  // revisits.
  virtual FailureOr<ImplOp>
  generateImpl(TraitOp trait,
               ClaimType wanted,
               OpBuilder &builder) const = 0;
};

// Composite that itself behaves like an ImplGenerator
class ImplGeneratorSet : public ImplGenerator {
  public:
    inline FailureOr<ImplOp>
    generateImpl(TraitOp trait,
                 ClaimType wanted,
                 OpBuilder &builder) const override {
      // return the first successful result of all generators
      for (const auto &g : generators) {
        // a generator may park the insertion point in the IR it generated,
        // so each attempt starts from the insertion point we were handed
        OpBuilder::InsertionGuard guard(builder);
        auto maybeImpl = g->generateImpl(trait, wanted, builder);
        if (succeeded(maybeImpl))
          return maybeImpl;
      }
      return failure();
    }

    inline ImplGeneratorSet &add(std::unique_ptr<ImplGenerator> g) {
      generators.emplace_back(std::move(g));
      return *this;
    }

    template<typename... Ts>
    ImplGeneratorSet &add() {
      (add(std::make_unique<Ts>()), ...);
      return *this;
    }

  private:
    SmallVector<std::unique_ptr<ImplGenerator>,4> generators;
};

/// A trait application as read in one module.
///
/// A spelling names its symbols in one symbol table, and two modules can spell
/// one application and mean two different impls of it, so what selection
/// settles is settled for that application under the module it was demanded in
/// and not for the spelling alone.
using ScopedApplication = std::pair<Operation *, TraitApplicationAttr>;

/// Why selection refused an application: the candidates it judged, those
/// whose assumptions hold -- none, or two or more -- and those whose do not. A
/// refusal is an answer like a selection, so asking again is told it, and a
/// caller reporting it names the same candidates the first ask judged.
struct Refusal {
  SmallVector<ImplOp> satisfiable;
  SmallVector<ImplOp> unsatisfiable;
};

/// An impl selection chose, the arguments its parameters take at the
/// application it was chosen for, and the height of the derivation it chose it
/// by: the number of obligation frames the deepest chain under it stood on, the
/// frame of the selection itself included.
struct ChosenImpl {
  ImplOp impl;
  SpecializationMap arguments;
  unsigned height;
  /// The proof of the application, deriving it from `impl` over the proofs
  /// selection chose for its where entries, once one stands; null before.
  FlatSymbolRefAttr proof = {};
};

// Memoization state for impl selection.
struct ResolutionMemo {
  // Maps a fully-concrete trait application, as read in one module, to the impl
  // selected for it. A selection read here stands as high above the chain
  // reading it as its derivation did, so the depth bound judges the
  // derivation and not the order selections were asked in.
  DenseMap<ScopedApplication, ChosenImpl> chosen;

  // Maps an application selection refused, as read in one module, to why. A
  // refusal is entered only where it is final -- reached without leaning on an
  // application further down the obligation chain than the one refused
  // (`provisionalBelow`) -- so an application in neither map is a question
  // selection has not yet answered, never a refusal.
  DenseMap<ScopedApplication, Refusal> refused;

  // The applications impl selection is part-way through, outermost first. A
  // repeat is a resolution cycle; the number of frames is how deep the
  // obligation chain has recursed, which is what bounds a chain whose every
  // step is a new application.
  SmallVector<ObligationFrame> visiting;

  // The shallowest frame of `visiting` a cycle guard refused a candidate at
  // since the innermost selection still running began, or UINT_MAX where none
  // was. A refusal computed while this stands below the depth the refused
  // application was asked at leans on a frame still part-way through, which
  // may yet be answered otherwise -- rustc's provisional result -- so it is not
  // entered in `refused`.
  unsigned provisionalBelow = UINT_MAX;

  // The greatest height of a selection answered under the candidate the
  // innermost selection still running is judging -- under its headers, while
  // it reads them -- which the height of a selection choosing that candidate
  // is one above.
  unsigned heightBelow = 0;

  // A memo for assumptionsSatisfiableFor
  // For every (ImplOp, TraitApplicationAttr) in this map, the ImplOp's
  // assumptions are known to be satisfiable for the given TraitApplicationAttr,
  // at the arguments it maps to.
  // We only memoize satisfiable results because new proofs appear in the IR
  // as resolution unfolds
  DenseMap<std::pair<ImplOp, TraitApplicationAttr>, SpecializationMap>
      assumptionsKnownSatisfiable;

  // The applications a generator has already supplied an impl for, in the
  // module it was supplied into. Generation supplies an impl the module lacks,
  // and a generated impl is a function of the application it was asked for, so
  // asking twice would publish a second op under the name the first already
  // holds. The impl supplied stands in the module, where the candidate scan
  // reads it.
  DenseSet<ScopedApplication> generatedFor;
};

/// What impl selection answers about one question: the answer; a refusal,
/// which selection keeps and an obligation left standing names; or an
/// overflow, the stage's hard error, which the resolver has already named
/// where it was met.
template <typename T>
class Answer {
public:
  Answer(T value) : state(std::move(value)) {}
  static Answer refusal() { return Answer(Refused{}); }
  static Answer overflow() { return Answer(Overflowed{}); }

  bool isAnswer() const { return std::holds_alternative<T>(state); }
  bool isOverflow() const { return std::holds_alternative<Overflowed>(state); }

  const T &operator*() const { return std::get<T>(state); }
  T &operator*() { return std::get<T>(state); }
  const T *operator->() const { return &std::get<T>(state); }
  T *operator->() { return &std::get<T>(state); }

  /// The answer, or failure where there is none: for a reader to whom a
  /// refusal and an overflow differ in nothing.
  FailureOr<T> orFailure() const {
    if (isAnswer())
      return **this;
    return failure();
  }

  /// This answer's refusal or overflow, as the answer to another question
  /// that stops where this one stopped. Holds no answer.
  template <typename U>
  Answer<U> stop() const {
    assert(!isAnswer() && "carrying an answer as a stop");
    return isOverflow() ? Answer<U>::overflow() : Answer<U>::refusal();
  }

private:
  struct Refused {};
  struct Overflowed {};
  explicit Answer(Refused) : state(Refused{}) {}
  explicit Answer(Overflowed) : state(Overflowed{}) {}

  std::variant<T, Refused, Overflowed> state;
};

/// What stops impl selection for the rest of a stage run, as it is named.
///
/// Two kinds go past the depth limit (`kInstantiationDepthLimit`): an
/// obligation chain, and the steps resolving a spelling's projections. rustc
/// bounds selection and normalization alike by its one recursion limit, and
/// names either as an overflow. The third is an impl a generator wrote for
/// another application than the one it was asked for: the memo's refusals are
/// final only because no impl generated later serves one, so no answer
/// selection keeps can stand after it.
class Overflow {
public:
  /// The obligation chain `chain` reaching `app`, with `height` frames
  /// standing at and below the frame reaching it.
  static Overflow obligations(TraitApplicationAttr app,
                              ArrayRef<ObligationFrame> chain,
                              unsigned height = 1) {
    return Overflow(app, Type(), chain, height, ImplOp());
  }

  /// The projections `spelled` spells, still changing after the limit's
  /// worth of resolution steps.
  static Overflow projectionSteps(Type spelled) {
    return Overflow(TraitApplicationAttr(), spelled, {}, 0, ImplOp());
  }

  /// `impl`, generated for `asked`, stating another application.
  static Overflow inexactImpl(ImplOp impl, TraitApplicationAttr asked) {
    return Overflow(asked, Type(), {}, 0, impl);
  }

  /// Names this overflow: at the generated impl it is about, or else at
  /// `anchor`, the demand that met it.
  void emit(Location anchor) const;

private:
  Overflow(TraitApplicationAttr app, Type spelled,
           ArrayRef<ObligationFrame> chain, unsigned height, ImplOp impl)
      : app(app), spelled(spelled), chain(chain), height(height), impl(impl) {}

  TraitApplicationAttr app;
  Type spelled;
  ArrayRef<ObligationFrame> chain;
  unsigned height;
  ImplOp impl;
};

/// The impl selected for a claim, paired with the normalized claim used for
/// selection.
///
/// Projection normalization is part of impl resolution. Callers that specialize
/// the selected impl must use this claim, not the original source spelling, so
/// selection and substitution agree on the same semantic type arguments.
struct ResolvedImpl {
  ImplOp impl;
  ClaimType selectedClaim;
  /// The arguments the impl's parameters take at `selectedClaim`, as the
  /// selection that chose it read them.
  SpecializationMap arguments;
};

/// One step of the resolution of a projection with a ground head: the impl
/// selection settled on for the projection's application, the arguments that
/// impl's parameters take at the claim selection chose it for, and the impl's
/// binding of the projected associated type at those arguments and at the
/// projection's own associated-type arguments as spelled.
///
/// The binding is specialized and nothing more: a projection it spells, the
/// impl's own or one an associated-type argument carried in, is a step of its
/// own. So `projection = binding` is exactly what the witness verifier's
/// specialization of the cited impl's binding reproduces. The only constructor
/// asks selection about the projection's own application, reads the arguments
/// off what selection settled, and specializes the binding itself, so a step
/// that exists states what the impl selected for its projection binds at the
/// arguments it takes there. Only impl selection constructs one.
class ProjectionResolution {
public:
  ProjectionType getProjection() const { return projection; }
  ImplOp getImpl() const { return impl; }
  const SpecializationMap &getArguments() const { return arguments; }
  Type getBinding() const { return binding; }

private:
  friend class ImplResolver;

  /// The step resolving `projection` through the impl `select` settles on for
  /// the projection's application, at the arguments selection read for it.
  /// Stops where selection does, and is refused where the impl binds no such
  /// associated type at the projection's associated-type arguments.
  static Answer<ProjectionResolution>
  get(ProjectionType projection,
      llvm::function_ref<Answer<ResolvedImpl>(ClaimType)> select,
      llvm::function_ref<InFlightDiagnostic()> err = nullptr);

  ProjectionResolution(ProjectionType projection, ImplOp impl,
                       SpecializationMap arguments, Type binding)
      : projection(projection), impl(impl), arguments(std::move(arguments)),
        binding(binding) {}

  ProjectionType projection;
  ImplOp impl;
  SpecializationMap arguments;
  Type binding;
};

struct EqualityResolution;
struct ProvenPremise;

/// One step of an endpoint's resolution, as its witness cites it: the equality
/// `projection = binding`, the impl selection chose for the projection's
/// application, the arguments its parameters take there, and the evidence for
/// each of that impl's where entries at those arguments, in order -- a proven
/// application at an application entry, and the resolution of the equality at
/// an equality entry.
struct ResolutionStep {
  TypeEqualityAttr equality;
  FlatSymbolRefAttr impl;
  ImplArgumentsAttr arguments;
  SmallVector<std::variant<std::shared_ptr<ProvenPremise>,
                           std::shared_ptr<EqualityResolution>>>
      premises;
};

/// The resolution of a monomorphic equality's two sides to one spelling: one
/// step per distinct projection resolved on the way.
struct EqualityResolution {
  TypeEqualityAttr equality;
  SmallVector<ResolutionStep> steps;
};

/// An application where entry discharged by a proof, as the evidence citing it
/// is written: `proven`, the proof's claim at the application its declaration
/// proves, and `entry`, the where entry's spelling, unproven, which
/// `respelling` carries `proven` to -- no step where the two are spelled alike.
struct ProvenPremise {
  ClaimType proven;
  ClaimType entry;
  SmallVector<ResolutionStep> respelling;
};

/// Builds at `builder`'s insertion point the witness of `premise`'s proof at
/// its own application, coerced to the entry's spelling by the witnesses of
/// its respelling steps where the two differ.
Value buildPremiseEvidence(OpBuilder &builder, Location loc,
                           const ProvenPremise &premise);

/// Builds at `builder`'s insertion point the evidence for `eq` from `steps`,
/// the steps resolving its sides to one ground spelling: refl for identical
/// sides; the sole step's witness where it proves `eq` as spelled; else the
/// composition of every step's witness, whose ground congruence closure carries
/// the sides together across every step. Each step's witness cites its impl
/// with one claim per where entry, a witness of the proof at an application
/// entry and the evidence built for the equality at an equality entry. The
/// result is closed: it reads no value from around it.
Value buildEqualityEvidence(OpBuilder &builder, Location loc,
                            TypeEqualityAttr eq,
                            ArrayRef<ResolutionStep> steps);

/// The template instantiations one stage run has cut.
///
/// Each instance is cut for a template at a call standing inside another
/// function, so the instances form a forest and what matters about one is how
/// many instances of the SAME template stand on the path that reaches it. A
/// template that instantiates itself at a larger type mints a distinct instance
/// at every step, so no cycle guard sees a repeat; that count is what tells
/// such a chain from a deep but finite nest of distinct templates, which Rust's
/// monomorphization collector counts the same way, per function.
///
/// An instance is named by the function it was cut into. Nothing erases a
/// function while the stage runs, so a recorded name stands for as long as the
/// chain does.
class InstantiationChain {
public:
  /// How many instances of `templateKey` stand on the chain reaching
  /// `instance`, counting `instance` itself. A function this has never
  /// recorded is a root and stands at zero.
  unsigned depthAt(Operation *instance, Attribute templateKey) const;

  /// Records that `instance` was cut for `templateKey` at a call inside
  /// `parent`.
  ///
  /// An instance already recorded keeps the chain it was first cut on, and an
  /// instance that is its own parent -- a call whose specialization reached the
  /// very function it stands in -- records nothing, so the chain stays a
  /// forest.
  void note(Operation *instance, Operation *parent, Attribute templateKey);

  /// The frames from the root down to `instance`, each a function and the
  /// template it was cut for. This is what a refusal at the depth limit names.
  SmallVector<std::pair<Operation *, Attribute>>
  chainTo(Operation *instance) const;

  /// Says a call refused to instantiate because the chain reached the limit.
  void noteLimitReached() { limitReached = true; }

  /// Whether any call refused on the depth limit. A greedy driver treats that
  /// refusal as a pattern that did not apply, so the stage reads this after its
  /// driver and fails rather than converging over the refusal.
  bool wasLimitReached() const { return limitReached; }

private:
  struct Frame {
    Operation *parent;
    Attribute templateKey;
  };

  DenseMap<Operation *, Frame> frames;
  bool limitReached = false;
};

/// Where impl selection is asked: the module whose symbol table the demand's
/// spelling names its trait and impls in -- the one whose proofs an answer may
/// cite, and the one a generated impl or a created proof belongs in -- and the
/// location of the op that demands it, where a refusal of the whole obligation
/// chain the demand opens is named. Built from the demanding op alone, so the
/// two cannot come from different places.
class SelectionSite {
public:
  /// The site of `demander`: the module anchoring its symbol uses, and its
  /// location.
  static SelectionSite of(Operation *demander) {
    return SelectionSite(getAnchorModule(demander), demander->getLoc());
  }

  ModuleOp scope;
  Location cause;

private:
  SelectionSite(ModuleOp scope, Location cause) : scope(scope), cause(cause) {}
};

/// ImplResolver is impl selection for one stage run: the one place that
/// chooses an impl for a concrete trait application, generates the impl a
/// module lacks, and writes the proofs of what it chose.
///
/// On construction, it discovers all loaded dialects that provide the
/// `GenerateImplsInterface` and asks them to populate its internal
/// `ImplGeneratorSet`. These generators are asked for an impl when no candidate
/// serves an application.
///
/// A rewrite that meets an obligation it needs settled asks here directly, at
/// the op that holds the obligation, as Rust's monomorphization collector asks
/// `Instance::resolve`: an allegation's proof (`resolveAndEnsureProofFor`) and
/// a ground projection's resolution (`resolveProjection`). Every answer is
/// memoized, so asking again answers alike: a selection, a proof, or a refusal
/// entered once it is final. A refusal is an answer -- an application nothing
/// serves stays spelled where it stands, for the stage's exit walk to name.
/// Answers are pure functions of the concrete types asked about and the
/// module's impls, and a generated impl is exact in its types, so generating
/// one never turns a unique candidate into an ambiguity.
///
/// This class may mutate the IR (e.g. by inserting `trait.proof` or
/// `trait.impl` ops) through the provided `OpBuilder`. It only builds ops,
/// never erases or replaces them, so no rewriter capability is required.
/// Callers running under a greedy pattern driver must still hand down that
/// driver's active `PatternRewriter`, whose listener enqueues the inserted ops
/// for the driver to revisit.
///
/// Every ask names the site it was demanded at (`SelectionSite`). Its module is
/// what the records are keyed under, so a demand raised inside a nested module
/// is never answered with a symbol only the module around it resolves.
class ImplResolver {
  public:
    /// Creates a new `ImplResolver` for the given `module`.
    /// Finds all loaded dialects that provide the `GenerateImplsInterface` and
    /// populates this `ImplResolver`'s `ImplGeneratorsSet`.
    explicit ImplResolver(ModuleOp module);

    /// Ensures canonical proof for a fully-concrete trait application `claim`.
    /// Resolution proceeds as follows:
    ///   1. If an unconditional ImplOp exists, its symbol proves the claim.
    ///   2. Otherwise, recursively resolve and ensure proofs for the impl's
    ///      where entries, then create (or reuse) a `trait.proof` op whose body
    ///      derives the claim from the impl over that evidence and whose symbol
    ///      proves it. The trait's requirements are the impl's to return.
    /// This function may mutate the IR via `builder`.
    ///
    /// Answers `claim` proven at the application it spells, naming the symbol
    /// (ImplOp or ProofOp) that proves exactly that spelling: the proof at
    /// the application selection resolves `claim` to, respelled as `claim`
    /// spells it where the two differ (`respellProof`). Refused if no unique
    /// and satisfiable impl can be found, naming why through `err` when it is
    /// given, however often it was asked before.
    Answer<ClaimType> resolveAndEnsureProofFor(ClaimType claim,
                                               const SelectionSite &site,
                                               OpBuilder &builder,
                                               llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// The proof of `to` resting on the evidence `proven` names: `proven`
    /// itself where it spells `to` already, else a cast of what it names to
    /// `to` (GHC's `$dX' = $dX |> co`), whose body is the witness of that proof
    /// or unconditional impl coerced by the steps selection resolves the two
    /// spellings' projections through -- the cast standing with that source
    /// and spelling, else one written. A cast is taken of the proof a cast
    /// names, never of the cast (`WitnessOp::verifySymbolUses` refuses a
    /// witness in a proof's body that names a cast), so every spelling of one
    /// proof is one cast of it. Not a selection: the impl the cited proof
    /// chose is kept, so a value crossing to another spelling carries the
    /// evidence it was given.
    /// Refused where selection does not carry the two spellings together;
    /// overflows where it does.
    Answer<ClaimType> respellProof(ClaimType proven, TraitApplicationAttr to,
                                   const SelectionSite &site,
                                   OpBuilder &builder);

    /// The evidence for `claim`, monomorphic, as a use spells it: the proof of
    /// the application selection resolves `claim` to, with the steps respelling
    /// that application as `claim` spells it, none where the two agree. A use
    /// builds from it the witness of that proof coerced by the steps
    /// (`buildPremiseEvidence`), so no proof is written at the use's own
    /// spelling: one is, by `respellProof`, only where a type names the
    /// spelling. Refused where selection refuses `claim`, naming why through
    /// `err` when it is given; overflows where selection does.
    Answer<ProvenPremise> resolveEvidenceFor(
        ClaimType claim, const SelectionSite &site, OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// The premise discharging the application entry `entry` by the proof
    /// `proof` names in `site`'s scope: that proof at the application its
    /// declaration proves -- a proof's own, an unconditional impl's header --
    /// and the steps respelling it as `entry`. Refused where the name resolves
    /// to no proof, or selection does not carry the two spellings together;
    /// overflows where selection does.
    Answer<ProvenPremise> resolvePremise(ClaimType entry,
                                         FlatSymbolRefAttr proof,
                                         const SelectionSite &site,
                                         OpBuilder &builder, unsigned depth);

    /// Resolves one step of a concrete ProjectionType: the impl the internal
    /// impl resolution pipeline selects for its application, the arguments
    /// that impl takes there, and its associated-type binding specialized at
    /// them.
    Answer<ProjectionResolution> resolveProjection(
        ProjectionType proj, const SelectionSite &site, OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Walks `ty` and replaces every projection `replacerFor` admits that
    /// selection resolves with its resolved type, to a fixed point; a
    /// projection selection refuses is left spelled as written. By default a
    /// projection is admitted once it is ground; under
    /// `makeGroundHeadProjectionReplacer`, once its head application is.
    /// Overflows where the resolution still changes after the depth limit's
    /// worth of projection steps (a binding that grows under it has no normal
    /// form), or where a step does.
    Answer<Type> resolveProjectionsIn(
        Type ty, const SelectionSite &site, OpBuilder &builder,
        decltype(&makeGroundProjectionReplacer) replacerFor =
            makeGroundProjectionReplacer);

    /// The ground spellings the sides of `eq` resolve to through selection at
    /// `site`, appending one step per distinct ground projection resolved on
    /// the way to `steps`; identical sides are read as spelled. The walk
    /// descends composites, so a projection nested inside one yields its step
    /// just as a top-level projection does, and runs to a fixed point because
    /// a binding may itself spell a projection. An equality entry of a
    /// resolving impl is resolved the same way, and must reach one spelling.
    /// Refused where a step is refused or a ground projection still stands;
    /// overflows where selection does or a side does not settle within the
    /// depth limit.
    Answer<std::pair<Type, Type>>
    resolveEquality(TypeEqualityAttr eq, const SelectionSite &site,
                    OpBuilder &builder, SmallVectorImpl<ResolutionStep> &steps,
                    llvm::function_ref<InFlightDiagnostic()> err = nullptr,
                    unsigned depth = 0);

    /// Whether an overflow (`Overflow`) was met over this resolver's span;
    /// the stage fails on it rather than naming what it left standing.
    bool hasOverflowed() const { return overflowed; }

    /// Names, through `err`, why selection refused the application of
    /// `obligation` -- a claim, or a projection's head -- in `scope`, over the
    /// candidates it recorded: none whose assumptions hold, or two or more.
    /// Names nothing where selection has entered no refusal. The refusal is
    /// what an obligation left standing does not say by itself; this asks
    /// selection nothing new.
    void nameRefusal(Type obligation, ModuleOp scope,
                     llvm::function_ref<InFlightDiagnostic()> err) const;

    /// The template instantiations cut over this resolver's span, which is one
    /// stage run.
    InstantiationChain &getInstantiationChain() { return instantiations; }

    /// The proof standing in `scope` whose body derives `app` from `impl`
    /// given, at its application entries in order, the proofs `subproofs`
    /// names; null where none stands. An equality entry is ground and has one
    /// answer, so it identifies nothing. An impl with no parameters and no
    /// where entries whose header spells `app` is its own proof. A read of
    /// `proofs`, the index of what each proof cites.
    ClaimType findProof(ModuleOp scope, ImplOp impl, TraitApplicationAttr app,
                        ArrayRef<FlatSymbolRefAttr> subproofs);

    /// Writes in `scope` the proof of `app` whose body derives `impl`'s header
    /// at `arguments` over one premise per entry of `entries`, `impl`'s where
    /// entries at those arguments: for each application entry, in order, a
    /// witness of the proof the next of `subproofs` names at the application
    /// that proof proves, and for each equality entry the evidence its `equalitySteps` build.
    /// Each spelling is written once, where it is decided: a witness whose
    /// application its entry spells otherwise, and a header spelling `app`
    /// otherwise, are coerced to the spelling they meet by the witnesses of the
    /// steps selection at `site` resolves their projections through. This is a
    /// derive transcribed, not a selection: it records nothing a selection
    /// reads. The proof stands at the end of `scope`, named by its impl and
    /// arguments as the module's symbol table admits. Refused where selection
    /// does not carry two such spellings to one; overflows where it does.
    Answer<ClaimType> writeProof(ModuleOp scope, ImplOp impl,
                                 TraitApplicationAttr app,
                                 const SpecializationMap &arguments,
                                 ArrayRef<ClaimType> entries,
                                 ArrayRef<FlatSymbolRefAttr> subproofs,
                                 ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
                                 const SelectionSite &site, OpBuilder &builder);

  private:
    /// What a proof in one module is identified by: the impl its derive cites,
    /// the application it proves and the proofs its application premises name
    /// -- or, for a cast, the proof or unconditional impl it casts and the
    /// application it respells it to, with no premise list.
    using ProofIdentity =
        std::tuple<Operation *, Operation *, TraitApplicationAttr, ArrayAttr>;

    /// The proof `identity` names in its module, null where none stands. The
    /// module's proofs are indexed on its first read, in module order, so the
    /// first proof of an identity is the one found; a proof written later is
    /// indexed as it is written (`indexProof`).
    FlatSymbolRefAttr lookupProof(const ProofIdentity &identity);

    /// Enters `proof`, standing in `scope`, under what it cites, unless a proof
    /// of that identity is entered already.
    void indexProof(ModuleOp scope, ProofOp proof);

    /// The proof of the application selection resolves `wanted` to, naming why
    /// selection refuses `wanted` through `err` when it is given.
    Answer<ClaimType> proveResolution(
        ClaimType wanted, const SelectionSite &site, OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err);

    /// The proof of `app`, the application selection resolved `wanted` to,
    /// through `impl` at `arguments`: the proof standing over the proofs of
    /// `impl`'s where entries at their resolutions, else one written. Refused
    /// where an entry is.
    Answer<ClaimType> proofAtResolution(
        ImplOp impl, TraitApplicationAttr app,
        const SpecializationMap &arguments, const SelectionSite &site,
        OpBuilder &builder, ClaimType wanted,
        llvm::function_ref<InFlightDiagnostic()> err);

    /// Finds the unique impl for the wanted claim and returns the normalized
    /// claim that was actually used for selection.
    Answer<ResolvedImpl> resolveImplFor(
        ClaimType wanted,
        const SelectionSite &site,
        OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// The steps carrying `spelled`'s type arguments to `resolved`'s through
    /// selection at `site`: those resolving the equality of each argument the
    /// two spell apart, whose sides selection must carry to one spelling,
    /// bounded below `depth` as `resolveEquality` is. Refused where it does
    /// not; overflows where selection does.
    Answer<SmallVector<ResolutionStep>>
    resolveRespelling(TraitApplicationAttr spelled,
                      TraitApplicationAttr resolved, const SelectionSite &site,
                      OpBuilder &builder, unsigned depth);


    /// `ty` with every projection `replacerFor` admits that selection
    /// resolves resolved, to a fixed point; none where the resolution still
    /// changes after the depth limit's worth of projection steps, which this
    /// names nothing about: a candidate's header is read through this, and a
    /// header with no normal form makes its impl no candidate rather than the
    /// stage's overflow. Overflows where a step does.
    Answer<std::optional<Type>> settleThroughSelection(
        Type ty, const SelectionSite &site, OpBuilder &builder,
        decltype(&makeGroundProjectionReplacer) replacerFor =
            makeGroundProjectionReplacer);

    /// Stops selection at the overflow `what`, met at `site`: the stage's
    /// hard error. Selection answers nothing more, nothing reached under it is
    /// entered in the memo, and it is named once per demand site.
    void overflow(const Overflow &what, const SelectionSite &site);

    /// `ty` with every monomorphic projection selection has already settled in
    /// `scope` resolved through what it settled; the rest left spelled as
    /// written. It asks selection nothing, so the stage's exit walk reads
    /// through it the key a refusal of a leftover obligation was entered
    /// under.
    Type readSettledProjectionsIn(Type ty, ModuleOp scope) const;

    /// One step of `proj`'s resolution through the impl selection has already
    /// settled on for its application in `scope`; fails where it has settled
    /// none (`readSettledProjectionsIn`).
    FailureOr<ProjectionResolution>
    readSettledProjection(ProjectionType proj, ModuleOp scope) const;

    /// Answers the arguments `impl`'s parameters take at `concreteSelf` where
    /// all of its where-clause assumptions are satisfiable when specialized
    /// for it, asked at `site` by the selection of `concreteSelf`, whose frame
    /// stands on the chain while it runs; refused where one is not, and
    /// overflows where judging one does.
    Answer<SpecializationMap> assumptionsSatisfiableFor(ImplOp impl,
                                                        ClaimType concreteSelf,
                                                        const SelectionSite &site,
                                                        OpBuilder &builder);

    ModuleOp module;
    ResolutionMemo memo;

    /// Every proof standing in a module the resolver has read proofs in, by
    /// what it cites (`ProofIdentity`): the index a lookup of a proof reads in
    /// place of a scan of the module's proofs.
    DenseMap<ProofIdentity, FlatSymbolRefAttr> proofs;
    DenseSet<Operation *> indexedScopes;

    bool overflowed = false;
    DenseSet<Location> overflowSites;

    InstantiationChain instantiations;
    ImplGeneratorSet generators;
};

} // end mlir::trait
