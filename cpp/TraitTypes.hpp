// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "DemandLedger.hpp"
#include "TraitAttributes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir { class OpBuilder; }

namespace mlir::trait {

// Generated interface declarations below mention these types before the
// concrete helper classes are defined in this header.
class TraitOp;
class InstantiationMap;
class UnificationMap;
class SpecializationMap;
class ProjectionBindings;
class EvidenceBindings;
class CallSubstitution;
class ImplResolver;
class ReadOnlyImplResolver;

}

#include <TraitTypeInterfaces.hpp.inc>

#define GET_TYPEDEF_CLASSES
#include <TraitTypes.hpp.inc>

namespace mlir { class AsmParser; }

namespace mlir::trait {

/// Parse one where-clause predicate: an application `@Trait[types...]` yielding a
/// TraitApplicationAttr, or an equality `!A = !B` yielding a checked
/// TypeEqualityAttr whose endpoints carry no proven claim, disambiguated by the
/// leading `@`. This is the single grammar
/// the claim type, the predicate array, and trait.assume each read; a `by @proof`
/// tail (allowed only on an application claim) is the caller's to add. Fails on a
/// malformed predicate, having emitted the diagnostic where the endpoints are
/// ill-formed.
FailureOr<Attribute> parseApplicationOrEqualityPredicate(AsmParser &p);

/// Rebuild an equality claim with `respell` applied to each endpoint, or nullopt
/// when `claim` is not an equality claim or neither endpoint changes. An equality
/// claim's endpoints live in hand-written storage the generic type replacer
/// cannot see, so a replacer that must reach them registers this rule for
/// ClaimType -- last, to take priority over any generic claim rule.
inline std::optional<Type> respellEqualityEndpoints(
    ClaimType claim, llvm::function_ref<Type(Type)> respell) {
  auto eq = claim.getEqualityAttr();
  if (!eq)
    return std::nullopt;
  Type newLhs = respell(eq.getLhs());
  Type newRhs = respell(eq.getRhs());
  if (newLhs == eq.getLhs() && newRhs == eq.getRhs())
    return std::nullopt;
  return Type(ClaimType::getEquality(claim.getContext(), newLhs, newRhs));
}

inline bool isPolymorphicType(Type root);
inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                                  Type root);
inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty);
inline void normalizeSubstitutionInPlace(llvm::DenseMap<Type,Type> &subst);

/// InstantiationMap: GenericTypeInterface -> UnificationTypeInterface.
///
/// Maps each generic type parameter encountered during instantiation to the
/// fresh unification variable allocated for that parameter. Reusing the mapping
/// preserves identity: repeated occurrences of the same generic instantiate to
/// the same fresh variable.
class InstantiationMap {
public:
  std::optional<UnificationTypeInterface> lookup(GenericTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(GenericTypeInterface key, UnificationTypeInterface value) {
    bindings[key] = value;
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  llvm::DenseMap<GenericTypeInterface, UnificationTypeInterface> bindings;
};

/// UnificationMap: UnificationTypeInterface -> Type.
///
/// Bindings accumulated while unifying two types. Keys are types that actively
/// participate in unification, such as inference variables and projections; the
/// values are the types they are known to equal.
class UnificationMap {
public:
  std::optional<Type> lookup(UnificationTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(UnificationTypeInterface key, Type value) { bindings[key] = value; }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  llvm::DenseMap<UnificationTypeInterface, Type> bindings;
};

/// SpecializationMap: GenericTypeInterface -> Type.
///
/// Concrete type arguments chosen for generic type parameters.
class SpecializationMap {
public:
  std::optional<Type> lookup(GenericTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(GenericTypeInterface key, Type value) {
    assert((!bindings.count(key) || bindings.lookup(key) == value) &&
           "specialization bindings must not be replaced with a different type");
    bindings[key] = value;
  }

  // A specialization is fully composed by construction, so one structural
  // substitution pass is enough.
  //
  // Applying a substitution resolves nothing. Stamping a concrete argument into
  // a projection spelling can turn a symbolic projection into a ground one,
  // and the result carries that projection still spelled as written: it reaches an
  // engine that could resolve it only if this caller goes on to unify with a
  // module or to stamp through the module-capable replacer.
  Type apply(Type ty) const { return applySubstitutionOnce(toTypeMap(), ty); }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

  static SpecializationMap fromTypeMap(const llvm::DenseMap<Type, Type> &subst) {
    SpecializationMap result;
    for (auto [key, value] : subst) {
      auto generic = dyn_cast<GenericTypeInterface>(key);
      assert(generic && "specialization keys must be generic types");
      result.bind(generic, value);
    }
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<GenericTypeInterface, Type> bindings;
};

/// ProjectionBindings: ProjectionType -> Type.
///
/// Concrete associated type results for projection types.
class ProjectionBindings {
public:
  std::optional<Type> lookup(ProjectionType key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(ProjectionType key, Type value) {
    assert((!bindings.count(key) || bindings.lookup(key) == value) &&
           "projection bindings must not be replaced with a different type");
    bindings[key] = value;
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<ProjectionType, Type> bindings;
};

/// EvidenceBindings: ClaimType -> ClaimType.
///
/// Maps unproven claim spellings to equivalent proven claim spellings discovered
/// while checking evidence.
class EvidenceBindings {
public:
  std::optional<ClaimType> lookup(ClaimType key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(ClaimType unproven, ClaimType proven) {
    assert(!unproven.isProven() && "evidence keys must be unproven claims");
    assert(proven.isProven() && "evidence values must be proven claims");
    assert((!bindings.count(unproven) || bindings.lookup(unproven) == proven) &&
           "evidence bindings must not be replaced with a different proof");
    size_t before = bindings.size();
    bindings[unproven] = proven;
    // Re-binding the same key to the same proof writes no new entry, so the
    // count is of the closure this map holds rather than of the calls it took.
    if (bindings.size() != before)
      countEvidenceBinding(bindings.size());
  }

  // Used by recursive proof verification to roll back an optimistic binding
  // when a nested obligation fails.
  void erase(ClaimType key) { bindings.erase(key); }

  bool empty() const { return bindings.empty(); }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<ClaimType, ClaimType> bindings;
};

/// What deriving each proven obligation produced, kept for as long as the proof
/// stands.
///
/// The evidence bindings a derivation writes are the closure of one proof: the
/// obligation it discharges bound to the claim proving it, then the same for
/// every obligation underneath. That closure is a fact about the proof, not
/// about the caller that asked for it, so it is kept once per normalized pair
/// rather than once per asking, and it is kept whatever else the fact base
/// does: an impl minted since can make a NEW application resolvable, but it
/// cannot change what the proof already standing over this one binds.
///
/// The key is the pair AS NORMALIZED -- the demanded obligation and the proven
/// value with their ground projections resolved -- rather than the pair as some
/// caller happened to spell it. Two callers reaching one obligation through
/// different projection spellings key it identically that way, which is what
/// lets one record answer both. The obligation is part of the key and not
/// derivable from the proven value: an obligation and the claim proving it need
/// not name the same application, and the closure's first binding is between
/// exactly those two.
///
/// The spellings inside a closure are the module's, and a sweep respells those,
/// so the record is transcribed with the module by the sweep that moves them --
/// storing them at a grade nothing respells would mean storing them without
/// their proofs, which is what the closure is for.
///
/// Entries are only ever added, and re-recording one is checked rather than
/// trusted: two derivations of one application that disagree are a fault this
/// must report, not a race to the map.
class ProofClosureRecord {
public:
  /// The evidence bindings one derivation wrote, in the order it wrote them.
  using Closure = SmallVector<std::pair<ClaimType, ClaimType>, 4>;

  /// What deriving `proven` for `unproven` produced, or nothing when no
  /// derivation of that pair has been recorded. Both sides are the normalized
  /// spellings.
  const Closure *lookup(ClaimType unproven, ClaimType proven) const {
    auto it = entries.find(std::make_pair(unproven, proven));
    return it == entries.end() ? nullptr : &it->second;
  }

  /// Whether a spelling is one nothing but a respelling can move.
  ///
  /// Two things leave a spelling open. A ground projection is one the impls
  /// standing when it was normalized could not resolve, so an impl generated
  /// since may resolve it and reach a different closure. A type variable is one
  /// the template it belongs to binds differently for each instance, so a pair
  /// carrying one is a coincidence of the instance being derived rather than a
  /// fact about the proof. A spelling with neither has no open question in it:
  /// the proof it names is settled, and the only thing that moves it afterwards
  /// is the sweep, which this record is transcribed by.
  static bool isSettled(ClaimType claim) {
    if (isPolymorphicType(Type(claim)))
      return false;
    bool open = false;
    Type(claim).walk([&](Type sub) {
      if (isa<ProjectionType>(sub))
        open = true;
    });
    return !open;
  }

  /// Whether the pair and every binding in `closure` are settled, which is the
  /// condition on holding a derivation for longer than the fact base stands.
  static bool isSettled(ClaimType unproven, ClaimType proven,
                        const Closure &closure) {
    if (!isSettled(unproven) || !isSettled(proven))
      return false;
    for (auto [boundUnproven, boundProven] : closure)
      if (!isSettled(boundUnproven) || !isSettled(boundProven))
        return false;
    return true;
  }

  /// Records `closure` as what deriving `proven` for `unproven` produces, and
  /// says whether this record now answers for the pair.
  ///
  /// A derivation that is not settled is refused: this answers for as long as
  /// the proof stands, and an unsettled derivation stands only until the next
  /// impl.
  ///
  /// A second derivation of one settled pair reaching a different closure is a
  /// pair this answers for no longer: the entry is withdrawn and the pair is
  /// refused from then on, so that what this holds is only ever what deriving
  /// would have produced.
  bool record(ClaimType unproven, ClaimType proven, Closure closure) {
    assert(isWellGraded(unproven, proven) &&
           "a recorded pair is an obligation and the claim proving it");
    if (!isSettled(unproven, proven, closure))
      return false;
    switch (place(entries, std::make_pair(unproven, proven),
                  std::move(closure))) {
    case Placement::Held:
      return true;
    case Placement::Agreed:
      // The pair was derived again and reached the answer already held. Nothing
      // needed deriving: this is the work a reader serving from the record is
      // meant to have stopped doing, so it is counted rather than passed over.
      countRecordedPairRederived();
      return true;
    case Placement::Withdrawn:
    case Placement::Refused:
      return false;
    }
    llvm_unreachable("a closure is placed, agreed with, withdrawn or refused");
  }

  /// How many pairs two closures disagreed over, which this therefore answers
  /// for no longer.
  size_t disputedCount() const { return disputed.size(); }

  /// Respells every key and every binding this holds through `replacer`, which
  /// is the same rewrite the sweep applies to the module.
  ///
  /// Two pairs can respell to one -- an unproven claim among the type arguments
  /// of both gains the same proof -- and the closures they carry are then two
  /// closures held for one pair. That pair meets the rule a pair derived twice
  /// meets: the closures are compared, equal ones leave it answered, differing
  /// ones withdraw it, and a pair already disputed takes neither. So what this
  /// holds after a transcription is still only what deriving would have
  /// produced.
  void respellWith(AttrTypeReplacer &replacer) {
    EntryMap respelled;
    respelled.reserve(entries.size());
    // The sweep's rewrite is the one that gives an unproven claim its proof, so
    // applying it to a spelling rewrites what is nested inside that spelling AND
    // the spelling itself. Only the first is wanted here: every position of this
    // record is an obligation or the claim proving one, and which it is says how
    // a reader will spell its ask. So each position keeps its own grade and
    // takes the interior rewrite -- an obligation stays an obligation whose type
    // arguments now name their proofs, which is exactly the spelling the next
    // ask arrives in.
    auto respellObligation = [&](ClaimType claim) {
      return cast<ClaimType>(replacer.replace(Type(claim))).asUnproven();
    };
    auto respellProof = [&](ClaimType claim) {
      return cast<ClaimType>(replacer.replace(Type(claim)));
    };
    auto respellPair = [&](const std::pair<ClaimType, ClaimType> &pair) {
      return std::make_pair(respellObligation(pair.first),
                            respellProof(pair.second));
    };
    // The disputes are transcribed first, because a disputed pair is one no
    // closure answers for again: an entry whose key respells onto a disputed
    // one is refused by it, rather than the two deciding it between them.
    llvm::DenseSet<Key> respelledDisputes;
    respelledDisputes.reserve(disputed.size());
    for (auto &key : disputed)
      respelledDisputes.insert(respellPair(key));
    disputed = std::move(respelledDisputes);
    for (auto &entry : entries) {
      Closure closure;
      closure.reserve(entry.second.size());
      for (auto &binding : entry.second) {
        // A closure is the set of bindings replaying it writes, and two
        // bindings that were distinct can respell alike. Keeping both would
        // make comparing closures stricter than comparing the bindings they
        // write, so a binding already in hand is not written again.
        Key respelledBinding = respellPair(binding);
        if (!llvm::is_contained(closure, respelledBinding))
          closure.push_back(respelledBinding);
      }
      // Nothing derived anything here, so the re-derivation the recording site
      // counts has no counterpart: two closures that agree leave the pair
      // answered and file nothing.
      place(respelled, respellPair(entry.first), std::move(closure));
    }
    entries = std::move(respelled);
    assert(gradesHold() && "transcribing must leave every position its grade");
  }

  size_t size() const { return entries.size(); }

private:
  /// An obligation and the claim proving it, which is what every key and every
  /// binding this holds is.
  using Key = std::pair<ClaimType, ClaimType>;
  using EntryMap = llvm::DenseMap<Key, Closure>;

  /// What placing a closure under a key left this holding.
  enum class Placement {
    /// The key held no closure and now holds this one.
    Held,
    /// The key held an equal closure, which is the one that stands.
    Agreed,
    /// The key held a differing closure, so neither stands.
    Withdrawn,
    /// The key is disputed, so it takes no closure.
    Refused,
  };

  /// Places `closure` under `key` in `into`, holding a key to one closure.
  ///
  /// Two closures held for one pair that disagree are a pair this cannot answer
  /// for: whichever answer it gave, the other closure would have been what
  /// deriving produced. The entry is withdrawn and the pair is refused from
  /// then on, so a reader gets no answer rather than the wrong one and the
  /// reader's own fallback is what covers it.
  Placement place(EntryMap &into, const Key &key, Closure closure) {
    if (disputed.contains(key))
      return Placement::Refused;
    auto [entry, inserted] = into.try_emplace(key, std::move(closure));
    if (inserted)
      return Placement::Held;
    if (entry->second == closure)
      return Placement::Agreed;
    into.erase(entry);
    disputed.insert(key);
    countProofClosureWithdrawn();
    return Placement::Withdrawn;
  }

  /// Whether a pair is an obligation paired with a claim proving it, which is
  /// what every key and every binding this holds is.
  static bool isWellGraded(ClaimType unproven, ClaimType proven) {
    return !unproven.isProven() && proven.isProven();
  }

  /// Whether every position this holds carries the grade its place demands.
  bool gradesHold() const {
    for (auto &entry : entries) {
      if (!isWellGraded(entry.first.first, entry.first.second))
        return false;
      for (auto [unproven, proven] : entry.second)
        if (!isWellGraded(unproven, proven))
          return false;
    }
    for (auto &key : disputed)
      if (!isWellGraded(key.first, key.second))
        return false;
    return true;
  }

  EntryMap entries;
  llvm::DenseSet<Key> disputed;
};

/// The proof derivations one span of resolution has completed, so that a
/// derivation performed once can be replayed rather than performed again.
///
/// Recursive proof verification derives an obligation once per call site,
/// because each call site's evidence map is born empty. A derivation's whole
/// output is the closure of bindings it writes into that map, so replaying that
/// closure into another map leaves it holding what deriving would have left it
/// holding. This is an acceptance shortcut and nothing else: a pair it has no
/// answer for is derived exactly as before.
///
/// A derivation reads the module. The ground-projection lookup resolves only
/// where exactly one candidate binds an application, so an impl minted since
/// can make an obligation newly resolvable or newly ambiguous and specialize it
/// differently. Every entry therefore names the fact base it was read from, and
/// an entry read from an earlier one is not an answer. Two events move that
/// fact base and neither moves with the other: impl selection minting a fact,
/// and a sweep respelling the module's copy of the facts -- a sweep records no
/// proof, so a count of facts cannot see it, and what a derivation reads are
/// spellings.
///
/// This holds no fact: everything in it is derivable again, which is what lets
/// a reader keep it through a handle that may not resolve and makes dropping an
/// entry always safe.
///
/// Beside it, and reached through it because every site that derives already
/// carries it, sits the record of what deriving each proven application
/// produces. That record answers for a pair however the caller spelled its
/// projections, and for as long as the proof stands; this memo answers for the
/// pair exactly as it arrived, and only until the next fact.
class ProofDerivationMemo {
public:
  /// The evidence bindings one derivation wrote, in the order it wrote them.
  using Closure = ProofClosureRecord::Closure;

  /// What deriving each proven application produces.
  ProofClosureRecord &getClosures() { return closures; }
  const ProofClosureRecord &getClosures() const { return closures; }

  /// The closure deriving `proven` for `unproven` produced, or nothing when no
  /// derivation of that pair is held against the fact base as it stands.
  const Closure *lookup(ClaimType unproven, ClaimType proven) const {
    auto it = entries.find(std::make_pair(unproven, proven));
    if (it == entries.end() || it->second.factBase != factBase)
      return nullptr;
    return &it->second.closure;
  }

  /// Holds `closure` as what deriving `proven` for `unproven` produced, against
  /// the fact base as it stands.
  void record(ClaimType unproven, ClaimType proven, Closure closure) {
    entries[std::make_pair(unproven, proven)] = Entry{std::move(closure),
                                                      factBase};
  }

  /// Says impl selection has minted a fact, so nothing derived before now was
  /// derived from the module as it stands.
  void noteFactWritten() { ++factBase; }

  /// Says a sweep has respelled the module's copy of the facts.
  void noteRespelling() { ++factBase; }

private:
  struct Entry {
    Closure closure;
    uint64_t factBase = 0;
  };

  llvm::DenseMap<std::pair<ClaimType, ClaimType>, Entry> entries;
  ProofClosureRecord closures;
  uint64_t factBase = 0;
};

/// ImplSpecialization: SpecializationMap + EvidenceBindings.
///
/// The complete set of type rewrites needed to specialize an impl method for a
/// proven self claim. Unlike CallSubstitution, this does not carry projection
/// bindings or require fixed-point closure.
class ImplSpecialization {
public:
  ImplSpecialization(SpecializationMap specialization,
                     EvidenceBindings evidenceBindings)
      : specialization(std::move(specialization)),
        evidenceBindings(std::move(evidenceBindings)) {}

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result = specialization.toTypeMap();
    for (auto [key, value] : evidenceBindings.toTypeMap())
      result[key] = value;
    return result;
  }

private:
  SpecializationMap specialization;
  EvidenceBindings evidenceBindings;
};

/// CallSubstitution: SpecializationMap + ProjectionBindings + EvidenceBindings.
///
/// The complete set of type rewrites needed to lower one call site, closed under
/// the projections and the proofs those rewrites expose.
///
/// The factory below is the only way to make one, so a substitution that exists
/// is one the read closed: every monomorphic projection the call spells is bound
/// to what impl selection settled for it, and every proven claim it spells is
/// bound together with everything that claim's proof binds underneath.
class CallSubstitution {
public:
  /// The closed substitution that lowers a call whose operands and results are
  /// `operandTypes` and `resultTypes` and whose callee signature is `formalTy`,
  /// starting from the parameter specialization the call's unification produced.
  ///
  /// The components expose bindings for one another -- a projection binding can
  /// rewrite a spelling into one that names a proof, and a proof binding can
  /// expose a projection in the claim it names -- so all three are chased
  /// together until no component grows. Every proven claim is read off the record
  /// of what deriving each pair produces, and only a pair no derivation has
  /// reached before is derived, through the same prover proof birth uses.
  ///
  /// Fails where the read cannot close it: a projection it cannot answer leaves
  /// the call spelling a type it cannot make concrete, and an obligation it
  /// cannot record has already reported itself.
  static FailureOr<CallSubstitution>
  forCall(SpecializationMap specialization, TypeRange operandTypes,
          TypeRange resultTypes, FunctionType formalTy, ModuleOp module,
          const ReadOnlyImplResolver &reading,
          llvm::function_ref<InFlightDiagnostic()> err = nullptr);

  const SpecializationMap &getSpecialization() const { return specialization; }

  // The components can expose bindings for one another, so call substitutions
  // must chase to a fixed point.
  Type apply(Type ty) const {
    return applySubstitutionToFixedPoint(toTypeMap(), ty);
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result = specialization.toTypeMap();
    for (auto [key, value] : projectionBindings.toTypeMap())
      result[key] = value;
    for (auto [key, value] : evidenceBindings.toTypeMap())
      result[key] = value;
    normalizeSubstitutionInPlace(result);
    return result;
  }

private:
  explicit CallSubstitution(SpecializationMap specialization)
      : specialization(std::move(specialization)) {}

  void discoverProjectionBindings(TypeRange types, ModuleOp module,
                                  const ReadOnlyImplResolver &reading,
                                  bool &declined);
  LogicalResult readEvidenceBindings(
      TypeRange types, ModuleOp module, const ReadOnlyImplResolver &reading,
      llvm::function_ref<InFlightDiagnostic()> err);

  size_t bindingCount() const {
    return specialization.bindingCount() + projectionBindings.bindingCount() +
           evidenceBindings.bindingCount();
  }

  SpecializationMap specialization;
  ProjectionBindings projectionBindings;
  EvidenceBindings evidenceBindings;
};

// The demand-walk accessor. An equality claim's endpoints are opaque to the
// generic type walk, so a reader that classifies theory content -- projections,
// claims -- must consult this to reach content nested only inside an endpoint.
// It invokes `callback` on both endpoints of every equality claim reachable in
// `root`; callers then walk those endpoints with their own classification. This
// is the binding read-side rule: every walk that classifies theory content
// consults the accessor for equality claims.
inline void walkEqualityEndpoints(Type root,
                                  llvm::function_ref<void(Type)> callback) {
  root.walk([&](Type sub) {
    if (auto claim = dyn_cast<ClaimType>(sub))
      if (auto eq = claim.getEqualityAttr()) {
        callback(eq.getLhs());
        callback(eq.getRhs());
      }
  });
}

// this walks a Type and looks for any occurrence of the given NeedleType.
// Equality-claim endpoints are opaque to Type::walk, so this also routes through
// the demand-walk accessor: a needle reachable only inside an equality endpoint
// is still found.
template<class NeedleType> bool containsType(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) {
    if (isa<NeedleType>(sub))
      found = true;
  });
  if (!found)
    walkEqualityEndpoints(ty, [&](Type endpoint) {
      if (containsType<NeedleType>(endpoint))
        found = true;
    });
  return found;
}

inline bool isPolymorphicType(Type root) {
  // fast path: if the root itself is a PolymorphicTypeInterface,
  // call its predicate
  if (auto p = dyn_cast<PolymorphicTypeInterface>(root)) {
    return p.isPolymorphic();
  }

  // otherwise, just walk the type
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance(); 

    if (auto p = dyn_cast<PolymorphicTypeInterface>(sub)) {
      if (p.isPolymorphic()) {
        found = true;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return found;
}

inline bool isMonomorphicType(Type ty) {
  return !isPolymorphicType(ty);
}

// A type is "ground" when it contains no PolymorphicTypeInterface nodes at all —
// no poly vars, no inference vars, no projections, no claims. Unlike
// isMonomorphicType, which asks whether any participant *reports* as polymorphic,
// isGroundType asks whether any participant *exists*. A monomorphic projection
// like !trait.proj<@Foo[i64], "Bar"> is monomorphic (no poly vars) but not
// ground (the projection still needs resolution).
inline bool isGroundType(Type root) {
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    if (isa<PolymorphicTypeInterface>(sub)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !found;
}

/// Refine implementation for ops whose `inferReturnTypes` refuses to mint
/// fresh PolyTypes: when inference fails, there is nothing to refine and
/// verification accepts the declared result types as-is (an opaque
/// polymorphic input determines nothing about the result). When inference
/// succeeds, the default compatibility check applies. Ops declare
/// InferTypeOpInterface with ["refineReturnTypes"] and delegate here.
template <typename ConcreteOp>
LogicalResult refineUnlessUnmintable(MLIRContext *ctx,
                                     std::optional<Location> location,
                                     ValueRange operands, DictionaryAttr attrs,
                                     OpaqueProperties properties,
                                     RegionRange regions,
                                     SmallVectorImpl<Type> &returnTypes) {
  SmallVector<Type, 4> inferred;
  if (failed(ConcreteOp::inferReturnTypes(ctx, location, operands, attrs,
                                          properties, regions, inferred)))
    return success();
  if (!ConcreteOp::isCompatibleReturnTypes(inferred, returnTypes))
    return emitOptionalError(
        location, "'", ConcreteOp::getOperationName(), "' op inferred type(s) ",
        inferred, " are incompatible with return type(s) of operation ",
        returnTypes);
  return success();
}

// returns true iff every PolymorphicTypeInterface inside `root` is polymorphic,
// and at least one such participant exists
inline bool isPurelyPolymorphicType(Type root) {
  bool sawPoly = false;

  // fast path: if the root itself is a PolymorphicTypeInterface
  // call its predicate
  if (auto p = dyn_cast<PolymorphicTypeInterface>(root)) {
    if (p.isMonomorphic())
      return false; // root participates and is monomorphic -> not purely polymorphic
    sawPoly = true; // root participates and is polymorphic
  }

  // otherwise, walk the type and check every participating subtype
  bool allParticipatingArePoly = true;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance();

    if (auto p = dyn_cast<PolymorphicTypeInterface>(sub)) {
      if (p.isPolymorphic()) {
        sawPoly = true;
        return WalkResult::advance();
      }
      // found a participant, monomorphic subtype -> fail
      allParticipatingArePoly = false;
      return WalkResult::interrupt();
    }

    return WalkResult::advance(); // non-participating types are ignored by design
  });

  // must have seen at least one one polymorphic participant, and none that are monomorphic
  return allParticipatingArePoly && sawPoly;
}

/// Instantiate a type with fresh inference variables.
///
/// For each GenericTypeInterface encountered in `t`, creates a fresh instance and
/// records the mapping in `inst`. The `idCounter` is used to generate unique IDs
/// for inference variables (should start at 0 for each instantiation context).
///
/// For structural types (e.g., FunctionType, TupleType), recursively instantiates
/// sub-elements and rebuilds the type.
///
/// For atomic types (e.g., i32, f64), returns the type unchanged.
///
/// This function is memoized via `inst` - if a GenericTypeInterface is encountered multiple
/// times within the same type structure, it maps to the same InferenceType.
Type instantiate(Type t, InstantiationMap& inst, uint64_t& idCounter);

inline void normalizeSubstitutionInPlace(llvm::DenseMap<Type,Type> &subst) {
  // Snapshot keys so we can mutate the map safely.
  llvm::SmallVector<Type, 8> keys;
  keys.reserve(subst.size());
  for (auto &kv : subst) keys.push_back(kv.first);

  // Path-compressed chase with simple cycle guard + memo.
  llvm::DenseMap<Type, Type> memo;
  llvm::SmallPtrSet<Type, 8> inStack;

  auto chase = [&](Type t, auto &chase_ref) -> Type {
    // If t doesn’t map anywhere, it’s a fixed point.
    auto it = subst.find(t);
    if (it == subst.end()) return t;

    // Already memoized?
    if (auto mit = memo.find(t); mit != memo.end()) return mit->second;

    // Cycle guard: if we re-enter t, bail by treating t as fixed.
    if (!inStack.insert(t).second) return t;

    Type to = chase_ref(it->second, chase_ref);  // recurse
    memo[t] = to;                                // path compression
    inStack.erase(t);
    return to;
  };

  for (Type k : keys) {
    Type v = chase(k, chase);
    if (v == k) {
      // Drop trivial self-map.
      subst.erase(k);
    } else {
      subst[k] = v; // Collapse k directly to its fixed point.
    }
  }

}

inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                              Type root) {
  SpecializationMap specialization;
  for (auto [key, value] : subst)
    if (auto generic = dyn_cast<GenericTypeInterface>(key))
      specialization.bind(generic, value);

  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      // GenericTypeInterface types own generic specialization entirely;
      // don't recurse into their result.
      Type specialized = generic.specializeWith(specialization);
      // A generic type that cannot spell its specialized form yields no type at
      // all. Stamping that into the enclosing type would leave a hole in it, so
      // the type stands as written and whatever consumes it next is what
      // reports that it never resolved.
      if (!specialized)
        specialized = t;
      return std::make_pair(specialized, WalkResult::skip());
    }

    // Otherwise, check the full mixed map for non-generic bindings such as
    // projections and evidence claims.
    if (auto it = subst.find(t); it != subst.end()) {
      return std::make_pair(it->second, WalkResult::advance());
    }

    return std::nullopt;
  });

  // Reach the equality endpoints the generic rule above cannot, applying the
  // same one-step substitution so a nested equality claim keeps no endpoint the
  // substitution still binds.
  replacer.addReplacement([&](ClaimType claim) -> std::optional<Type> {
    return respellEqualityEndpoints(claim, [&](Type t) {
      return applySubstitutionOnce(subst, t);
    });
  });

  return replacer.replace(root);
}

/// Applies `subst` repeatedly until it reaches a fixed point, so the returned
/// type carries no component that `subst` would still rewrite. The fixed
/// point is over `subst` alone; a projection whose base grounds under the
/// substitution stays a (now-resolvable) projection for the resolution
/// patterns.
inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty) {
  Type cur = ty;
  while (true) {
    Type next = applySubstitutionOnce(subst, cur);
    if (!next || next == cur) break;
    cur = next;
  }
  return cur;
}

/// Applies a GAT substitution: maps each type in `typeParams` to the
/// corresponding type in `assocTypeArgs`, then substitutes into `boundType`.
/// Returns the original `boundType` unchanged if `typeParams` is empty.
inline Type applyGATSubstitution(ArrayAttr typeParams,
                                 ArrayRef<Type> assocTypeArgs,
                                 Type boundType) {
  if (!typeParams || typeParams.empty())
    return boundType;
  assert(typeParams.size() == assocTypeArgs.size() &&
         "GAT arity mismatch: typeParams and assocTypeArgs must have the same size");
  DenseMap<Type,Type> gatSubst;
  for (auto [param, arg] : llvm::zip(typeParams, assocTypeArgs))
    gatSubst[cast<TypeAttr>(param).getValue()] = arg;
  return applySubstitutionToFixedPoint(gatSubst, boundType);
}

inline FailureOr<DenseMap<Type,Type>> composeSubstitutions(const DenseMap<Type,Type> &f,
                                                           const DenseMap<Type,Type> &g,
                                                           llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
  DenseMap<Type,Type> fog;

  for (const auto &[k, v] : f) {
    // rewrite v by g to a fixed point
    auto rewritten = applySubstitutionToFixedPoint(g, v);

    auto [it, inserted] = fog.try_emplace(k, rewritten);
    if (!inserted && it->second != rewritten) {
      if (err) err() << "conflicting substitution for " << k
                     << ": " << it->second << " vs " << rewritten;
      return failure();
    }
  }
  return fog;
}

/// Attempts to update `subst` so that the parameter type `formal`
/// is unified with the argument type `actual`.
///
/// This function applies the current substitution mapping to both `formal`
/// and `actual` before comparison. If the normalized types are identical,
/// the substitution is unchanged and the call succeeds.
///
/// Otherwise, `formal` is examined to determine how `actual` can serve as
/// its substitute:
///   - If `formal` implements `UnficationTypeInterface`, its
///     `unify` logic is invoked to extend `subst`.
///   - If `formal` and `actual` have the same type constructor and arity,
///     substitution recurses on their immediate subtypes.
///   - Otherwise, the types are considered incompatible and an error is
///     reported via `emitError`, if provided.
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst);

/// As above, but discards the resulting substitution
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics *and* the resulting substitution
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module);

/// Attempts to build a substitution which is the inverse of subst by mapping values in subst to keys
inline FailureOr<DenseMap<Type,Type>> invertSubstitution(
    const DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
  DenseMap<Type,Type> inverted;
  for (const auto &[k, v] : subst) {
    auto [it, inserted] = inverted.try_emplace(v, k);
    if (!inserted && it->second != k) {
      if (err) err() << "substitution is not injective: conflicting inverse for "
                     << v << ": " << it->second << " vs " << k;
      return failure();
    }
  }
  return inverted;
}

/// Compute the substitution that specializes a possibly polymorphic `formal`
/// type so it unifies with an `actual` type.
///
/// This is the main helper for checking uses of polymorphic functions or values
/// against a concrete call site or expected signature:
///
///  * **Instantiation.** Replace every generic parameter found in both `formal`
///    and `actual` with fresh inference variables, so unification works even if
///    `actual` itself contains generics.
///  * **Unification.** Solve constraints so the instantiated `formal` and
///    instantiated `actual` become equal, producing a mapping from inference
///    variables to concrete types.
///  * **Back-projection.** Compose the inference solution back through the
///    instantiation map to yield a map from the original generics in `formal`
///    to fully resolved types. Any generics that came from `actual` remain as
///    generics; no inference variables remain.
///  * **Normalization.** Chase and collapse substitution chains so the map is
///    stable (no trivial self-maps, no stale inference variables).
///
/// The returned map always has keys that are the generic placeholders occurring
/// in `formal`. Values are “ground” relative to inference (no `!trait.infer`
/// left), though they may still mention generics if the `actual` side was also
/// generic.
///
/// Returns `failure()` if the two types cannot be unified. If `err` is supplied,
/// a diagnostic is emitted on failure.
FailureOr<SpecializationMap> buildSpecialization(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

// this walks an Attribute and looks for any occurrence of the given NeedleType
template<class NeedleType> bool containsType(Attribute attr) {
  bool found = false;
  attr.walk([&](Attribute sub) {
    if (auto ta = dyn_cast<TypeAttr>(sub)) {
      if (containsType<NeedleType>(ta.getValue()))
        found = true;
    }
  });
  return found;
}

// this walks an Operation and looks for any occurrence of the given NeedleType
// note that this search does not recurse into child operations
template<class NeedleType> bool opMentionsType(Operation *op) {
  // inspect operands
  for (Type t : op->getOperandTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect result types
  for (Type t : op->getResultTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect block arguments
  for (Region& r : op->getRegions())
    for (Block& b : r)
      for (Value arg : b.getArguments())
        if (containsType<NeedleType>(arg.getType()))
          return true;

  // inspect attributes
  for (NamedAttribute attr : op->getAttrs())
    if (containsType<NeedleType>(attr.getValue()))
      return true;

  return false;
}

/// Collects distinct generic types appearing anywhere in `ty`.
///
/// Claim and projection types store their trait application as an attribute, so
/// this helper descends through those application arguments explicitly instead
/// of relying only on MLIR's structural type walk.
inline SmallVector<GenericTypeInterface,4> getGenericTypesIn(Type ty) {
  SmallVector<GenericTypeInterface, 4> result;
  DenseSet<Type> seen;

  auto collect = [&](Type ty, auto &collectRef) -> void {
    if (auto generic = dyn_cast<GenericTypeInterface>(ty)) {
      if (seen.insert(generic).second)
        result.push_back(generic);
    }

    if (auto claim = dyn_cast<ClaimType>(ty)) {
      if (auto eq = claim.getEqualityAttr()) {
        // The equality arm's endpoints are opaque to the structural walk, so
        // descend them explicitly to collect any generic hiding inside.
        collectRef(eq.getLhs(), collectRef);
        collectRef(eq.getRhs(), collectRef);
      } else {
        for (Type arg : claim.getTraitApplication().getTypeArgs())
          collectRef(arg, collectRef);
      }
    } else if (auto projection = dyn_cast<ProjectionType>(ty)) {
      for (Type arg : projection.getTraitApplication().getTypeArgs())
        collectRef(arg, collectRef);
      for (Type arg : projection.getAssocTypeArgs())
        collectRef(arg, collectRef);
    }

    ty.walkImmediateSubElements(
        /*walkAttrsFn=*/[](Attribute) {},
        /*walkTypesFn=*/[&](Type subTy) {
          collectRef(subTy, collectRef);
        });
  };

  collect(ty, collect);
  return result;
}

/// Whether `ty` spells a projection whose resolution is determined but not yet
/// written: a `ProjectionType` with no type variable left inside it.
///
/// A step that reads a spelling and cannot revisit what it read asks this
/// first. Mangling a name is the case that matters: the name is computed from
/// the spelling and nothing later recomputes it, so a name mangled while a
/// projection still stands is a name for a type the module no longer has once
/// the projection resolves.
///
/// The test is narrower than groundness on purpose. A type argument carrying a
/// proven claim is not ground and never becomes ground, so a step deferring on
/// groundness would defer forever; what it must wait for is the projection
/// alone. Claim and projection types carry their trait application as an
/// attribute, so this descends through those arguments explicitly rather than
/// relying only on the structural type walk.
///
/// Defined out of line so that a dialect asking it links one symbol rather than
/// the type identities this dialect's own library carries.
bool mentionsMonomorphicProjection(Type ty);

/// Verify that a `proven` claim soundly proves the (possibly still polymorphic)
/// `unproven` claim and extend `subst` with a mapping when appropriate.
///
/// Notes:
/// - `unproven` must be an unproven obligation; a proven `unproven` is a caller
///   error and is rejected with a diagnostic.
/// - Only records a mapping when converting an unproven form to its proven form;
///   no-op if `unproven == proven`.
/// - Recursively checks trait requirements and impl assumptions, ensuring all
///   subproofs are consistent and present.
///
/// `origin` names the caller: this recorder normalizes both claims through the
/// ground-projection lookup and normalizes the impl's obligations, so it raises
/// demand, and it runs both inside the stage and inside a proof op's verifier.
/// It has no default, so a new caller states which it is.
///
/// `memo`, when given, is consulted for the pair before anything else is done
/// with it and holds what this derivation produces. It is the stage's, and one
/// thread's: a verifier runs on a worker thread and passes none. Like `origin`
/// it has no default, so a new caller states whether it has one.
LogicalResult verifyAndRecordProof(ClaimType unproven,
                                   ClaimType proven,
                                   ModuleOp module,
                                   EvidenceBindings &bindings,
                                   DemandOrigin origin,
                                   ProofDerivationMemo *memo,
                                   llvm::function_ref<InFlightDiagnostic()> err);

/// Walks `ty` and binds every proof the types it spells name.
///
/// For every `ClaimType` node inside `ty` that carries a proof (i.e.
/// `isProven()`), this binds its unproven form (`claim.asUnproven()`) to the
/// proven claim itself, and binds whatever that claim's proof binds underneath.
/// If a conflicting binding for the same unproven key already exists, returns
/// failure and emits an error through `err`.
///
/// `origin` names the caller, which every proof this walk verifies is verified
/// under, and `memo` is what each of those verifications is served from and
/// held in. Neither has a default, so a new caller states both.
LogicalResult bindProofsIn(Type ty,
                                    ModuleOp module,
                                    EvidenceBindings &bindings,
                                    DemandOrigin origin,
                                    ProofDerivationMemo *memo,
                                    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// Resolve every ground projection in `ty` by module-visible impl
/// lookup, leaving non-ground and unresolvable projections spelled as written.
///
/// This is a read-only lookup: it selects the unique existing impl whose self
/// application matches a ground projection's trait application, reads that
/// impl's associated-type binding, and substitutes. Exactly one matching impl
/// is required; two or more decline. A conditional impl (nonempty assumptions)
/// may be that one match -- selecting it is mechanical name resolution, and a
/// legal program has already discharged the projection's head claim, which is
/// what its premise witnesses. It never mints proofs, generates impls, or
/// mutates IR, so it is safe to run inside a verifier.
///
/// `origin` names the caller, which the signature otherwise says nothing about.
/// It classifies the demand this call raises: a verifier's demand is counted,
/// a stage's is recorded. It has no default, so a new caller states which it
/// is rather than inheriting an answer.
///
/// `topLevelMissReasons`, when given, receives one bit per LookupMissReason on
/// which a projection of `ty` itself (not one reached inside a candidate probe)
/// declined. A caller that goes on to accept the unresolved type reads it to
/// say which class of the residual tolerance the accept fell in.
Type resolveGroundProjectionsByLookup(Type ty, ModuleOp module,
                                      DemandOrigin origin,
                                      unsigned *topLevelMissReasons = nullptr);

/// How many irreducible projection crossings the residual tolerance has
/// accepted in this process, and how those accepts split by the tolerance
/// site's own taxonomy. The four class counts partition the total. Reported
/// beside the demand census, whose population it overlaps but does not belong
/// to.
uint64_t residualToleranceAcceptCount();
uint64_t residualToleranceAcceptsGeneratorPendingCount();
uint64_t residualToleranceAcceptsMultiCandidateCount();
uint64_t residualToleranceAcceptsHypothesisCount();
uint64_t residualToleranceAcceptsMixedOrOtherCount();

std::string generateMangledNameSuffixFor(TypeRange typeArgs);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const DenseMap<Type,Type> &subst,
    ArrayRef<GenericTypeInterface> typeParams);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams);

} // end mlir::trait
