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
/// leading `@`. This is the single grammar for a where-clause predicate; a
/// `by @proof` tail (allowed only on an application claim) is the caller's to
/// add. Fails on a malformed predicate, having emitted the diagnostic where the
/// endpoints are ill-formed.
FailureOr<Attribute> parseApplicationOrEqualityPredicate(AsmParser &p);

/// The clone rule for equality evidence: rebuild an equality claim with
/// `respell` applied to each endpoint, atomically, through the checked
/// constructor. Answers nullopt when `claim` is not an equality claim, and
/// otherwise always skips the result's interior -- an endpoint moves through
/// this rule or not at all, so a replacer registering it must not let its own
/// walk reach the endpoints afterwards.
inline std::optional<std::pair<Type, WalkResult>> respellEqualityEndpoints(
    ClaimType claim, llvm::function_ref<Type(Type)> respell) {
  auto eq = claim.getEqualityAttr();
  if (!eq)
    return std::nullopt;
  Type newLhs = respell(eq.getLhs());
  Type newRhs = respell(eq.getRhs());
  if (newLhs == eq.getLhs() && newRhs == eq.getRhs())
    return std::make_pair(Type(claim), WalkResult::skip());
  return std::make_pair(
      Type(ClaimType::getEquality(claim.getContext(), newLhs, newRhs)),
      WalkResult::skip());
}

/// A replacer whose equality endpoints are a leaf.
///
/// An equality's endpoints are ordinary sub-elements, so every walk reaches
/// them -- which is what lets the framework's symbol-user driver see a
/// reference standing in one. What no replacer may do is move one: an endpoint
/// that received a stamped proof would be exactly the state
/// `TypeEqualityAttr::get` refuses, so two individually-correct rewrites would
/// kill a legal program. The one attribute-level rule this registers returns
/// the equality unchanged and skips its interior, which makes every replacer
/// built from it a reader of endpoints and never a writer. The rule sits on the
/// attribute rather than on the claim because a bare `TypeEqualityAttr` stands
/// in attribute positions with no claim around it: the witness attribute of an
/// equality-arm `trait.witness`, and the `assumptions` and `witnesses` arrays of
/// `trait.impl`. The one sanctioned mover is `respellEqualityEndpoints`.
AttrTypeReplacer makeEndpointSealedReplacer();

/// The sealed replacer above plus the one rule every ground-projection rewrite
/// registers: a polymorphic projection is left standing -- it stands for as many
/// types as its variables have instances, so no resolver owes it an answer --
/// and a ground one is resolved through `hop`, which declines by answering
/// nullopt.
AttrTypeReplacer makeGroundProjectionReplacer(
    std::function<std::optional<Type>(ProjectionType)> hop);

inline bool isPolymorphicType(Type root);
inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                                  Type root);
inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty);

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
  // a projection spelling can turn a symbolic projection into a ground one, and
  // the result carries that projection still spelled as written: what resolves
  // it is a later reading through the caller's established context, or a stamp
  // through the module-capable replacer.
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
    bindings[unproven] = proven;
  }

  // Used by the recursive proof derivation to roll back an optimistic binding
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

/// Whether a spelling is one nothing but a respelling can move.
///
/// Two things leave a spelling open. A ground projection is one the impls
/// standing when it was read could not resolve, so an impl generated since may
/// resolve it and reach a different type. A type variable is one the template it
/// belongs to binds differently for each instance, so what it spells is a
/// coincidence of the instance being read rather than a fact. A spelling with
/// neither has no open question in it, and a comparison of two such spellings is
/// a verdict nothing later revises.
inline bool spellingIsSettled(Type ty) {
  if (isPolymorphicType(ty))
    return false;
  bool open = false;
  ty.walk([&](Type sub) {
    if (isa<ProjectionType>(sub))
      open = true;
  });
  return !open;
}

/// Whether an equality premise read at a citation is one that citation cannot
/// decide, so the instances made of it decide it instead.
///
/// A reading carrying a type variable is a template's: the variable stands for
/// whatever each instance binds it to, and the instance is where the premise is
/// read. A reading with no variable in it is decided here even where it spells a
/// projection nothing resolves -- no instance moves that spelling either, so a
/// premise it leaves unequal is a premise that does not hold, and the impl
/// stating it does not apply. This is the one judgment impl selection, a proof,
/// a witness and a derive all read their premises by; `OpenPremise` says where
/// a reading this leaves open is decided.
inline bool premiseDefersToInstances(Type lhs, Type rhs) {
  return isPolymorphicType(lhs) || isPolymorphicType(rhs);
}

/// Where an equality premise a citation leaves open is decided.
enum class OpenPremise {
  /// The instances made of this template. The variable the reading carries
  /// stands for whatever each instance binds it to, and the clone reads the
  /// premise at the arguments that instance supplies.
  DecidedAtInstances,

  /// Here or nowhere. A proof op states its impl's premises at the claim it
  /// stands over, and a citation of that proof reads nothing inside it, so a
  /// premise the claim leaves open is one no later reading decides.
  RefusedHere
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
/// lets one record answer both.
///
/// The module the derivation read is part of the key as well. A spelling names
/// its symbols in one symbol table, and two modules can spell one claim
/// identically and mean two different proofs of it, so what a derivation
/// answers for is that claim under the module it was read from and not the
/// spelling alone.
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

  /// What deriving `proven` for `unproven` under `anchor` produced, or nothing
  /// when no derivation of that pair has been recorded. Both sides are the
  /// normalized spellings, and `anchor` is the module they were read from.
  const Closure *lookup(Operation *anchor, ClaimType unproven,
                        ClaimType proven) const {
    auto it = entries.find(Key{anchor, unproven, proven});
    return it == entries.end() ? nullptr : &it->second;
  }

  /// Whether a claim is one nothing but a respelling can move, which is the
  /// condition on holding what deriving it produced: an open spelling is one an
  /// impl generated since, or the instance a template is cut at, can move to
  /// another type, and the derivation would reach another closure there.
  static bool isSettled(ClaimType claim) {
    return spellingIsSettled(Type(claim));
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
  bool record(Operation *anchor, ClaimType unproven, ClaimType proven,
              Closure closure) {
    assert(isWellGraded(unproven, proven) &&
           "a recorded pair is an obligation and the claim proving it");
    if (!isSettled(unproven, proven, closure))
      return false;
    switch (place(entries, Key{anchor, unproven, proven}, std::move(closure))) {
    case Placement::Held:
    case Placement::Agreed:
      return true;
    case Placement::Withdrawn:
    case Placement::Refused:
      return false;
    }
    llvm_unreachable("a closure is placed, agreed with, withdrawn or refused");
  }

  /// Respells every key and every binding this holds under `anchor` through
  /// `replacer`, which is the same rewrite the sweep applies to that module.
  ///
  /// A derivation read under another module keeps its spellings: the rewrite
  /// names proof symbols one symbol table resolves, and what another module
  /// spells the same way it proves its own way.
  ///
  /// Two pairs can respell to one -- an unproven claim among the type arguments
  /// of both gains the same proof -- and the closures they carry are then two
  /// closures held for one pair. That pair meets the rule a pair derived twice
  /// meets: the closures are compared, equal ones leave it answered, differing
  /// ones withdraw it, and a pair already disputed takes neither. So what this
  /// holds after a transcription is still only what deriving would have
  /// produced.
  void respellWith(AttrTypeReplacer &replacer, Operation *anchor) {
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
    auto respellBinding = [&](const std::pair<ClaimType, ClaimType> &binding) {
      return std::make_pair(respellObligation(binding.first),
                            respellProof(binding.second));
    };
    // A key's module is the symbol table its spellings name, which a sweep
    // rewriting types does not move.
    auto respellKey = [&](const Key &key) {
      return Key{std::get<0>(key), respellObligation(std::get<1>(key)),
                 respellProof(std::get<2>(key))};
    };
    // The disputes are transcribed first, because a disputed pair is one no
    // closure answers for again: an entry whose key respells onto a disputed
    // one is refused by it, rather than the two deciding it between them.
    llvm::DenseSet<Key> respelledDisputes;
    respelledDisputes.reserve(disputed.size());
    for (auto &key : disputed)
      respelledDisputes.insert(std::get<0>(key) == anchor ? respellKey(key)
                                                          : key);
    disputed = std::move(respelledDisputes);
    for (auto &entry : entries) {
      if (std::get<0>(entry.first) != anchor) {
        place(respelled, entry.first, entry.second);
        continue;
      }
      Closure closure;
      closure.reserve(entry.second.size());
      for (auto &binding : entry.second) {
        // A closure is the set of bindings replaying it writes, and two
        // bindings that were distinct can respell alike. Keeping both would
        // make comparing closures stricter than comparing the bindings they
        // write, so a binding already in hand is not written again.
        std::pair<ClaimType, ClaimType> transcribed = respellBinding(binding);
        if (!llvm::is_contained(closure, transcribed))
          closure.push_back(transcribed);
      }
      place(respelled, respellKey(entry.first), std::move(closure));
    }
    entries = std::move(respelled);
    assert(gradesHold() && "transcribing must leave every position its grade");
  }

private:
  /// A module, an obligation read under it, and the claim proving that
  /// obligation, which is what every key this holds is. A binding inside a
  /// closure is the obligation and the claim alone: every binding a derivation
  /// wrote was read under the key's own module.
  using Key = std::tuple<Operation *, ClaimType, ClaimType>;
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
      if (!isWellGraded(std::get<1>(entry.first), std::get<2>(entry.first)))
        return false;
      for (auto [unproven, proven] : entry.second)
        if (!isWellGraded(unproven, proven))
          return false;
    }
    for (auto &key : disputed)
      if (!isWellGraded(std::get<1>(key), std::get<2>(key)))
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

  /// The closure deriving `proven` for `unproven` under `anchor` produced, or
  /// nothing when no derivation of that pair is held against the fact base as it
  /// stands. A spelling names its symbols in one symbol table, so the module the
  /// derivation read is part of what it answers for.
  const Closure *lookup(Operation *anchor, ClaimType unproven,
                        ClaimType proven) const {
    auto it = entries.find(Key{anchor, unproven, proven});
    if (it == entries.end() || it->second.factBase != factBase)
      return nullptr;
    return &it->second.closure;
  }

  /// Holds `closure` as what deriving `proven` for `unproven` under `anchor`
  /// produced, against the fact base as it stands.
  void record(Operation *anchor, ClaimType unproven, ClaimType proven,
              Closure closure) {
    entries[Key{anchor, unproven, proven}] = Entry{std::move(closure),
                                                   factBase};
  }

  /// Says impl selection has minted a fact, so nothing derived before now was
  /// derived from the module as it stands.
  void noteFactWritten() { ++factBase; }

  /// Says a sweep has respelled the module's copy of the facts.
  void noteRespelling() { ++factBase; }

private:
  /// A module and the pair a caller asked about under it.
  using Key = std::tuple<Operation *, ClaimType, ClaimType>;

  struct Entry {
    Closure closure;
    uint64_t factBase = 0;
  };

  llvm::DenseMap<Key, Entry> entries;
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

  /// The arguments the impl's own parameters took, without the proven spellings
  /// its evidence carries. What names an instance is read from this alone.
  const SpecializationMap &getSpecialization() const { return specialization; }

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
  /// starting from the arguments the call supplies for the callee's parameters.
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

  // A projection binding can rewrite a spelling into one that names a proof and
  // a proof binding can expose a projection, so the ground half of this map is
  // chased until it settles. Both kinds of key name a ground spelling, so no
  // chain through them reaches a key from its own value; the parameter bindings
  // ride along under keys nothing the chase mints spells again.
  Type apply(Type ty) const {
    return applySubstitutionToFixedPoint(toTypeMap(), ty);
  }

  // The three components key disjoint kinds of type -- a parameter, a
  // projection, a claim -- so the union holds every binding each one made under
  // the key it was made for. A variable therefore keeps the value bound to it,
  // which is what an equality endpoint reading it must see; a chain through a
  // projection or evidence key resolves because readers apply this map to a
  // fixed point.
  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result = specialization.toTypeMap();
    for (auto [key, value] : projectionBindings.toTypeMap())
      result[key] = value;
    for (auto [key, value] : evidenceBindings.toTypeMap())
      result[key] = value;
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

// Whether any occurrence of NeedleType is reachable in `ty`. An equality
// claim's endpoints are ordinary sub-elements, so the structural walk reaches a
// needle standing inside one.
template<class NeedleType> bool containsType(Type ty) {
  return ty.walk([](Type sub) {
             return isa<NeedleType>(sub) ? WalkResult::interrupt()
                                         : WalkResult::advance();
           }).wasInterrupted();
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
// no poly vars, no projections, no claims. Unlike
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

inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                              Type root) {
  SpecializationMap specialization;
  for (auto [key, value] : subst)
    if (auto generic = dyn_cast<GenericTypeInterface>(key))
      specialization.bind(generic, value);

  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
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

  // Move the equality endpoints the seal above holds as a leaf, applying the
  // generic-keyed part of the map alone: an endpoint receives variable
  // bindings, never a projection or evidence binding resolved inside it, which
  // a witness verifier's single-substitution instance check would break.
  llvm::DenseMap<Type, Type> genericKeyed = specialization.toTypeMap();
  replacer.addReplacement(
      [genericKeyed](ClaimType claim)
          -> std::optional<std::pair<Type, WalkResult>> {
    return respellEqualityEndpoints(claim, [&](Type t) {
      return applySubstitutionOnce(genericKeyed, t);
    });
  });

  return replacer.replace(root);
}

/// The pass budget the substitution fixed point spends before it gives up.
///
/// The chase settles in as many passes as the longest chain of keys it binds
/// through -- a small number, because the keys it is for are ground spellings: a
/// resolved projection and a proven claim each name a type, and naming one can
/// expose another, but nothing along such a chain is reached from its own value.
/// A key that is (a type variable bound to a spelling mentioning that same
/// variable) grows one level per pass and never settles, which is why parameter
/// bindings are stamped once instead of chased; this bound stops any growth a
/// caller still lets through well before it exhausts the stack in the structural
/// rewrite, and the depth it reaches stays walkable.
constexpr unsigned kSubstitutionFixedPointMaxPasses = 256;

/// How many times one declaration may occur on the chain of instantiations or
/// obligations that reaches the work in hand.
///
/// A template that instantiates itself at a larger type, and an impl whose
/// where-clause demands its own trait at a larger type, both make progress at
/// every step: each step is a new declaration instance or a new application, so
/// no cycle guard sees a repeat and the recursion runs until the machine stops
/// it. Counting occurrences of ONE declaration along the chain is what tells
/// such a chain from a deep but finite one, which nests distinct declarations.
/// Rust bounds the same two recursions the same way, at the same default
/// (`recursion_limit`, 128).
constexpr unsigned kInstantiationDepthLimit = 128;

/// How many frames from each end of a chain a refusal names. A chain at the
/// depth limit is a hundred-odd frames of the same shape; its ends say where it
/// started and what it grew into, and the frames between them say nothing more.
constexpr size_t kChainEndsNamed = 3;

/// Attaches the ends of `chain` to `diagnostic`, one note per frame through
/// `name`, with a note standing in for the frames between them.
template <typename FrameT>
void nameChainEnds(InFlightDiagnostic &diagnostic, ArrayRef<FrameT> chain,
                   llvm::function_ref<void(InFlightDiagnostic &, FrameT)> name) {
  if (chain.size() <= 2 * kChainEndsNamed) {
    for (const FrameT &frame : chain)
      name(diagnostic, frame);
    return;
  }
  for (const FrameT &frame : chain.take_front(kChainEndsNamed))
    name(diagnostic, frame);
  diagnostic.attachNote() << "... " << chain.size() - 2 * kChainEndsNamed
                          << " more frame(s) elided";
  for (const FrameT &frame : chain.take_back(kChainEndsNamed))
    name(diagnostic, frame);
}

/// Refuses an obligation chain that has reached the depth limit for one trait,
/// naming the chain that reaches `app`.
///
/// Every frame on the chain can be a distinct application, so the cycle guard
/// never fires on a chain whose obligations keep growing the type they ask
/// about. This is what stops it, and the chain is what says where the growth
/// came from. The refusal stands at the demand it was raised under, or at
/// `anchor` where no demand names a place.
///
/// The chain holds the applications an obligation walk is part-way through,
/// outermost first, whichever walk it is: impl selection deriving a candidate's
/// where clause and a proof derivation descending its subproofs count frames the
/// same way, so one obligation chain has one bound.
LogicalResult checkObligationChainDepth(ArrayRef<TraitApplicationAttr> chain,
                                        TraitApplicationAttr app,
                                        Location anchor);

/// Applies `subst` repeatedly until it reaches a fixed point, so the returned
/// type carries no component that `subst` would still rewrite. The fixed
/// point is over `subst` alone; a projection whose base grounds under the
/// substitution stays a (now-resolvable) projection for the resolution
/// patterns.
///
/// For ground keys alone -- a resolved projection, a proven claim -- where one
/// rewrite exposes another. A map keyed by a declaration's parameters is
/// stamped with `instantiate` instead: chasing one re-reads what a parameter
/// stood for as though it were the declaration's own spelling, so a parameter
/// whose argument mentions that parameter grows one level per pass. What such a
/// map hands back here is the partial the budget stopped at, spelled as
/// written, which every comparison downstream declines on.
inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty) {
  Type cur = ty;
  for (unsigned pass = 0; pass != kSubstitutionFixedPointMaxPasses; ++pass) {
    Type next = applySubstitutionOnce(subst, cur);
    if (!next || next == cur)
      break;
    cur = next;
  }
  return cur;
}

/// The classes a set of type equalities carves out of the types they mention.
///
/// An equality is not a directed rule: `A = B` and `B = A` say one thing, and a
/// set of equalities relates types symmetrically and transitively. Each class
/// has one canonical member -- the least under the structural order below -- and
/// a normalizer reads the equalities by rewriting every member of a class to
/// that one member. The rewrite settles because the member it lands on is fixed
/// for the class, where a directed rule carries `A` to `B` and back forever as
/// soon as both orientations stand.
///
/// The canonical member of a class is its least under this order, in this
/// sequence: fewer projections first, then fewer types named, then a type
/// mentioning no type variable before one that does, then the spelling itself.
/// Each key reads only the types' own structure, so the member a class is
/// headed by is the same in every process and under every allocation. Fewer
/// projections first is what makes the rewrite resolve projections rather than
/// introduce them, and fewer types next is what keeps a rewrite from growing
/// what it rewrites.
class TypeEquivalence {
public:
  /// Records that `a` and `b` are the same type, interning both.
  void assumeEqual(Type a, Type b) { unite(intern(a), intern(b)); }

  unsigned size() const { return terms.size(); }

  /// The index this structure knows `t` by, interning it if it is new.
  unsigned intern(Type t);

  /// The index of the canonical member of the class `id` falls in.
  unsigned findCanonical(unsigned id);

  /// Joins the classes of `a` and `b`.
  void unite(unsigned a, unsigned b);

  Type termAt(unsigned id) const { return terms[id]; }

  /// Every member that is not its class's canonical one, mapped to that one:
  /// the substitution a normalizer applies to rewrite a spelling to the one
  /// spelling its class stands for.
  llvm::DenseMap<Type, Type> substitutionToCanonicalMembers();

private:
  /// Whether `a` precedes `b` under the order the class doc states. The keys
  /// are computed on demand and kept, because a union asks for them at most
  /// once per member and the spelling key is the expensive one.
  bool precedes(unsigned a, unsigned b);

  /// How a member orders against the others: how many projections it spells,
  /// how many types its spelling names, whether it mentions a type variable,
  /// and the spelling itself. A type prints as something, so an empty
  /// `spelling` is one not yet printed -- the three keys before it decide every
  /// comparison but the one between two types of the same shape.
  struct OrderKey {
    unsigned projections;
    unsigned types;
    bool mentionsVariable;
    std::string spelling;
  };
  OrderKey &orderKeyOf(unsigned id);

  llvm::DenseMap<Type, unsigned> ids;
  SmallVector<Type> terms;
  SmallVector<unsigned> parent;
  SmallVector<std::optional<OrderKey>> orderKeys;
};

/// Applies a GAT substitution: maps each type in `typeParams` to the
/// corresponding type in `assocTypeArgs`, then substitutes into `boundType`.
/// Returns the original `boundType` unchanged if `typeParams` is empty.
///
/// The associated type's parameters are stamped once. An argument is a term of
/// whoever spelled the projection, so a label it shares with one of these
/// parameters names that caller's variable and not this declaration's.
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
  return applySubstitutionOnce(gatSubst, boundType);
}

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
        // This walk ignores attributes, so the endpoints an equality holds in
        // its own attribute are descended explicitly to collect any generic
        // standing inside.
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

/// The declaration parameter `ty` is an occurrence of, or null when it is none.
///
/// A type answers for itself through `getParameterAtom`; the parameter is the
/// one label that answer carries. A label carries only itself; a kind-
/// constraining wrapper carries the label it constrains; a computed type such
/// as `!coord.weak_product<A,B>` carries two, which makes it a composite the
/// caller decomposes rather than a parameter it binds.
inline GenericTypeInterface getParameterOccurrence(Type ty) {
  auto generic = dyn_cast<GenericTypeInterface>(ty);
  if (!generic)
    return {};
  Type atom = generic.getParameterAtom();
  GenericTypeInterface label;
  for (GenericTypeInterface inside : getGenericTypesIn(atom)) {
    if (getGenericTypesIn(Type(inside)).size() != 1)
      continue;
    if (label)
      return {};
    label = inside;
  }
  return label;
}

/// The type parameters `ty` binds, in first-occurrence order.
///
/// This is the reader every declaration's parameter list comes from: the
/// distinct parameters the generics `ty` spells are occurrences of, each
/// counted once at the position it first appears.
inline SmallVector<GenericTypeInterface, 4> getTypeParametersIn(Type ty) {
  SmallVector<GenericTypeInterface, 4> result;
  DenseSet<Type> seen;
  for (GenericTypeInterface generic : getGenericTypesIn(ty)) {
    GenericTypeInterface parameter = getParameterOccurrence(Type(generic));
    if (parameter && seen.insert(Type(parameter)).second)
      result.push_back(parameter);
  }
  return result;
}

/// The first `!trait.poly` label no type standing in `op`'s declaration spells.
///
/// A label names a position in the declaration that binds it, so a rewrite that
/// introduces a variable into a declaration already written -- the state of a
/// fold body, the intermediate tuple a flat map is cut into -- must not spell a
/// label that declaration, or any body already inside it, binds: a substitution
/// over either is keyed by those labels and would rewrite the new variable along
/// with them. Reading the labels standing in the declaration and taking the next
/// one is what keeps the new variable the rewrite's own.
///
/// The declaration is the outermost operation below the enclosing module that
/// `op` stands in -- the function, trait or impl whose parameters a substitution
/// is keyed by -- or `op` itself when it stands in none.
///
/// A declaration built from nothing needs none of this: its own parameters are
/// labelled 0, 1, ... by their position in its header, and its body is read
/// against that header alone.
unsigned firstUnusedPolyLabel(Operation *op);

/// The established context a comparison reads both sides through: an impl's own
/// bindings and premises inside a verifier, the recorded facts inside the
/// stage. A caller with no context passes none, which leaves both sides spelled
/// as written. Failure means the rewrite has no normal form.
using Normalizer = llvm::function_ref<FailureOr<Type>(Type)>;

/// One optional type argument per parameter of a declaration, dense by the
/// declaration's own parameter order.
///
/// A slot is filled at the first position that determines it and compared for
/// identity at every later one, so a parameter occurring twice admits only an
/// actual spelling one type at both. Nothing here decides a match: filling a
/// slot wrongly can only make the rebuilt declaration differ from the actual,
/// which the comparison refuses.
class TypeArguments {
public:
  explicit TypeArguments(ArrayRef<GenericTypeInterface> parameters)
      : parameters(parameters.begin(), parameters.end()),
        slots(parameters.size()) {}

  ArrayRef<GenericTypeInterface> getParameters() const { return parameters; }

  /// Whether the declaration binds `parameter`.
  bool binds(GenericTypeInterface parameter) const {
    return indexOf(parameter).has_value();
  }

  /// Fills `parameter`'s slot with `value`, or checks that it already holds
  /// exactly that. Fails on a parameter this declaration does not bind and on a
  /// second, differing value.
  LogicalResult assign(GenericTypeInterface parameter, Type value,
                       llvm::function_ref<InFlightDiagnostic()> err);

  /// The argument `parameter` took, or nothing when it took none or when the
  /// declaration does not bind it.
  std::optional<Type> lookup(GenericTypeInterface parameter) const {
    auto index = indexOf(parameter);
    return index ? slots[*index] : std::nullopt;
  }

  /// Whether every parameter has an argument.
  bool complete() const {
    return llvm::all_of(slots, [](const std::optional<Type> &slot) {
      return slot.has_value();
    });
  }

  /// The substitution these arguments spell: each filled slot keyed by its
  /// parameter. An empty slot leaves its parameter standing.
  SpecializationMap toSpecialization() const {
    SpecializationMap result;
    for (auto [index, parameter] : llvm::enumerate(parameters))
      if (slots[index])
        result.bind(parameter, *slots[index]);
    return result;
  }

private:
  std::optional<size_t> indexOf(GenericTypeInterface parameter) const {
    for (auto [index, declared] : llvm::enumerate(parameters))
      if (declared == parameter)
        return index;
    return std::nullopt;
  }

  SmallVector<GenericTypeInterface, 4> parameters;
  SmallVector<std::optional<Type>, 4> slots;
};

/// `declared` with its parameters replaced by `args`, in one pass.
///
/// A term substituted in is never revisited, so a callee's parameter and a
/// caller's that happen to share a spelling stay apart and a substitution that
/// maps a parameter into a type mentioning it terminates instead of growing.
inline Type instantiate(Type declared, const SpecializationMap &args) {
  return args.apply(declared);
}

/// Reads `actual` for the arguments `formal`'s parameters take, filling `args`.
///
/// The walk runs in lockstep. A formal parameter occurrence takes the actual
/// subterm standing opposite it; the actual side is never read as a pattern, so
/// nothing it spells is narrowed to fit. A formal claim recurses through its
/// predicate by position, blind to the proof either side carries. Every other
/// node recurses only where the two sides carry the same constructor, the same
/// attributes and the same child count; where they diverge the reading stops,
/// because it runs before either side is normalized and has nothing to learn
/// there.
///
/// A projection is not injective, so nothing is read out of the position it
/// stands in; its arguments are read only against the same projection on the
/// actual side, and only once every position outside a projection has had its
/// say. A parameter read twice at two spellings keeps the first, for the same
/// reason both readings are safe: the two may yet be one type. So this cannot
/// refuse anything, which is what makes `verifyEqualAfterInstantiation` the
/// only verdict.
void extractTypeArguments(Type formal, Type actual, TypeArguments &args);

/// Whether `formal` instantiated at `args` is `actual`.
///
/// Both sides are stripped of the proofs their claims carry -- comparison is
/// modulo the proof, permanently -- and read through `normalize`, so two
/// spellings the caller's established context makes one compare equal. The
/// verdict is then identity on interned types: a parameter no argument filled
/// stands, and a projection the context cannot reduce is equal to itself alone.
LogicalResult verifyEqualAfterInstantiation(
    Type formal, const SpecializationMap &args, Type actual,
    Normalizer normalize, llvm::function_ref<InFlightDiagnostic()> err);

/// The arguments carrying `formal` to `actual`, where `parameters` are the
/// parameters the declaration `formal` comes from binds: read them off the
/// actual, then check the instantiated declaration is the actual.
FailureOr<SpecializationMap> matchDeclaration(
    ArrayRef<GenericTypeInterface> parameters, Type formal, Type actual,
    Normalizer normalize, llvm::function_ref<InFlightDiagnostic()> err);

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

/// How many requirements `claim` carries: the `where` predicates of the trait
/// it applies, plus -- when `claim` is proven -- the assumptions of the impl its
/// proof cites. An equality claim applies no trait, so it requires nothing.
FailureOr<uint64_t> getClaimRequirementCount(
    ClaimType claim,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

/// The requirement `claim` carries at `index`, in the one order a projection
/// indexes them by: the trait's `where` predicates in declaration order, then --
/// when `claim` is proven -- the assumptions of the impl its proof cites, which
/// is the order that proof already names its subproofs in. The requirement is
/// instantiated at the claim's arguments; an application requirement of a proven
/// claim carries the provider of the subproof discharging it, and an equality
/// requirement never carries a provider. This is the one reading a
/// `trait.project` hop is checked against. Refuses an index past the last
/// requirement.
FailureOr<ClaimType> getClaimRequirementAt(
    ClaimType claim,
    ModuleOp module,
    uint64_t index,
    llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

/// The verdict of reading one citation.
enum class Citation {
  /// The declaration the cited symbol holds, read at the arguments the
  /// obligation supplies, rebuilds the obligation, and an impl named directly
  /// applies there.
  Carries,

  /// A side still spells a projection the impls standing now leave unresolved,
  /// so nothing decides this citation yet: the obligation stands unproven for
  /// impl selection to derive and for the leftover walk to refuse.
  Declined,

  /// The declaration rebuilds some other application, or an impl named directly
  /// does not apply at this one. `err` carries the reason.
  Refused
};

/// Whether the evidence `proven` names discharges the obligation `unproven`,
/// judged at that one claim and no deeper.
///
/// An obligation is discharged only by evidence for that same application, and
/// the evidence is the DECLARATION the cited symbol holds -- a blanket impl and
/// a proof written over type variables each stand for every instance of theirs.
/// So the judgment is whether that declaration, read at the arguments this
/// obligation supplies, rebuilds the obligation. The claim the citation is
/// spelled with is built from the obligation, so it says nothing here; only the
/// declaration does. An impl named directly must also apply where it is cited:
/// it takes no subproof, so its equality premises are read here or nowhere.
///
/// Nothing is read inside a cited proof. A proof op decides its own premises
/// and its own citations at the claim it stands over, and one whose premises
/// are all decided there holds at every instance of it, so a citation of it
/// needs this top-level match alone.
///
/// `origin` names the caller: the readings here normalize through the
/// ground-projection lookup, so they raise demand.
Citation verifyCitation(ClaimType unproven, ClaimType proven, ModuleOp module,
                        DemandOrigin origin,
                        llvm::function_ref<InFlightDiagnostic()> err);

/// Derives the whole tree standing under `proven` and extends `bindings` with a
/// mapping for every obligation it discharges, which is what a clone needs to
/// respell the claims it carries. A verifier asks `verifyCitation` about one
/// claim instead; this is the reader that goes underneath.
///
/// Notes:
/// - `unproven` must be an unproven obligation; a proven `unproven` is a caller
///   error and is rejected with a diagnostic.
/// - Only records a mapping when converting an unproven form to its proven form;
///   no-op if `unproven == proven`.
/// - Recursively reads trait requirements and impl assumptions, ensuring all
///   subproofs are consistent and present.
///
/// `origin` names the caller: this recorder normalizes both claims through the
/// ground-projection lookup and normalizes the impl's obligations, so it raises
/// demand. It has no default, so a new caller states which it is.
///
/// `memo`, when given, is consulted for the pair before anything else is done
/// with it and holds what this derivation produces. It is the stage's, and one
/// thread's. Like `origin` it has no default, so a new caller states whether it
/// has one.
LogicalResult verifyAndRecordProof(ClaimType unproven,
                                   ClaimType proven,
                                   ModuleOp module,
                                   EvidenceBindings &bindings,
                                   DemandOrigin origin,
                                   ProofDerivationMemo *memo,
                                   llvm::function_ref<InFlightDiagnostic()> err);

/// Refuses every citation the claims `ty` spells that does not discharge the
/// obligation it stands on, each read at its own claim and no deeper.
///
/// A proven claim spells evidence for one application, and what the spelling
/// asserts is that the declaration the evidence holds carries to that
/// application. What the evidence proves underneath was decided at the proof op
/// holding it. Two spellings proving one claim by different symbols are two
/// names for one fact, so nothing is carried across the claims here.
///
/// A citation nothing standing now decides is left to the leftover walk, which
/// refuses an obligation no round resolves.
///
/// `origin` names the caller: the readings raise demand.
LogicalResult verifyCitationsIn(Type ty, ModuleOp module, DemandOrigin origin,
                                llvm::function_ref<InFlightDiagnostic()> err);

/// Walks `ty` and binds every proof the types it spells name.
///
/// For every `ClaimType` node inside `ty` that carries a proof (i.e.
/// `isProven()`), this binds its unproven form (`claim.asUnproven()`) to the
/// proven claim itself, and binds whatever that claim's proof binds underneath.
/// If a conflicting binding for the same unproven key already exists, returns
/// failure and emits an error through `err`.
///
/// `origin` names the caller, which every proof this walk derives is derived
/// under, and `memo` is what each of those derivations is served from and held
/// in. Neither has a default, so a new caller states both.
LogicalResult bindProofsIn(Type ty,
                                    ModuleOp module,
                                    EvidenceBindings &bindings,
                                    DemandOrigin origin,
                                    ProofDerivationMemo *memo,
                                    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// The module that anchors symbol lookups for `anchor`: the operation itself
/// when it is the module, otherwise its enclosing module (null if it has none).
/// A symbol-use verifier reached through an attribute or type interface recovers
/// its lookup scope this way.
ModuleOp getAnchorModule(Operation *anchor);

/// Which projections a lookup is licensed to resolve.
enum class LookupScope {
  /// Only a projection whose arguments are all concrete. Its resolution is a
  /// fact about the program: the spelling names one type, and rewriting it into
  /// that type is sound in any position, including one whose result is stamped
  /// into IR.
  Ground,

  /// Also a projection whose arguments still carry variables, when the
  /// projection's own spelling determines which impl serves it: exactly one impl
  /// matches, and the match binds only that impl's type parameters -- never a
  /// variable the projection spells. Such a projection resolves the same way for
  /// every instance of its variables, so the two spellings denote one type
  /// whatever inference goes on to choose. A projection whose spelling would
  /// have to be narrowed to fit an impl determines nothing (inference may narrow
  /// it another way) and is left as written. A resolution under this scope is
  /// read to compare a spelling, never to serve it: it feeds a comparison, not a
  /// position that stamps the resolved type into IR.
  Determined
};

/// Resolve every projection in `ty` that `scope` licenses by module-visible impl
/// lookup, leaving the rest spelled as written.
///
/// This is a read-only lookup: it selects the unique existing impl whose self
/// application matches a projection's trait application, reads that impl's
/// associated-type binding, and substitutes. Exactly one matching impl is
/// required; two or more decline. A conditional impl (nonempty assumptions) may
/// be that one match -- selecting it is mechanical name resolution, and a legal
/// program has already discharged the projection's head claim, which is what its
/// premise witnesses. It never mints proofs, generates impls, or mutates IR, so
/// it is safe to run inside a verifier.
///
/// `origin` names the caller, which the signature otherwise says nothing about.
/// A verifier's demand stays local; a stage demand enters the preparation queue.
/// It has no default, so each caller states which applies.
///
Type resolveProjectionsByLookup(Type ty, ModuleOp module, DemandOrigin origin,
                                LookupScope scope);

/// The fallible sibling of the resolver above, for a caller reached from
/// untrusted IR that must refuse a nonconverging projection rather than decline
/// on it. A projection whose resolution never grounds -- a cyclic
/// associated-type binding across impls -- yields failure here (surfaced through
/// `emitError` for a live demand, silent under a cross-check) instead of the
/// infallible entry's spelled-as-written partial. The proof and call-signature
/// verifiers thread it so a hostile cycle fails verification cleanly, never
/// aborting the process. Success returns the ground normal form exactly as the
/// infallible entry would.
FailureOr<Type> resolveProjectionsByLookup(
    Type ty, ModuleOp module, DemandOrigin origin, LookupScope scope,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// Rewrites `ty` with `step` until its spelling stops changing, handing back the
/// driver's partial at a rewrite that never does.
///
/// Resolving a projection substitutes the selected impl's associated-type
/// binding, and that binding may itself be spelled as a projection -- an impl
/// whose associated type forwards through its own type parameter (`type Element
/// = B::Element`) binds one. One rewrite leaves such a spelling standing a
/// second would resolve, so a projection normal form is a fixed point of `step`
/// and never the result of a single pass. Every normalizer reaches it here, so
/// the spelling one hands back is the spelling all of them do -- which is what
/// lets a demand and an impl's self application be compared for equality at
/// all.
Type normalizeProjectionsToFixedPoint(Type ty, ModuleOp module,
                                      llvm::function_ref<Type(Type)> step);

/// The fallible driver `normalizeProjectionsToFixedPoint` wraps: rewrites `ty`
/// with `step` until its spelling stops changing and reports success, or reports
/// failure at a rewrite that never settles instead of stopping the compilation.
/// `out` receives the fixed point on success and the still-changing partial
/// normal form on failure, so a caller owning its own diagnostic -- an op-
/// attached error over an impl-local rule step that must not reach the fatal
/// module-level reporter -- names the type that would not converge.
LogicalResult tryNormalizeProjectionsToFixedPoint(
    Type ty, llvm::function_ref<Type(Type)> step, Type &out);

/// A normalizer over the impls a module holds: a ground projection exactly one
/// of them binds reduces to what it binds, which is the same answer in every
/// position that spells it. This is a reading of committed facts and not of the
/// evidence an operation carries, so a caller that may only read the latter
/// does not build one.
class GroundProjectionLookup {
public:
  /// `err`, when given, receives the diagnostic for a resolution chain with no
  /// normal form, which is the one way this reading fails.
  GroundProjectionLookup(ModuleOp module, DemandOrigin origin,
                         llvm::function_ref<InFlightDiagnostic()> err = nullptr)
      : module(module), origin(origin), err(err) {}

  FailureOr<Type> operator()(Type ty) const {
    return resolveProjectionsByLookup(ty, module, origin, LookupScope::Ground,
                                      err);
  }

private:
  ModuleOp module;
  DemandOrigin origin;
  llvm::function_ref<InFlightDiagnostic()> err;
};

std::string generateMangledNameSuffixFor(TypeRange typeArgs);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const DenseMap<Type,Type> &subst,
    ArrayRef<GenericTypeInterface> typeParams);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams);

} // end mlir::trait
