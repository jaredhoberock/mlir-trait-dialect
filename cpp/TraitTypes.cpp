// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "DemandLedger.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/xxhash.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <TraitTypeInterfaces.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <TraitTypes.cpp.inc>

#define DEBUG_TYPE "trait-residual-tolerance"

// Counts each irreducible projection-vs-rigid crossing the module-capable
// unify entry accepts without a binding (the residual tolerance). Licensed
// behavior, so it is a statistic rather than an error; a nonzero value under
// -stats makes regrowth of that population visible in-tree.
STATISTIC(numResidualToleranceAccepts,
          "irreducible projection crossings accepted by the residual tolerance");

// The same population split by which class of the tolerance site's own taxonomy
// each accept fell in, so that a zero in one class can be read as a discharged
// clause rather than lost in the aggregate. A generator-pending accept is a
// ground base no impl binds yet; a multi-candidate accept is a ground base
// several impls bind; a hypothesis accept is a still-symbolic base resolvable
// only through a frame hypothesis; a mixed-or-other accept is a ground base
// whose projections declined on several arms at once, or on an arm that is
// neither headline case. The four partition the aggregate: each accept bumps
// exactly one of them beside it.
STATISTIC(numResidualToleranceAcceptsGeneratorPending,
          "residual-tolerance accepts on a ground base no impl binds yet");
STATISTIC(numResidualToleranceAcceptsMultiCandidate,
          "residual-tolerance accepts on a ground base several impls bind");
STATISTIC(numResidualToleranceAcceptsHypothesis,
          "residual-tolerance accepts on a still-symbolic base");
STATISTIC(numResidualToleranceAcceptsMixedOrOther,
          "residual-tolerance accepts on a ground base declining on several or "
          "non-headline arms");

namespace mlir::trait {

uint64_t residualToleranceAcceptCount() {
  return numResidualToleranceAccepts.getValue();
}

uint64_t residualToleranceAcceptsGeneratorPendingCount() {
  return numResidualToleranceAcceptsGeneratorPending.getValue();
}

uint64_t residualToleranceAcceptsMultiCandidateCount() {
  return numResidualToleranceAcceptsMultiCandidate.getValue();
}

uint64_t residualToleranceAcceptsHypothesisCount() {
  return numResidualToleranceAcceptsHypothesis.getValue();
}

uint64_t residualToleranceAcceptsMixedOrOtherCount() {
  return numResidualToleranceAcceptsMixedOrOther.getValue();
}

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <TraitTypes.cpp.inc>
  >();
}

std::string hashToSuffix(StringRef input) {
  uint64_t hash = llvm::xxHash64(input);
  std::string result;
  llvm::raw_string_ostream out(result);
  out << llvm::format("_h%016" PRIx64, hash);
  out.flush();
  return result;
}

std::string generateMangledNameSuffixFor(TypeRange typeArgs) {
  if (typeArgs.empty()) return "";

  std::string full;
  llvm::raw_string_ostream os(full);
  for (Type ty : typeArgs)
    os << "_" << ty;
  os.flush();

  return hashToSuffix(full);
}

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const DenseMap<Type,Type> &subst,
    ArrayRef<GenericTypeInterface> typeParams) {
  SmallVector<Type> concreteTypes;
  for (auto ty : typeParams)
    concreteTypes.push_back(applySubstitutionToFixedPoint(subst, ty));
  return generateMangledNameSuffixFor(concreteTypes);
}

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams) {
  SmallVector<Type> concreteTypes;
  for (auto ty : typeParams)
    concreteTypes.push_back(subst.apply(ty));
  return generateMangledNameSuffixFor(concreteTypes);
}


//===----------------------------------------------------------------------===//
// Ground projection resolution
//===----------------------------------------------------------------------===//

namespace {

/// Stops the compilation at a ground projection whose resolution does not
/// terminate, naming the demand it arose under.
///
/// The rewrite this reports has no normal form, so there is no spelling to
/// return and no diagnostic a later stage could attach to a spelling it never
/// received. The enclosing demand is what names where the projection was asked
/// about; outside a stage span there is no enclosing demand and the module is
/// all there is to name.
///
/// No in-tree program reaches this, and the checks in front of it are why. An
/// impl whose own associated-type binding projects back through itself
/// resolves to a spelling equal to the demand, so the lookup makes no progress
/// and the leftover walk reports the projection as unresolved; a binding cycle
/// across two impls either grows the type until the substitution that stamps
/// it runs out of stack, or oscillates without growing and is caught by the
/// rewrite budget of the driver that keeps re-deriving it. This stands behind
/// all three.
[[noreturn]] void reportUnnormalizableGroundProjection(Type ty,
                                                       unsigned iterations,
                                                       ModuleOp module) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "ground projection normalization did not converge within "
         << iterations << " iterations for type " << ty << ", demanded at ";
  if (std::optional<Location> anchor = currentDemandAnchor())
    stream << *anchor;
  else
    stream << module.getLoc();
  llvm::report_fatal_error(Twine(message));
}

} // namespace

Type resolveGroundProjectionsByLookup(Type ty, ModuleOp module,
                                      DemandOrigin origin,
                                      unsigned *topLevelMissReasons) {
  if (!module)
    return ty;

  // Candidate impls per trait application, memoized for this resolution. The
  // lookup mutates no impls, so the memo stays valid across the fixed-point
  // iterations below, and it is scoped to this call so nothing outside observes
  // it -- repeated projections over the same application skip the module scan.
  DenseMap<TraitApplicationAttr, SmallVector<ImplOp>> candidateCache;

  // The demands this call recorded. Established only when the postcondition
  // below is armed, so an ordinary call carries no per-call state at all.
  const bool checkRecordingCoverage = DemandLedger::isPostconditionEnabled() &&
                                      isDemandRecordingActive() &&
                                      recordsToLedger(origin);
  DenseSet<Type> recordedDemands;

  AttrTypeReplacer replacer;
  replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
    // Impl enumeration below builds each candidate's self-claim substitution,
    // which unifies, which re-enters this callback. The guard makes that
    // re-entry visible, so a demand raised about a candidate is told apart from
    // the demand this call was asked about.
    LookupProbeScope probe;

    auto declineWith = [&](LookupMissReason reason) {
      recordLookupMiss(Type(proj), reason, origin, probe.getEnclosingDepth());
      if (checkRecordingCoverage)
        recordedDemands.insert(Type(proj));
      // A caller classifying an accept wants the arms the projections of `ty`
      // itself declined on, so only the outermost walk contributes. A decline
      // inside a candidate probe (a nonzero enclosing depth) is about a
      // candidate the partition may discard, not about `ty`.
      if (topLevelMissReasons && probe.getEnclosingDepth() == 0)
        *topLevelMissReasons |= 1u << static_cast<unsigned>(reason);
      return std::optional<Type>(std::nullopt);
    };

    // Only a projection whose arguments are all concrete has a determined
    // resolution. A projection over a still-symbolic base stays spelled as
    // written.
    if (isPolymorphicType(proj))
      return std::nullopt;

    ClaimType claim = proj.asClaim();
    TraitApplicationAttr app = claim.getTraitApplication();

    // Read-only selection: resolve only when exactly one existing impl binds
    // this application. Two or more matches, and impl generation, are left to
    // the resolver. The single match may be conditional (a nonempty assumptions
    // list): selecting it is mechanical name resolution, not premise evaluation,
    // and a legal program has already discharged this ground projection's head
    // claim -- the premise the conditional impl carries.
    auto it = candidateCache.find(app);
    if (it == candidateCache.end()) {
      auto trait = app.getTrait(module, nullptr);
      if (failed(trait))
        return declineWith(LookupMissReason::TraitSymbolNotFound);
      it = candidateCache.insert({app, trait->getCandidateImplsFor(claim)}).first;
    }
    const SmallVector<ImplOp> &candidates = it->second;
    if (candidates.size() != 1)
      return declineWith(candidates.empty()
                             ? LookupMissReason::NoCandidateImpl
                             : LookupMissReason::MultipleCandidateImpls);
    ImplOp impl = candidates.front();

    SmallVector<Type> assocTypeArgs(proj.getAssocTypeArgs());
    auto binding = impl.specializeAssociatedTypeBinding(
        proj.getAssocName().getValue(), assocTypeArgs);
    if (failed(binding))
      return declineWith(LookupMissReason::AssociatedTypeBindingFailed);
    auto subst = impl.buildSubstitutionForSelfClaim(claim);
    if (failed(subst))
      return declineWith(LookupMissReason::SelfClaimSubstitutionFailed);
    return applySubstitutionToFixedPoint(subst->toTypeMap(), *binding);
  });

  // A resolved binding may itself expose a ground projection, so run to a
  // fixed point.
  constexpr unsigned maxIterations = 64;
  Type previous;
  for (unsigned i = 0; i != maxIterations && ty != previous; ++i) {
    previous = ty;
    ty = replacer.replace(ty);
  }

  // Reaching the iteration cap while the type is still changing means the
  // lookup rewrite has no fixed point (a cyclic or oscillating resolution).
  // What the loop reached is a partial normal form, and every caller either
  // compares a spelling against it or stamps it into a specialized instance, so
  // handing it back would turn a resolution that does not terminate into a
  // spelling mismatch or a mis-specialized monomorph somewhere else entirely.
  // The compilation stops at the demand that would not normalize instead.
  if (ty != previous)
    reportUnnormalizableGroundProjection(ty, maxIterations, module);

  // Every monomorphic projection this call leaves standing is a demand no impl
  // served, so a recording site must have observed it. A survivor with no
  // record is a gap in the ledger's wiring, not a fault in the program, so it
  // goes to the census channel: an error here would fail a correct compile.
  //
  // No in-tree program makes this fire, and none is written to: the two halves
  // are complementary by construction. Every exit that leaves a projection
  // spelled as written goes through declineWith, which records and then adds to
  // the set this walk consults, so a survivor the callback visited is in the
  // set. What remains is a projection the callback never visited -- one the
  // replacer does not reach -- which is the gap the check exists to find and
  // which no module can be written to produce today. Deleting an arm's record,
  // or the set, makes it fire: the census lit rows carry an implicit negative
  // pin over their whole output, so they fail on the first unhooked line.
  if (checkRecordingCoverage)
    ty.walk([&](Type sub) {
      auto proj = dyn_cast<ProjectionType>(sub);
      if (!proj || isPolymorphicType(proj) || recordedDemands.contains(sub))
        return;
      reportUnhookedMint(sub);
    });

  return ty;
}

//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

int nextPolyTypeId() {
  static std::atomic<int> counter{-1};
  return counter.fetch_sub(1, std::memory_order_relaxed);
}

PolyType PolyType::getUnique(MLIRContext* ctx) {
  return PolyType::get(ctx, nextPolyTypeId());
}

Type PolyType::instantiate(InstantiationMap &inst, uint64_t &idCounter) {
  auto self = cast<GenericTypeInterface>(*this);

  // check memo first - if we've already instantiated this PolyType, return it
  if (auto existing = inst.lookup(self))
    return *existing;

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++);
  inst.bind(self, cast<UnificationTypeInterface>(fresh));
  return fresh;
}

Type PolyType::specializeWith(const SpecializationMap &subst) const {
  auto self = cast<GenericTypeInterface>(*this);
  if (auto replacement = subst.lookup(self))
    return *replacement;
  return *this;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <unique> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    uniqueId = nextPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'unique'");
      return Type();
    }
    
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>'");
    return Type();
  }

  return PolyType::get(ctx, uniqueId);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getUniqueId() << ">";
}


//===----------------------------------------------------------------------===//
// InferenceType
//===----------------------------------------------------------------------===//

LogicalResult InferenceType::unify(
  Type other,
  ModuleOp /*module*/,
  UnificationMap &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;
  auto selfKey = cast<UnificationTypeInterface>(self);

  // normalize
  other = applySubstitutionOnce(subst.toTypeMap(), other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto existing = subst.lookup(selfKey)) {
    if (*existing != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << *existing
                            << ", cannot bind to " << other;
      return failure();
    }
    return success();
  }

  // occurs check: forbid T := f(..., T, ...) to avoid cycles
  auto occursIn = [](Type needle, Type haystack) {
    bool hit = false;
    haystack.walk([&](Type t) {
      if (!hit && t == needle) hit = true;
    });
    return hit;
  };

  if (occursIn(self, other)) {
    if (err) err() << "recursive substitution: " << self
                   << " occurs in " << other;
    return failure();
  }

  // bind the variable
  subst.bind(selfKey, other);
  return success();
}


//===----------------------------------------------------------------------===//
// ClaimType
//===----------------------------------------------------------------------===//


// Recover the module that anchors symbol lookups: the operation verification
// reached, or that operation itself when it is the anchoring symbol table.
static ModuleOp getAnchorModule(Operation *anchor) {
  if (!anchor)
    return {};
  if (auto module = dyn_cast<ModuleOp>(anchor))
    return module;
  return anchor->getParentOfType<ModuleOp>();
}

// Entry point for the upstream SymbolUserTypeInterface: symbol-table
// verification invokes this for every claim reachable from an operation, and an
// owning op may call it directly. The module is recovered from the anchoring
// operation and diagnostics are anchored there.
LogicalResult ClaimType::verifySymbolUses(Operation *op,
                                          SymbolTableCollection &symbolTable) const {
  ModuleOp module = getAnchorModule(op);
  if (!module)
    return op->emitError() << "cannot verify " << *this
                           << ": anchor operation is not nested in a module";
  auto err = [&] { return op->emitError(); };

  // verify trait application
  if (failed(getTraitApplication().verifySymbolUses(op, symbolTable)))
    return failure();

  // if there's a proof, verify that it points to a valid symbol
  if (auto proof = getProof())
    if (failed(ProofOp::getProofOpOrUnconditionalImplOp(module, proof, err)))
      return failure();

  return success();
}

Type ClaimType::parse(AsmParser& p) {
  MLIRContext *ctx = p.getContext();

  if (p.parseLess())
    return {};

  TraitApplicationAttr app = mlir::dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app)
    return {};

  FlatSymbolRefAttr proof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proof))
      return {};
  }

  if (p.parseGreater())
    return {};

  return ClaimType::get(ctx, app, proof);
}

void ClaimType::print(AsmPrinter& p) const {
  p << "<";
  getTraitApplication().print(p);
  if (isProven()) {
    p << " by " << getProof();
  }
  p << ">";
}

bool ClaimType::isPolymorphic() const {
  // a !trait.claim<@Trait[Types...]> is polymorphic if any of its type arguments are polymorphic
  return llvm::any_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  });
}

/// Verifies that two recorded proofs for the same obligation are coherent.
///
/// Proof recording keys on the demanded obligation, normalized to its ground
/// form before recording, so every path that reaches one obligation keys and
/// records it identically, and the candidate arrives already normalized at its
/// recording site. A second observation is coherent exactly when its candidate
/// equals the recorded proof literally. Any residual disagreement -- a
/// different proof symbol, or a spelling that does not match after
/// normalization -- is an incoherent proof mapping.
static LogicalResult verifyEquivalentRecordedProof(
    ClaimType unproven,
    ClaimType recorded,
    ClaimType candidate,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (recorded == candidate)
    return success();

  if (err) err() << "inconsistent proof mapping: " << unproven
                 << " is already bound to " << recorded
                 << ", but attempted to bind " << candidate;
  return failure();
}

namespace {

/// What one node of a derivation produced.
///
/// A node's closure is its own binding followed by its children's, which is
/// what replaying it into another evidence map has to write. A node is complete
/// when this derivation computed all of that: a child that exited early on a
/// binding this derivation did not itself write contributes a closure nobody
/// here knows, and neither it nor anything above it can be held.
struct DerivedNode {
  ProofDerivationMemo::Closure closure;
  bool complete = true;

  /// Adds one binding, which a node already carrying it has already written.
  ///
  /// A closure is the SET of bindings replaying it writes, kept in derivation
  /// order for a reader. One obligation can be reached through two of a proof's
  /// subtrees, and whether the second reaching writes it again or exits early on
  /// the first is decided by the order the caller's own map was filled in -- so
  /// keeping a binding once is what makes two derivations of one pair produce
  /// one closure.
  void add(ClaimType unproven, ClaimType proven) {
    if (written.insert(std::make_pair(unproven, proven)).second)
      closure.emplace_back(unproven, proven);
  }

  void addAll(const ProofDerivationMemo::Closure &other) {
    for (auto [unproven, proven] : other)
      add(unproven, proven);
  }

  /// Takes `other` as this node's whole closure, which a node that wrote
  /// nothing of its own does when another derivation already holds it.
  void take(const ProofDerivationMemo::Closure &other) {
    closure.clear();
    written.clear();
    addAll(other);
  }

private:
  llvm::DenseSet<std::pair<ClaimType, ClaimType>> written;
};

/// The nodes one top-level derivation has completed, held until it succeeds.
///
/// Nothing is put in the memo while the derivation that produced it is still
/// running. A node reached through an ancestor's optimistic binding was derived
/// under an assumption that ancestor can still take back, and the map the
/// derivation writes into is rolled back with it; publishing on the outermost
/// success is what keeps the memo from outliving an assumption that failed.
///
/// A node is also what a later node of the same derivation exits early on, so
/// this is indexed by the normalized obligation the early exit looks up as well
/// as by the pair the memo is keyed on.
class DerivationStaging {
public:
  void hold(ClaimType keyUnproven, ClaimType keyProven,
            ClaimType normalizedUnproven, ClaimType normalizedProven,
            const ProofDerivationMemo::Closure &closure) {
    byNormalizedObligation[normalizedUnproven] = held.size();
    held.push_back(
        Held{keyUnproven, keyProven, normalizedUnproven, normalizedProven,
             closure});
  }

  /// What deriving the obligation now bound to `normalizedUnproven` produced,
  /// when this derivation is what bound it.
  const ProofDerivationMemo::Closure *
  lookupDerived(ClaimType normalizedUnproven) const {
    auto it = byNormalizedObligation.find(normalizedUnproven);
    if (it == byNormalizedObligation.end())
      return nullptr;
    return &held[it->second].closure;
  }

  /// Publishes every node into the memo of spelling pairs, and every node's
  /// closure into the record of what deriving its pair produces.
  ///
  /// The record decides for itself what it can keep: an unsettled derivation and
  /// a pair two derivations disagree over are both refused there.
  void publishInto(ProofDerivationMemo &memo) const {
    ProofClosureRecord &closures = memo.getClosures();
    for (const Held &node : held) {
      memo.record(node.keyUnproven, node.keyProven, node.closure);
      countProofDerivationRecorded();
      (void)closures.record(node.normalizedUnproven, node.normalizedProven,
                            node.closure);
    }
  }

private:
  struct Held {
    ClaimType keyUnproven;
    ClaimType keyProven;
    ClaimType normalizedUnproven;
    ClaimType normalizedProven;
    ProofDerivationMemo::Closure closure;
  };

  SmallVector<Held, 8> held;
  llvm::DenseMap<ClaimType, size_t> byNormalizedObligation;
};

} // namespace

static LogicalResult deriveProof(ClaimType unproven, ClaimType proven,
                                 ModuleOp module, EvidenceBindings &bindings,
                                 DemandOrigin origin,
                                 ProofDerivationMemo *memo,
                                 DerivationStaging &staging,
                                 DerivedNode &derived,
                                 llvm::function_ref<InFlightDiagnostic()> err);

/// Writes a closure a derivation already produced into `bindings`.
///
/// Every entry is looked up before it is written, because a differing
/// re-binding is a program the compiler must diagnose rather than an
/// impossibility it may assume: the same obligation can arrive proven by two
/// symbols, and that is the incoherent proof mapping the derivation this
/// replaces reports at its own early exit.
static LogicalResult replayClosure(const ProofDerivationMemo::Closure &closure,
                                   EvidenceBindings &bindings,
                                   llvm::function_ref<InFlightDiagnostic()> err) {
  for (auto [unproven, proven] : closure) {
    if (auto existing = bindings.lookup(unproven)) {
      if (failed(verifyEquivalentRecordedProof(unproven, *existing, proven, err)))
        return failure();
      continue;
    }
    bindings.bind(unproven, proven);
  }
  return success();
}

/// Reports where the closure held for `(unproven, proven)` and the closure
/// deriving that pair again produces differ.
///
/// The held closure is what the replay writes in place of a derivation, so
/// deriving the pair again is the statement being tested. That re-derivation is
/// work the compilation does not do, so it runs as a cross-check and is counted
/// nowhere. A disagreement is a gap in this dialect's own reasoning and never a
/// fault in the program being compiled, so it goes to the census channel.
static void checkHeldClosureAgrees(ClaimType unproven, ClaimType proven,
                                   ModuleOp module, DemandOrigin origin,
                                   const ProofDerivationMemo::Closure &held) {
  if (!DemandLedger::isPostconditionEnabled())
    return;

  auto report = [&](const Twine &derived) {
    llvm::errs() << demandCensusProofDerivationDisagreementPrefix
                 << " unproven=" << unproven << " proven=" << proven
                 << " held=" << held.size() << " derived=" << derived << "\n";
  };

  DemandCrossCheckScope checking;
  EvidenceBindings scratch;
  DerivationStaging staging;
  DerivedNode derived;
  if (failed(deriveProof(unproven, proven, module, scratch, origin,
                         /*memo=*/nullptr, staging, derived, /*err=*/nullptr)))
    return report("<no derivation>");

  if (derived.closure != held)
    report(Twine(derived.closure.size()) + " entries, differing");
}

/// Derives one node of a proof, extending `bindings` with everything the node's
/// own claim and its obligations bind.
static LogicalResult deriveProof(ClaimType unproven, ClaimType proven,
                                 ModuleOp module, EvidenceBindings &bindings,
                                 DemandOrigin origin,
                                 ProofDerivationMemo *memo,
                                 DerivationStaging &staging,
                                 DerivedNode &derived,
                                 llvm::function_ref<InFlightDiagnostic()> err) {
  countProofVerification();

  // the proven side must carry a proof
  if (!proven.isProven()) {
    if (err) err() << "expected proven claim, but found " << proven;
    return failure();
  }

  // the unproven side must be an unproven obligation: it is the recording key,
  // and a proven claim here would trip the bindings.bind precondition
  // downstream. Reject it with a diagnostic instead of reaching that assert.
  if (unproven.isProven()) {
    if (err) err() << "expected unproven obligation, but found proven claim "
                   << unproven;
    return failure();
  }

  // The memo answers for the pair as it arrived, before either side is
  // normalized: that is the pair a caller holds, and answering here is what
  // skips the two normalizations below as well as the derivation under them.
  ClaimType askedUnproven = unproven;
  ClaimType askedProven = proven;
  if (memo) {
    // What the record of per-application closures would have answered here is
    // counted beside the memo's own answer, because a reader that must not
    // derive at all has only the record to serve from and this is where its
    // coverage of the memo's population is measured. The spelling asked with is
    // the one such a reader holds, so it is asked with the same one.
    if (memo->getClosures().lookup(askedUnproven, askedProven))
      countProofClosureAnswered();
    else
      countProofClosureUnanswered();

    if (const auto *closure = memo->lookup(askedUnproven, askedProven)) {
      countProofDerivationMemoHit();
      checkHeldClosureAgrees(askedUnproven, askedProven, module, origin,
                             *closure);
      derived.take(*closure);
      return replayClosure(*closure, bindings, err);
    }
    countProofDerivationMemoMiss();
  }

  // Normalize both the demanded obligation (the recording key) and the proven
  // value before recording. Requirement obligations arrive at their stamped
  // declaration projections; resolving those ground redexes means every path
  // that reaches the same obligation keys it identically and records the same
  // proven spelling, so a second observation matches the first literally
  // instead of reconciling two equivalent spellings.
  {
    Type normalizedProven =
        resolveGroundProjectionsByLookup(proven, module, origin);
    proven = cast<ClaimType>(normalizedProven);

    Type normalizedUnproven =
        resolveGroundProjectionsByLookup(unproven, module, origin);
    unproven = cast<ClaimType>(normalizedUnproven);
  }

  // early exit if we've already recorded this obligation. The same proof may
  // be observed through multiple equivalent claim spellings, so validate proof
  // coherence instead of requiring syntactic claim equality.
  if (auto existing = bindings.lookup(unproven)) {
    countProofVerificationEarlyExit();
    if (failed(verifyEquivalentRecordedProof(unproven, *existing, proven, err)))
      return failure();
    // What this node would have written is already written. When this
    // derivation is what wrote it, that closure is in hand and stands for this
    // node's; when something before this derivation wrote it, the record of
    // what deriving the application produces is where the closure is, because
    // it is kept per application rather than per derivation. Only where neither
    // has it does the node go undescribed, and nothing containing it can be
    // held either.
    if (const auto *closure = staging.lookupDerived(unproven)) {
      derived.take(*closure);
    } else if (const auto *recorded =
                   memo ? memo->getClosures().lookup(unproven, proven)
                        : nullptr) {
      countProofDerivationRecovered();
      derived.take(*recorded);
    } else {
      derived.complete = false;
    }
    return success();
  }

  // look up the trait and its requirements using the unproven claim
  auto trait = unproven.getTraitApplication().getTrait(module, err);
  if (failed(trait)) return failure();

  // inspect the proof symbol on the proven side
  auto symOp = ProofOp::getProofOpOrUnconditionalImplOp(module, proven.getProof(), err);
  if (failed(symOp)) return failure();

  // if it's an impl op, check that the trait has no requirements
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (trait->hasRequirements()) {
      if (err) err() << "impl provides no subproof for trait requirements";
      return failure();
    }

    // success: bind the whole claim so that later normalization keeps the proof
    bindings.bind(unproven, proven);
    // A leaf: the binding it wrote is the whole of what deriving it produces.
    derived.add(unproven, proven);
    staging.hold(askedUnproven, askedProven, unproven, proven, derived.closure);
    return success();
  }

  // otherwise the symbol must be a ProofOp
  auto proof = dyn_cast<ProofOp>(*symOp);

  // check that the proof's claim can specialize to match proven. Both name this
  // one committed proof, so reducing ground redexes its spellings mint is a
  // computation over the proof's own facts, not a spelling comparison -- the
  // recorder is a ratified minting point that reads module facts (the real
  // module drives the ground-redex resolution inside unification).
  if (failed(buildSpecialization(proof.getProvenClaim(), proven, module, err)))
    return failure();

  // Use the proof's concrete claim (projections resolved) rather than the
  // unproven claim (which may still contain projections). Example:
  // unproven = @D[A[i32]::Out, A[f32]::Out], concrete = @D[i64, i64].
  // An impl like @D[poly, poly] can unify with @D[i64, i64] but not with
  // @D[A[i32]::Out, A[f32]::Out] (the two projections are structurally
  // different even though both resolve to i64).
  auto obligations = proof.getImpl().specializeObligationsAsClaimsFor(
      proof.getProvenClaim().asUnproven(), origin, err);
  if (failed(obligations)) return failure();

  // get the subproof claims (also checks arity against obligations)
  auto subproofs = proof.verifyAndGetSubproofClaims(origin, err);
  if (failed(subproofs)) return failure();

  // Bind optimistically before recursing so that coinductive self-references
  // (where an obligation resolves back to the same claim) hit the early exit
  // at the top of this function instead of diverging.
  bindings.bind(unproven, proven);
  derived.add(unproven, proven);

  // recurse over obligations
  for (auto [ob, sub] : llvm::zip(*obligations, *subproofs)) {
    DerivedNode child;
    if (failed(deriveProof(ob, sub, module, bindings, origin, memo, staging,
                           child, err))) {
      bindings.erase(unproven);
      return failure();
    }
    derived.complete &= child.complete;
    if (derived.complete)
      derived.addAll(child.closure);
  }

  if (derived.complete) {
    staging.hold(askedUnproven, askedProven, unproven, proven, derived.closure);
  } else {
    derived.take({});
    countProofDerivationNotRecorded();
  }
  return success();
}

LogicalResult verifyAndRecordProof(
    ClaimType unproven,
    ClaimType proven,
    ModuleOp module,
    EvidenceBindings &bindings,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  DerivationStaging staging;
  DerivedNode derived;
  if (failed(deriveProof(unproven, proven, module, bindings, origin, memo,
                         staging, derived, err)))
    return failure();

  // Everything this derivation completed goes into the memo together, now that
  // the derivation it was completed under has returned success.
  if (memo)
    staging.publishInto(*memo);
  return success();
}

/// Walk `root` and record substitution entries for every proven claim
/// found within it.  This maps the unproven claim to the proven claim.
/// These entries are used during unification so that
/// `applySubstitutionToFixedPoint` can normalize claims before
/// per-type unification dispatch.
LogicalResult recordProofBindingsIn(
    Type root,
    ModuleOp module,
    EvidenceBindings &bindings,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  LogicalResult status = success();

  root.walk([&](Type node) {
    if (status.failed()) return;

    if (auto claim = dyn_cast<ClaimType>(node)) {
      if (claim.isProven())
        if (failed(verifyAndRecordProof(claim.asUnproven(), claim, module,
                                        bindings, origin, memo, err)))
          status = failure();
    }
  });

  return status;
}

void ClaimType::getProjections(
    ModuleOp module,
    SmallVectorImpl<ClaimType>& result) {
  // identity
  result.push_back(*this);

  // trait requirements
  auto trait = getTraitApplication().getTraitOrAbort(module, "ClaimType::getProjections: couldn't find trait");
  auto specRequirements = trait.specializeRequirementsAsClaimsFor(*this);
  if (succeeded(specRequirements))
    result.append(*specRequirements);

  // proven impl assumptions
  if (isProven()) {
    if (auto proof = SymbolTable::lookupNearestSymbolFrom<ProofOp>(module, getProof())) {
      auto specAssumptions = proof.getImpl().specializeAssumptionsAsClaimsFor(*this);
      if (succeeded(specAssumptions))
        result.append(*specAssumptions);
    }
  }
}

static LogicalResult unifyTypeRange(ArrayRef<Type> formalTypes,
                                    ArrayRef<Type> actualTypes,
                                    ModuleOp module,
                                    UnificationMap &subst,
                                    llvm::function_ref<InFlightDiagnostic()> err);

LogicalResult ClaimType::unify(
    Type other,
    ModuleOp module,
    UnificationMap& subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize formal first
  Type formalNormTy = applySubstitutionOnce(subst.toTypeMap(), *this);
  ClaimType formal = mlir::dyn_cast<ClaimType>(formalNormTy);

  // if formal is no longer a ClaimType, delegate to generic path
  if (!formal)
    return trait::unify(formalNormTy, other, module, subst, err);

  // normalize actual second
  Type normActualTy = applySubstitutionOnce(subst.toTypeMap(), other);
  ClaimType actual = mlir::dyn_cast<ClaimType>(normActualTy);

  // if actual isn't a claim, it's an immediate mismatch
  if (!actual) {
    if (err) {
      err() << "expected !trait.claim, but found " << normActualTy;
    }
    return failure();
  }

  // do claim-specific checks below

  auto formalApp = formal.getTraitApplication();
  auto actualApp = actual.getTraitApplication();

  // same trait?
  if (formalApp.getTraitName() != actualApp.getTraitName()) {
    if (err) err() << "trait mismatch: expected " << formalApp.getTraitName()
                   << ", but found " << actualApp.getTraitName();
    return failure();
  }

  // check proofs
  auto formalProof = formal.getProof();
  auto actualProof = actual.getProof();
  if (formalProof && actualProof && formalProof != actualProof) {
    if (err) err() << "proof mismatch: expected " << formalProof
                   << ", but found " << actualProof;
    return failure();
  }
  if (formalProof && !actualProof) {
    if (err) err() << "cannot unify proven claim with unproven claim";
    return failure();
  }

  return unifyTypeRange(formalApp.getTypeArgs(), actualApp.getTypeArgs(), module,
                        subst, err);
}


//===----------------------------------------------------------------------===//
// ProjectionType
//===----------------------------------------------------------------------===//

bool ProjectionType::isPolymorphic() const {
  return llvm::any_of(getTraitApplication().getTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  }) || llvm::any_of(getAssocTypeArgs(), [](Type ty) {
    return mlir::trait::isPolymorphicType(ty);
  });
}

Type ProjectionType::parse(AsmParser &p) {
  MLIRContext *ctx = p.getContext();

  if (p.parseLess())
    return {};

  // parse @Trait[Types...]
  TraitApplicationAttr app = mlir::dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app)
    return {};

  if (p.parseComma())
    return {};

  // parse "AssocName"
  StringAttr assocName;
  if (p.parseAttribute(assocName))
    return {};

  // parse optional , [gat_args...]
  SmallVector<Type> assocTypeArgs;
  if (succeeded(p.parseOptionalComma())) {
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
          Type ty;
          if (p.parseType(ty)) return failure();
          assocTypeArgs.push_back(ty);
          return success();
        })))
      return {};
  }

  if (p.parseGreater())
    return {};

  return ProjectionType::get(ctx, app, assocName, assocTypeArgs);
}

void ProjectionType::print(AsmPrinter &p) const {
  p << "<";
  getTraitApplication().print(p);
  p << ", " << getAssocName();
  if (!getAssocTypeArgs().empty()) {
    p << ", [";
    llvm::interleaveComma(getAssocTypeArgs(), p, [&](Type ty) {
      p.printType(ty);
    });
    p << "]";
  }
  p << ">";
}

// Entry point for the upstream SymbolUserTypeInterface. A projection carries the
// same trait application as its claim, so verification delegates to that claim.
LogicalResult ProjectionType::verifySymbolUses(Operation *op,
                                               SymbolTableCollection &symbolTable) const {
  return asClaim().verifySymbolUses(op, symbolTable);
}

//===----------------------------------------------------------------------===//
// unify
//===----------------------------------------------------------------------===//

static LogicalResult unifyTypeRange(ArrayRef<Type> formalTypes,
                                    ArrayRef<Type> actualTypes,
                                    ModuleOp module,
                                    UnificationMap &subst,
                                    llvm::function_ref<InFlightDiagnostic()> err) {
  if (formalTypes.size() != actualTypes.size()) {
    if (err)
      err() << "type arity mismatch: expected " << formalTypes.size()
            << " type arguments, but found " << actualTypes.size();
    return failure();
  }

  for (auto [formal, actual] : llvm::zip(formalTypes, actualTypes)) {
    if (failed(trait::unify(formal, actual, module, subst, err)))
      return failure();
  }

  return success();
}

/// Collect exactly the immediate child Types and Attributes of `ty`. If `ty` has no sub‐elements,
/// returns empty vectors.
static std::pair<SmallVector<Type, 4>, SmallVector<Attribute, 4>> getImmediateSubElements(Type ty) {
  SmallVector<Type, 4> childTypes;
  SmallVector<Attribute, 4> childAttrs;
  ty.walkImmediateSubElements(
      /*walkAttrsFn=*/[&](Attribute subAttr) {
        childAttrs.push_back(subAttr);
      },
      /*walkTypesFn=*/[&](Type subTy) {
        childTypes.push_back(subTy);
      });
  return std::pair(childTypes, childAttrs);
}

static LogicalResult unifyStructurally(Type formal,
                                       Type actual,
                                       ModuleOp module,
                                       UnificationMap &subst,
                                       llvm::function_ref<InFlightDiagnostic()> err) {
  if (formal == actual) return success();

  // check for same
  // 1. type constructor
  // 2. subelement arity
  // 3. attribute equality
  // and then recurse on children, if there are any
  auto [formalSubTys, formalSubAttrs] = getImmediateSubElements(formal);
  auto [actualSubTys, actualSubAttrs] = getImmediateSubElements(actual);

  bool formalHasSubs = !formalSubTys.empty() || !formalSubAttrs.empty();
  bool actualHasSubs = !actualSubTys.empty() || !actualSubAttrs.empty();

  // if neither side is decomposable, they're unequal leaves -> mismatch
  // if only one side is decomposable, constructors differ in structure -> mismatch
  if (!formalHasSubs || !actualHasSubs) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // the constructor and arity of subelements of both types must match before recursing
  if (formal.getTypeID() != actual.getTypeID() ||
      formalSubTys.size() != actualSubTys.size() ||
      formalSubAttrs.size() != actualSubAttrs.size()) {
    if (err) err() << "type mismatch: expected " << formal
                   << " but found " << actual;
    return failure();
  }

  // The attributes of both types must match exactly before recursing on
  // child types. XXX: this treats attributes as opaque, so it will not find
  // and unify types stored inside type-bearing attributes.
  for (auto [f, a] : llvm::zip(formalSubAttrs, actualSubAttrs)) {
    if (f != a) {
      if (err) err() << "attribute mismatch: expected " << f
                     << " but found " << a;
      return failure();
    }
  }

  // Recurse on each sub type pair
  for (auto [f, a] : llvm::zip(formalSubTys, actualSubTys)) {
    if (failed(unify(f, a, module, subst, err)))
      return failure();
  }

  return success();
}

/// Records a monomorphic projection the unifier let stand: it equated the two
/// sides, or bound a variable to the projection, without asking any impl what
/// the projection resolves to. These sit apart from the lookup's miss arms --
/// nothing here consulted the lookup at all.
///
/// The test is a root test on purpose. A projection nested inside two
/// aggregates the unifier found literally equal goes unobserved, because
/// walking every equality would put a type traversal on the unifier's hottest
/// path; such a projection is observed here anyway whenever the two sides are
/// not already equal, since the structural recursion then brings it to this
/// entry on its own.
///
/// The unifier's signature names no caller, so `module` classifies the demand:
/// the module-free comparator is what a verifier holding no module reaches,
/// while a caller carrying one is the stage or a committed-fact match inside a
/// verifier, which the stage's suspension brackets cover.
static void observeUnifierAcceptance(Type ty, ModuleOp module) {
  // Cheapest test first: with both switches off there is no observer at all,
  // which is what keeps this off the unifier's equality path.
  if (!DemandLedger::areObservationsEnabled())
    return;
  if (!isa<ProjectionType>(ty) || isPolymorphicType(ty))
    return;
  recordUnifierAcceptance(ty, module ? DemandOrigin::Unification
                                     : DemandOrigin::ModuleFreeComparison);
}

/// Unify a projection type with another type. Two entries reach here: a
/// module-free comparison (a verifier passes no module) and a module-capable
/// resolution (a pass or a committed-fact substitution build passes the
/// module).
///
///  - Projection vs projection: require the same symbolic projection head, then
///    recurse through trait application and associated-type arguments. This
///    allows nested projections to justify equivalent spellings.
///  - Projection vs a free inference variable it does not occur in: bind the
///    variable to the projection.
///  - Projection vs any other type: under the module-capable entry, resolve the
///    projection if a unique impl binds it and unify the result; under the
///    module-free entry, an unresolved crossing is a strict mismatch and is
///    rejected. Only the module-capable entry, on an irreducible crossing no
///    committed fact determines, tolerates it (see the residual note below).
LogicalResult ProjectionType::unify(
    Type other,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // Projection trait applications carry type arguments inside an attribute,
  // so structural attribute equality is too strict. Compare the symbolic
  // projection head, then recurse through the type arguments.
  if (auto otherProj = mlir::dyn_cast<ProjectionType>(other)) {
    auto formalApp = getTraitApplication();
    auto actualApp = otherProj.getTraitApplication();
    if (formalApp.getTraitName() != actualApp.getTraitName() ||
        getAssocName() != otherProj.getAssocName()) {
      if (err)
        err() << "projection mismatch: expected " << *this << " but found "
              << otherProj;
      return failure();
    }

    observeUnifierAcceptance(*this, module);
    if (failed(unifyTypeRange(formalApp.getTypeArgs(), actualApp.getTypeArgs(),
                              module, subst, err)))
      return failure();
    return unifyTypeRange(getAssocTypeArgs(), otherProj.getAssocTypeArgs(),
                          module, subst, err);
  }

  // projection vs non-projection.
  //
  // A projection is an opaque type function whose value is fixed only by claim
  // evidence, not by unification. Against a free inference variable that does
  // not occur inside this projection there is a sound choice -- bind the
  // variable to the projection -- so delegate to the variable's own unifier. A
  // variable that DOES occur inside is NOT bound here: a projection is a
  // resolvable function, so `V = proj<...V...>` is a forwarding equation (V is a
  // fixpoint of the resolution), not an infinite type, and must not trip the
  // variable unifier's occurs check. It falls to the module-free rejection or
  // the module-capable resolution below rather than binding.
  if (auto otherVar = mlir::dyn_cast<InferenceType>(other)) {
    bool occurs = false;
    Type(*this).walk([&](Type t) {
      if (t == other) occurs = true;
    });
    if (!occurs) {
      observeUnifierAcceptance(*this, module);
      return otherVar.unify(*this, module, subst, err);
    }
  }

  // A projection all of whose arguments are concrete and whose trait application
  // a unique module-visible impl binds has one determined resolution. Binding a
  // variable mid-solve mints such ground redexes (binding V:=i64 turns
  // proj<@Prod[V]> into the ground proj<@Prod[i64]>), so a caller carrying a
  // module -- a pass, or a committed-fact substitution build -- resolves them
  // here and unifies the resolved type against `other`, catching a real mismatch
  // against the resolved concrete spelling. A verifier compares spellings with
  // no module (the module-free comparator); an equality check performs no module
  // lookup, so this step is skipped and an unresolved crossing is a strict
  // mismatch below.
  // The arms a ground base declined on, kept so an accept below can be classed.
  unsigned groundMissReasons = 0;
  if (isMonomorphicType(*this) && module) {
    Type resolved = resolveGroundProjectionsByLookup(
        *this, module, DemandOrigin::Unification, &groundMissReasons);
    if (resolved != Type(*this))
      return trait::unify(resolved, other, module, subst, err);
  }

  // The projection did not resolve and meets a rigid non-projection type. The
  // module-free comparator (a verifier) holds no evidence for the equality:
  // spellings must be identical after substitution, so reject the crossing.
  // Counted rather than recorded: this is a demand raised where there is no
  // module to read facts from, which is outside any stage population by
  // construction, and it is the exact crossing the tolerance below accepts when
  // a module is in hand -- the two counts read against each other.
  //
  // No test reaches this arm. The module-free comparator has one caller, the
  // cast op's input/result consistency check, which replaces every projection
  // over the claim's own application with a variable; what is left to reach
  // here is a projection over some other application meeting a different type
  // across the two, which is an ill-formed cast no row spells.
  if (!module) {
    countModuleFreeProjectionRejection();
    if (err)
      err() << "projection mismatch: expected " << *this << " but found "
            << other;
    return failure();
  }

  // XXX TODO(residual tolerance): the module-capable entry reached an
  // irreducible crossing that no committed fact determines here and accepts it
  // without a binding. This entry runs both at pass time and inside verifiers on
  // committed-fact matches (witness, proof, derive, and per-candidate
  // enumeration), so the tolerance is not pass-exclusive. Three classes survive
  // here, each with its own end condition:
  //   - Generator-pending grounds: a concrete base whose impl a downstream
  //     generator has not yet synthesized (the prelude's Convergence machinery).
  //     Empty on the stage by construction -- generation precedes the lowering
  //     that runs the ground lookup, so a base reaching it there has every impl
  //     it will get -- and nonzero on the comparisons a verifier raises against
  //     committed facts before the stage begins, which generation has not
  //     reached. Acceptance ends when that second population reaches zero. The
  //     census reads the two apart in its
  //     residual-tolerance-accepts-generator-pending and
  //     residual-tolerance-accepts-before-the-stage-generator-pending columns.
  //   - Hypothesis-resolvable projections: a still-symbolic base resolvable only
  //     through a frame hypothesis (a where-clause equality). The witnessable
  //     part ends when the crossing is witnessed at its cast site; the
  //     un-witnessable subclass (a hypothesis with no recordable provider) is
  //     PERMANENT, so acceptance ends here only if committed builds provably
  //     never receive that subclass.
  //   - Ground multi-candidate crossings: a ground base several impls bind.
  //     Resolution is premise-partitioned and belongs to the resolver alone,
  //     never to this comparison.
  if (!isCrossChecking()) {
    ++numResidualToleranceAccepts;
    // Split the accept by the tolerance site's taxonomy so law 5's zero clause
    // can be read against one class. A still-symbolic base never reached the
    // ground lookup (the block above skips a non-monomorphic type), so it is the
    // hypothesis class by itself. A ground base declined on the lookup's arms,
    // and a single headline arm names its class; several arms at once, or an arm
    // that is neither headline case, is neither generator-pending nor
    // multi-candidate and goes to the mixed-or-other class.
    if (!isMonomorphicType(*this)) {
      ++numResidualToleranceAcceptsHypothesis;
    } else if (groundMissReasons ==
               (1u << unsigned(LookupMissReason::NoCandidateImpl))) {
      ++numResidualToleranceAcceptsGeneratorPending;
    } else if (groundMissReasons ==
               (1u << unsigned(LookupMissReason::MultipleCandidateImpls))) {
      ++numResidualToleranceAcceptsMultiCandidate;
    } else {
      ++numResidualToleranceAcceptsMixedOrOther;
    }
  }
  return success();
}

/// Attempt to unify `formal` with `actual`, extending `subst` with any
/// new bindings that make them equal under substitution.
///
/// Both sides are first normalized by applying `subst` to a fixed point.
/// After that we check for trivial equality and then choose how to drive
/// unification:
///
/// Priority of unifiers:
///  1. **Formal first** — If the formal side implements
///     `UnificationTypeInterface`, we let it drive unification. This gives
///     formal-side types (inference variables, projections, claims) first
///     refusal to decide how to handle the match.
///  2. **Actual second** — If the actual side implements
///     `UnificationTypeInterface`, we let it drive. This handles the
///     symmetric case (e.g., inference variable on the actual side).
///  3. **Structural fallback** — Otherwise we fall back to generic
///     shape-by-shape unification for non-unifiable types.
///
/// Returns success if the two types can be made equal under an extended `subst`.
/// On failure, nothing is recorded and `err` (if provided) will be invoked to
/// emit a diagnostic.
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // normalize both types by applying the current substitution
  formal = applySubstitutionToFixedPoint(subst.toTypeMap(), formal);
  actual = applySubstitutionToFixedPoint(subst.toTypeMap(), actual);

  // if the normalized types are equal, unification succeeds
  if (formal == actual) {
    observeUnifierAcceptance(formal, module);
    return success();
  }

  // formal-side unifier takes priority
  if (auto formalUnifier = dyn_cast<UnificationTypeInterface>(formal))
    return formalUnifier.unify(actual, module, subst, err);

  // actual-side unifier
  if (auto actualUnifier = dyn_cast<UnificationTypeInterface>(actual))
    return actualUnifier.unify(formal, module, subst, err);

  // structural fallback
  return unifyStructurally(formal, actual, module, subst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst) {
  auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
  return unify(formal, actual, module, subst, errFn);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  UnificationMap discardedSubst;
  return unify(formal, actual, module, discardedSubst, err);
}

LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module) {
  UnificationMap discardedSubst;
  return unify(formal, actual, module, discardedSubst);
}


//===----------------------------------------------------------------------===//
// instantiate
//===----------------------------------------------------------------------===//

Type instantiate(Type root, InstantiationMap &inst, uint64_t &idCounter) {
  AttrTypeReplacer r;
  r.addReplacement([&](Type t) -> std::optional<Type> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      return generic.instantiate(inst, idCounter);
    }
    return std::nullopt;
  });

  // this walks into types nested inside attributes (e.g., trait applications)
  // and replaces all GenericTypeInterface types according to (and extending) inst
  return r.replace(root);
}


FailureOr<SpecializationMap> buildSpecialization(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // instantiate generics on both sides with the same instantiation map
  InstantiationMap genToInfer;
  uint64_t idCounter = 0;
  Type iformal = instantiate(formal, genToInfer, idCounter);
  Type iactual = instantiate(actual, genToInfer, idCounter);

  // get the inverse instantiation map as well
  auto inferToGen = invertSubstitution(genToInfer.toTypeMap(), err);
  if (failed(inferToGen)) return failure();

  // unify the instantiated formal and actual types
  UnificationMap inferToType;
  if (failed(unify(iformal, iactual, module, inferToType, err)))
    return failure();

  // compose (gen -> infer) o (infer -> type)
  auto composed = composeSubstitutions(genToInfer.toTypeMap(), inferToType.toTypeMap(), err);
  if (failed(composed)) return failure();

  // compose again with inferToGen to map any remaining unsolved
  // inference variables originating from actual back to their
  // original generics
  auto result = composeSubstitutions(*composed, *inferToGen, err);
  if (failed(result)) return failure();

  normalizeSubstitutionInPlace(*result);
  return SpecializationMap::fromTypeMap(*result);
}

} // end mlir::trait
