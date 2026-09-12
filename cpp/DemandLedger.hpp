// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>

namespace mlir::trait {

/// The lookup failure used by preparation and projection compatibility.
enum class LookupMissReason : uint8_t {
  /// The projection's trait application names no trait in the module.
  TraitSymbolNotFound,
  /// No impl in the module binds the projection's trait application.
  NoCandidateImpl,
  /// Two or more impls bind it, so selection is a premise partition the
  /// read-only lookup does not perform.
  MultipleCandidateImpls,
  /// The unique impl has no usable binding for the projected associated type.
  AssociatedTypeBindingFailed,
  /// The unique impl's self claim does not specialize to the demanded claim.
  SelfClaimSubstitutionFailed,
};

/// Distinguishes a stage obligation from verification work.
enum class DemandOrigin : uint8_t {
  /// The module-capable replacer that stamps a specialized monomorph.
  MonomorphStampOut,
  /// The obligation recorder normalizing both sides before recording a proof.
  ProofRecording,
  /// Matching a declaration against a use, reducing a ground projection either
  /// side spells so the two meet at one grade.
  DeclarationMatch,
  /// A call site specializing its callee at pass time.
  CallSiteSpecialization,
  /// A read of the recorded facts falling back to the module's impls for a
  /// projection impl selection has settled nothing for.
  RecordedFactRead,
  /// A call op's verifier comparing its formal and actual signatures.
  CallSignatureVerification,
  /// A proof op's verifier walking the proof structure it declares.
  ProofVerification,
};

/// Whether this origin can add a pending stage obligation.
///
/// A verifier runs outside any stage and may run on a worker thread, so its
/// demand never enters the stage preparation queue.
inline bool recordsToLedger(DemandOrigin origin) {
  switch (origin) {
  case DemandOrigin::MonomorphStampOut:
  case DemandOrigin::ProofRecording:
  case DemandOrigin::DeclarationMatch:
  case DemandOrigin::CallSiteSpecialization:
  case DemandOrigin::RecordedFactRead:
    return true;
  case DemandOrigin::CallSignatureVerification:
  case DemandOrigin::ProofVerification:
    return false;
  }
  return false;
}

enum class DemandSkip : uint8_t {
  /// Nothing: every op the module holds, trait and impl headers included.
  Nothing,
  /// Trait, impl and proof ops and their whole subtrees. What they spell -- a
  /// trait's own requirements, an impl's assumptions, the projections in either
  /// header -- stands for good and is nothing to serve.
  Infrastructure,
  /// Those and a still-polymorphic template function besides. A template's
  /// spelling is resolved when it is cloned for a concrete instance.
  Foreign,
};

/// The demands `root` spells: its monomorphic projections, and the unproven
/// monomorphic claims something is still meant to prove.
///
/// Result and block-argument types are what the stage's own leftover sweeps
/// walk, and an operand type is its producer's result type, so between them they
/// cover the types preparation must serve. `inAttributes` adds the
/// projections an op carries as attribute data instead, which no leftover sweep
/// reads and a pattern that rewrites an op's whole dictionary does.
///
/// The two sides carry their own skip because they answer different questions;
/// each caller states which it is asking.
/// The result is in the order the walk found them, so a caller putting these to
/// impl selection asks in an order one run repeats.
///
/// `origins`, when given, receives where each demand was first found spelled.
/// A caller that puts one to impl selection names that place while it does, so
/// what the ask raises underneath is attributed to the op carrying the spelling
/// rather than to the module.
llvm::SetVector<Type> demandsSpelledIn(ModuleOp module, bool inAttributes,
                                       DemandSkip projections,
                                       DemandSkip claims,
                                       DenseMap<Type, Location> *origins = nullptr);

/// Pending obligations and diagnostic frames owned by one instantiation.
class DemandLedger {
public:
  void record(Type demand, unsigned missArms);
  ArrayRef<Type> getDrainableDemands() const { return demands.getArrayRef(); }
  unsigned getDrainableArms(Type demand) const { return arms.lookup(demand); }

  void pushFrame(Type demand);
  void pushFrame(Location origin);
  void popFrame();
  size_t getFrameDepth() const { return frames.size(); }
  std::optional<Location> getInnermostFrameOrigin() const {
    if (frames.empty())
      return std::nullopt;
    return frames.back();
  }

  /// Reports every drained key the stage neither served nor left standing, and
  /// fails when there is one.
  ///
  /// A round takes a key off the drain when nothing it could ask later would
  /// settle it differently: impl selection resolved it, or refused it on the
  /// arm no later resolution overturns and left its spelling for the stage's
  /// leftover walks to report. A key that is neither served nor still spelled
  /// was taken and dropped, and nothing downstream would say so.
  LogicalResult checkDrainedKeysSettled(ModuleOp module,
                                        const DenseSet<Type> &drained,
                                        const DenseSet<Type> &served) const;

  /// Reports every drainable demand still spelled at stage exit that no round
  /// served, and fails when there is one.
  ///
  /// checkDrainedKeysSettled covers the demands a round took off the drain and
  /// then lost; this covers the complementary case, a drainable demand no round
  /// settled -- one deferred to a round that never came, or one whose surviving
  /// spelling the stage's leftover-op walks do not reach because it lives on a
  /// block argument or in an attribute.
  LogicalResult checkStandingDemandsServed(
      ModuleOp module, const DenseSet<Type> &served) const;

private:
  llvm::SetVector<Type> demands;
  llvm::DenseMap<Type, unsigned> arms;
  SmallVector<Location, 8> frames;
};

std::optional<Location> currentDemandAnchor();

/// Installs one stage demand context on this thread.
class DemandLedgerScope {
public:
  explicit DemandLedgerScope(DemandLedger &ledger);
  ~DemandLedgerScope();

  DemandLedgerScope(const DemandLedgerScope &) = delete;
  DemandLedgerScope &operator=(const DemandLedgerScope &) = delete;

private:
  DemandLedger *previous;
};

/// Excludes verification from the active stage demand context.
class DemandRecordingSuspension {
public:
  DemandRecordingSuspension();
  ~DemandRecordingSuspension();

  DemandRecordingSuspension(const DemandRecordingSuspension &) = delete;
  DemandRecordingSuspension &operator=(const DemandRecordingSuspension &) = delete;

private:
  DemandLedger *previous;
};

/// Carries the enclosing demand and its source through nested resolution.
class DemandFrame {
public:
  explicit DemandFrame(Type demand);
  explicit DemandFrame(Location origin);
  ~DemandFrame();

  DemandFrame(const DemandFrame &) = delete;
  DemandFrame &operator=(const DemandFrame &) = delete;

private:
  DemandLedger *ledger;
};

/// Distinguishes a root lookup from a candidate probe.
class LookupProbeScope {
public:
  LookupProbeScope();
  ~LookupProbeScope();

  LookupProbeScope(const LookupProbeScope &) = delete;
  LookupProbeScope &operator=(const LookupProbeScope &) = delete;

  /// How many lookup callbacks were already running when this one started.
  unsigned getEnclosingDepth() const { return enclosingDepth; }

private:
  unsigned enclosingDepth;
};

/// Excludes obligations of discardable candidates from pending work.
class SpeculationScope {
public:
  SpeculationScope();
  ~SpeculationScope();

  SpeculationScope(const SpeculationScope &) = delete;
  SpeculationScope &operator=(const SpeculationScope &) = delete;

private:
  bool previous;
};

/// Queue only real stage demands; probes, speculation and verifiers stay local.
void recordLookupMiss(Type demand, LookupMissReason reason, DemandOrigin origin,
                      unsigned enclosingDepth);
void recordResolverProjectionMiss(Type demand);
void recordReadOnlyResolverMiss(Type demand);

} // namespace mlir::trait
