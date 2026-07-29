// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "DemandLedger.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/Statistic.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdlib>

#define DEBUG_TYPE "trait-demand"

// The stage population, and the parts of it that a drain would have to leave
// alone. A verifier's demand is counted here and nowhere else, so the two
// populations stay separable under -stats.
STATISTIC(numDemandObservations,
          "demands observed inside the monomorphization stage");
STATISTIC(numSpeculativeDemandObservations,
          "demands observed while the resolver partitioned candidates");
STATISTIC(numProbeInternalDemandObservations,
          "demands observed inside a candidate probe");
STATISTIC(numUnattributedDemandObservations,
          "demands observed with no enclosing demand frame");
STATISTIC(numVerifierLookupMisses,
          "ground projection lookup misses raised by a verifier");
STATISTIC(numResolverProjectionMisses,
          "projections the resolver's engine failed and its caller left spelled");
STATISTIC(numUnifierAcceptances,
          "monomorphic projections the unifier accepted without resolving");
STATISTIC(numWithheldCallClaims,
          "calls whose claim withheld the license to consult module facts");
STATISTIC(numObligationNormalizations,
          "obligations normalized against one impl's own local bindings");
STATISTIC(numObligationNormalizationMisses,
          "monomorphic projections obligation normalization left standing");
STATISTIC(numVerifierObligationNormalizations,
          "projection normalizations run inside a cast verifier");
STATISTIC(numModuleFreeProjectionRejections,
          "projection crossings the module-free comparator rejected");
STATISTIC(numModulelessRegionProjections,
          "monomorphic projections stamped into a region specialized with no module");
STATISTIC(numUnhookedMints,
          "monomorphic projections that survived the lookup unrecorded");
STATISTIC(numServedDrainableKeys,
          "drainable keys the stage went on to serve");

namespace mlir::trait {

// The census partitions the population by engine and by miss arm, so its
// tables are sized by these counts and would silently drop a column if an
// enumerator were added without them.
static_assert(static_cast<unsigned>(DemandEngine::WithheldCallClaim) + 1 ==
                  numDemandEngines,
              "numDemandEngines must count every DemandEngine");
static_assert(static_cast<unsigned>(
                  LookupMissReason::SelfClaimSubstitutionFailed) +
                      1 ==
                  numLookupMissReasons,
              "numLookupMissReasons must count every LookupMissReason");

namespace {

/// The sink, the lookup's re-entry depth and the speculation flag are ambient
/// because the components that record reach no resolver: the lookup's signature
/// names a type and a module, and the unifier's names neither.
thread_local DemandLedger *ambientLedger = nullptr;
thread_local unsigned ambientLookupDepth = 0;
thread_local bool ambientSpeculating = false;

const char *engineName(DemandEngine engine) {
  switch (engine) {
  case DemandEngine::GroundProjectionLookup: return "lookup-miss";
  case DemandEngine::UnifierAcceptance: return "unifier-acceptance";
  case DemandEngine::ResolverProjectionEngine: return "resolver-engine-miss";
  case DemandEngine::ObligationNormalization: return "obligation-normalization";
  case DemandEngine::WithheldCallClaim: return "withheld-call-claim";
  }
  return "unknown";
}

const char *missReasonName(LookupMissReason reason) {
  switch (reason) {
  case LookupMissReason::TraitSymbolNotFound: return "trait-symbol-not-found";
  case LookupMissReason::NoCandidateImpl: return "no-candidate-impl";
  case LookupMissReason::MultipleCandidateImpls: return "multiple-candidate-impls";
  case LookupMissReason::AssociatedTypeBindingFailed: return "assoc-binding-failed";
  case LookupMissReason::SelfClaimSubstitutionFailed: return "self-claim-substitution-failed";
  }
  return "unknown";
}

/// The three classes one observation's flags fall into. They partition the
/// observations, which is what makes the census counts add up: an observation
/// that is both speculative and probe-internal is a probe, since the probe is
/// the nearer reason its demand may never be raised again.
enum FlagClass { RealObservation = 0, SpeculativeObservation, ProbeObservation };

FlagClass classOf(DemandFlags flags) {
  if (flags.probeInternal) return ProbeObservation;
  if (flags.speculative) return SpeculativeObservation;
  return RealObservation;
}

const char *flagClassName(FlagClass cls) {
  switch (cls) {
  case RealObservation: return "real";
  case SpeculativeObservation: return "speculative";
  case ProbeObservation: return "probe-internal";
  }
  return "unknown";
}

bool environmentVariableIsSet(const char *name) {
  const char *value = ::getenv(name);
  return value && *value;
}

const bool recordingEnabled =
    environmentVariableIsSet(demandCensusEnvironmentVariable);
const bool postconditionEnabled =
    environmentVariableIsSet(demandCensusCheckEnvironmentVariable);

/// With both switches off no scope installs a sink and nothing is ever
/// recorded, so every guard in this file tests this first and the recording
/// sites never reach thread-local storage. Several of them sit on the stage's
/// hottest paths -- the unifier's equality check, the claim substitution
/// rebuild -- where that one skipped access is the difference between an
/// observer that costs nothing and one that costs a little everywhere.
const bool observationsEnabled = recordingEnabled || postconditionEnabled;

/// Records `demand` on the installed ledger, if any, having already checked
/// that one is installed.
void recordOnAmbientLedger(Type demand, DemandObservationKind kind) {
  DemandFlags flags;
  flags.speculative = ambientSpeculating;
  ambientLedger->record(demand, kind, flags, ambientLookupDepth);
}

/// Writes one line naming a drainable key the stage served after recording it.
void reportServedDemand(Type demand) {
  ++numServedDrainableKeys;
  llvm::errs() << demandCensusServedPrefix << " type=" << demand << "\n";
}

} // namespace

//===----------------------------------------------------------------------===//
// Switches
//===----------------------------------------------------------------------===//

bool DemandLedger::isRecordingEnabled() { return recordingEnabled; }

bool DemandLedger::isPostconditionEnabled() { return postconditionEnabled; }

bool DemandLedger::areObservationsEnabled() { return observationsEnabled; }

//===----------------------------------------------------------------------===//
// The ledger
//===----------------------------------------------------------------------===//

DemandLedger::DemandLedger() {
  statisticsAtBirth.residualToleranceAccepts = residualToleranceAcceptCount();
  statisticsAtBirth.verifierLookupMisses = numVerifierLookupMisses.getValue();
  statisticsAtBirth.verifierObligationNormalizations =
      numVerifierObligationNormalizations.getValue();
  statisticsAtBirth.resolverProjectionMisses =
      numResolverProjectionMisses.getValue();
  statisticsAtBirth.unifierAcceptances = numUnifierAcceptances.getValue();
  statisticsAtBirth.obligationNormalizations =
      numObligationNormalizations.getValue();
  statisticsAtBirth.obligationNormalizationMisses =
      numObligationNormalizationMisses.getValue();
  statisticsAtBirth.withheldCallClaims = numWithheldCallClaims.getValue();
  statisticsAtBirth.moduleFreeProjectionRejections =
      numModuleFreeProjectionRejections.getValue();
  statisticsAtBirth.modulelessRegionProjections =
      numModulelessRegionProjections.getValue();
}

void DemandLedger::record(Type demand, DemandObservationKind kind,
                          DemandFlags flags, unsigned depth) {
  // A polymorphic type has no determined resolution, so no component declined
  // to resolve one: every recording site tests this before it asks.
  assert(isMonomorphicType(demand) &&
         "a demand is a monomorphic type some component declined to resolve");

  flags.probeInternal = depth > 0;
  flags.unattributed = frames.empty();

  Location origin = frames.empty() ? Location(UnknownLoc::get(demand.getContext()))
                                   : frames.back().origin;
  Type parent = frames.empty() ? Type() : frames.back().demand;

  keys.insert(demand);
  auto it = records.try_emplace(demand, origin, parent, depth).first;
  DemandRecord &record = it->second;

  // Provenance belongs to the observation and a key keeps one observation's
  // worth of it. A key first seen inside a probe, or with no frame, would
  // otherwise report that asking forever -- including to a drain, which reports
  // at the origin. The first observation that is neither a probe nor a
  // speculation and carries a frame replaces it, and is then what the key
  // keeps.
  if (!record.provenanceIsReal && flags.isReal() && !flags.unattributed) {
    record.origin = origin;
    record.parent = parent;
    record.depth = depth;
    record.provenanceIsReal = true;
  }

  ++record.observations;
  record.engines |= 1u << static_cast<unsigned>(kind.getEngine());
  if (flags.isReal())
    record.realEngines |= 1u << static_cast<unsigned>(kind.getEngine());
  if (auto reason = kind.getMissReason())
    record.missReasons |= 1u << static_cast<unsigned>(*reason);
  record.real |= flags.isReal();
  record.speculative |= flags.speculative;
  record.probeInternal |= flags.probeInternal;
  record.unattributed |= flags.unattributed;

  unsigned cls = classOf(flags);
  ++observationsByEngine[static_cast<unsigned>(kind.getEngine())][cls];
  if (auto reason = kind.getMissReason())
    ++observationsByArm[static_cast<unsigned>(*reason)][cls];

  ++observationCount;
  ++numDemandObservations;
  if (flags.speculative) ++numSpeculativeDemandObservations;
  if (flags.probeInternal) ++numProbeInternalDemandObservations;
  if (flags.unattributed) ++numUnattributedDemandObservations;
}

const DemandRecord *DemandLedger::lookup(Type demand) const {
  auto it = records.find(demand);
  return it == records.end() ? nullptr : &it->second;
}

void DemandLedger::pushFrame(Type demand) {
  Location origin = frames.empty()
                        ? Location(UnknownLoc::get(demand.getContext()))
                        : frames.back().origin;
  frames.push_back(Frame{demand, origin});
}

void DemandLedger::pushFrame(Location origin) {
  frames.push_back(Frame{Type(), origin});
}

void DemandLedger::popFrame() {
  assert(!frames.empty() && "demand frames must be popped by their own guard");
  frames.pop_back();
}

void DemandLedger::dumpCensus() const {
  llvm::raw_ostream &os = llvm::errs();

  // Per-key lines carry the whole entry, attribution included: a reader asking
  // which demands the stage failed to serve reads these, and a drain would
  // report at the origin they name.
  uint64_t keysByClass[3] = {};
  uint64_t armKeys[numLookupMissReasons][3] = {};
  uint64_t engineKeys[numDemandEngines] = {};
  uint64_t drainableKeys = 0;
  uint64_t unattributedKeys = 0;

  for (Type key : getKeys()) {
    const DemandRecord *record = lookup(key);
    assert(record && "every ledger key has a record");

    DemandFlags flags;
    flags.speculative = record->speculative;
    flags.probeInternal = record->probeInternal;
    // A key that carries any unflagged observation is classed real however many
    // probes also observed it, which is the same rule that makes it drainable.
    FlagClass cls = record->real ? RealObservation : classOf(flags);

    ++keysByClass[cls];
    if (record->isDrainable()) ++drainableKeys;
    if (record->unattributed) ++unattributedKeys;

    os << demandCensusDemandPrefix << " flags=";
    bool first = true;
    auto emitFlag = [&](const char *name) {
      if (!first) os << ",";
      os << name;
      first = false;
    };
    if (record->real) emitFlag("real");
    if (record->speculative) emitFlag("speculative");
    if (record->probeInternal) emitFlag("probe-internal");
    if (record->unattributed) emitFlag("unattributed");

    os << " drainable=" << (record->isDrainable() ? "yes" : "no")
       << " observations=" << record->observations
       << " depth=" << record->depth << " kinds=";
    first = true;
    for (unsigned e = 0; e != numDemandEngines; ++e) {
      if (!(record->engines & (1u << e))) continue;
      ++engineKeys[e];
      if (!first) os << ",";
      os << engineName(static_cast<DemandEngine>(e));
      first = false;
    }
    os << " arms=";
    first = true;
    for (unsigned a = 0; a != numLookupMissReasons; ++a) {
      if (!(record->missReasons & (1u << a))) continue;
      ++armKeys[a][cls];
      if (!first) os << ",";
      os << missReasonName(static_cast<LookupMissReason>(a));
      first = false;
    }
    if (first) os << "-";
    os << " parent=";
    if (record->parent)
      os << record->parent;
    else
      os << "-";
    os << " origin=" << record->origin << " type=" << key << "\n";
  }

  uint64_t partitioned = 0;
  for (unsigned e = 0; e != numDemandEngines; ++e) {
    uint64_t total = 0;
    for (unsigned c = 0; c != 3; ++c) total += observationsByEngine[e][c];
    partitioned += total;
    os << demandCensusEnginePrefix << " "
       << engineName(static_cast<DemandEngine>(e))
       << " keys=" << engineKeys[e] << " observations=" << total;
    for (unsigned c = 0; c != 3; ++c)
      os << " " << flagClassName(static_cast<FlagClass>(c)) << "="
         << observationsByEngine[e][c];
    os << "\n";
  }
  for (unsigned a = 0; a != numLookupMissReasons; ++a) {
    uint64_t total = 0;
    uint64_t armKeyTotal = 0;
    for (unsigned c = 0; c != 3; ++c) {
      total += observationsByArm[a][c];
      armKeyTotal += armKeys[a][c];
    }
    os << demandCensusArmPrefix << " "
       << missReasonName(static_cast<LookupMissReason>(a))
       << " keys=" << armKeyTotal << " observations=" << total;
    for (unsigned c = 0; c != 3; ++c)
      os << " " << flagClassName(static_cast<FlagClass>(c)) << "="
         << observationsByArm[a][c];
    os << "\n";
  }

  // The engine partition covers every observation exactly once, so a census
  // whose parts do not add up has lost a record rather than merely miscounted.
  assert(partitioned == observationCount &&
         "the census partition must cover every observation");
  (void)partitioned;

  os << demandCensusSummaryPrefix << " keys=" << keys.size()
     << " observations=" << observationCount
     << " drainable-keys=" << drainableKeys
     << " unattributed-keys=" << unattributedKeys;
  for (unsigned c = 0; c != 3; ++c)
    os << " " << flagClassName(static_cast<FlagClass>(c)) << "-keys="
       << keysByClass[c];
  os << "\n";

  // Two counter lines, each labelled by the population its numbers cover. Every
  // column names one event and carries a name no other column carries:
  // `verifier` columns count events only a verifier raises, `total` columns
  // count events wherever they happen, the stage and the verifiers together. No
  // column is a part of another, so the two lines are read side by side rather
  // than differenced.
  os << demandCensusCounterPrefix << " verifier"
     << " lookup-misses="
     << numVerifierLookupMisses.getValue() -
            statisticsAtBirth.verifierLookupMisses
     << " cast-normalizations="
     << numVerifierObligationNormalizations.getValue() -
            statisticsAtBirth.verifierObligationNormalizations
     << "\n";

  os << demandCensusCounterPrefix << " total"
     << " residual-tolerance-accepts="
     << residualToleranceAcceptCount() -
            statisticsAtBirth.residualToleranceAccepts
     << " resolver-engine-misses="
     << numResolverProjectionMisses.getValue() -
            statisticsAtBirth.resolverProjectionMisses
     << " unifier-acceptances="
     << numUnifierAcceptances.getValue() - statisticsAtBirth.unifierAcceptances
     << " obligation-normalizations="
     << numObligationNormalizations.getValue() -
            statisticsAtBirth.obligationNormalizations
     << " obligation-normalization-misses="
     << numObligationNormalizationMisses.getValue() -
            statisticsAtBirth.obligationNormalizationMisses
     << " withheld-call-claims="
     << numWithheldCallClaims.getValue() - statisticsAtBirth.withheldCallClaims
     << " module-free-rejections="
     << numModuleFreeProjectionRejections.getValue() -
            statisticsAtBirth.moduleFreeProjectionRejections
     << " moduleless-region-projections="
     << numModulelessRegionProjections.getValue() -
            statisticsAtBirth.modulelessRegionProjections
     << "\n";

  os << demandCensusScanPrefix
     << " claim-substitution-rebuilds=" << claimSubstitutionRebuilds
     << " claim-substitution-entries-walked=" << claimSubstitutionEntriesWalked
     << " proof-scans=" << proofScans
     << " proof-scan-entries=" << proofScanEntries
     << " proof-collision-scans=" << proofCollisionScans
     << " proof-collision-scan-entries=" << proofCollisionScanEntries << "\n";
}

void DemandLedger::reportServedDrainableKeys(ModuleOp module) const {
  if (!postconditionEnabled)
    return;

  // What the module still spells. Result and block-argument types are what the
  // stage's own leftover-projection sweep walks, and an operand type is its
  // producer's result type, so between them they cover every type a later round
  // would find to serve.
  DenseSet<Type> standing;
  auto collect = [&](Type root) {
    root.walk([&](Type sub) {
      if (isa<ProjectionType>(sub) && isMonomorphicType(sub))
        standing.insert(sub);
    });
  };
  module.walk([&](Operation *op) {
    for (Type ty : op->getResultTypes())
      collect(ty);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          collect(arg.getType());
  });

  for (Type key : getKeys()) {
    const DemandRecord *record = lookup(key);
    assert(record && "every ledger key has a record");
    if (!record->isDrainable())
      continue;
    // An engine that refused the demand outright would refuse it again, so the
    // key stands whether or not the spelling that carried it survived.
    if (record->missReasons)
      continue;

    bool stillSpelled = false;
    key.walk([&](Type sub) {
      if (standing.contains(sub))
        stillSpelled = true;
    });
    // No test reaches the report: every drainable key recorded so far is either
    // exempt above or still spelled at stage exit, which is the drainability
    // rule holding. A row that reports here is one the rule over-admitted.
    if (!stillSpelled)
      reportServedDemand(key);
  }
}

//===----------------------------------------------------------------------===//
// Reaching the ledger
//===----------------------------------------------------------------------===//

bool isDemandRecordingActive() {
  return observationsEnabled && ambientLedger != nullptr;
}

DemandLedgerScope::DemandLedgerScope(DemandLedger &ledger)
    : previous(nullptr), installed(observationsEnabled) {
  if (installed) {
    previous = ambientLedger;
    ambientLedger = &ledger;
  }
}

DemandLedgerScope::~DemandLedgerScope() {
  if (!installed)
    return;
  assert(ambientLedger &&
         ambientLedger->getFrameDepth() == 0 &&
         "every demand frame opened inside a stage span must close inside it");
  ambientLedger = previous;
}

DemandRecordingSuspension::DemandRecordingSuspension() : previous(nullptr) {
  if (!observationsEnabled)
    return;
  assert((!ambientLedger || ambientLedger->getFrameDepth() == 0) &&
         "a demand frame must not span a suspension");
  previous = ambientLedger;
  ambientLedger = nullptr;
}

DemandRecordingSuspension::~DemandRecordingSuspension() {
  if (observationsEnabled)
    ambientLedger = previous;
}

DemandFrame::DemandFrame(Type demand)
    : ledger(isDemandRecordingActive() ? ambientLedger : nullptr) {
  if (ledger)
    ledger->pushFrame(demand);
}

DemandFrame::DemandFrame(Location origin)
    : ledger(isDemandRecordingActive() ? ambientLedger : nullptr) {
  if (ledger)
    ledger->pushFrame(origin);
}

DemandFrame::~DemandFrame() {
  // Popped on the ledger this frame pushed to, which is not necessarily the one
  // installed now: a suspension between the two would have replaced it.
  if (ledger)
    ledger->popFrame();
}

LookupProbeScope::LookupProbeScope()
    : enclosingDepth(observationsEnabled ? ambientLookupDepth++ : 0) {}

LookupProbeScope::~LookupProbeScope() {
  if (observationsEnabled)
    --ambientLookupDepth;
}

SpeculationScope::SpeculationScope() : previous(false) {
  if (!observationsEnabled)
    return;
  previous = ambientSpeculating;
  ambientSpeculating = true;
}

SpeculationScope::~SpeculationScope() {
  if (observationsEnabled)
    ambientSpeculating = previous;
}

//===----------------------------------------------------------------------===//
// Recording sites
//===----------------------------------------------------------------------===//

void recordLookupMiss(Type demand, LookupMissReason reason, DemandOrigin origin,
                      unsigned enclosingDepth) {
  if (!recordsToLedger(origin)) {
    ++numVerifierLookupMisses;
    return;
  }
  if (!isDemandRecordingActive())
    return;

  DemandFlags flags;
  flags.speculative = ambientSpeculating;
  ambientLedger->record(demand, DemandObservationKind::lookupMiss(reason), flags,
                        enclosingDepth);
}

void recordUnifierAcceptance(Type demand, DemandOrigin origin) {
  ++numUnifierAcceptances;
  if (!recordsToLedger(origin) || !isDemandRecordingActive())
    return;
  recordOnAmbientLedger(demand, DemandObservationKind::unifierAcceptance());
}

void recordResolverProjectionMiss(Type demand) {
  ++numResolverProjectionMisses;
  if (!isDemandRecordingActive())
    return;
  recordOnAmbientLedger(demand,
                        DemandObservationKind::resolverProjectionEngine());
}

void recordObligationNormalizationMiss(Type demand, DemandOrigin origin) {
  ++numObligationNormalizationMisses;
  if (!recordsToLedger(origin) || !isDemandRecordingActive())
    return;
  recordOnAmbientLedger(demand,
                        DemandObservationKind::obligationNormalization());
}

void countObligationNormalization() { ++numObligationNormalizations; }

void recordWithheldCallClaim(Type demand) {
  if (!isDemandRecordingActive())
    return;
  recordOnAmbientLedger(demand, DemandObservationKind::withheldCallClaim());
}

void countWithheldCallClaim() { ++numWithheldCallClaims; }

void countModuleFreeProjectionRejection() {
  ++numModuleFreeProjectionRejections;
}

void countModulelessRegionProjection() { ++numModulelessRegionProjections; }

void countVerifierObligationNormalization() {
  ++numVerifierObligationNormalizations;
}

void countClaimSubstitutionRebuild(size_t entries) {
  if (isDemandRecordingActive())
    ambientLedger->countClaimSubstitutionRebuild(entries);
}

void countProofScan(size_t entries) {
  if (isDemandRecordingActive())
    ambientLedger->countProofScan(entries);
}

void countProofCollisionScan(size_t entries) {
  if (isDemandRecordingActive())
    ambientLedger->countProofCollisionScan(entries);
}

void reportUnhookedMint(Type demand) {
  ++numUnhookedMints;
  llvm::errs() << demandCensusUnhookedPrefix << " type=" << demand << "\n";
}

} // end mlir::trait
