// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "DemandLedger.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::trait {
namespace {
thread_local DemandLedger *ambientLedger = nullptr;
thread_local unsigned ambientLookupDepth = 0;
thread_local bool ambientSpeculating = false;

/// Only a real unresolved obligation can cause later preparation work.
void recordPending(Type demand, unsigned missArms, unsigned depth) {
  if (!ambientLedger || ambientSpeculating || depth)
    return;
  ambientLedger->record(demand, missArms);
}
} // namespace

void DemandLedger::record(Type demand, unsigned missArms) {
  assert(isMonomorphicType(demand) && "pending demands have a concrete type");
  demands.insert(demand);
  arms[demand] |= missArms;
  // The first frame that names a place keeps it: a demand is raised where an
  // engine first read a spelling it could not settle, and a later read of the
  // same demand stands wherever that later reader does.
  if (std::optional<Location> origin = getInnermostFrameOrigin())
    raisedAt.try_emplace(demand, *origin);
}

void DemandLedger::pushFrame(Type demand) {
  Location origin = frames.empty()
                        ? Location(UnknownLoc::get(demand.getContext()))
                        : frames.back();
  frames.push_back(origin);
}

void DemandLedger::pushFrame(Location origin) {
  frames.push_back(origin);
}

void DemandLedger::popFrame() {
  assert(!frames.empty() && "demand frames must be popped by their own guard");
  frames.pop_back();
}

llvm::SetVector<Type> demandsSpelledIn(ModuleOp module, bool inAttributes,
                                       DemandSkip projections,
                                       DemandSkip claims,
                                       DenseMap<Type, Location> *origins) {
  llvm::SetVector<Type> spelled;
  auto skips = [](DemandSkip discipline, Operation *op) {
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return true;
    if (discipline != DemandSkip::Foreign)
      return false;
    if (auto func = dyn_cast<func::FuncOp>(op))
      return isPolymorphicType(Type(func.getFunctionType()));
    return false;
  };

  Operation *spellingOp = nullptr;
  auto note = [&](Type demand) {
    if (spelled.insert(demand) && origins && spellingOp)
      origins->try_emplace(demand, spellingOp->getLoc());
  };
  // Note every monomorphic projection reachable in a type. An equality claim's
  // endpoints are ordinary sub-elements, so a projection standing inside one --
  // even one itself inside a further equality claim -- is still demanded and its
  // impl generated. `root` may be a Type or an Attribute.
  auto collect = [&](auto root) {
    root.walk([&](Type sub) {
      if (isa<ProjectionType>(sub) && isMonomorphicType(sub))
        note(sub);
      return WalkResult::advance();
    });
  };
  module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (skips(projections, op))
      return WalkResult::skip();
    spellingOp = op;
    for (Type ty : op->getResultTypes())
      collect(ty);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          collect(arg.getType());
    if (inAttributes)
      collect(Attribute(op->getAttrDictionary()));
    return WalkResult::advance();
  });

  auto collectClaims = [&](Type root) {
    root.walk([&](Type sub) {
      auto claim = dyn_cast<ClaimType>(sub);
      // Only application claims are impl-resolution demands. An equality claim is
      // established by trait.witness, not by selecting an impl, so it is never a
      // demand the resolver serves -- and it carries no trait application to
      // resolve for.
      if (claim && claim.isApplication() && !claim.isProven() &&
          claim.isMonomorphic())
        note(sub);
    });
  };
  module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (skips(claims, op))
      return WalkResult::skip();
    spellingOp = op;
    for (Type ty : op->getResultTypes())
      collectClaims(ty);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          collectClaims(arg.getType());
    return WalkResult::advance();
  });
  return spelled;
}

LogicalResult
DemandLedger::checkStandingDemandsServed(ModuleOp module,
                                         const DenseSet<Type> &served) const {
  // A demand spelled only inside trait infrastructure or a still-polymorphic
  // template is resolved on cloning. Like the leftover-projection sweep, this
  // check excludes obligations that preparation does not serve.
  llvm::SetVector<Type> spelled = demandsSpelledIn(
      module, /*inAttributes=*/true, DemandSkip::Foreign,
      DemandSkip::Foreign);

  bool standing = false;
  for (Type key : getDrainableDemands()) {
    // Only real demands enter the queue; probes and speculation are excluded.
    if (served.contains(key))
      continue;
    // The one refusal no later resolution overturns leaves its demand standing
    // on purpose; selection named the ambiguity where the demand stood, so this
    // walk passes it over. The failed lookup preserves that arm alongside its
    // demand.
    if (getDrainableArms(key) &
        (1u << static_cast<unsigned>(LookupMissReason::MultipleCandidateImpls)))
      continue;

    bool stillSpelled = false;
    key.walk([&](Type sub) {
      if (spelled.contains(sub))
        stillSpelled = true;
    });
    if (!stillSpelled)
      continue;

    standing = true;
    module.emitError()
        << "instantiate-monomorphs left the demand " << key
        << " standing and never served it";
  }
  return failure(standing);
}

std::optional<Location> currentDemandAnchor() {
  if (!ambientLedger)
    return std::nullopt;
  return ambientLedger->getInnermostFrameOrigin();
}

DemandLedgerScope::DemandLedgerScope(DemandLedger &ledger)
    : previous(ambientLedger) {
  ambientLedger = &ledger;
}

DemandLedgerScope::~DemandLedgerScope() {
  assert(ambientLedger &&
         ambientLedger->getFrameDepth() == 0 &&
         "every demand frame opened inside a stage span must close inside it");
  ambientLedger = previous;
}

DemandRecordingSuspension::DemandRecordingSuspension()
    : previous(ambientLedger) {
  assert((!ambientLedger || ambientLedger->getFrameDepth() == 0) &&
         "a demand frame must not span a suspension");
  ambientLedger = nullptr;
}

DemandRecordingSuspension::~DemandRecordingSuspension() {
  ambientLedger = previous;
}

DemandFrame::DemandFrame(Type demand) : ledger(ambientLedger) {
  if (ledger)
    ledger->pushFrame(demand);
}

DemandFrame::DemandFrame(Location origin) : ledger(ambientLedger) {
  if (ledger)
    ledger->pushFrame(origin);
}

DemandFrame::~DemandFrame() {
  // Popped on the ledger this frame pushed to, which is not necessarily the one
  // installed now: a suspension between the two would have replaced it.
  if (ledger)
    ledger->popFrame();
}

LookupProbeScope::LookupProbeScope() : enclosingDepth(ambientLookupDepth++) {}

LookupProbeScope::~LookupProbeScope() { --ambientLookupDepth; }

SpeculationScope::SpeculationScope() : previous(ambientSpeculating) {
  ambientSpeculating = true;
}

SpeculationScope::~SpeculationScope() { ambientSpeculating = previous; }

void recordLookupMiss(Type demand, LookupMissReason reason, DemandOrigin origin,
                      unsigned enclosingDepth) {
  if (recordsToLedger(origin))
    recordPending(demand, 1u << static_cast<unsigned>(reason), enclosingDepth);
}

void recordResolverProjectionMiss(Type demand) {
  recordPending(demand, 0, ambientLookupDepth);
}

void recordReadOnlyResolverMiss(Type demand) {
  recordPending(demand, 0, ambientLookupDepth);
}

} // namespace mlir::trait
