// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_ATTR_INTERFACES_TD
#define TRAIT_ATTR_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

// The identity of an impl rule: a family of impls a dialect's generator
// produces, implemented by an attribute of that dialect.
def RuleAttrInterface : AttrInterface<"RuleAttrInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    An attribute implementing this interface names an impl rule: a family of
    impls one dialect's generator produces, each an instance of the rule at one
    application. The meaning of the rule is that dialect's; this dialect reads
    the identity alone, and trusts it. A generated `trait.impl` carries the
    rule it is an instance of as its provenance, and a `trait.allege` names the
    rule asserting its claim; no verifier compares the two, and impl selection
    proves an allegation's claim by whichever impl it selects. The interface
    declares no method: implementing it is the whole statement that the
    attribute names a rule, as an attribute implementing a memory-space or
    target interface states it is one.
  }];
}

#endif // TRAIT_ATTR_INTERFACES_TD
