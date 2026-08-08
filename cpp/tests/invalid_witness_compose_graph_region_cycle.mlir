// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The composition arm's evidence is another claim value, so its validity rests
// on its premises. A region without SSA dominance -- a module body is a graph
// region -- lets a premise be the op's own result: two composes each cite the
// other, and each would pass its local entailment check because its premise IS
// its result, grounding a false equality on nothing. Requiring an SSA-dominance
// region refuses the cycle, so the induction bottoms out at the
// certificate- and refl-anchored leaves that admit no false equality.

// The verifier refuses the first op of the cycle it reaches; that one refusal
// is enough to reject the module and stops verification, so a single diagnostic
// is expected.
// expected-error @+1 {{a composition witness must be in a region that enforces SSA dominance}}
%a = trait.witness compose(%b) : (!trait.claim<i32 = i64>) : !trait.claim<i32 = i64>
%b = trait.witness compose(%a) : (!trait.claim<i32 = i64>) : !trait.claim<i32 = i64>
