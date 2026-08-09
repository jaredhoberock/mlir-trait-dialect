// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" | FileCheck %s

// A trait.derive discharges its impl's assumptions by exact spelling. A
// conditional impl whose where-clause carries a ground projection stands in a
// never-instantiated template, so the derive survives to the pass boundary with
// its derived claim still polymorphic. The instantiation rounds prove and
// resolve the operand's projection to its ground spelling (the proven-claim
// respell rewrites @Other[@HasPart[i64]::Part] to @Other[f32] by @Other_f32),
// while the derive verifier recomputes the assumption by pure substitution and
// leaves the projection symbolic. The reconciliation walk bridges the resolved
// operand back to the expected spelling with a coerce citing the per-hop
// proj-resolve certificate, so the verifier's exact compare holds; without the
// bridge the module would not verify after the pass.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T7 = !trait.poly<7>

trait.trait @HasPart[!S] {
  trait.assoc_type @Part
}

trait.impl @HasPart_i64 for @HasPart[i64] {
  trait.assoc_type @Part = f32
}

trait.trait @Other[!S] {}

trait.impl @Other_f32 for @Other[f32] {}

trait.trait @Tr[!S] {}

trait.impl @CondImpl for @Tr[!X] where [@Other[!trait.proj<@HasPart[i64], "Part">]] {}

// The witness proving the projection resolves, the coerce carrying the resolved
// operand back to the projection spelling, and the derive taking that bridged
// operand, all standing in the surviving template.
// CHECK: %[[HOP:.*]] = trait.witness proj_resolve !trait.proj<@HasPart[i64], "Part"> resolves f32 by @HasPart_i64
// CHECK: %[[BRIDGED:.*]] = trait.coerce %{{.*}} : !trait.claim<@Other[f32] by @Other_f32> to !trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">]> via (%[[HOP]])
// CHECK: trait.derive @Tr[!trait.poly<7>] from @CondImpl given(%[[BRIDGED]])
func.func private @template(
  %op: !trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">]>
) -> !trait.claim<@Tr[!T7]> {
  %d = trait.derive @Tr[!T7] from @CondImpl given(%op)
    : (!trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">]>)
  return %d : !trait.claim<@Tr[!T7]>
}
