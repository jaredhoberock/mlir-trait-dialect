// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Building a specialization mints a fresh inference variable for each generic,
// starting past every inference id the formal and actual already spell so that
// "fresh" cannot alias a variable in hand. A scan that stopped at an equality
// claim would miss a variable spelled only inside an endpoint -- and the mint
// would then hand a sibling generic the very id the endpoint holds, conflating
// two variables. The impl's self claim below spells `!trait.poly<0>` in the
// first argument and, inside the equality endpoint of the second,
// `!trait.infer<0>`. The derive binds the poly to `f64` and the
// endpoint variable to `i32`; the two stay distinct only when the scan counts the
// endpoint id, so the mint for `!trait.poly<0>` starts past it. Miss that id and
// the poly mints back to `!trait.infer<0>`, the endpoint aliases it, and the two
// bindings collide as one variable pulled to both `f64` and `i32`.

!P = !trait.poly<0>
!I = !trait.infer<0>

trait.trait private @Trait[!trait.poly<0>, !trait.poly<1>] {}
trait.trait private @Need[!trait.poly<0>] {}

trait.impl private @I for @Trait[!P, !trait.claim<!I = i32>] where [@Need[!P]] {}

// CHECK-LABEL: func.func @derive_over_a_sealed_endpoint
// CHECK: trait.derive @Trait[f64, !trait.claim<i32 = i32>] from @I
func.func @derive_over_a_sealed_endpoint(%n: !trait.claim<@Need[f64]>) {
  %d = trait.derive @Trait[f64, !trait.claim<i32 = i32>] from @I given(%n)
    : (!trait.claim<@Need[f64]>)
  return
}
