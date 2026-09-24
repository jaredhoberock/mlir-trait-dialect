// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Deriving `@Call[tuple<i32, f32>, i64]` from `@wrap` reads its equality
// premise `Call[i32, i64]::Output = f32` through the impls the module holds,
// which selects among every `@Call` impl for `@Call[i32, i64]`. Each candidate
// has its parameters read off its header and its where clause. `@wrap`'s
// header leaves `!F` and `!Y` open against `i32`, and its equality determines
// `!Y` only once `!F` is known: `Call[!F, i64]::Output` still names the
// unsettled `!F`, so it says nothing and is not normalized. Normalizing it
// would select among the `@Call` impls for `@Call[!F, i64]`, `@wrap` among
// them, whose equality names an unsettled parameter again: the selection would
// not end. `@wrap` is refused by its header and `@direct` resolves the premise.

!S = !trait.poly<0>
!A = !trait.poly<1>
!F = !trait.poly<2>
!Y = !trait.poly<3>
!B = !trait.poly<4>

trait.trait private @Call[!S, !A] {
  trait.assoc_type @Output
}

trait.impl private @wrap for @Call[tuple<!F, !Y>, !B] where [
  @Call[!F, !B],
  !trait.proj<@Call[!F, !B], "Output"> = !Y
] {
  trait.assoc_type @Output = !Y
}

trait.impl private @direct for @Call[i32, i64] {
  trait.assoc_type @Output = f32
}

// CHECK-LABEL: func.func private @derive
// CHECK: trait.derive @Call[tuple<i32, f32>, i64] from @wrap
func.func private @derive(%c: !trait.claim<@Call[i32, i64]>) {
  %d = trait.derive @Call[tuple<i32, f32>, i64] from @wrap given(%c) : (!trait.claim<@Call[i32, i64]>)
  return
}
