// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// An impl's own false equality assumption is inert only while its projection
// stays symbolic. @T2_i64 assumes proj<@Sib[i64],"Elem"> = i1, and it declares a
// premise -- citing the conditional @Sib_cond, its @X[i64] assumption supplied
// by a discharge citation -- that resolves the projection to i64. The own-equality
// birth check replays the premise, reduces the endpoint to the ground value
// i64, and refuses the ground mismatch against i1, even though the impl never
// consumes the equality. The acceptance is the symbolic case alone.

!S = !trait.poly<0>

trait.trait @X[!S] {}
trait.impl @X_i64 for @X[i64] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}
trait.impl @Sib_cond for @Sib[i64] where [@X[i64]] {
  trait.assoc_type @Elem = i64
}

trait.trait @T2[!S] {
  func.func private @id(!S) -> !S
}

// expected-error @below {{does not satisfy its own equality predicate #trait<equality!trait.proj<@Sib[i64], "Elem"> = i1>: 'i64' and 'i1' are not the same type}}
trait.impl @T2_i64 for @T2[i64] where [!trait.proj<@Sib[i64], "Elem"> = i1]
    premises [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i64 by @Sib_cond>]
    discharges [#trait<witness @X[i64] by @X_i64>] {
  func.func @id(%x: i64) -> i64 {
    return %x : i64
  }
}
