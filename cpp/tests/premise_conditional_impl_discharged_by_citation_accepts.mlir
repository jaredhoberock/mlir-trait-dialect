// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A premise cites a CONDITIONAL sibling impl whose assumption the citing impl's
// own where clause does NOT cover. @Sib_i64_cond binds Sib[i64]::Elem = i32 and
// assumes @Y[i64]; @Host_i64 declares no where clause, so arm (i) cannot
// discharge @Y[i64]. A declared discharge citation names @Y_i64 -- an
// unconditional impl of @Y[i64] -- as the discharger, so arm (ii) supplies the
// assumption, the premise replays, and the expected signature reduces to
// (i64) -> i32 and matches. The discharger is named; the audit resolves the
// symbol and never scans the module for one.

!S = !trait.poly<0>

trait.trait @Y[!S] {}

trait.impl @Y_i64 for @Y[i64] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64_cond for @Sib[i64] where [@Y[i64]] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// CHECK: trait.impl @Host_i64 for @Host[i64]premises [#trait<certificate!trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond>]discharges [#trait<discharge@Y[i64] by @Y_i64>]
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<certificate !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond>]
    discharges [#trait<discharge @Y[i64] by @Y_i64>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
