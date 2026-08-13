// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait's header requires a sibling-projection equality (@Sib[Self]::Elem = f32)
// that is FALSE: the only impl of @Sib[i64] binds Elem = i64. This hand-written
// impl declares no premise, so the endpoint stays symbolic at impl verification and the check
// defers -- a non-ground projection cannot be decided against the impl's bindings.
// Nothing here consumes the equality, so no coerce reaches the erase barrier and
// the false requirement is checked nowhere: the same inert corner Rust's
// unsatisfiable where-clause bounds occupy, accepted by design. A front end
// emitting this impl would instead declare a premise resolving @Sib[i64]::Elem to
// i64, and the ground mismatch i64 != f32 would then refuse it at impl verification.

!S = !trait.poly<0>

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64 for @Sib[i64] {
  trait.assoc_type @Elem = i64
}

trait.trait @T[!S] where [!trait.proj<@Sib[!S], "Elem"> = f32] {
  func.func private @id(!S) -> !S
}

// CHECK: trait.impl @T_i64
trait.impl @T_i64 for @T[i64] {
  func.func @id(%x: i64) -> i64 {
    return %x : i64
  }
}
