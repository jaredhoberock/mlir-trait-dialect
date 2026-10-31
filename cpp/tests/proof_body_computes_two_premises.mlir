// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A proof's body computes one claim per where entry of the impl it derives
// from and returns the derive. @pair proves A[tuple<i32, i64>] from @A_pair
// with a witness for each of its two entries; the instance of the impl's
// function the call of @A.a reaches through @pair reads both, each through
// the clone of the premise the proof computed for it, and takes the receiver
// alone.

// INSTANCE:      func.func private @A_pair.a_{{h[0-9a-f]+}}(%{{.*}}: !trait.claim<@A[tuple<i32, i64>] by @pair>) -> i64 {
// INSTANCE-NEXT:   %[[P:.*]] = trait.witness @B_i32 for @B[i32]
// INSTANCE-NEXT:   %[[Q:.*]] = trait.witness @B_i64 for @B[i64]
// INSTANCE-NEXT:   call @B_i32.b_{{.*}}(%[[P]])
// INSTANCE-NEXT:   call @B_i64.b_{{.*}}(%[[Q]])
// INSTANCE:      func.func @main() -> i64 {
// INSTANCE-NEXT:   %[[W:.*]] = trait.witness @pair for @A[tuple<i32, i64>]
// INSTANCE-NEXT:   call @A_pair.a_{{h[0-9a-f]+}}(%[[W]])

// CHECK: {{^}}12{{$}}

!P = !trait.poly<0>
!Q = !trait.poly<1>
trait.trait private @B(!trait.claim<@B[!P]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!P]>) -> i64
trait.trait private @A(!trait.claim<@A[!P]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!P]>) -> i64
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>) -> i64 {
  %c = arith.constant 2 : i64
  return %c : i64
}
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) members [@B.b -> @B_i64.b] {
}
func.func private @B_i64.b(%self: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 10 : i64
  return %c : i64
}
trait.impl private @A_pair(%self: !trait.claim<@A[tuple<!P, !Q>]>, %p: !trait.claim<@B[!P]>, %q: !trait.claim<@B[!Q]>) members [@A.a -> @A_pair.a] {
}
func.func private @A_pair.a(%self: !trait.claim<@A[tuple<!P, !Q>]>, %p: !trait.claim<@B[!P]>, %q: !trait.claim<@B[!Q]>) -> i64 {
  %x = trait.func.call @B.b(%p) : (!trait.claim<@B[!P]>) -> i64
  %y = trait.func.call @B.b(%q) : (!trait.claim<@B[!Q]>) -> i64
  %r = arith.addi %x, %y : i64
  return %r : i64
}
trait.proof private @pair {
  %p = trait.witness @B_i32 for @B[i32]
  %q = trait.witness @B_i64 for @B[i64]
  %d = trait.derive @A[tuple<i32, i64>] from @A_pair[i32, i64] given(%p, %q) : (!trait.claim<@B[i32] by @B_i32>, !trait.claim<@B[i64] by @B_i64>)
  trait.return %d : !trait.claim<@A[tuple<i32, i64>]>
}
func.func @main() -> i64 {
  %w = trait.witness @pair for @A[tuple<i32, i64>]
  %r = trait.func.call @A.a(%w) : (!trait.claim<@A[tuple<i32, i64>] by @pair>) -> i64
  return %r : i64
}
