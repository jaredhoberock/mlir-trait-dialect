// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not mlir-opt -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Evidence that returns to its own question through another impl: @A_i32's
// evidence for @B[i32] is @C_i32's, and @C_i32's is @A_i32's. Each impl reads
// another's return, so each verifies; a projection that reads the cycle reads
// @A_i32's return, which projects @C_i32's, which projects @A_i32's again: the
// evidence has no base, so the projection is never inlined, and the stage
// names it where it stands.

// CHECK: :[[@LINE+19]]:8: error: unproven monomorphic claim '!trait.claim<@B[i32]>' after instantiate-monomorphs
// CHECK: note: its source '!trait.claim<@A[i32] by @A_i32>' names @A_i32, whose evidence for requirement 0 has no base: it is read through the returns of @A_i32, @C_i32, @A_i32 back to a requirement it stands for

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %c = trait.witness @C_i32 for @C[i32]
  %b = trait.project %c[0] : !trait.claim<@C[i32] by @C_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}

// -----

// Evidence that returns to its own question through two requirements of one
// impl: requirement 0 is requirement 1 and requirement 1 is requirement 0, both
// read off a witness of the impl itself, and the impl is refused where it is
// declared.

// CHECK: error: 'trait.impl' op returns evidence for requirement 0 that projects the impl's own application

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> (!trait.claim<@B[!T]>, !trait.claim<@B[!T]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %x = trait.project %a[1] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  %y = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %x, %y : !trait.claim<@B[i32]>, !trait.claim<@B[i32]>
}

// -----

// The first cycle read inside an instance: @callee's projection off its proven
// parameter reads @A_i32's return, which projects @C_i32's, around the cycle.
// The cut inlines no evidence with no base, so the projection stands in the
// instance and the stage names it there.

// CHECK: :[[@LINE+18]]:8: error: unproven monomorphic claim '!trait.claim<@B[i32]>' after instantiate-monomorphs
// CHECK: note: its source '!trait.claim<@A[i32] by @A_i32>' names @A_i32, whose evidence for requirement 0 has no base: it is read through the returns of @A_i32, @C_i32, @A_i32 back to a requirement it stands for

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %c = trait.witness @C_i32 for @C[i32]
  %b = trait.project %c[0] : !trait.claim<@C[i32] by @C_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>) {
  %a = trait.witness @A_i32 for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
func.func private @callee(%c: !trait.claim<@A[!T]>) -> i64 {
  %b = trait.project %c[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  %v = trait.method.call %b @B[!T]::@v() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %a = trait.witness @A_i32 for @A[i32]
  %v = trait.func.call @callee(%a) : (!trait.claim<@A[i32] by @A_i32>) -> i64
  return %v : i64
}

// -----

// The cycle through where arguments: @A_i32's return projects its where
// argument @C[i32], which proof @PA gives it as @PC's claim, and @C_i32's
// return projects its where argument @A[i32], which @PC gives it as @PA's.
// The reading follows what each proof gave the impl it derives from, so the
// evidence has no base.

// CHECK: note: its source '!trait.claim<@A[i32] by @PA>' names @PA, whose evidence for requirement 0 has no base: it is read through the returns of @A_i32, @C_i32, @A_i32 back to a requirement it stands for

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%self: !trait.claim<@C[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>, %c: !trait.claim<@C[i32]>) {
  %b = trait.project %c[0] : !trait.claim<@C[i32]> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.impl private @C_i32(%self: !trait.claim<@C[i32]>, %a: !trait.claim<@A[i32]>) {
  %b = trait.project %a[0] : !trait.claim<@A[i32]> -> !trait.claim<@B[i32]>
  trait.return %b : !trait.claim<@B[i32]>
}
trait.proof private @PA {
  %c = trait.witness @PC for @C[i32]
  %d = trait.derive @A[i32] from @A_i32 given(%c) : (!trait.claim<@C[i32] by @PC>)
  trait.return %d : !trait.claim<@A[i32]>
}
trait.proof private @PC {
  %a = trait.witness @PA for @A[i32]
  %d = trait.derive @C[i32] from @C_i32 given(%a) : (!trait.claim<@A[i32] by @PA>)
  trait.return %d : !trait.claim<@C[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @PA for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @PA> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}

// -----

// A cycle behind a projection standing on another: @X returns the projection
// of the requirement @Y returns at @C[i32], and @Y returns @X's own
// derivation at @A[i32]. Reading the inner projection first commits the outer
// one to @X at @A[i32] again, so the evidence has no base.

// CHECK: error: unproven monomorphic claim '!trait.claim<@B[i32]>' after instantiate-monomorphs
// CHECK: note: its source '!trait.claim<@A[i32] by @X>' names @X, whose evidence for requirement 0 has no base: it is read through the returns of @X, @Y, @X back to a requirement it stands for

!T = !trait.poly<0>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.trait private @A(%s: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.trait private @C(%s: !trait.claim<@C[!T]>) -> !trait.claim<@A[!T]> {}
trait.impl private @Y(%s: !trait.claim<@C[i32]>) {
  %x = trait.derive @A[i32] from @X given()
  trait.return %x : !trait.claim<@A[i32]>
}
trait.impl private @X(%s: !trait.claim<@A[i32]>) {
  %c = trait.derive @C[i32] from @Y given()
  %p = trait.project %c[0] : !trait.claim<@C[i32]> -> !trait.claim<@A[i32]>
  %r = trait.project %p[0] : !trait.claim<@A[i32]> -> !trait.claim<@B[i32]>
  trait.return %r : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @X for @A[i32]
  %b = trait.project %a[0] : !trait.claim<@A[i32] by @X> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}

// -----

// Requirement evidence read through one hundred and twenty-nine returns, each
// a projection of the next impl's, before a witness: past the depth limit the
// reading overflows, as any obligation chain that deep does.

// CHECK: error: overflow evaluating the requirement {{.*}}@A[i129, i32]{{.*}}: 129 obligations stand on the chain that reaches it

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.method @v() -> i64 }
trait.impl private @Base(%s: !trait.claim<@B[i32]>) { trait.method @v() -> i64 { %v = arith.constant 7 : i64 trait.return %v : i64 } }
trait.trait private @A(%s: !trait.claim<@A[!T,!U]>) -> !trait.claim<@B[!U]> {}
// REPEAT 1 129: trait.impl private @I{k}(%s: !trait.claim<@A[i{k},i32]>) { %a = trait.witness @I{k+1} for @A[i{k+1},i32] %b = trait.project %a[0] : !trait.claim<@A[i{k+1},i32] by @I{k+1}> -> !trait.claim<@B[i32]> trait.return %b : !trait.claim<@B[i32]> }
trait.impl private @I130(%s: !trait.claim<@A[i130,i32]>) { %b = trait.witness @Base for @B[i32] trait.return %b : !trait.claim<@B[i32] by @Base> }
func.func @main() -> i64 {
  %a = trait.witness @I1 for @A[i1,i32]
  %b = trait.project %a[0] : !trait.claim<@A[i1,i32] by @I1> -> !trait.claim<@B[i32]>
  %v = trait.method.call %b @B[i32]::@v() : () -> i64
  return %v : i64
}
