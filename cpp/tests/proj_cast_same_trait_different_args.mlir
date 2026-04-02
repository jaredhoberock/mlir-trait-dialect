// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Bug: proj_cast's proven-claim verifier resolves ALL projections
// whose associated type name matches the impl, ignoring type
// arguments. When a type contains A[i32]::Out and A[i64]::Out,
// evidence for A[i32] incorrectly resolves both.
//
// The proj_cast below should pass verification: the verifier
// resolves A[i32]::Out to f32, leaves A[i64]::Out unresolved
// (different type args), and defers. Instead it resolves BOTH
// to f32 and compares f32 != f64, producing a spurious error.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @A[!S] {
  trait.assoc_type @Out
}

trait.trait @Pair[!S, !T] {
}

trait.impl @A_i32 for @A[i32] {
  trait.assoc_type @Out = f32
}

trait.impl @A_i64 for @A[i64] {
  trait.assoc_type @Out = f64
}

trait.proof @A_i32_proof proves @A_i32 for @A[i32] given []

// CHECK-LABEL: func.func @main
// CHECK: return
func.func @main() -> i32 {
  %claim = trait.allege @Pair[
    !trait.proj<@A[i32], "Out">,
    !trait.proj<@A[i64], "Out">
  ]

  %a_i32 = trait.witness @A_i32_proof for @A[i32]

  // This should pass: only A[i32]::Out should be resolved.
  // A[i64]::Out should be left alone (different type args).
  %cast = trait.proj.cast %claim, %a_i32
    : !trait.claim<@Pair[!trait.proj<@A[i32], "Out">, !trait.proj<@A[i64], "Out">]>
    to !trait.claim<@Pair[f32, f64]>
    by !trait.claim<@A[i32] by @A_i32_proof>

  %c0 = arith.constant 0 : i32
  return %c0 : i32
}
