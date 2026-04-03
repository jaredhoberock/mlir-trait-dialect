// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Verify that trait.func.call accepts a proven claim whose type args
// are projections from different type params. The call passes
// D[A[i32]::Out, A[f32]::Out] (proven) to a callee expecting
// D[A[poly<4>]::Out, A[poly<5>]::Out]. The verifier must unify
// poly<4>=i32 and poly<5>=f32 independently.

// CHECK-LABEL: func.func @main
// CHECK: return
module {
  trait.trait @D[!trait.poly<0>, !trait.poly<1>] {}
  trait.trait @A[!trait.poly<2>] { trait.assoc_type @Out }
  trait.impl @D_impl for @D[!trait.poly<3>, !trait.poly<3>] {}
  trait.impl @A_i32 for @A[i32] { trait.assoc_type @Out = i64 }
  trait.impl @A_f32 for @A[f32] { trait.assoc_type @Out = i64 }
  trait.proof @A_p proves @A_i32 for @A[i32] given []
  trait.proof @D_p proves @D_impl for @D[i64, i64] given []

  func.func nested @f(%x: !trait.poly<4>, %y: !trait.poly<5>,
    %d: !trait.claim<@D[!trait.proj<@A[!trait.poly<4>], "Out">, !trait.proj<@A[!trait.poly<5>], "Out">]>
  ) -> i32 { %0 = arith.constant 0 : i32 return %0 : i32 }

  func.func @main() -> i32 {
    %x = arith.constant 0 : i32
    %y = arith.constant 0.0 : f32
    %ev = trait.witness @A_p for @A[i32]
    %d = trait.witness @D_p for @D[i64, i64]
    %d1 = trait.proj.cast %d, %ev
      : !trait.claim<@D[i64, i64] by @D_p>
      to !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>
      by !trait.claim<@A[i32] by @A_p>
    // This fails: proven claim with projections from different type args
    %r = trait.func.call @f(%x, %y, %d1)
      : (i32, f32, !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>)
      -> i32
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
