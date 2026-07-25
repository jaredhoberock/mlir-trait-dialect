// RUN: mlir-opt %s -verify-diagnostics

// A trait.proj.cast is justified only for projections over its claim's own trait
// application. Here the cast unresolves @D[i64, i64] into
// @D[@A[i32]::Out, @A[f32]::Out] but names only an @A[i32] claim. That claim
// resolves @A[i32]::Out to i64, but it says nothing about @A[f32]::Out, which is
// a projection over a different application. The @A[f32]::Out spelling survives
// resolution and no longer matches the i64 the input carries there, so the cast
// is rejected: one claim does not justify a projection over another application.

module {
  trait.trait @D[!trait.poly<0>, !trait.poly<1>] {}
  trait.trait @A[!trait.poly<2>] { trait.assoc_type @Out }
  trait.impl @D_impl for @D[!trait.poly<3>, !trait.poly<3>] {}
  trait.impl @A_i32 for @A[i32] { trait.assoc_type @Out = i64 }
  trait.impl @A_f32 for @A[f32] { trait.assoc_type @Out = i64 }
  trait.proof @A_p proves @A_i32 for @A[i32] given []
  trait.proof @D_p proves @D_impl for @D[i64, i64] given []

  func.func @main() -> i32 {
    %d = trait.witness @D_p for @D[i64, i64]
    %ev = trait.witness @A_p for @A[i32]
    // expected-error @below {{does not match resolved result type}}
    %d1 = trait.proj.cast %d, %ev
      : !trait.claim<@D[i64, i64] by @D_p>
      to !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>
      by !trait.claim<@A[i32] by @A_p>
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
