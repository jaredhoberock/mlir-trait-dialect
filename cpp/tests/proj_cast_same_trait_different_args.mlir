// RUN: mlir-opt %s -verify-diagnostics

// A proven proj.cast resolves only projections over its claim's exact trait
// application, type arguments included: evidence for @A[i32] resolves
// @A[i32]::Out but not @A[i64]::Out. Here the cast names an @A[i32] claim yet
// spells both @A[i32]::Out and @A[i64]::Out in its input. @A[i32]::Out resolves
// to f32, but @A[i64]::Out is a projection over a different application and
// survives; it no longer matches the f64 the result carries there, so the cast
// is rejected rather than resolving @A[i64]::Out with the wrong evidence.

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

func.func @main() -> i32 {
  %claim = trait.allege @Pair[
    !trait.proj<@A[i32], "Out">,
    !trait.proj<@A[i64], "Out">
  ]

  %a_i32 = trait.witness @A_i32_proof for @A[i32]

  // expected-error @below {{does not match resolved result type}}
  %cast = trait.proj.cast %claim, %a_i32
    : !trait.claim<@Pair[!trait.proj<@A[i32], "Out">, !trait.proj<@A[i64], "Out">]>
    to !trait.claim<@Pair[f32, f64]>
    by !trait.claim<@A[i32] by @A_i32_proof>

  %c0 = arith.constant 0 : i32
  return %c0 : i32
}
