// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// A call can carry the same logical obligation along two paths:
//
//   * as a requirement of another proof
//   * as an explicit where-clause argument whose type preserves a projection
//
// The verifier should reject genuinely incoherent proofs, but it must accept
// equivalent proven claim spellings when the same proof symbol independently
// specializes to both forms.

module {
  trait.trait @T0[!trait.poly<0>] {
    trait.assoc_type @A
  }

  trait.trait @T1[!trait.poly<1>] {}

  trait.trait @T2[!trait.poly<2>] where [@T1[!trait.proj<@T0[!trait.poly<2>], "A">]] {}

  trait.impl @T1_i64 for @T1[i64] {}

  trait.impl @T0_i64 for @T0[i64] {
    trait.assoc_type @A = i64
  }

  trait.impl @T2_i64 for @T2[i64] {}

  trait.proof @T2_i64_p proves @T2_i64 for @T2[i64] given [@T1_i64]

  func.func @f(
    %x: !trait.poly<3>,
    %t2: !trait.claim<@T2[!trait.poly<3>]>,
    %t1: !trait.claim<@T1[!trait.proj<@T0[!trait.poly<3>], "A">]>
  ) -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }

  // CHECK-LABEL: func.func @main
  // CHECK-NOT: trait.func.call
  // CHECK: return
  func.func @main() -> i32 {
    %x = arith.constant 1 : i64
    %t2 = trait.witness @T2_i64_p for @T2[i64]
    %t1 = trait.witness @T1_i64 for @T1[i64]
    %t0 = trait.witness @T0_i64 for @T0[i64]
    %t1_projected = trait.proj.cast %t1, %t0
      : !trait.claim<@T1[i64] by @T1_i64>
      to !trait.claim<@T1[!trait.proj<@T0[i64], "A">] by @T1_i64>
      by !trait.claim<@T0[i64] by @T0_i64>
    %r = trait.func.call @f(%x, %t2, %t1_projected)
      : (i64,
         !trait.claim<@T2[i64] by @T2_i64_p>,
         !trait.claim<@T1[!trait.proj<@T0[i64], "A">] by @T1_i64>)
      -> i32
    return %r : i32
  }
}
