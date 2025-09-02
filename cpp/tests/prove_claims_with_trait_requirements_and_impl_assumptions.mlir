// RUN: mlir-opt -pass-pipeline='builtin.module(prove-claims-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK: trait.trait @
trait.trait @A[!A] {}

!Ai = !trait.poly<1>
// CHECK: trait.impl @A_impl
trait.impl @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK: trait.trait @B
trait.trait @B[!B] {}

!Bi = !trait.poly<3>
// CHECK: trait.impl @B_impl
trait.impl @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK: trait.trait @C
trait.trait @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK: trait.impl @C_impl
trait.impl @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

func.func @foo(%x: i8) -> i1 {
  // CHECK: trait.witness @C_impl_i8_p for @C[i8]
  %c = trait.allege @C[i8]
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK: trait.proof @A_impl_i8_p proves @A_impl for @A[i8] given []
// CHECK: trait.proof @B_impl_i8_p proves @B_impl for @B[i8] given []
// CHECK: trait.proof @C_impl_i8_p proves @C_impl for @C[i8] given [@A_impl_i8_p, @B_impl_i8_p]
