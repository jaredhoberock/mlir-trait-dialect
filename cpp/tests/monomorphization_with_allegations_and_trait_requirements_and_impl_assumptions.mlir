// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK-NOT: trait.trait @A
trait.trait @A[!A] {}

!Ai = !trait.poly<1>
// CHECK-NOT: trait.impl @A_impl
trait.impl @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK-NOT: trait.trait @B
trait.trait @B[!B] {}

!Bi = !trait.poly<3>
// CHECK-NOT: trait.impl @B_impl
trait.impl @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK-NOT: trait.trait @C
trait.trait @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK-NOT: trait.impl @C_impl
trait.impl @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

// CHECK-LABEL: func.func @foo
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @foo(%x: i8) -> i1 {
  %c = trait.allege @C[i8]
  // CHECK: call @C_impl_i8_method
  %res = trait.method.call %c @C[i8]::@method(%x)
    :  (!C) -> i1
    as (i8) -> i1
  return %res : i1
}

// CHECK-NOT: trait.proof
