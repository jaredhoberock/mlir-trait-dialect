// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T0 = !trait.poly<0>
// CHECK-NOT: trait.trait @A
trait.trait @A [!T0] {
  func.func private @method_a(!T0) -> i32
}

!T1 = !trait.poly<1>
// CHECK-NOT: trait.trait @B
trait.trait @B [!T1] {
  func.func private @method_b(!T1) -> i32
}

// CHECK-NOT: trait.impl for @B[i32]
trait.impl for @B[i32] {
  func.func @method_b(%arg0: i32) -> i32 {
    %res = arith.constant 1 : i32
    return %res : i32
  }
}

// CHECK-NOT: trait.impl for @B[i8]
trait.impl for @B[i8] {
  func.func @method_b(%arg: i8) -> i32 {
    %res = arith.constant 1 : i32
    return %res : i32
  }
}


!T2 = !trait.poly<2>
// CHECK-NOT: trait.impl @A_impl_poly
trait.impl @A_impl_poly for @A[!T2] where [@B[!T2]] {
  func.func @method_a(%arg0: !T2) -> i32 {
    %0 = trait.assume @B[!T2]
    %1 = trait.method.call @B::@method_b<%0> (%arg0) : (!T1) -> i32 as !trait.claim<@B[!T2]> (!T2) -> i32
    return %1 : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @test() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %p_i32 = trait.allege @A[i32]
  // CHECK: call @A_impl_poly_i32_method_a
  %res0 = trait.method.call @A::@method_a<%p_i32> (%c42_i32) : (!T0) -> i32 as !trait.claim<@A[i32]> (i32) -> i32

  %c7_i8 = arith.constant 7 : i8
  %p_i8 = trait.allege @A[i8]
  // CHECK: call @A_impl_poly_i8_method_a
  %res1 = trait.method.call @A::@method_a<%p_i8> (%c7_i8) : (!T0) -> i32 as !trait.claim<@A[i8]> (i8) -> i32

  %res = arith.addi %res0, %res1 : i32

  return %res : i32
}

// CHECK-NOT: trait.proof
