// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T = !trait.poly<0>
trait.trait @Get[!T] {
  // method returns the trait's type parameter
  func.func private @get() -> !T
}

trait.impl for @Get[i32] {
  func.func @get() -> i32 {
    %c = arith.constant 0 : i32
    return %c : i32
  }
}

// The method's formal return is !T, but the call site writes its actual result as !R.
// Monomorphization must unify and ground !R := i32.
!A = !trait.poly<1>
!R = !trait.poly<2>
func.func @return_existential_type(%claim: !trait.claim<@Get[!A]>) -> !R {
  %res = trait.method.call %claim @Get[!A]::@get()
    :  () -> !T
    as () -> !R
  return %res : !R
}

func.func @bar() -> i32 {
  %a = trait.allege @Get[i32]
  %res = trait.func.call @return_existential_type(%a)
    :  (!trait.claim<@Get[!A]>) -> !R
    as (!trait.claim<@Get[i32]>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @bar() -> i32
// CHECK:       return {{.*}} : i32
// CHECK-NOT:   trait.trait
// CHECK-NOT:   trait.impl
// CHECK-NOT:   trait.func.call
// CHECK-NOT:   trait.method.call
// CHECK-NOT:   trait.allege
// CHECK-NOT:   trait.assume
// CHECK-NOT:   trait.witness
// CHECK-NOT:   trait.project
// CHECK-NOT:   !trait.poly<
