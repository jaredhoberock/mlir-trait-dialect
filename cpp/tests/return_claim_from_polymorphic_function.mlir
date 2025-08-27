// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// this trait returns some type from get
!R = !trait.poly<0>
trait.trait @Get[!R] {
  func.func private @get() -> !R
}

// this trait will be used in an impl where below
!A = !trait.poly<1>
trait.trait @Assumption[!A] {}

// a blanket impl for @Assumption for all types
trait.impl for @Assumption[!A] {}

// this impl returns an assumption claim from @get
trait.impl for @Get[!trait.claim<@Assumption[i32]>] where [
  @Assumption[i32]
] {
  func.func @get() -> !trait.claim<@Assumption[i32]> {
    %res = trait.assume @Assumption[i32]
    return %res : !trait.claim<@Assumption[i32]>
  }
}

// this polymorphic function calls get and returns its result
// CHECK-LABEL: func.func @"call_get_
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @call_get(%c: !trait.claim<@Get[!R]>) -> !R {
  %res = trait.method.call %c @Get[!R]::@get()
    :  () -> !R
    as () -> !R
  return %res : !R
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
// test that we can call a polymorphic function that returns a claim
func.func @test() {
  // allege an impl for @Get exists which returns this type of claim
  %a = trait.allege @Get[!trait.claim<@Assumption[i32]>]

  // call a polymorphic function that returns the !trait.claim
  trait.func.call @call_get(%a)
    :  (!trait.claim<@Get[!R]>) -> !R
    as (!trait.claim<@Get[!trait.claim<@Assumption[i32]>]>) -> !trait.claim<@Assumption[i32]>

  return
}

// CHECK-NOT: trait.assume
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.project
// CHECK-NOT: trait.proof
// CHECK-NOT: trait.trait
