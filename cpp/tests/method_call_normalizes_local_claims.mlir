// RUN: mlir-opt %s | FileCheck %s

!T = !trait.poly<0>
!U = !trait.poly<1>
!Output = !trait.proj<@Value[i64], "Output">

trait.trait @Value[!T] {
  trait.assoc_type @Output
}

trait.impl @Value_i64 for @Value[i64] {
  trait.assoc_type @Output = i64
}

trait.trait @Trait[!T] {
  trait.assoc_type @First
  trait.assoc_type @Second
  func.func private @get(!T) -> !trait.proj<@Trait[!T], "First">
}

trait.impl @Trait_impl for @Trait[!T] {
  trait.assoc_type @First = !trait.proj<@Trait[!T], "Second">
  trait.assoc_type @Second = !Output
  func.func @get(%self: !T) -> !Output {
    %result = ub.poison : !Output
    return %result : !Output
  }
}

trait.trait @Fn[!T, !U] {
  trait.assoc_type @Output
}

trait.trait @FnUni[!T, !U] {
}

trait.trait @Map[!T] {
  func.func private @map(
    !T,
    !U,
    !trait.claim<@FnUni[!U, !T]>
  ) -> !trait.proj<@Fn[!U, !T], "Output">
}

trait.impl @Fn_i1_i64 for @Fn[i1, i64] {
  trait.assoc_type @Output = !Output
}

trait.impl @FnUni_i1_i64 for @FnUni[i1, i64] where [@Fn[i1, i64]] {
}

// CHECK-LABEL: func.func @method_result_normalizes_chained_bindings
// CHECK: trait.method.call
func.func @method_result_normalizes_chained_bindings(%value: i64) -> !Output {
  %claim = trait.derive @Trait[i64] from @Trait_impl given()
  %result = trait.method.call %claim @Trait[i64]::@get(%value)
    : (i64) -> !Output
  return %result : !Output
}

// CHECK-LABEL: func.func @method_result_normalizes_after_binding_input_generics
// CHECK: trait.method.call
func.func @method_result_normalizes_after_binding_input_generics(%value: i64) -> !Output {
  %map = trait.allege @Map[i64] unsafe
  %fn = trait.witness @Fn_i1_i64 for @Fn[i1, i64]
  %fn_uni = trait.derive @FnUni[i1, i64] from @FnUni_i1_i64 given(%fn)
    : (!trait.claim<@Fn[i1, i64] by @Fn_i1_i64>)
  %f = arith.constant false
  %result = trait.method.call %map @Map[i64]::@map(%value, %f, %fn_uni)
    : (i64, i1, !trait.claim<@FnUni[i1, i64]>) -> !Output
  return %result : !Output
}
