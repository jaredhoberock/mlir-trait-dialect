// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// A projected claim argument should retain enough proof information during
// monomorphization for method calls through that claim to lower. The call to
// @callee passes @Trait[i64, i64] through a trait.proj.cast to match the
// formal @Trait[T, Outer[i64]::Item] claim. Instantiating @callee should still
// lower the call to @Trait::@method.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @Trait[!S, !T] {
  func.func private @method(!S, !T) -> i64
}

trait.trait @Outer[!S] {
  trait.assoc_type @Item
}

func.func private @callee(%t: !T,
    %outer: !trait.claim<@Outer[i64]>,
    %claim: !trait.claim<@Trait[!T, !trait.proj<@Outer[i64], "Item">]>) -> i64 {
  %x = arith.constant 1 : i64
  %px = trait.proj.cast %x, %outer
    : i64 to !trait.proj<@Outer[i64], "Item">
    by !trait.claim<@Outer[i64]>
  %result = trait.method.call %claim @Trait[!T, !trait.proj<@Outer[i64], "Item">]::@method(%t, %px)
    : (!T, !trait.proj<@Outer[i64], "Item">) -> i64
  return %result : i64
}

trait.impl @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.impl @Trait_i64 for @Trait[i64, i64] {
  func.func @method(%self: i64, %x: i64) -> i64 {
    return %x : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.method.call
// CHECK-NOT: !trait.claim
// CHECK: call @callee
// CHECK: return
func.func @main() -> i64 {
  %outer = trait.witness @Outer_i64 for @Outer[i64]
  %trait = trait.witness @Trait_i64 for @Trait[i64, i64]
  %projected = trait.proj.cast %trait, %outer
    : !trait.claim<@Trait[i64, i64] by @Trait_i64>
    to !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>
    by !trait.claim<@Outer[i64] by @Outer_i64>
  %x = arith.constant 0 : i64
  %result = trait.func.call @callee(%x, %outer, %projected)
    : (i64, !trait.claim<@Outer[i64] by @Outer_i64>,
       !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>) -> i64
  return %result : i64
}
