// RUN: mlir-opt %s -verify-diagnostics

!S = !trait.poly<0>
!F = !trait.poly<1>

trait.trait @Fn[!F] {
  trait.assoc_type @Output
  trait.assoc_type @Other
}

trait.trait @SameAs[!S, !F] {}

trait.trait @Trait[!S] {
  func.func private @method(
    !S,
    !trait.claim<@SameAs[
      !trait.proj<@Fn[!S], "Output">,
      !trait.proj<@Fn[!S], "Output">
    ]>
  ) -> i32
}

// expected-error @below {{projection mismatch}}
// expected-error @below {{method 'method' has incompatible signature}}
trait.impl @Trait_i32 for @Trait[i32] {
  func.func @method(
    %self: i32,
    %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[i32], "Other">,
      !trait.proj<@Fn[i32], "Other">
    ]>
  ) -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
