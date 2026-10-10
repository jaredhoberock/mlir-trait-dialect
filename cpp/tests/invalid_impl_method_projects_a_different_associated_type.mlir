// RUN: mlir-opt %s -verify-diagnostics

!S = !trait.poly<0>
!F = !trait.poly<1>

trait.trait private @Fn(%self: !trait.claim<@Fn[!trait.poly<0>]>) {
  trait.assoc_type @Output
  trait.assoc_type @Other
}

trait.trait private @SameAs(%self: !trait.claim<@SameAs[!S, !F]>) {}

trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) {
  trait.method @method(
    !S,
    !trait.claim<@SameAs[
      !trait.proj<@Fn[!S], "Output">,
      !trait.proj<@Fn[!S], "Output">
    ]>
  ) -> i32
}

// expected-error @below {{method 'method' has incompatible signature}}
trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>) {
  trait.method @method(
    %self: i32,
    %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[i32], "Other">,
      !trait.proj<@Fn[i32], "Other">
    ]>
  ) -> i32 {
    %c0 = arith.constant 0 : i32
    trait.return %c0 : i32
  }
}
