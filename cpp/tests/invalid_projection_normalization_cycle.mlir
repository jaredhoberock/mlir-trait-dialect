// RUN: mlir-opt %s -verify-diagnostics

!T = !trait.poly<0>

trait.trait private @Trait(%self: !trait.claim<@Trait[!T]>) {
  trait.assoc_type @Output
  trait.method @method(
    !T,
    !trait.proj<@Trait[!T], "Output">
  ) -> !trait.proj<@Trait[!T], "Output">
}

// expected-error @below {{projection normalization did not converge}}
trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>) {
  trait.assoc_type @Output = tuple<!trait.proj<@Trait[i32], "Output">>
  trait.method @method(
      %self: i32,
      %value: !trait.proj<@Trait[i32], "Output">
  ) -> !trait.proj<@Trait[i32], "Output"> {
    trait.return %value : !trait.proj<@Trait[i32], "Output">
  }
}
