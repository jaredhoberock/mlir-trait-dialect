// RUN: mlir-opt %s -verify-diagnostics

!T = !trait.poly<0>

trait.trait @Trait[!T] {
  trait.assoc_type @Output
  func.func private @method(
    !T,
    !trait.proj<@Trait[!T], "Output">
  ) -> !trait.proj<@Trait[!T], "Output">
}

// expected-error @below {{projection normalization did not converge}}
trait.impl @Trait_i32 for @Trait[i32] {
  trait.assoc_type @Output = tuple<!trait.proj<@Trait[i32], "Output">>
  func.func @method(
      %self: i32,
      %value: !trait.proj<@Trait[i32], "Output">
  ) -> !trait.proj<@Trait[i32], "Output"> {
    return %value : !trait.proj<@Trait[i32], "Output">
  }
}
