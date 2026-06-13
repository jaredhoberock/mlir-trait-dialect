// RUN: mlir-opt %s -verify-diagnostics

!T = !trait.poly<0>

trait.trait @Trait[!T] {
  trait.assoc_type @Output
}

trait.impl @Trait_i32 for @Trait[i32] {
  trait.assoc_type @Output = tuple<!trait.proj<@Trait[i32], "Output">>
}

trait.proof @Proof proves @Trait_i32 for @Trait[i32] given []

func.func @cast(%value: !trait.proj<@Trait[i32], "Output">) {
  %claim = trait.witness @Proof for @Trait[i32]
  // expected-error @below {{projection normalization did not converge}}
  %cast = trait.proj.cast %value, %claim
    : !trait.proj<@Trait[i32], "Output"> to !trait.proj<@Trait[i32], "Output">
    by !trait.claim<@Trait[i32] by @Proof>
  return
}
