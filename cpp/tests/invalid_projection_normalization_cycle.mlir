// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics

!T = !trait.poly<0>

trait.trait private @Trait(!trait.claim<@Trait[!T]>) assoc {Output = 0} members [@Trait.method]
func.func private @Trait.method(!trait.claim<@Trait[!T]>, !T, !trait.proj<@Trait[!T], "Output">) -> !trait.proj<@Trait[!T], "Output">

// expected-error @below {{projection normalization did not converge}}
trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>) bindings {Output = tuple<!trait.proj<@Trait[i32], "Output">>} members [@Trait.method -> @Trait_i32.method] {
}
func.func private @Trait_i32.method(%self_claim: !trait.claim<@Trait[i32]>, %self: i32, %value: !trait.proj<@Trait[i32], "Output">) -> !trait.proj<@Trait[i32], "Output"> {
  return %value : !trait.proj<@Trait[i32], "Output">
}
