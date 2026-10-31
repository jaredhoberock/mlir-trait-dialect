// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics

!S = !trait.poly<0>
!F = !trait.poly<1>

trait.trait private @Fn(!trait.claim<@Fn[!trait.poly<0>]>) assoc {Output = 0, Other = 0}

trait.trait private @SameAs(!trait.claim<@SameAs[!S, !F]>)

trait.trait private @Trait(!trait.claim<@Trait[!S]>) members [@Trait.method]
func.func private @Trait.method(!trait.claim<@Trait[!S]>, !S, !trait.claim<@SameAs[
      !trait.proj<@Fn[!S], "Output">,
      !trait.proj<@Fn[!S], "Output">
    ]>) -> i32

// expected-error @below {{member @Trait.method has incompatible signature}}
trait.impl private @Trait_i32(%self_claim: !trait.claim<@Trait[i32]>) members [@Trait.method -> @Trait_i32.method] {
}
func.func private @Trait_i32.method(%self_claim: !trait.claim<@Trait[i32]>, %self: i32, %same: !trait.claim<@SameAs[
      !trait.proj<@Fn[i32], "Other">,
      !trait.proj<@Fn[i32], "Other">
    ]>) -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}
