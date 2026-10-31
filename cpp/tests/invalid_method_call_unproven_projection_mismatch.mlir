// RUN: mlir-opt %s -verify-diagnostics

!T = !trait.poly<0>
!X = !trait.proj<@Broad[i64], "Output">

trait.trait private @Unwrap(!trait.claim<@Unwrap[!T]>) assoc {Output = 0} members [@Unwrap.unwrap]
func.func private @Unwrap.unwrap(!trait.claim<@Unwrap[!T]>, !T) -> !trait.proj<@Unwrap[!T], "Output">

trait.trait private @Broad(!trait.claim<@Broad[!T]>) assoc {Output = 0}

trait.impl private @Broad_i64(%self: !trait.claim<@Broad[i64]>) bindings {Output = i64} {
}

trait.impl private @Unwrap_i64(%self_claim: !trait.claim<@Unwrap[i64]>) bindings {Output = !X} members [@Unwrap.unwrap -> @Unwrap_i64.unwrap] {
}
func.func private @Unwrap_i64.unwrap(%self_claim: !trait.claim<@Unwrap[i64]>, %self: i64) -> !X {
  %result = ub.poison : !X
  return %result : !X
}

func.func @unproven_claim_does_not_normalize(
    %claim: !trait.claim<@Unwrap[!X]>,
    %value: !X) -> !X {
  // expected-error @below {{type mismatch: expected '(!trait.claim<@Unwrap[!trait.proj<@Broad[i64], "Output">]>, !trait.proj<@Broad[i64], "Output">) -> !trait.proj<@Unwrap[!trait.proj<@Broad[i64], "Output">], "Output">' but found '(!trait.claim<@Unwrap[!trait.proj<@Broad[i64], "Output">]>, !trait.proj<@Broad[i64], "Output">) -> !trait.proj<@Broad[i64], "Output">'}}
  %result = trait.func.call @Unwrap.unwrap(%claim, %value) : (!trait.claim<@Unwrap[!X]>, !X) -> !X
  return %result : !X
}
