// RUN: mlir-opt %s -verify-diagnostics

!T = !trait.poly<0>
!X = !trait.proj<@Broad[i64], "Output">

trait.trait private @Unwrap[!T] {
  trait.assoc_type @Output
  func.func private @unwrap(!T) -> !trait.proj<@Unwrap[!T], "Output">
}

trait.trait private @Broad[!T] {
  trait.assoc_type @Output
}

trait.impl private @Broad_i64 for @Broad[i64] {
  trait.assoc_type @Output = i64
}

trait.impl private @Unwrap_i64 for @Unwrap[i64] {
  trait.assoc_type @Output = !X
  func.func @unwrap(%self: i64) -> !X {
    %result = ub.poison : !X
    return %result : !X
  }
}

func.func @unproven_claim_does_not_normalize(
    %claim: !trait.claim<@Unwrap[!X]>,
    %value: !X) -> !X {
  // expected-error @below {{type mismatch: expected '(!trait.proj<@Broad[i64], "Output">) -> !trait.proj<@Unwrap[!trait.proj<@Broad[i64], "Output">], "Output">' but found '(!trait.proj<@Broad[i64], "Output">) -> !trait.proj<@Broad[i64], "Output">'}}
  %result = trait.method.call %claim @Unwrap[!X]::@unwrap(%value)
    : (!X) -> !X
  return %result : !X
}
