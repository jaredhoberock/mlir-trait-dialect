// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The derive's premise rests on @pb, a root that proves @Bound at the spelling
// @Trait[i64]::Output rather than at i64, the spelling the derive's where entry
// resolves to. The proof the stage writes for the derive names the root and
// respells it to the entry in its own body, as every witness in a proof's body
// names a root, and the method the derive's impl calls through that premise
// runs @Bound_i64's. The row pins a shape, not a change: the stage ran it as
// well before transcription cited a premise's root.

// CHECK: {{^}}5{{$}}

!S = !trait.poly<0>
trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) {
  trait.assoc_type @Output
}
trait.trait private @Bound(%self: !trait.claim<@Bound[!S]>) { trait.method @v() -> i64 }
trait.impl private @Trait_impl(%self: !trait.claim<@Trait[i64]>) {
  trait.assoc_type @Output = i64
}
trait.impl private @Bound_i64(%self: !trait.claim<@Bound[i64]>) {
  trait.method @v() -> i64 {
    %v = arith.constant 5 : i64
    trait.return %v : i64
  }
}
trait.proof private @pb {
  %d = trait.derive @Bound[i64] from @Bound_i64 given()
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %d : !trait.claim<@Bound[i64]> to !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]> via (%e) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  trait.return %c : !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>
}
trait.trait private @Need(%self: !trait.claim<@Need[!S]>) { trait.method @w() -> i64 }
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) {
  trait.method @w() -> i64 {
    %r = trait.method.call %b @Bound[!S]::@v() : () -> i64
    trait.return %r : i64
  }
}
func.func @main() -> i64 {
  %b = trait.witness @pb for @Bound[!trait.proj<@Trait[i64], "Output">]
  %eq = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">] by @pb> to !trait.claim<@Bound[i64]> via (%eq) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  %d = trait.derive @Need[i64] from @Need_from[i64] given(%c) : (!trait.claim<@Bound[i64]>)
  %r = trait.method.call %d @Need[i64]::@w() : () -> i64
  return %r : i64
}
