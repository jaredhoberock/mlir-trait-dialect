// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @Trait requires itself at its associated type and @Leaf, so a proof of @I
// discharges the first requirement by a proof of @I again: @P7 cites itself
// and @Seven, @P9 cites itself and @Nine. @f derives @Other from @W given its
// parameter, which the call supplies by @P9, while @PW discharges @W's entry
// by @P7. Compared entry by entry, the two proofs meet their own cycle again
// and differ at @Leaf, so they are two pieces of evidence: the comparison
// ends, and the derive is refused rather than run through @P7.

// CHECK: error: 'trait.derive' op is given '!trait.claim<@Trait[i32] by @P9>' at where-clause entry 0, which @PW discharges by @P7 instead
// CHECK-NOT: error:

!T = !trait.poly<0>
trait.trait private @Leaf[!T] { trait.method @value() -> i64 }
trait.impl private @Seven for @Leaf[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Leaf[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Trait[!T] where [@Trait[!trait.proj<@Trait[!T], "Sub">], @Leaf[!T]] {
  trait.assoc_type @Sub
}
trait.impl private @I for @Trait[i32] {
  trait.assoc_type @Sub = i32
}
trait.proof private @P7 proves @I[] for @Trait[i32] given [@P7, @Seven]
trait.proof private @P9 proves @I[] for @Trait[i32] given [@P9, @Nine]
trait.trait private @Other[!T] { trait.method @method() -> i64 }
trait.impl private @W for @Other[!T] where [@Trait[!T]] {
  trait.method @method() -> i64 {
    %t = trait.assume 0 : !trait.claim<@Trait[!T]>
    %l = trait.project %t[1] : !trait.claim<@Trait[!T]> -> !trait.claim<@Leaf[!T]>
    %v = trait.method.call %l @Leaf[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @PW proves @W[!T = i32] for @Other[i32] given [@P7]
func.func private @f(%p: !trait.claim<@Trait[!T]>) -> i64 {
  %d = trait.derive @Other[!T] from @W[!T = !T] given(%p) : (!trait.claim<@Trait[!T]>)
  %v = trait.method.call %d @Other[!T]::@method() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %p = trait.witness @P9 for @Trait[i32]
  %v = trait.func.call @f(%p) : (!trait.claim<@Trait[i32] by @P9>) -> i64
  return %v : i64
}
