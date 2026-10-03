// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not mlir-opt -split-input-file -pass-pipeline='builtin.module(resolve-impls-trait)' 2>&1 | FileCheck %s

// @a and @b cite each other, a coinductive pair, and @c1 through @c127 chain
// down to @b, so the derivation from @c1 holds @c1 ... @c127, @b and @a, one
// hundred and twenty-nine obligations, the shallowest past the limit. Followed
// from @a first, the height below @b counts @a zero because @a stands on the
// chain above it; that height holds only on chains through @a and is not kept,
// so the chain from @c1 reaches @a anew and is refused. Here the pair stands
// first.

// CHECK: error: overflow evaluating the requirement {{.*}}@A[i32]{{.*}}: 128 obligations stand on the chain that reaches it

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>, %b: !trait.claim<@B[i32]>) {}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>, %a: !trait.claim<@A[i32]>) {}
trait.proof private @a {
  %b = trait.witness @b for @B[i32]
  %d = trait.derive @A[i32] from @A_i32 given(%b) : (!trait.claim<@B[i32] by @b>)
  trait.return %d : !trait.claim<@A[i32]>
}
trait.proof private @b {
  %a = trait.witness @a for @A[i32]
  %d = trait.derive @B[i32] from @B_i32 given(%a) : (!trait.claim<@A[i32] by @a>)
  trait.return %d : !trait.claim<@B[i32]>
}
// REPEAT 1 127: trait.trait private @C{k}(%self: !trait.claim<@C{k}[!trait.poly<0>]>) {}
// REPEAT 1 126: trait.impl private @C{k}_i32(%self: !trait.claim<@C{k}[i32]>, %n: !trait.claim<@C{k+1}[i32]>) {}
trait.impl private @C127_i32(%self: !trait.claim<@C127[i32]>, %b: !trait.claim<@B[i32]>) {}
// REPEAT 1 126: trait.proof private @c{k} { %n = trait.witness @c{k+1} for @C{k+1}[i32] %d = trait.derive @C{k}[i32] from @C{k}_i32 given(%n) : (!trait.claim<@C{k+1}[i32] by @c{k+1}>) trait.return %d : !trait.claim<@C{k}[i32]> }
trait.proof private @c127 {
  %b = trait.witness @b for @B[i32]
  %d = trait.derive @C127[i32] from @C127_i32 given(%b) : (!trait.claim<@B[i32] by @b>)
  trait.return %d : !trait.claim<@C127[i32]>
}

// -----

// The same proofs with the chain standing first.

// CHECK: error: overflow evaluating the requirement {{.*}}@A[i32]{{.*}}: 128 obligations stand on the chain that reaches it

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>, %b: !trait.claim<@B[i32]>) {}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>, %a: !trait.claim<@A[i32]>) {}
// REPEAT 1 127: trait.trait private @C{k}(%self: !trait.claim<@C{k}[!trait.poly<0>]>) {}
// REPEAT 1 126: trait.impl private @C{k}_i32(%self: !trait.claim<@C{k}[i32]>, %n: !trait.claim<@C{k+1}[i32]>) {}
trait.impl private @C127_i32(%self: !trait.claim<@C127[i32]>, %b: !trait.claim<@B[i32]>) {}
// REPEAT 1 126: trait.proof private @c{k} { %n = trait.witness @c{k+1} for @C{k+1}[i32] %d = trait.derive @C{k}[i32] from @C{k}_i32 given(%n) : (!trait.claim<@C{k+1}[i32] by @c{k+1}>) trait.return %d : !trait.claim<@C{k}[i32]> }
trait.proof private @c127 {
  %b = trait.witness @b for @B[i32]
  %d = trait.derive @C127[i32] from @C127_i32 given(%b) : (!trait.claim<@B[i32] by @b>)
  trait.return %d : !trait.claim<@C127[i32]>
}
trait.proof private @a {
  %b = trait.witness @b for @B[i32]
  %d = trait.derive @A[i32] from @A_i32 given(%b) : (!trait.claim<@B[i32] by @b>)
  trait.return %d : !trait.claim<@A[i32]>
}
trait.proof private @b {
  %a = trait.witness @a for @A[i32]
  %d = trait.derive @B[i32] from @B_i32 given(%a) : (!trait.claim<@A[i32] by @a>)
  trait.return %d : !trait.claim<@B[i32]>
}
