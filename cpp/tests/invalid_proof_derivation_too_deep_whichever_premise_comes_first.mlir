// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not mlir-opt -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @r derives @R[i32] over @X[i32], by @x, and @C1[i32], by @c1; @c1 through
// @c126 chain down to @x, and @x cites @y1, which cites @Y2_i32. Through the
// chain the derivation stands 128 obligations deep at @y1, the shallowest
// past the limit, through @x directly only three. The height below a pair is
// kept wherever it is first followed and read again where the pair is reached
// deeper, so the chain is refused whichever premise the derive lists first.
// Here @X comes first.

// CHECK: error: overflow evaluating the requirement {{.*}}@Y1[i32]{{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @R(%self: !trait.claim<@R[!T]>) {}
trait.trait private @X(%self: !trait.claim<@X[!T]>) {}
trait.trait private @Y1(%self: !trait.claim<@Y1[!T]>) {}
trait.trait private @Y2(%self: !trait.claim<@Y2[!T]>) {}
// REPEAT 1 126: trait.trait private @C{k}(%self: !trait.claim<@C{k}[!trait.poly<0>]>) {}
trait.impl private @R_i32(%self: !trait.claim<@R[i32]>, %x: !trait.claim<@X[i32]>, %c: !trait.claim<@C1[i32]>) {}
trait.impl private @X_i32(%self: !trait.claim<@X[i32]>, %y: !trait.claim<@Y1[i32]>) {}
trait.impl private @Y1_i32(%self: !trait.claim<@Y1[i32]>, %y: !trait.claim<@Y2[i32]>) {}
trait.impl private @Y2_i32(%self: !trait.claim<@Y2[i32]>) {}
// REPEAT 1 125: trait.impl private @C{k}_i32(%self: !trait.claim<@C{k}[i32]>, %n: !trait.claim<@C{k+1}[i32]>) {}
trait.impl private @C126_i32(%self: !trait.claim<@C126[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.proof private @r {
  %x = trait.witness @x for @X[i32]
  %c = trait.witness @c1 for @C1[i32]
  %d = trait.derive @R[i32] from @R_i32 given(%x, %c) : (!trait.claim<@X[i32] by @x>, !trait.claim<@C1[i32] by @c1>)
  trait.return %d : !trait.claim<@R[i32]>
}
trait.proof private @x {
  %y = trait.witness @y1 for @Y1[i32]
  %d = trait.derive @X[i32] from @X_i32 given(%y) : (!trait.claim<@Y1[i32] by @y1>)
  trait.return %d : !trait.claim<@X[i32]>
}
trait.proof private @y1 {
  %y = trait.witness @Y2_i32 for @Y2[i32]
  %d = trait.derive @Y1[i32] from @Y1_i32 given(%y) : (!trait.claim<@Y2[i32] by @Y2_i32>)
  trait.return %d : !trait.claim<@Y1[i32]>
}
// REPEAT 1 125: trait.proof private @c{k} { %n = trait.witness @c{k+1} for @C{k+1}[i32] %d = trait.derive @C{k}[i32] from @C{k}_i32 given(%n) : (!trait.claim<@C{k+1}[i32] by @c{k+1}>) trait.return %d : !trait.claim<@C{k}[i32]> }
trait.proof private @c126 {
  %x = trait.witness @x for @X[i32]
  %d = trait.derive @C126[i32] from @C126_i32 given(%x) : (!trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@C126[i32]>
}

// -----

// The same derivation with @C1 listed first.

// CHECK: error: overflow evaluating the requirement {{.*}}@Y1[i32]{{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @R(%self: !trait.claim<@R[!T]>) {}
trait.trait private @X(%self: !trait.claim<@X[!T]>) {}
trait.trait private @Y1(%self: !trait.claim<@Y1[!T]>) {}
trait.trait private @Y2(%self: !trait.claim<@Y2[!T]>) {}
// REPEAT 1 126: trait.trait private @C{k}(%self: !trait.claim<@C{k}[!trait.poly<0>]>) {}
trait.impl private @R_i32(%self: !trait.claim<@R[i32]>, %c: !trait.claim<@C1[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.impl private @X_i32(%self: !trait.claim<@X[i32]>, %y: !trait.claim<@Y1[i32]>) {}
trait.impl private @Y1_i32(%self: !trait.claim<@Y1[i32]>, %y: !trait.claim<@Y2[i32]>) {}
trait.impl private @Y2_i32(%self: !trait.claim<@Y2[i32]>) {}
// REPEAT 1 125: trait.impl private @C{k}_i32(%self: !trait.claim<@C{k}[i32]>, %n: !trait.claim<@C{k+1}[i32]>) {}
trait.impl private @C126_i32(%self: !trait.claim<@C126[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.proof private @r {
  %x = trait.witness @x for @X[i32]
  %c = trait.witness @c1 for @C1[i32]
  %d = trait.derive @R[i32] from @R_i32 given(%c, %x) : (!trait.claim<@C1[i32] by @c1>, !trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@R[i32]>
}
trait.proof private @x {
  %y = trait.witness @y1 for @Y1[i32]
  %d = trait.derive @X[i32] from @X_i32 given(%y) : (!trait.claim<@Y1[i32] by @y1>)
  trait.return %d : !trait.claim<@X[i32]>
}
trait.proof private @y1 {
  %y = trait.witness @Y2_i32 for @Y2[i32]
  %d = trait.derive @Y1[i32] from @Y1_i32 given(%y) : (!trait.claim<@Y2[i32] by @Y2_i32>)
  trait.return %d : !trait.claim<@Y1[i32]>
}
// REPEAT 1 125: trait.proof private @c{k} { %n = trait.witness @c{k+1} for @C{k+1}[i32] %d = trait.derive @C{k}[i32] from @C{k}_i32 given(%n) : (!trait.claim<@C{k+1}[i32] by @c{k+1}>) trait.return %d : !trait.claim<@C{k}[i32]> }
trait.proof private @c126 {
  %x = trait.witness @x for @X[i32]
  %d = trait.derive @C126[i32] from @C126_i32 given(%x) : (!trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@C126[i32]>
}
