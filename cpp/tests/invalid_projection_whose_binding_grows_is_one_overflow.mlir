// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | mlir-opt -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// @Hg binds @H[T]::E to @H[tuple<T>]::E, so @H[i32]::E grows at every
// resolution and has no normal form: past the depth limit's worth of
// projection steps it is an overflow. Whatever reaches it -- an allegation
// spelling it, an impl's where equality selection checks, an equality alleged
// -- names it once, and the stage stops there.

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E = !trait.proj<@H[tuple<!T>], "E"> }
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.method @value() -> i64 }
trait.impl private @I2(%s: !trait.claim<@A[i64]>) { trait.method @value() -> i64 { %c = arith.constant 9 : i64 trait.return %c : i64 } }
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<@A[!trait.proj<@H[i32], "E">]>': 128 projection steps stand on the chain}}
  %a = trait.allege @A[!trait.proj<@H[i32], "E">]
  %v = trait.method.call %a @A[!trait.proj<@H[i32], "E">]::@value() : () -> i64
  return %v : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E = !trait.proj<@H[tuple<!T>], "E"> }
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.method @value() -> i64 }
trait.impl private @I2(%s: !trait.claim<@A[!T]>, %e: !trait.claim<!trait.proj<@H[!T], "E"> = i64>) { trait.method @value() -> i64 { %c = arith.constant 9 : i64 trait.return %c : i64 } }
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<!trait.proj<@H[i32], "E"> = i64>': 128 projection steps stand on the chain}}
  %a = trait.allege @A[i32]
  %v = trait.method.call %a @A[i32]::@value() : () -> i64
  return %v : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E = !trait.proj<@H[tuple<!T>], "E"> }
func.func private @need(!trait.claim<!trait.proj<@H[i32], "E"> = i64>)
func.func @main() {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<!trait.proj<@H[i32], "E"> = i64>': 128 projection steps stand on the chain}}
  %e = trait.allege !trait.proj<@H[i32], "E"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@H[i32], "E"> = i64>) -> ()
  return
}

// -----

// A finite chain that stands past the depth limit: @H[i1]::E is bound to
// @H[i2]::E, and so on to @H[i130]::E, which is i32. One hundred and thirty
// steps are past the limit's worth, the same overflow as a growing binding.

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.method @value() -> i64 }
trait.impl private @A_i32(%s: !trait.claim<@A[i32]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }
// REPEAT 1 129: trait.impl private @H{k}(%s: !trait.claim<@H[i{k}]>) { trait.assoc_type @E = !trait.proj<@H[i{k+1}], "E"> }
trait.impl private @H130(%s: !trait.claim<@H[i130]>) { trait.assoc_type @E = i32 }
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<@A[!trait.proj<@H[i1], "E">]>': 128 projection steps stand on the chain}}
  %a = trait.allege @A[!trait.proj<@H[i1], "E">]
  %v = trait.method.call %a @A[!trait.proj<@H[i1], "E">]::@value() : () -> i64
  return %v : i64
}
