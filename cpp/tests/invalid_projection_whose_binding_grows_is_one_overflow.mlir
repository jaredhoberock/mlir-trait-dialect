// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | mlir-opt -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// @Hg binds @H[T]::E to @H[tuple<T>]::E, so @H[i32]::E grows at every
// resolution and has no normal form: past the depth limit's worth of
// projection steps it is an overflow. Whatever reaches it -- an allegation
// spelling it, an impl's where equality selection checks, an equality alleged
// -- names it once, and the stage stops there.

!T = !trait.poly<0>
trait.trait private @H(!trait.claim<@H[!T]>) assoc {E = 0}
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) bindings {E = !trait.proj<@H[tuple<!T>], "E">} {
}
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
func.func private @A.value(!trait.claim<@A[!T]>) -> i64
trait.impl private @I2(%s: !trait.claim<@A[i64]>) members [@A.value -> @I2.value] {
}
func.func private @I2.value(%s: !trait.claim<@A[i64]>) -> i64 {
  %c = arith.constant 9 : i64 return %c : i64
}
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<@A[!trait.proj<@H[i32], "E">]>': a chain of steps resolving its projections does not end within 128 steps}}
  %a = trait.allege @A[!trait.proj<@H[i32], "E">]
  %v = trait.func.call @A.value(%a) : (!trait.claim<@A[!trait.proj<@H[i32], "E">]>) -> i64
  return %v : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @H(!trait.claim<@H[!T]>) assoc {E = 0}
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) bindings {E = !trait.proj<@H[tuple<!T>], "E">} {
}
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
func.func private @A.value(!trait.claim<@A[!T]>) -> i64
trait.impl private @I2(%s: !trait.claim<@A[!T]>, %e: !trait.claim<!trait.proj<@H[!T], "E"> = i64>) members [@A.value -> @I2.value] {
}
func.func private @I2.value(%s: !trait.claim<@A[!T]>, %e: !trait.claim<!trait.proj<@H[!T], "E"> = i64>) -> i64 {
  %c = arith.constant 9 : i64 return %c : i64
}
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<!trait.proj<@H[i32], "E"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
  %a = trait.allege @A[i32]
  %v = trait.func.call @A.value(%a) : (!trait.claim<@A[i32]>) -> i64
  return %v : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @H(!trait.claim<@H[!T]>) assoc {E = 0}
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) bindings {E = !trait.proj<@H[tuple<!T>], "E">} {
}
func.func private @need(!trait.claim<!trait.proj<@H[i32], "E"> = i64>)
func.func @main() {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<!trait.proj<@H[i32], "E"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
  %e = trait.allege !trait.proj<@H[i32], "E"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@H[i32], "E"> = i64>) -> ()
  return
}

// -----

// A finite chain that stands past the depth limit: @H[i1]::E is bound to
// @H[i2]::E, and so on to @H[i130]::E, which is i32. One hundred and thirty
// steps are past the limit's worth, the same overflow as a growing binding.

!T = !trait.poly<0>
trait.trait private @H(!trait.claim<@H[!T]>) assoc {E = 0}
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
func.func private @A.value(!trait.claim<@A[!T]>) -> i64
trait.impl private @A_i32(%s: !trait.claim<@A[i32]>) members [@A.value -> @A_i32.value] {
}
func.func private @A_i32.value(%s: !trait.claim<@A[i32]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}
// REPEAT 1 129: trait.impl private @H{k}(%s: !trait.claim<@H[i{k}]>) bindings {E = !trait.proj<@H[i{k+1}], "E">} {}
trait.impl private @H130(%s: !trait.claim<@H[i130]>) bindings {E = i32} {
}
func.func @main() -> i64 {
  // expected-error @+1 {{overflow evaluating the requirement '!trait.claim<@A[!trait.proj<@H[i1], "E">]>': a chain of steps resolving its projections does not end within 128 steps}}
  %a = trait.allege @A[!trait.proj<@H[i1], "E">]
  %v = trait.func.call @A.value(%a) : (!trait.claim<@A[!trait.proj<@H[i1], "E">]>) -> i64
  return %v : i64
}
