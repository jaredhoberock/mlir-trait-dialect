// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Erase deletes nothing for being a template. What leaves the module leaves
// because nothing names it: the symbol-dce closing the monomorphize pipeline
// collects every template whole, and with it every module-level helper only a
// template named. A helper reachable from a live clone stands; one reachable
// only from a template that was never instantiated goes with the template,
// which by-kind deletion leaves behind as dead code.
//
// Three shapes stand beside that: an impl named nowhere but inside a type (a
// coerce's result type), which the barrier erases before the collector reads
// the module, so collection runs on a module whose type-held references are
// already gone; a mutually citing pair of proofs, which the collector takes as
// a cycle nothing roots, with no proof-specific rule; and a trait whose second
// impl no call ever selects.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' \
// RUN:   | FileCheck %s --implicit-check-not=@unused_helper \
// RUN:     --implicit-check-not=@dead_template --implicit-check-not=@Show_i64 \
// RUN:     --implicit-check-not=@Tag --implicit-check-not=@Ping \
// RUN:     --implicit-check-not=@holds_evidence_in_its_type \
// RUN:     --implicit-check-not=trait.trait --implicit-check-not=trait.impl \
// RUN:     --implicit-check-not=trait.proof --implicit-check-not=trait.witness \
// RUN:     --implicit-check-not=trait.coerce --implicit-check-not=!trait.claim \
// RUN:     --implicit-check-not=!trait.proj

!S = !trait.poly<0>

trait.trait private @Show[!S] {
  func.func private @show(!S) -> i32
}

trait.impl private @Show_i32 for @Show[i32] {
  func.func private @show(%x: i32) -> i32 {
    return %x : i32
  }
}

// no call selects this impl, so its method is never cloned
trait.impl private @Show_i64 for @Show[i64] {
  func.func private @show(%x: i64) -> i32 {
    %c = arith.constant 0 : i32
    return %c : i32
  }
}

// Two module-level helpers, alike but for who reaches them: @live_template is
// instantiated from main, so its clone keeps @used_helper alive, while
// @dead_template is instantiated by nothing and @unused_helper is left named
// only by it.
func.func private @used_helper(%x: i32) -> i32 {
  return %x : i32
}

func.func private @unused_helper(%x: i32) -> i32 {
  return %x : i32
}

func.func private @live_template(%c: !trait.claim<@Show[!S]>, %x: !S) -> i32 {
  %r = trait.method.call %c @Show[!S]::@show(%x) : (!S) -> i32
  %h = func.call @used_helper(%r) : (i32) -> i32
  return %h : i32
}

func.func private @dead_template(%x: !S) -> i32 {
  %c = arith.constant 1 : i32
  %h = func.call @unused_helper(%c) : (i32) -> i32
  return %h : i32
}

// @Tag, its impl and its proof are named from one place: a type. The parameter
// spelling and the coerce's result type below are the whole of it, and the
// barrier takes both, so the collector reads a module in which nothing mentions
// them at all.
trait.trait private @Tag[!S] {
}

trait.impl private @Tag_i32 for @Tag[i32] {
}

trait.proof private @Tag_p proves @Tag_i32 for @Tag[i32] given []

func.func private @holds_evidence_in_its_type(%e: !trait.claim<@Tag[i32] by @Tag_p>) -> i32 {
  %same = trait.coerce %e : !trait.claim<@Tag[i32] by @Tag_p> to !trait.claim<@Tag[i32] by @Tag_p>
  %z = arith.constant 0 : i32
  return %z : i32
}

// A coinductive pair: each proof discharges the other's obligation and nothing
// outside the pair names either, so the two are an unrooted cycle.
trait.trait private @Ping[!S] where [@Ping[!trait.proj<@Ping[!S], "Other">]] {
  trait.assoc_type @Other
}

trait.impl private @Ping_i32 for @Ping[i32] {
  trait.assoc_type @Other = i64
}

trait.impl private @Ping_i64 for @Ping[i64] {
  trait.assoc_type @Other = i32
}

trait.proof private @Ping_i32_p proves @Ping_i32 for @Ping[i32] given [@Ping_i64_p]
trait.proof private @Ping_i64_p proves @Ping_i64 for @Ping[i64] given [@Ping_i32_p]

// The method clone, the helper the live clone reaches, the clone itself and
// main are the whole of what stands; the implicit-check-nots on the RUN line
// hold every region between them clear of the rest.
// CHECK: func.func private @Show_i32_show
// CHECK: func.func private @used_helper
// CHECK: func.func private @live_template
// CHECK: call @Show_i32_show
// CHECK: call @used_helper
// CHECK: func.func @main
// CHECK: call @live_template
func.func @main() -> i32 {
  %x = arith.constant 7 : i32
  %e = trait.allege @Show[i32]
  %r = trait.func.call @live_template(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]} : (!trait.claim<@Show[i32]>, i32) -> i32
  return %r : i32
}
