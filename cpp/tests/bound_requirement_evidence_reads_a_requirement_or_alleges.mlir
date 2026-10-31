// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// An impl's evidence member for a quantified requirement computes the
// requirement's evidence with ordinary operations over the impl's block
// arguments, its leading parameters: a requirement of a premise read by
// position, a premise's own evidence member called at the binder's variables (with the binder's premise
// passed on), a derive of another impl, or an allegation of a claim a compiler
// rule decides, which the requirement's use proves at its instance. Each
// result is spelled through the impl's own binding, so the value computed at
// the binding's type is coerced by the impl's own resolution of the projection.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
!B = !trait.poly<3>

trait.trait private @Sup0(!trait.claim<@Sup0[!S]>)
trait.trait private @Sub0(!trait.claim<@Sub0[!S]>) -> !trait.claim<@Sup0[!S]>
trait.trait private @Marker(!trait.claim<@Marker[!S]>)

trait.trait private @Has(!trait.claim<@Has[!S]>) assoc {A = 1} members [@Has.requirement_0]
func.func private @Has.requirement_0(!trait.claim<@Has[!S]>) -> !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>

// `impl<P: Sub0> Has for (P,) { type A<X> = P; }`: `Sup0[P]` is requirement 0
// of the premise `Sub0[P]`.
trait.impl private @Has_sub(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %sub0: !trait.claim<@Sub0[!trait.poly<0>]>) bindings {A = !trait.poly<0>} members [@Has.requirement_0 -> @Has_sub.requirement_0] {
}
func.func private @Has_sub.requirement_0(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %sub0: !trait.claim<@Sub0[!trait.poly<0>]>) -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
  %sup = trait.project %sub0[0] : !trait.claim<@Sub0[!trait.poly<0>]> -> !trait.claim<@Sup0[!trait.poly<0>]>
  %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.poly<0> by @Has_sub[!trait.poly<0>] given(%sub0) : (!trait.claim<@Sub0[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>
  %r = trait.coerce %sup : !trait.claim<@Sup0[!trait.poly<0>]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>)
  return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
}

// `impl<P: Has> Has for (P, P) { type A<X> = P::A<X>; }`: the bound is the
// premise's own evidence member at the binder's variable.
trait.impl private @Has_fwd(%self: !trait.claim<@Has[tuple<!trait.poly<0>, !trait.poly<0>>]>, %has: !trait.claim<@Has[!trait.poly<0>]>) bindings {A = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>} members [@Has.requirement_0 -> @Has_fwd.requirement_0] {
}
func.func private @Has_fwd.requirement_0(%self: !trait.claim<@Has[tuple<!trait.poly<0>, !trait.poly<0>>]>, %has: !trait.claim<@Has[!trait.poly<0>]>) -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]> {
  %sup = trait.func.call @Has.requirement_0(%has) : (!trait.claim<@Has[!trait.poly<0>]>) -> !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
  %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]> by @Has_fwd[!trait.poly<0>] given(%has) : (!trait.claim<@Has[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>>
  %r = trait.coerce %sup : !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>>)
  return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]>
}

// A quantified requirement with a premise, read off a premise's evidence
// member with the binder's own premise passed on.
trait.trait private @Gen(!trait.claim<@Gen[!S]>) assoc {B = 1} members [@Gen.requirement_0]
func.func private @Gen.requirement_0(!trait.claim<@Gen[!S]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]>
trait.impl private @Gen_fwd(%self: !trait.claim<@Gen[tuple<!trait.poly<0>>]>, %gen: !trait.claim<@Gen[!trait.poly<0>]>) bindings {B = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>} members [@Gen.requirement_0 -> @Gen_fwd.requirement_0] {
}
func.func private @Gen_fwd.requirement_0(%self: !trait.claim<@Gen[tuple<!trait.poly<0>>]>, %gen: !trait.claim<@Gen[!trait.poly<0>]>, %m: !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]> {
  %marker = trait.func.call @Gen.requirement_0(%gen, %m) : (!trait.claim<@Gen[!trait.poly<0>]>, !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]>
  %b = trait.witness proj_resolve !trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> resolves !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]> by @Gen_fwd[!trait.poly<0>] given(%gen) : (!trait.claim<@Gen[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>>
  %r = trait.coerce %marker : !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]> to !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]> via (%b) : (!trait.claim<!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>>)
  return %r : !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]>
}

// A conclusion a compiler rule decides is alleged.
trait.trait private @Rule(!trait.claim<@Rule[!S]>)
trait.trait private @Holds(!trait.claim<@Holds[!S]>) assoc {C = 1} members [@Holds.requirement_0]
func.func private @Holds.requirement_0(!trait.claim<@Holds[!S]>) -> !trait.claim<@Rule[!trait.proj<@Holds[!trait.poly<0>], "C", [!trait.poly<1>]>]>
trait.impl private @Holds_i32(%self: !trait.claim<@Holds[i32]>) bindings {C = tuple<i64, i64>} members [@Holds.requirement_0 -> @Holds_i32.requirement_0] {
}
func.func private @Holds_i32.requirement_0(%self: !trait.claim<@Holds[i32]>) -> !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]> {
  %r = trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]
  return %r : !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]>
}

// A requirement read off an evidence member of an impl the body derives,
// discharging that impl's premise with the body's own.
trait.trait private @Goal(!trait.claim<@Goal[!S]>)
trait.trait private @Mid(!trait.claim<@Mid[!S]>) -> !trait.claim<@Goal[!S]>
trait.trait private @Base(!trait.claim<@Base[!S]>) assoc {A = 1} members [@Base.requirement_0]
func.func private @Base.requirement_0(!trait.claim<@Base[!S]>) -> !trait.claim<@Mid[!trait.proj<@Base[!trait.poly<0>], "A", [!trait.poly<1>]>]>
trait.impl private @Base_p(%self: !trait.claim<@Base[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) bindings {A = !trait.poly<0>} members [@Base.requirement_0 -> @Base_p.requirement_0] {
}
func.func private @Base_p.requirement_0(%self: !trait.claim<@Base[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
  %a = trait.witness proj_resolve !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.poly<0> by @Base_p[!trait.poly<0>] given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>
  %r = trait.coerce %mid : !trait.claim<@Mid[!trait.poly<0>]> to !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>)
  return %r : !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
}
trait.trait private @Dst(!trait.claim<@Dst[!S]>) assoc {A = 1} members [@Dst.requirement_0]
func.func private @Dst.requirement_0(!trait.claim<@Dst[!S]>) -> !trait.claim<@Goal[!trait.proj<@Dst[!trait.poly<0>], "A", [!trait.poly<1>]>]>
trait.impl private @Dst_p(%self: !trait.claim<@Dst[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) bindings {A = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>} members [@Dst.requirement_0 -> @Dst_p.requirement_0] {
}
func.func private @Dst_p.requirement_0(%self: !trait.claim<@Dst[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) -> !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
  %base = trait.derive @Base[tuple<!trait.poly<0>>] from @Base_p[!trait.poly<0>] given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>)
  %m = trait.func.call @Base.requirement_0(%base) : (!trait.claim<@Base[tuple<!trait.poly<0>>]>) -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
  %goal = trait.project %m[0] : !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> -> !trait.claim<@Goal[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
  %a = trait.witness proj_resolve !trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> by @Dst_p[!trait.poly<0>] given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>>
  %r = trait.coerce %goal : !trait.claim<@Goal[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> to !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>>)
  return %r : !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
}

// CHECK-LABEL: func.func private @Has_sub.requirement_0
// CHECK-SAME: %[[SUB0:[^:]*]]: !trait.claim<@Sub0[!trait.poly<0>]>
// CHECK: trait.project %[[SUB0]][0] : <@Sub0[!trait.poly<0>]> -> <@Sup0[!trait.poly<0>]>
// CHECK-LABEL: func.func private @Has_fwd.requirement_0
// CHECK-SAME: %[[HAS:[^:]*]]: !trait.claim<@Has[!trait.poly<0>]>
// CHECK: trait.func.call @Has.requirement_0(%[[HAS]])
// CHECK-LABEL: func.func private @Gen_fwd.requirement_0
// CHECK-SAME: %[[GEN:[^:]*]]: !trait.claim<@Gen[!trait.poly<0>]>, %[[M:[^:]*]]: !trait.claim<@Marker[!trait.poly<1>]>
// CHECK: trait.func.call @Gen.requirement_0(%[[GEN]], %[[M]])
// CHECK-LABEL: func.func private @Holds_i32.requirement_0
// CHECK: trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]
// CHECK-LABEL: func.func private @Base_p.requirement_0
// CHECK-SAME: %[[MIDB:[^:]*]]: !trait.claim<@Mid[!trait.poly<0>]>
// CHECK: trait.coerce %[[MIDB]]
// CHECK-LABEL: func.func private @Dst_p.requirement_0
// CHECK-SAME: %[[MIDD:[^:]*]]: !trait.claim<@Mid[!trait.poly<0>]>
// CHECK: %[[BASE:.*]] = trait.derive @Base[tuple<!trait.poly<0>>] from @Base_p[!trait.poly<0>] given(%[[MIDD]])
// CHECK: %[[MID:.*]] = trait.func.call @Base.requirement_0(%[[BASE]])
// CHECK: trait.project %[[MID]][0]
