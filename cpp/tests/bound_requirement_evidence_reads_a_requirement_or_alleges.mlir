// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// An impl's evidence method for a quantified requirement computes the
// requirement's evidence with ordinary operations over the impl's block
// arguments: a requirement of a premise read by position, a premise's own
// evidence method called at the binder's variables (with the binder's premise
// passed on), a derive of another impl, or an allegation of a claim a compiler
// rule decides, which the requirement's use proves at its instance. Each
// result is spelled through the impl's own binding, so the value computed at
// the binding's type is coerced by the impl's own resolution of the projection.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
!B = !trait.poly<3>

trait.trait private @Sup0(%self: !trait.claim<@Sup0[!S]>) {}
trait.trait private @Sub0(%self: !trait.claim<@Sub0[!S]>) -> !trait.claim<@Sup0[!S]> {}
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}

trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
}

// `impl<P: Sub0> Has for (P,) { type A<X> = P; }`: `Sup0[P]` is requirement 0
// of the premise `Sub0[P]`.
trait.impl private @Has_sub(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %sub0: !trait.claim<@Sub0[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = !trait.poly<0>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
    %sup = trait.project %sub0[0] : !trait.claim<@Sub0[!trait.poly<0>]> -> !trait.claim<@Sup0[!trait.poly<0>]>
    %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.poly<0> by @Has_sub given(%sub0) : (!trait.claim<@Sub0[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>
    %r = trait.coerce %sup : !trait.claim<@Sup0[!trait.poly<0>]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>)
    trait.return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
  }
}

// `impl<P: Has> Has for (P, P) { type A<X> = P::A<X>; }`: the bound is the
// premise's own evidence method at the binder's variable.
trait.impl private @Has_fwd(%self: !trait.claim<@Has[tuple<!trait.poly<0>, !trait.poly<0>>]>, %has: !trait.claim<@Has[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]> {
    %sup = trait.method.call %has @Has[!trait.poly<0>]::@requirement_0() : () -> !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]>
    %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]> by @Has_fwd given(%has) : (!trait.claim<@Has[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>>
    %r = trait.coerce %sup : !trait.claim<@Sup0[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>>)
    trait.return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "A", [!trait.poly<1>]>]>
  }
}

// A quantified requirement with a premise, read off a premise's evidence
// method with the binder's own premise passed on.
trait.trait private @Gen(%self: !trait.claim<@Gen[!S]>) {
  trait.assoc_type @B<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]>
}
trait.impl private @Gen_fwd(%self: !trait.claim<@Gen[tuple<!trait.poly<0>>]>, %gen: !trait.claim<@Gen[!trait.poly<0>]>) {
  trait.assoc_type @B<[!trait.poly<1>]> = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>
  trait.method @requirement_0(%m: !trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]> {
    %marker = trait.method.call %gen @Gen[!trait.poly<0>]::@requirement_0(%m) : (!trait.claim<@Marker[!trait.poly<1>]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]>
    %b = trait.witness proj_resolve !trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> resolves !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]> by @Gen_fwd given(%gen) : (!trait.claim<@Gen[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>>
    %r = trait.coerce %marker : !trait.claim<@Marker[!trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>]> to !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]> via (%b) : (!trait.claim<!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]> = !trait.proj<@Gen[!trait.poly<0>], "B", [!trait.poly<1>]>>)
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Gen[tuple<!trait.poly<0>>], "B", [!trait.poly<1>]>]>
  }
}

// A conclusion a compiler rule decides is alleged.
trait.trait private @Rule(%self: !trait.claim<@Rule[!S]>) {}
trait.trait private @Holds(%self: !trait.claim<@Holds[!S]>) {
  trait.assoc_type @C<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[!trait.poly<0>], "C", [!trait.poly<1>]>]>
}
trait.impl private @Holds_i32(%self: !trait.claim<@Holds[i32]>) {
  trait.assoc_type @C<[!trait.poly<0>]> = tuple<i64, i64>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]> {
    %r = trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]
    trait.return %r : !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]>
  }
}

// A requirement read off an evidence method of an impl the body derives,
// discharging that impl's premise with the body's own.
trait.trait private @Goal(%self: !trait.claim<@Goal[!S]>) {}
trait.trait private @Mid(%self: !trait.claim<@Mid[!S]>) -> !trait.claim<@Goal[!S]> {}
trait.trait private @Base(%self: !trait.claim<@Base[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Mid[!trait.proj<@Base[!trait.poly<0>], "A", [!trait.poly<1>]>]>
}
trait.impl private @Base_p(%self: !trait.claim<@Base[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = !trait.poly<0>
  trait.method @requirement_0() -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
    %a = trait.witness proj_resolve !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.poly<0> by @Base_p given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>
    %r = trait.coerce %mid : !trait.claim<@Mid[!trait.poly<0>]> to !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.poly<0>>)
    trait.return %r : !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
  }
}
trait.trait private @Dst(%self: !trait.claim<@Dst[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Goal[!trait.proj<@Dst[!trait.poly<0>], "A", [!trait.poly<1>]>]>
}
trait.impl private @Dst_p(%self: !trait.claim<@Dst[tuple<!trait.poly<0>>]>, %mid: !trait.claim<@Mid[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>
  trait.method @requirement_0() -> !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> {
    %base = trait.derive @Base[tuple<!trait.poly<0>>] from @Base_p given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>)
    %m = trait.method.call %base @Base[tuple<!trait.poly<0>>]::@requirement_0() : () -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
    %goal = trait.project %m[0] : !trait.claim<@Mid[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> -> !trait.claim<@Goal[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
    %a = trait.witness proj_resolve !trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> resolves !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> by @Dst_p given(%mid) : (!trait.claim<@Mid[!trait.poly<0>]>) : !trait.claim<!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>>
    %r = trait.coerce %goal : !trait.claim<@Goal[!trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> to !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]> via (%a) : (!trait.claim<!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]> = !trait.proj<@Base[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>>)
    trait.return %r : !trait.claim<@Goal[!trait.proj<@Dst[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>]>
  }
}

// CHECK-LABEL: trait.impl private @Has_sub
// CHECK: trait.project %sub0[0] : <@Sub0[!trait.poly<0>]> -> <@Sup0[!trait.poly<0>]>
// CHECK-LABEL: trait.impl private @Has_fwd
// CHECK: trait.method.call %has @Has[!trait.poly<0>]::@requirement_0()
// CHECK-LABEL: trait.impl private @Gen_fwd
// CHECK: trait.method.call %gen @Gen[!trait.poly<0>]::@requirement_0(%arg0)
// CHECK-LABEL: trait.impl private @Holds_i32
// CHECK: trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<0>]>]
// CHECK-LABEL: trait.impl private @Base_p
// CHECK: trait.coerce %mid
// CHECK-LABEL: trait.impl private @Dst_p
// CHECK: %[[BASE:.*]] = trait.derive @Base[tuple<!trait.poly<0>>] from @Base_p given(%mid)
// CHECK: %[[MID:.*]] = trait.method.call %[[BASE]] @Base[tuple<!trait.poly<0>>]::@requirement_0()
// CHECK: trait.project %[[MID]][0]
