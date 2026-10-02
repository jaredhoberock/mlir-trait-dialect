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
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[!S], "A", [!B]>]>
}

// `impl<P: Sub0> Has for (P,) { type A<X> = P; }`: `Sup0[P]` is requirement 0
// of the premise `Sub0[P]`.
trait.impl private @Has_sub(%self: !trait.claim<@Has[tuple<!P>]>, %sub0: !trait.claim<@Sub0[!P]>) {
  trait.assoc_type @A<[!X]> = !P
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P>], "A", [!B]>]> {
    %sup = trait.project %sub0[0] : !trait.claim<@Sub0[!P]> -> !trait.claim<@Sup0[!P]>
    %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!P>], "A", [!B]> resolves !P by @Has_sub given(%sub0) : (!trait.claim<@Sub0[!P]>) : !trait.claim<!trait.proj<@Has[tuple<!P>], "A", [!B]> = !P>
    %r = trait.coerce %sup : !trait.claim<@Sup0[!P]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P>], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!P>], "A", [!B]> = !P>)
    trait.return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P>], "A", [!B]>]>
  }
}

// `impl<P: Has> Has for (P, P) { type A<X> = P::A<X>; }`: the bound is the
// premise's own evidence method at the binder's variable.
trait.impl private @Has_fwd(%self: !trait.claim<@Has[tuple<!P, !P>]>, %has: !trait.claim<@Has[!P]>) {
  trait.assoc_type @A<[!X]> = !trait.proj<@Has[!P], "A", [!X]>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P, !P>], "A", [!B]>]> {
    %sup = trait.method.call %has @Has[!P]::@requirement_0() : () -> !trait.claim<@Sup0[!trait.proj<@Has[!P], "A", [!B]>]>
    %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!P, !P>], "A", [!B]> resolves !trait.proj<@Has[!P], "A", [!B]> by @Has_fwd given(%has) : (!trait.claim<@Has[!P]>) : !trait.claim<!trait.proj<@Has[tuple<!P, !P>], "A", [!B]> = !trait.proj<@Has[!P], "A", [!B]>>
    %r = trait.coerce %sup : !trait.claim<@Sup0[!trait.proj<@Has[!P], "A", [!B]>]> to !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P, !P>], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Has[tuple<!P, !P>], "A", [!B]> = !trait.proj<@Has[!P], "A", [!B]>>)
    trait.return %r : !trait.claim<@Sup0[!trait.proj<@Has[tuple<!P, !P>], "A", [!B]>]>
  }
}

// A quantified requirement with a premise, read off a premise's evidence
// method with the binder's own premise passed on.
trait.trait private @Gen(%self: !trait.claim<@Gen[!S]>) {
  trait.assoc_type @B<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!S], "B", [!B]>]>
}
trait.impl private @Gen_fwd(%self: !trait.claim<@Gen[tuple<!P>]>, %gen: !trait.claim<@Gen[!P]>) {
  trait.assoc_type @B<[!X]> = !trait.proj<@Gen[!P], "B", [!X]>
  trait.method @requirement_0(%m: !trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Gen[tuple<!P>], "B", [!B]>]> {
    %marker = trait.method.call %gen @Gen[!P]::@requirement_0(%m) : (!trait.claim<@Marker[!B]>) -> !trait.claim<@Marker[!trait.proj<@Gen[!P], "B", [!B]>]>
    %b = trait.witness proj_resolve !trait.proj<@Gen[tuple<!P>], "B", [!B]> resolves !trait.proj<@Gen[!P], "B", [!B]> by @Gen_fwd given(%gen) : (!trait.claim<@Gen[!P]>) : !trait.claim<!trait.proj<@Gen[tuple<!P>], "B", [!B]> = !trait.proj<@Gen[!P], "B", [!B]>>
    %r = trait.coerce %marker : !trait.claim<@Marker[!trait.proj<@Gen[!P], "B", [!B]>]> to !trait.claim<@Marker[!trait.proj<@Gen[tuple<!P>], "B", [!B]>]> via (%b) : (!trait.claim<!trait.proj<@Gen[tuple<!P>], "B", [!B]> = !trait.proj<@Gen[!P], "B", [!B]>>)
    trait.return %r : !trait.claim<@Marker[!trait.proj<@Gen[tuple<!P>], "B", [!B]>]>
  }
}

// A conclusion a compiler rule decides is alleged.
trait.trait private @Rule(%self: !trait.claim<@Rule[!S]>) {}
trait.trait private @Holds(%self: !trait.claim<@Holds[!S]>) {
  trait.assoc_type @C<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[!S], "C", [!B]>]>
}
trait.impl private @Holds_i32(%self: !trait.claim<@Holds[i32]>) {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!B]>]> {
    %r = trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!B]>]
    trait.return %r : !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!B]>]>
  }
}

// A requirement read off an evidence method of an impl the body derives,
// discharging that impl's premise with the body's own.
trait.trait private @Goal(%self: !trait.claim<@Goal[!S]>) {}
trait.trait private @Mid(%self: !trait.claim<@Mid[!S]>) -> !trait.claim<@Goal[!S]> {}
trait.trait private @Base(%self: !trait.claim<@Base[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Mid[!trait.proj<@Base[!S], "A", [!B]>]>
}
trait.impl private @Base_p(%self: !trait.claim<@Base[tuple<!P>]>, %mid: !trait.claim<@Mid[!P]>) {
  trait.assoc_type @A<[!X]> = !P
  trait.method @requirement_0() -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!P>], "A", [!B]>]> {
    %a = trait.witness proj_resolve !trait.proj<@Base[tuple<!P>], "A", [!B]> resolves !P by @Base_p given(%mid) : (!trait.claim<@Mid[!P]>) : !trait.claim<!trait.proj<@Base[tuple<!P>], "A", [!B]> = !P>
    %r = trait.coerce %mid : !trait.claim<@Mid[!P]> to !trait.claim<@Mid[!trait.proj<@Base[tuple<!P>], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Base[tuple<!P>], "A", [!B]> = !P>)
    trait.return %r : !trait.claim<@Mid[!trait.proj<@Base[tuple<!P>], "A", [!B]>]>
  }
}
trait.trait private @Dst(%self: !trait.claim<@Dst[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Goal[!trait.proj<@Dst[!S], "A", [!B]>]>
}
trait.impl private @Dst_p(%self: !trait.claim<@Dst[tuple<!P>]>, %mid: !trait.claim<@Mid[!P]>) {
  trait.assoc_type @A<[!X]> = !trait.proj<@Base[tuple<!P>], "A", [!X]>
  trait.method @requirement_0() -> !trait.claim<@Goal[!trait.proj<@Dst[tuple<!P>], "A", [!B]>]> {
    %base = trait.derive @Base[tuple<!P>] from @Base_p given(%mid) : (!trait.claim<@Mid[!P]>)
    %m = trait.method.call %base @Base[tuple<!P>]::@requirement_0() : () -> !trait.claim<@Mid[!trait.proj<@Base[tuple<!P>], "A", [!B]>]>
    %goal = trait.project %m[0] : !trait.claim<@Mid[!trait.proj<@Base[tuple<!P>], "A", [!B]>]> -> !trait.claim<@Goal[!trait.proj<@Base[tuple<!P>], "A", [!B]>]>
    %a = trait.witness proj_resolve !trait.proj<@Dst[tuple<!P>], "A", [!B]> resolves !trait.proj<@Base[tuple<!P>], "A", [!B]> by @Dst_p given(%mid) : (!trait.claim<@Mid[!P]>) : !trait.claim<!trait.proj<@Dst[tuple<!P>], "A", [!B]> = !trait.proj<@Base[tuple<!P>], "A", [!B]>>
    %r = trait.coerce %goal : !trait.claim<@Goal[!trait.proj<@Base[tuple<!P>], "A", [!B]>]> to !trait.claim<@Goal[!trait.proj<@Dst[tuple<!P>], "A", [!B]>]> via (%a) : (!trait.claim<!trait.proj<@Dst[tuple<!P>], "A", [!B]> = !trait.proj<@Base[tuple<!P>], "A", [!B]>>)
    trait.return %r : !trait.claim<@Goal[!trait.proj<@Dst[tuple<!P>], "A", [!B]>]>
  }
}

// CHECK-LABEL: trait.impl private @Has_sub
// CHECK: trait.project %sub0[0] : <@Sub0[!trait.poly<2>]> -> <@Sup0[!trait.poly<2>]>
// CHECK-LABEL: trait.impl private @Has_fwd
// CHECK: trait.method.call %has @Has[!trait.poly<2>]::@requirement_0()
// CHECK-LABEL: trait.impl private @Gen_fwd
// CHECK: trait.method.call %gen @Gen[!trait.poly<2>]::@requirement_0(%arg0)
// CHECK-LABEL: trait.impl private @Holds_i32
// CHECK: trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!trait.poly<3>]>]
// CHECK-LABEL: trait.impl private @Base_p
// CHECK: trait.coerce %mid
// CHECK-LABEL: trait.impl private @Dst_p
// CHECK: %[[BASE:.*]] = trait.derive @Base[tuple<!trait.poly<2>>] from @Base_p given(%mid)
// CHECK: %[[MID:.*]] = trait.method.call %[[BASE]] @Base[tuple<!trait.poly<2>>]::@requirement_0()
// CHECK: trait.project %[[MID]][0]
