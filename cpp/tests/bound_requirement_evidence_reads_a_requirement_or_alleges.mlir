// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A bound requirement's evidence can read a requirement of the application
// another body proves -- a supertrait of the impl's premise, the binder of a
// premise's own trait at the binder's variables, a premise's bound requirement
// with its premise discharged -- or allege an application a compiler rule
// decides, which the requirement's use proves at its instance.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>

trait.trait private @Sup0[!S] {}
trait.trait private @Sub0[!S] where [@Sup0[!S]] {}
trait.trait private @Marker[!S] {}

trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Sup0[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}

// `impl<P: Sub0> Has for (P,) { type A<X> = P; }`: `Sup0[P]` is requirement 0
// of the premise `Sub0[P]`.
trait.impl private @Has_sub for @Has[tuple<!P>] where [@Sub0[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 of where 0>] {
  trait.assoc_type @A<[!X]> = !P
}

// `impl<P: Has> Has for (P, P) { type A<X> = P::A<X>; }`: the bound is the
// premise's own binder at the binder's variable.
trait.impl private @Has_fwd for @Has[tuple<!P, !P>] where [@Has[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] of where 0>] {
  trait.assoc_type @A<[!X]> = !trait.proj<@Has[!P], "A", [!X]>
}

// A binder with a premise, read off a premise at the binder's variable with
// the binder's own premise discharged.
trait.trait private @Gen[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Gen[!S], "B", [!trait.bound<0>]>]] {
  trait.assoc_type @B<[!X]>
}
trait.impl private @Gen_fwd for @Gen[tuple<!P>] where [@Gen[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] given [premise 0] of where 0>] {
  trait.assoc_type @B<[!X]> = !trait.proj<@Gen[!P], "B", [!X]>
}

// A conclusion a compiler rule decides is alleged.
trait.trait private @Rule[!S] {}
trait.trait private @Holds[!S] where [forall [!trait.bound<0>] -> @Rule[!trait.proj<@Holds[!S], "C", [!trait.bound<0>]>]] {
  trait.assoc_type @C<[!X]>
}
trait.impl private @Holds_i32 for @Holds[i32]
    witnesses [#trait<witness requirement 0 by allege @Rule[tuple<i64, i64>]>] {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
}

// A hop off a hop off a citation discharging the cited impl's premise: each
// `given` belongs to the body it follows, and the text reads back the same.
trait.trait private @Goal[!S] {}
trait.trait private @Mid[!S] where [@Goal[!S]] {}
trait.trait private @Base[!S] where [forall [!trait.bound<0>] -> @Mid[!trait.proj<@Base[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Base_p for @Base[tuple<!P>] where [@Mid[!P]]
    witnesses [#trait<witness requirement 0 by where 0>] {
  trait.assoc_type @A<[!X]> = !P
}
trait.trait private @Dst[!S] where [forall [!trait.bound<0>] -> @Goal[!trait.proj<@Dst[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Dst_p for @Dst[tuple<!P>] where [@Mid[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 of requirement 0 for [!trait.bound<0>] of @Base_p[!P = !P] given [where 0]>] {
  trait.assoc_type @A<[!X]> = !trait.proj<@Base[tuple<!P>], "A", [!X]>
}

// CHECK: witnesses [#trait<witness requirement 0 by requirement 0 of where 0>]
// CHECK: witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] of where 0>]
// CHECK: witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] given [premise 0] of where 0>]
// CHECK: witnesses [#trait<witness requirement 0 by allege @Rule[tuple<i64, i64>]>]
// CHECK: witnesses [#trait<witness requirement 0 by where 0>]
// CHECK: witnesses [#trait<witness requirement 0 by requirement 0 of requirement 0 for [!trait.bound<0>] of @Base_p[!trait.poly<2> = !trait.poly<2>] given [where 0]>]
