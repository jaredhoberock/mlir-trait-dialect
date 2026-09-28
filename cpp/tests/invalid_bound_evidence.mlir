// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An impl of a trait with a bound requirement states evidence for it.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{states no evidence for bound requirement 0 of trait '@Has'}}
trait.impl private @Has_i32 for @Has[i32] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// Evidence cites a bound requirement of the trait by its position.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [@Marker[!S]] {}
// expected-error @below {{states bound evidence for requirement 0, which is not a bound requirement of trait '@Has'}}
trait.impl private @Has_i1 for @Has[i1]
    bound_evidence [#trait<bound_evidence 0: forall [!X] -> @Marker[tuple<i1, !X>] by @Marker_i1>] {}

// -----

// The evidence states the requirement at the impl's arguments.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [forall [!X] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{states bound evidence for requirement 0 as #trait<bound forall [!trait.poly<1>] -> @Marker[!trait.proj<@Has[i64], "A", [!trait.poly<1>]>]>, but the requirement at this impl is #trait<bound forall [!trait.poly<1>] -> @Marker[!trait.proj<@Has[i32], "A", [!trait.poly<1>]>]>}}
trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] -> @Marker[!trait.proj<@Has[i64], "A", [!X]>] by @Marker_i1>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// An impl cited as evidence is an impl of the conclusion there.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i64 for @Marker[i64] {}
trait.trait private @Has[!S] where [forall [!X] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{does not prove #trait<application@Marker[!trait.proj<@Has[i32], "A", [!trait.poly<1>]>]>: it states another predicate}}
trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] -> @Marker[!trait.proj<@Has[i32], "A", [!X]>] by @Marker_i64>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// A binding that ignores the binder's premise is not proved by it: `A<X> = X`
// states Marker of every X only where X is Marker.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Other[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Other[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{it states another predicate}}
trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Other[!X]] -> @Marker[!trait.proj<@Has[i32], "A", [!X]>] by premise 0>] {
  trait.assoc_type @A<[!X]> = !X
}

// -----

// A premise position past the binder's premises names none.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{the binder states 1 premises}}
trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[i32], "A", [!X]>] by premise 1>] {
  trait.assoc_type @A<[!X]> = !X
}

// -----

// An impl cited with a premise discharges each of that impl's where-clause
// entries in turn.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_wrap for @Marker[tuple<!P>] where [@Marker[!P]] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{the cited impl's where clause has 1 entries, and the evidence discharges 0}}
trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[i32], "A", [!X]>] by @Marker_wrap[!P = !X]>] {
  trait.assoc_type @A<[!X]> = tuple<!X>
}
