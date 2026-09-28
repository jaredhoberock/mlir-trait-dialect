// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An impl of a trait with a bound requirement states a witness for it.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{states no witness for bound requirement 0 of trait '@Has'}}
trait.impl private @Has_i32 for @Has[i32] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// A requirement witness names a bound requirement of the trait by position.

!S = !trait.poly<0>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [@Marker[!S]] {}
// expected-error @below {{states a witness for requirement 0, which is not a bound requirement of trait '@Has'}}
trait.impl private @Has_i1 for @Has[i1] witnesses [#trait<witness requirement 0 by @Marker_i1>] {}

// -----

// One witness per bound requirement.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{states a witness for requirement 0 twice}}
trait.impl private @Has_i32 for @Has[i32]
    witnesses [#trait<witness requirement 0 by @Marker_i1>, #trait<witness requirement 0 by @Marker_i1>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// An impl cited as evidence is an impl of the conclusion there.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i64 for @Marker[i64] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{does not prove #trait<application@Marker[!trait.proj<@Has[i32], "A", [!trait.bound<0>]>]>: it states another predicate}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by @Marker_i64>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// A binding that ignores the binder's premise is not proved by it: `A<X> = X`
// states Marker of every X only where X is Marker.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Other[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] where [@Other[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{it states another predicate}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by premise 0>] {
  trait.assoc_type @A<[!X]> = !X
}

// -----

// A premise position past the binder's premises names none.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{the binder states 1 premises}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by premise 1>] {
  trait.assoc_type @A<[!X]> = !X
}

// -----

// A where-clause position past the impl's where clause names no entry.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{the impl's where clause has 0 entries}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by where 0>] {
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
trait.trait private @Has[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{the cited impl's where clause has 1 entries, and the evidence discharges 0}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by @Marker_wrap[!P = !trait.bound<0>]>] {
  trait.assoc_type @A<[!X]> = tuple<!X>
}

// -----

// An impl proves an application, not an equality.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.impl private @Marker_i1 for @Marker[i1] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> !trait.proj<@Has[!S], "A", [!trait.bound<0>]> = !trait.bound<0>] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{an impl proves only a trait application}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by @Marker_i1>] {
  trait.assoc_type @A<[!X]> = !X
}

// -----

// Reflexivity proves an equality whose sides are one type through the impl's
// own bindings, and no other.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> !trait.proj<@Has[!S], "A", [!trait.bound<0>]> = !trait.bound<0>] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{evidence does not prove #trait<equality!trait.proj<@Has[i32], "A", [!trait.bound<0>]> = !trait.bound<0>>: its two sides are two types}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by refl>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{reflexivity proves only an equality}}
trait.impl private @Has_i32 for @Has[i32] witnesses [#trait<witness requirement 0 by refl>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// A witness of an application or an equality cites the impl that witnesses it:
// a premise or reflexivity stands only under a bound requirement.

trait.trait private @A[!trait.poly<0>] {}
// expected-error @below {{a witness of an application or an equality cites the impl that witnesses it}}
trait.impl private @A_i32 for @A[i32] witnesses [#trait<witness @A[i64] by premise 0>] {}

// -----

// Only a bound requirement's body discharges the premises of the impl it cites.

trait.trait private @A[!trait.poly<0>] {}
trait.impl private @A_i64 for @A[i64] {}
// expected-error @below {{only a bound requirement's witness discharges the premises of the impl it cites}}
trait.impl private @A_i32 for @A[i32] witnesses [#trait<witness @A[i64] by @A_i64 given [premise 0]>] {}

// -----

// A requirement hop names a requirement of the application its body proves,
// by position.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Sup0[!S] {}
trait.trait private @Sub0[!S] where [@Sup0[!S]] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Sup0[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{requirement index 1 is out of range}}
trait.impl private @Has_sub for @Has[tuple<!P>] where [@Sub0[!P]]
    witnesses [#trait<witness requirement 0 by requirement 1 of where 0>] {
  trait.assoc_type @A<[!X]> = !P
}

// -----

// A hop reads the requirement of the trait its body proves, not of another.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Sup0[!S] {}
trait.trait private @Other[!S] {}
trait.trait private @Sub1[!S] where [@Other[!S]] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Sup0[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{it states another predicate}}
trait.impl private @Has_sub for @Has[tuple<!P>] where [@Sub1[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 of where 0>] {
  trait.assoc_type @A<[!X]> = !P
}

// -----

// A hop reads a requirement off a trait application; reflexivity proves none.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Sup0[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Sup0[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
// expected-error @below {{reflexivity proves only the equality its position names}}
trait.impl private @Has_i32 for @Has[i32]
    witnesses [#trait<witness requirement 0 by requirement 0 of refl>] {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// A hop to a bound requirement discharges each premise it states there.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Marker[!S] {}
trait.trait private @Gen[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Gen[!S], "B", [!trait.bound<0>]>]] {
  trait.assoc_type @B<[!X]>
}
// expected-error @below {{requirement 0 states 1 premises, and the evidence discharges 0}}
trait.impl private @Gen_fwd for @Gen[tuple<!P>] where [@Gen[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] of where 0>] {
  trait.assoc_type @B<[!X]> = !trait.proj<@Gen[!P], "B", [!X]>
}

// -----

// A hop to a bound requirement takes one argument per variable it binds.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Marker[!S] {}
trait.trait private @Gen[!S] where [forall [!trait.bound<0>] -> @Marker[!trait.proj<@Gen[!S], "B", [!trait.bound<0>]>]] {
  trait.assoc_type @B<[!X]>
}
// expected-error @below {{requirement 0 binds 1 variables, and 0 arguments are supplied}}
trait.impl private @Gen_fwd for @Gen[tuple<!P>] where [@Gen[!P]]
    witnesses [#trait<witness requirement 0 by requirement 0 of where 0>] {
  trait.assoc_type @B<[!X]> = !trait.proj<@Gen[!P], "B", [!X]>
}

// -----

// An allegation states the predicate its position names.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Rule[!S] {}
trait.trait private @Holds[!S] where [forall [!trait.bound<0>] -> @Rule[!trait.proj<@Holds[!S], "C", [!trait.bound<0>]>]] {
  trait.assoc_type @C<[!X]>
}
// expected-error @below {{it states another predicate}}
trait.impl private @Holds_i32 for @Holds[i32]
    witnesses [#trait<witness requirement 0 by allege @Rule[i64]>] {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
}

// -----

// An allegation stands only under a bound requirement.

trait.trait private @A[!trait.poly<0>] {}
// expected-error @below {{a witness of an application or an equality cites the impl that witnesses it}}
trait.impl private @A_i32 for @A[i32] witnesses [#trait<witness @A[i64] by allege @A[i64]>] {}

// -----

// A witness body ends where its last arm ends; text after it belongs to no
// body and is refused rather than dropped. `where 0` takes no `given`.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Goal[!S] {}
trait.trait private @Mid[!S] where [@Goal[!S]] {}
trait.trait private @Base[!S] where [forall [!trait.bound<0>] -> @Mid[!trait.proj<@Base[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.trait private @Dst[!S] where [forall [!trait.bound<0>] -> @Goal[!trait.proj<@Dst[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Dst_i32 for @Dst[i32] where [@Base[i32]]
    // expected-error @below {{expected the end of the attribute}}
    witnesses [#trait<witness requirement 0 by requirement 0 of requirement 0 for [!trait.bound<0>] of where 0 given [where 0]>] {
  trait.assoc_type @A<[!X]> = !trait.proj<@Base[i32], "A", [!X]>
}
