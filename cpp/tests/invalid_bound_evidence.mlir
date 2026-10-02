// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A quantified requirement is a required method of its trait returning a claim,
// so an impl of the trait defines it.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
// expected-error @below {{missing implementation for required method 'requirement_0' of trait '@Has'}}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i1
}

// -----

// The evidence the method returns is evidence of its conclusion at the impl:
// an impl of another application proves another predicate.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i1
  trait.method @requirement_0() -> !trait.claim<@Marker[i1]> {
    %m = trait.witness @Marker_i64 for @Marker[i64]
    // expected-error @below {{type of return operand 0 ('!trait.claim<@Marker[i64] by @Marker_i64>') doesn't match method result type ('!trait.claim<@Marker[i1]>') in method @requirement_0}}
    trait.return %m : !trait.claim<@Marker[i64] by @Marker_i64>
  }
}

// -----

// A binding that ignores the binder's premise is not proved by it: `A<X> = X`
// states Marker of every X only where X is Marker, and the method holds Other.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.trait private @Other(%self: !trait.claim<@Other[!S]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Other[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = !X
  trait.method @requirement_0(%o: !trait.claim<@Other[!X]>) -> !trait.claim<@Marker[!X]> {
    // expected-error @below {{type of return operand 0 ('!trait.claim<@Other[!trait.poly<1>]>') doesn't match method result type ('!trait.claim<@Marker[!trait.poly<1>]>') in method @requirement_0}}
    trait.return %o : !trait.claim<@Other[!X]>
  }
}

// -----

// A derive of an impl supplies one claim per entry of that impl's where clause.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.impl private @Marker_wrap(%self: !trait.claim<@Marker[tuple<!P>]>, %marker: !trait.claim<@Marker[!P]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = tuple<!X>
  trait.method @requirement_0(%m: !trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[tuple<!X>]> {
    // expected-error @below {{impl '@Marker_wrap' has 1 where entries, and the citation supplies 0 claims}}
    %d = trait.derive @Marker[tuple<!X>] from @Marker_wrap given()
    trait.return %d : !trait.claim<@Marker[tuple<!X>]>
  }
}

// -----

// Reflexivity proves an equality whose sides are one type through the impl's
// own bindings, and no other.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<!trait.proj<@Has[!S], "A", [!X]> = !X>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i1
  trait.method @requirement_0() -> !trait.claim<i1 = !X> {
    // expected-error @below {{a refl witness requires identical endpoints, found 'i1' and '!trait.poly<1>'}}
    %e = trait.witness refl : !trait.claim<i1 = !X>
    trait.return %e : !trait.claim<i1 = !X>
  }
}

// -----

// A requirement read off a where argument names a requirement of that
// argument's trait by position.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Sup0(%self: !trait.claim<@Sup0[!S]>) {}
trait.trait private @Sub0(%self: !trait.claim<@Sub0[!S]>) -> !trait.claim<@Sup0[!S]> {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_sub(%self: !trait.claim<@Has[tuple<!P>]>, %sub0: !trait.claim<@Sub0[!P]>) {
  trait.assoc_type @A<[!X]> = !P
  trait.method @requirement_0() -> !trait.claim<@Sup0[!P]> {
    // expected-error @below {{requirement index 1 is out of range: '!trait.claim<@Sub0[!trait.poly<2>]>' has 1 requirements}}
    %s = trait.project %sub0[1] : !trait.claim<@Sub0[!P]> -> !trait.claim<@Sup0[!P]>
    trait.return %s : !trait.claim<@Sup0[!P]>
  }
}
