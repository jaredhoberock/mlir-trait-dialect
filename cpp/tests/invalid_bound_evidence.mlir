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
  trait.assoc_type @A<[!trait.poly<0>]> = i1
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
  trait.assoc_type @A<[!trait.poly<0>]> = i1
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
  trait.assoc_type @A<[!trait.poly<0>]> = !trait.poly<0>
  trait.method @requirement_0(%o: !trait.claim<@Other[!trait.poly<0>]>) -> !trait.claim<@Marker[!trait.poly<0>]> {
    // expected-error @below {{type of return operand 0 ('!trait.claim<@Other[!trait.poly<0>]>') doesn't match method result type ('!trait.claim<@Marker[!trait.poly<0>]>') in method @requirement_0}}
    trait.return %o : !trait.claim<@Other[!trait.poly<0>]>
  }
}

// -----

// A derive of an impl supplies one claim per entry of that impl's where clause.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {}
trait.impl private @Marker_wrap(%self: !trait.claim<@Marker[tuple<!trait.poly<0>>]>, %marker: !trait.claim<@Marker[!trait.poly<0>]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0(!trait.claim<@Marker[!X]>) -> !trait.claim<@Marker[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!trait.poly<0>]> = tuple<!trait.poly<0>>
  trait.method @requirement_0(%m: !trait.claim<@Marker[!trait.poly<0>]>) -> !trait.claim<@Marker[tuple<!trait.poly<0>>]> {
    // expected-error @below {{impl '@Marker_wrap' has 1 where entries, and the citation supplies 0 claims}}
    %d = trait.derive @Marker[tuple<!trait.poly<0>>] from @Marker_wrap given()
    trait.return %d : !trait.claim<@Marker[tuple<!trait.poly<0>>]>
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
  trait.assoc_type @A<[!trait.poly<0>]> = i1
  trait.method @requirement_0() -> !trait.claim<i1 = !trait.poly<0>> {
    // expected-error @below {{a refl witness requires identical endpoints, found 'i1' and '!trait.poly<0>'}}
    %e = trait.witness refl : !trait.claim<i1 = !trait.poly<0>>
    trait.return %e : !trait.claim<i1 = !trait.poly<0>>
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
trait.impl private @Has_sub(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>, %sub0: !trait.claim<@Sub0[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = !trait.poly<0>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.poly<0>]> {
    // expected-error @below {{requirement index 1 is out of range: '!trait.claim<@Sub0[!trait.poly<0>]>' has 1 requirements}}
    %s = trait.project %sub0[1] : !trait.claim<@Sub0[!trait.poly<0>]> -> !trait.claim<@Sup0[!trait.poly<0>]>
    trait.return %s : !trait.claim<@Sup0[!trait.poly<0>]>
  }
}
