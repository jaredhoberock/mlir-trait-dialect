// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A generic associated type's bound is a requirement quantified over the
// associated type's own parameter: `type A<X>: Marker where X: Marker`. Each
// impl states the evidence that it holds for every argument, one body of each
// form: an impl that holds everywhere, the binder's premise, the impl's own
// where-clause entry, an impl whose premise the binder's premise discharges,
// the impl stating it, and reflexivity.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>

trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}

trait.impl private @Marker_i1 for @Marker[i1] {}
trait.impl private @Marker_wrap for @Marker[tuple<!P>] where [@Marker[!P]] {}

trait.impl private @Has_i32 for @Has[i32]
    witnesses [#trait<witness requirement 0 by @Marker_i1>] {
  trait.assoc_type @A<[!X]> = i1
}

trait.impl private @Has_i64 for @Has[i64]
    witnesses [#trait<witness requirement 0 by premise 0>] {
  trait.assoc_type @A<[!X]> = !X
}

trait.impl private @Has_tuple for @Has[tuple<!P>] where [@Marker[!P]]
    witnesses [#trait<witness requirement 0 by where 0>] {
  trait.assoc_type @A<[!X]> = !P
}

trait.impl private @Has_f32 for @Has[f32]
    witnesses [#trait<witness requirement 0 by @Marker_wrap[!P = !trait.bound<0>] given [premise 0]>] {
  trait.assoc_type @A<[!X]> = tuple<!X>
}

trait.trait private @Self[!S] where [forall [!trait.bound<0>] -> @Self[!trait.proj<@Self[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Self_i32 for @Self[i32]
    witnesses [#trait<witness requirement 0 by @Self_i32>] {
  trait.assoc_type @A<[!X]> = i32
}

// An equality conclusion is proved by reflexivity when the impl's own binding
// makes its two sides one type.
trait.trait private @Same[!S] where [forall [!trait.bound<0>] -> !trait.proj<@Same[!S], "A", [!trait.bound<0>]> = !trait.bound<0>] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Same_i32 for @Same[i32]
    witnesses [#trait<witness requirement 0 by refl>] {
  trait.assoc_type @A<[!X]> = !X
}

// CHECK: trait.trait private @Has[!trait.poly<0>] where [forall [!trait.bound<0>] where [@Marker[!trait.bound<0>]] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.bound<0>]>]]
// CHECK: witnesses [#trait<witness requirement 0 by @Marker_i1>]
// CHECK: by premise 0>]
// CHECK: by where 0>]
// CHECK: by @Marker_wrap[!trait.poly<2> = !trait.bound<0>] given [premise 0]>]
// CHECK: by @Self_i32>]
// CHECK: witnesses [#trait<witness requirement 0 by refl>]
