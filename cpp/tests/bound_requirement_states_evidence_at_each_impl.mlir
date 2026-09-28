// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A generic associated type's bound is a requirement quantified over the
// associated type's own parameter: `type A<X>: Marker where X: Marker`. Each
// impl states the evidence that it holds for every argument, one body of each
// form: an impl that holds everywhere, the binder's premise, the impl's own
// where-clause entry, an impl whose premise the binder's premise discharges,
// and the impl stating it.

!S = !trait.poly<0>
!X = !trait.poly<1>
!P = !trait.poly<2>

trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}

trait.impl private @Marker_i1 for @Marker[i1] {}
trait.impl private @Marker_wrap for @Marker[tuple<!P>] where [@Marker[!P]] {}

trait.impl private @Has_i32 for @Has[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[i32], "A", [!X]>] by @Marker_i1>] {
  trait.assoc_type @A<[!X]> = i1
}

trait.impl private @Has_i64 for @Has[i64]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[i64], "A", [!X]>] by premise 0>] {
  trait.assoc_type @A<[!X]> = !X
}

trait.impl private @Has_tuple for @Has[tuple<!P>] where [@Marker[!P]]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[tuple<!P>], "A", [!X]>] by where 0>] {
  trait.assoc_type @A<[!X]> = !P
}

trait.impl private @Has_f32 for @Has[f32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[f32], "A", [!X]>] by @Marker_wrap[!P = !X] given [premise 0]>] {
  trait.assoc_type @A<[!X]> = tuple<!X>
}

trait.trait private @Self[!S] where [forall [!X] -> @Self[!trait.proj<@Self[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Self_i32 for @Self[i32]
    bound_evidence [#trait<bound_evidence 0: forall [!X] -> @Self[!trait.proj<@Self[i32], "A", [!X]>] by @Self_i32>] {
  trait.assoc_type @A<[!X]> = i32
}

// CHECK: trait.trait private @Has[!trait.poly<0>] where [forall [!trait.poly<1>] where [@Marker[!trait.poly<1>]] -> @Marker[!trait.proj<@Has[!trait.poly<0>], "A", [!trait.poly<1>]>]]
// CHECK: bound_evidence [#trait<bound_evidence 0: forall [!trait.poly<1>] where [@Marker[!trait.poly<1>]] -> @Marker[!trait.proj<@Has[i32], "A", [!trait.poly<1>]>] by @Marker_i1>]
// CHECK: by premise 0>]
// CHECK: by where 0>]
// CHECK: by @Marker_wrap[!trait.poly<2> = !trait.poly<1>] given [premise 0]>]
// CHECK: by @Self_i32>]
