// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

trait.trait @Trait0[!trait.poly<0>, !trait.poly<1>] {
  trait.assoc_type @Output
}

trait.trait @Trait1[!trait.poly<2>] {
  // expected-error @+1 {{function 'method' result type contains type parameter '!trait.poly<4>' that is not determined by any input type}}
  func.func nested @method(
    !trait.poly<2>,
    !trait.poly<3>
  ) -> tuple<!trait.proj<@Trait0[!trait.poly<4>, !trait.poly<2>], "Output">>
}

// -----

trait.trait @Trait0[!trait.poly<0>, !trait.poly<1>] {}

trait.trait @Trait1[!trait.poly<2>] {
  func.func nested @method(
    !trait.poly<2>,
    !trait.claim<@Trait0[!trait.poly<3>, !trait.poly<2>]>
  ) -> !trait.poly<3>
}
