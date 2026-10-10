// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

trait.trait private @Trait0(%self: !trait.claim<@Trait0[!trait.poly<0>, !trait.poly<1>]>) {
  trait.assoc_type @Output
}

trait.trait private @Trait1(%self: !trait.claim<@Trait1[!trait.poly<0>]>) {
  // expected-error @+1 {{function 'method' result type contains type parameter '!trait.poly<4>' that is not determined by any input type}}
  trait.method @method(
    !trait.poly<0>,
    !trait.poly<3>
  ) -> tuple<!trait.proj<@Trait0[!trait.poly<4>, !trait.poly<0>], "Output">>
}

// -----

trait.trait private @Trait0(%self: !trait.claim<@Trait0[!trait.poly<0>, !trait.poly<1>]>) {}

trait.trait private @Trait1(%self: !trait.claim<@Trait1[!trait.poly<0>]>) {
  trait.method @method(
    !trait.poly<0>,
    !trait.claim<@Trait0[!trait.poly<3>, !trait.poly<0>]>
  ) -> !trait.poly<3>
}
