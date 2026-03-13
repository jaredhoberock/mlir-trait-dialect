// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

!A = !trait.poly<0>
!B = !trait.poly<1>

trait.trait @Foo[!A] {
  trait.assoc_type @Out
}

trait.trait @Bar[!B] {
  trait.assoc_type @Out
}

func.func @bogus_cast(
    %val: !trait.proj<@Foo[!A], "Out">,
    %claim: !trait.claim<@Bar[!B]>
) -> !trait.proj<@Foo[!B], "Out"> {
  // expected-error @below {{claim trait '@Bar' does not match any projection trait in input or result types}}
  %cast = trait.proj.cast %val, %claim
    : !trait.proj<@Foo[!A], "Out">
    to !trait.proj<@Foo[!B], "Out">
    by !trait.claim<@Bar[!B]>
  return %cast : !trait.proj<@Foo[!B], "Out">
}
