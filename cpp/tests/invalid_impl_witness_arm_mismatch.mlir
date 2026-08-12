// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An impl's two witness arrays share one attribute shape but split by predicate
// arm. The `premises` array carries projection-resolution certificates, so every
// entry must witness an equality. An application-headed witness names an
// obligation, which belongs in `discharges`; the impl verifier refuses it here.

!S = !trait.poly<0>

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}
trait.trait @Host[!S] {}

// expected-error @below {{an impl premise must witness an equality}}
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<witness @Sib[i64] by @Sib_i64>] {
}

// -----

// Symmetric refusal for the other array: the `discharges` array carries the
// obligations an impl supplies, so every entry must witness a trait application.
// An equality-headed witness is a certificate, which belongs in `premises`.

!S = !trait.poly<0>

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}
trait.trait @Host[!S] {}

// expected-error @below {{an impl discharge must witness a trait application}}
trait.impl @Host_i64 for @Host[i64]
    discharges [#trait<witness !trait.proj<@Sib[i64], "Elem"> = i64 by @Sib_i64>] {
}
