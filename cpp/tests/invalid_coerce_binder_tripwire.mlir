// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The closure decides ground equational entailment: it treats every
// !trait.poly as a distinct constant, never binding one variable to another. An
// equality about !trait.poly<0> says nothing about !trait.poly<1>, so a coerce
// between !trait.poly<1> and i64 citing only !trait.poly<0> = i64 is refused.
// This is the tripwire if the endpoint grammar ever grows binders at this layer.

func.func @binder(%v: !trait.poly<1>, %e: !trait.claim<!trait.poly<0> = i64>) -> i64 {
  // expected-error @below {{are not equal under the cited equalities}}
  %c = trait.coerce %v : !trait.poly<1> to i64 via (%e) : (!trait.claim<!trait.poly<0> = i64>)
  return %c : i64
}
