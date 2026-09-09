// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A malformed application reports parser errors without asserting on its null result.
// RUN: mlir-opt %s -verify-diagnostics

module {
  // expected-error @+2 {{expected attribute value}}
  // expected-error @+1 {{expected a TraitApplicationAttr}}
  %w = trait.witness @impl for nope
}
