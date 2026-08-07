// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The equality arm never carries a proof receipt. A `by @...` receipt on an
// equality claim would fall into the asymmetric proof comparison the arm
// exists to avoid, so the parser refuses it.

// expected-error @below {{an equality claim may not carry a proof receipt}}
func.func private @f(!trait.claim<i32 = i64 by @p>)
