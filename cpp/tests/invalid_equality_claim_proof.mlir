// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The equality arm never carries a proof. A `by @...` proof on an equality
// claim would fall into the asymmetric proof comparison the arm exists to
// avoid, so the parser refuses it.

// expected-error @below {{an equality claim may not carry a proof}}
func.func private @f(!trait.claim<i32 = i64 by @p>)
