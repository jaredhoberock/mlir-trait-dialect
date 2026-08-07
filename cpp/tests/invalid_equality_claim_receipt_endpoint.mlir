// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An equality endpoint must be receipt-free. An endpoint spelling a proven
// claim would carry a proof receipt into the arm that forbids one, so the
// checked constructor refuses it.

// expected-error @below {{type-equality endpoints must be receipt-free}}
func.func private @f(!trait.claim<!trait.claim<@Trait[i64] by @Trait_impl> = i64>)
