// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// An equality endpoint must contain no proven claim. An endpoint spelling a
// proven claim would carry a proof into the arm that forbids one, so the
// checked constructor refuses it.

// expected-error @below {{a type-equality endpoint must not contain a proven claim}}
func.func private @f(!trait.claim<!trait.claim<@Trait[i64] by @Trait_impl> = i64>)
