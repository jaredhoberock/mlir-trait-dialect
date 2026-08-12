// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// An impl's `witnesses` array carries `#trait.witness` entries and nothing else.
// The element-type constraint is the array's only shape rule -- each arm (an
// equality-armed certificate or an application-armed discharge) is read at use,
// not policed here -- so a non-witness element is the one refusal at this level.

!S = !trait.poly<0>

trait.trait @Host[!S] {}

// expected-error @below {{attribute 'witnesses' failed to satisfy constraint: array of impl witnesses}}
trait.impl @Host_i64 for @Host[i64]
    witnesses ["not a witness"] {
}
