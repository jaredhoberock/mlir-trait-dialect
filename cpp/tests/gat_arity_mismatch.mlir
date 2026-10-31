// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A generic associated type's own parameters are a count its trait declares,
// and an impl's binding takes that count by position, so only a projection
// states how many arguments it supplies: one supplying another number than the
// associated type takes is diagnosed where it stands.

!S = !trait.poly<0>

trait.trait private @HasGAT(!trait.claim<@HasGAT[!S]>) assoc {Item = 1}

// expected-error @+1 {{projection '!trait.proj<@HasGAT[i32], "Item">' supplies 0 argument(s) to associated type 'Item', which takes 1}}
func.func private @takes(%x: !trait.proj<@HasGAT[i32], "Item">)

// -----

!S = !trait.poly<0>

trait.trait private @OneParam(!trait.claim<@OneParam[!S]>) assoc {Item = 1}

// expected-error @+1 {{projection '!trait.proj<@OneParam[i32], "Item", [i64, i8]>' supplies 2 argument(s) to associated type 'Item', which takes 1}}
func.func private @takes(%x: !trait.proj<@OneParam[i32], "Item", [i64, i8]>)
