// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @I binds @A[i8]::Out to i32, and returns for its trait's requirement
// reflexive evidence coerced into @A[i8]::Out = i64 through an allegation of
// that equality. A projection of the requirement is replaced by the coercion,
// cloned with the equalities it cites, so the allegation stands where the
// projection stood and is decided there, and refused.

// CHECK: :[[@LINE+6]]:{{[0-9]+}}: error: 'trait.allege' op alleges '!trait.proj<@A[i8], "Out">' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<!trait.proj<@A[!T], "Out"> = i64> { trait.assoc_type @Out }
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
 trait.assoc_type @Out = i32
 %e = trait.allege !trait.proj<@A[i8], "Out"> = i64
 %r = trait.witness refl : !trait.claim<i64 = i64>
 %c = trait.coerce %r : !trait.claim<i64 = i64> to !trait.claim<!trait.proj<@A[i8], "Out"> = i64> via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
 trait.return %c : !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
}
func.func @main() -> i64 {
 %a = trait.witness @I for @A[i8]
 %e = trait.project %a[0] : !trait.claim<@A[i8] by @I> -> !trait.claim<!trait.proj<@A[i8], "Out"> = i64>
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<!trait.proj<@A[i8], "Out"> = i64>)
 return %r : i64
}

// -----

// An application claim returned through a coercion that cites a false
// equality: the coercion's allegation is cloned with it where the projection
// stood, and refused there.

// CHECK: :[[@LINE+8]]:{{[0-9]+}}: error: 'trait.allege' op alleges 'i32' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @m() -> i64 }
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) { trait.method @m() -> i64 { %c = arith.constant 9 : i64 trait.return %c : i64 } }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
 %b = trait.witness @B_i32 for @B[i32]
 %e = trait.allege i32 = i64
 %c = trait.coerce %b : !trait.claim<@B[i32] by @B_i32> to !trait.claim<@B[i32]> via (%e) : (!trait.claim<i32 = i64>)
 trait.return %c : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
 %a = trait.witness @A_i32 for @A[i32]
 %b = trait.project %a[0] : !trait.claim<@A[i32] by @A_i32> -> !trait.claim<@B[i32] by @B_i32>
 %v = trait.method.call %b @B[i32]::@m() : () -> i64 by @B_i32
 return %v : i64
}
