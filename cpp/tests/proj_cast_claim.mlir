// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test trait.proj.cast where input/result are claim types containing projections.
// The impl method derives @Inner[i64] from @Outer[i32] via assume + project + proj.cast,
// then calls @Inner[i64]::@id through the resolved claim.

!S = !trait.poly<0>

trait.trait @Inner[!S] {
  func.func private @id(!S) -> !S
}

trait.trait @Outer[!S] where [@Inner[!trait.proj<@Outer[!S], "Assoc">]] {
  trait.assoc_type @Assoc
  func.func private @get(!S) -> !trait.proj<@Outer[!S], "Assoc">
}

trait.impl @Inner_i64 for @Inner[i64] {
  func.func @id(%self: i64) -> i64 {
    return %self : i64
  }
}

trait.impl @Outer_i32 for @Outer[i32] {
  trait.assoc_type @Assoc = i64

  func.func @get(%self: i32) -> i64 {
    %outer = trait.assume @Outer[i32]
    %inner_unresolved = trait.project %outer : @Outer[i32] to @Inner[!trait.proj<@Outer[i32], "Assoc">]
    %inner = trait.proj.cast %inner_unresolved, %outer
        : !trait.claim<@Inner[!trait.proj<@Outer[i32], "Assoc">]>
        to !trait.claim<@Inner[i64]>
        by !trait.claim<@Outer[i32]>
    %ext = arith.extsi %self : i32 to i64
    %res = trait.method.call %inner @Inner[i64]::@id(%ext) : (i64) -> i64
    return %res : i64
  }
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: !trait.proj
// CHECK-NOT: !trait.claim
// CHECK: call @
// CHECK: return %{{.*}} : i64
func.func @main() -> i64 {
  %x = arith.constant 3 : i32
  %a = trait.allege @Outer[i32]
  %r = trait.method.call %a @Outer[i32]::@get(%x) : (i32) -> i64
  return %r : i64
}
