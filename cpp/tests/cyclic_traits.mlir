// RUN: mlir-opt %s -pass-pipeline='builtin.module(verify-acyclic-traits)' -verify-diagnostics -split-input-file

// -----
// Mutual cycle: A <-> B (with an unrelated acyclic edge from A to C).
// expected-error @+1 {{cycle in trait `given` clause}}
trait.trait @A[!trait.poly<0>] given [
  @B[!trait.poly<0>],
  @C[!trait.poly<0>]
] {}

trait.trait @B[!trait.poly<0>] given [
  @A[!trait.poly<0>]
] {}

trait.trait @C[!trait.poly<0>] {} // acyclic leaf

// -----
// 3-node cycle: X -> Y -> Z -> X
// expected-error @+1 {{cycle in trait `given` clause}}
trait.trait @X[!trait.poly<0>] given [
  @Y[!trait.poly<0>]
] {}

trait.trait @Y[!trait.poly<0>] given [
  @Z[!trait.poly<0>]
] {}

trait.trait @Z[!trait.poly<0>] given [
  @X[!trait.poly<0>]
] {}

// -----
// Non-trivial, with branching and irrelevant parameters (types don’t break cycles):
//   T1 -> T2[!S], T4[!S]   (T4 is acyclic noise)
//   T2 -> T3[!S]
//   T3 -> T1[!U]           (back to T1 with a different param index)
// expected-error @+1 {{cycle in trait `given` clause}}
trait.trait @T1[!trait.poly<0>] given [
  @T2[!trait.poly<0>],
  @T4[!trait.poly<0>]
] {}

trait.trait @T2[!trait.poly<0>] given [
  @T3[!trait.poly<0>]
] {}

trait.trait @T3[!trait.poly<0>] given [
  @T1[!trait.poly<0>]
] {}

trait.trait @T4[!trait.poly<0>] {} // acyclic leaf
