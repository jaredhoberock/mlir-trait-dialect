// RUN: opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait @Add
// CHECK: trait.method @add(!trait.self, !trait.self) -> !trait.self

trait.trait @Add {
  trait.method @add(!trait.self, !trait.self) -> !trait.self
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait @PartialEq
// CHECK: trait.method @eq(!trait.self, !trait.self) -> i1
// CHECK: trait.method @neq(!trait.self, !trait.self) -> i1

trait.trait @PartialEq {
  trait.method @eq(!trait.self, !trait.self) -> i1
  trait.method @neq(!trait.self, !trait.self) -> i1
}
