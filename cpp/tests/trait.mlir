// RUN: opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait @Add
// CHECK: func.func private @add(!trait.self, !trait.self) -> !trait.self

trait.trait @Add {
  func.func private @add(!trait.self, !trait.self) -> !trait.self
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait @PartialEq
// CHECK: func.func private @eq(!trait.self, !trait.self) -> i1
// CHECK: func.func private @neq(!trait.self, !trait.self) -> i1

trait.trait @PartialEq {
  func.func private @eq(!trait.self, !trait.self) -> i1
  func.func private @neq(!trait.self, !trait.self) -> i1
}
