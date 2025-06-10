// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @foo
// CHECK-1: !trait.poly<0>
// CHECK-3: !trait.poly<-1>
// CHECK-NOT: !trait.poly<1>
// CHECK-NOT: !trait.poly<-2>
!T = !trait.poly<0>
!U = !trait.poly<fresh>
func.func @foo(%a: !T, %b: !U) -> !U {
  return %b : !U
}
