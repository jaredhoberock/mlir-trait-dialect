// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Sixty-four traits whose impls require one another in a ring, the last one
// asking for the first at tuple<T>: selection asks about @P1[i32], @P2[i32],
// ... @P64[i32], @P1[tuple<i32>], and around again forever. Every frame is a
// new application, and consecutive frames name different traits, so what stops
// the descent is how many frames stand on the chain and not how often any one
// trait recurs among them -- a ring of sixty-four traits is refused at the
// depth a ring of two meets, and the stack the walk runs on is never at issue.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it
// CHECK: note: required by {{.*}}@P1[i32]
// CHECK: note: required by {{.*}}@P2[i32]
// CHECK: note: required by {{.*}}@P3[i32]
// CHECK: note: {{.*}} more frame(s) elided

trait.trait private @P1[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P2[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P3[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P4[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P5[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P6[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P7[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P8[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P9[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P10[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P11[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P12[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P13[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P14[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P15[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P16[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P17[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P18[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P19[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P20[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P21[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P22[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P23[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P24[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P25[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P26[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P27[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P28[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P29[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P30[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P31[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P32[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P33[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P34[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P35[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P36[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P37[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P38[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P39[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P40[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P41[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P42[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P43[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P44[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P45[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P46[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P47[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P48[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P49[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P50[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P51[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P52[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P53[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P54[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P55[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P56[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P57[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P58[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P59[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P60[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P61[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P62[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P63[!trait.poly<0>] { func.func private @m() -> i64 }
trait.trait private @P64[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @P1_all for @P1[!trait.poly<0>] where [@P2[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P2[!trait.poly<0>]
    %v = trait.method.call %a @P2[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P2_all for @P2[!trait.poly<0>] where [@P3[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P3[!trait.poly<0>]
    %v = trait.method.call %a @P3[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P3_all for @P3[!trait.poly<0>] where [@P4[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P4[!trait.poly<0>]
    %v = trait.method.call %a @P4[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P4_all for @P4[!trait.poly<0>] where [@P5[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P5[!trait.poly<0>]
    %v = trait.method.call %a @P5[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P5_all for @P5[!trait.poly<0>] where [@P6[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P6[!trait.poly<0>]
    %v = trait.method.call %a @P6[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P6_all for @P6[!trait.poly<0>] where [@P7[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P7[!trait.poly<0>]
    %v = trait.method.call %a @P7[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P7_all for @P7[!trait.poly<0>] where [@P8[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P8[!trait.poly<0>]
    %v = trait.method.call %a @P8[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P8_all for @P8[!trait.poly<0>] where [@P9[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P9[!trait.poly<0>]
    %v = trait.method.call %a @P9[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P9_all for @P9[!trait.poly<0>] where [@P10[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P10[!trait.poly<0>]
    %v = trait.method.call %a @P10[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P10_all for @P10[!trait.poly<0>] where [@P11[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P11[!trait.poly<0>]
    %v = trait.method.call %a @P11[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P11_all for @P11[!trait.poly<0>] where [@P12[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P12[!trait.poly<0>]
    %v = trait.method.call %a @P12[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P12_all for @P12[!trait.poly<0>] where [@P13[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P13[!trait.poly<0>]
    %v = trait.method.call %a @P13[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P13_all for @P13[!trait.poly<0>] where [@P14[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P14[!trait.poly<0>]
    %v = trait.method.call %a @P14[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P14_all for @P14[!trait.poly<0>] where [@P15[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P15[!trait.poly<0>]
    %v = trait.method.call %a @P15[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P15_all for @P15[!trait.poly<0>] where [@P16[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P16[!trait.poly<0>]
    %v = trait.method.call %a @P16[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P16_all for @P16[!trait.poly<0>] where [@P17[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P17[!trait.poly<0>]
    %v = trait.method.call %a @P17[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P17_all for @P17[!trait.poly<0>] where [@P18[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P18[!trait.poly<0>]
    %v = trait.method.call %a @P18[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P18_all for @P18[!trait.poly<0>] where [@P19[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P19[!trait.poly<0>]
    %v = trait.method.call %a @P19[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P19_all for @P19[!trait.poly<0>] where [@P20[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P20[!trait.poly<0>]
    %v = trait.method.call %a @P20[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P20_all for @P20[!trait.poly<0>] where [@P21[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P21[!trait.poly<0>]
    %v = trait.method.call %a @P21[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P21_all for @P21[!trait.poly<0>] where [@P22[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P22[!trait.poly<0>]
    %v = trait.method.call %a @P22[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P22_all for @P22[!trait.poly<0>] where [@P23[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P23[!trait.poly<0>]
    %v = trait.method.call %a @P23[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P23_all for @P23[!trait.poly<0>] where [@P24[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P24[!trait.poly<0>]
    %v = trait.method.call %a @P24[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P24_all for @P24[!trait.poly<0>] where [@P25[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P25[!trait.poly<0>]
    %v = trait.method.call %a @P25[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P25_all for @P25[!trait.poly<0>] where [@P26[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P26[!trait.poly<0>]
    %v = trait.method.call %a @P26[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P26_all for @P26[!trait.poly<0>] where [@P27[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P27[!trait.poly<0>]
    %v = trait.method.call %a @P27[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P27_all for @P27[!trait.poly<0>] where [@P28[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P28[!trait.poly<0>]
    %v = trait.method.call %a @P28[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P28_all for @P28[!trait.poly<0>] where [@P29[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P29[!trait.poly<0>]
    %v = trait.method.call %a @P29[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P29_all for @P29[!trait.poly<0>] where [@P30[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P30[!trait.poly<0>]
    %v = trait.method.call %a @P30[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P30_all for @P30[!trait.poly<0>] where [@P31[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P31[!trait.poly<0>]
    %v = trait.method.call %a @P31[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P31_all for @P31[!trait.poly<0>] where [@P32[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P32[!trait.poly<0>]
    %v = trait.method.call %a @P32[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P32_all for @P32[!trait.poly<0>] where [@P33[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P33[!trait.poly<0>]
    %v = trait.method.call %a @P33[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P33_all for @P33[!trait.poly<0>] where [@P34[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P34[!trait.poly<0>]
    %v = trait.method.call %a @P34[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P34_all for @P34[!trait.poly<0>] where [@P35[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P35[!trait.poly<0>]
    %v = trait.method.call %a @P35[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P35_all for @P35[!trait.poly<0>] where [@P36[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P36[!trait.poly<0>]
    %v = trait.method.call %a @P36[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P36_all for @P36[!trait.poly<0>] where [@P37[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P37[!trait.poly<0>]
    %v = trait.method.call %a @P37[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P37_all for @P37[!trait.poly<0>] where [@P38[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P38[!trait.poly<0>]
    %v = trait.method.call %a @P38[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P38_all for @P38[!trait.poly<0>] where [@P39[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P39[!trait.poly<0>]
    %v = trait.method.call %a @P39[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P39_all for @P39[!trait.poly<0>] where [@P40[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P40[!trait.poly<0>]
    %v = trait.method.call %a @P40[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P40_all for @P40[!trait.poly<0>] where [@P41[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P41[!trait.poly<0>]
    %v = trait.method.call %a @P41[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P41_all for @P41[!trait.poly<0>] where [@P42[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P42[!trait.poly<0>]
    %v = trait.method.call %a @P42[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P42_all for @P42[!trait.poly<0>] where [@P43[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P43[!trait.poly<0>]
    %v = trait.method.call %a @P43[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P43_all for @P43[!trait.poly<0>] where [@P44[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P44[!trait.poly<0>]
    %v = trait.method.call %a @P44[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P44_all for @P44[!trait.poly<0>] where [@P45[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P45[!trait.poly<0>]
    %v = trait.method.call %a @P45[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P45_all for @P45[!trait.poly<0>] where [@P46[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P46[!trait.poly<0>]
    %v = trait.method.call %a @P46[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P46_all for @P46[!trait.poly<0>] where [@P47[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P47[!trait.poly<0>]
    %v = trait.method.call %a @P47[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P47_all for @P47[!trait.poly<0>] where [@P48[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P48[!trait.poly<0>]
    %v = trait.method.call %a @P48[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P48_all for @P48[!trait.poly<0>] where [@P49[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P49[!trait.poly<0>]
    %v = trait.method.call %a @P49[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P49_all for @P49[!trait.poly<0>] where [@P50[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P50[!trait.poly<0>]
    %v = trait.method.call %a @P50[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P50_all for @P50[!trait.poly<0>] where [@P51[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P51[!trait.poly<0>]
    %v = trait.method.call %a @P51[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P51_all for @P51[!trait.poly<0>] where [@P52[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P52[!trait.poly<0>]
    %v = trait.method.call %a @P52[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P52_all for @P52[!trait.poly<0>] where [@P53[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P53[!trait.poly<0>]
    %v = trait.method.call %a @P53[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P53_all for @P53[!trait.poly<0>] where [@P54[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P54[!trait.poly<0>]
    %v = trait.method.call %a @P54[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P54_all for @P54[!trait.poly<0>] where [@P55[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P55[!trait.poly<0>]
    %v = trait.method.call %a @P55[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P55_all for @P55[!trait.poly<0>] where [@P56[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P56[!trait.poly<0>]
    %v = trait.method.call %a @P56[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P56_all for @P56[!trait.poly<0>] where [@P57[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P57[!trait.poly<0>]
    %v = trait.method.call %a @P57[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P57_all for @P57[!trait.poly<0>] where [@P58[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P58[!trait.poly<0>]
    %v = trait.method.call %a @P58[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P58_all for @P58[!trait.poly<0>] where [@P59[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P59[!trait.poly<0>]
    %v = trait.method.call %a @P59[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P59_all for @P59[!trait.poly<0>] where [@P60[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P60[!trait.poly<0>]
    %v = trait.method.call %a @P60[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P60_all for @P60[!trait.poly<0>] where [@P61[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P61[!trait.poly<0>]
    %v = trait.method.call %a @P61[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P61_all for @P61[!trait.poly<0>] where [@P62[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P62[!trait.poly<0>]
    %v = trait.method.call %a @P62[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P62_all for @P62[!trait.poly<0>] where [@P63[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P63[!trait.poly<0>]
    %v = trait.method.call %a @P63[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P63_all for @P63[!trait.poly<0>] where [@P64[!trait.poly<0>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P64[!trait.poly<0>]
    %v = trait.method.call %a @P64[!trait.poly<0>]::@m() : () -> i64
    return %v : i64
  }
}
trait.impl private @P64_all for @P64[!trait.poly<0>] where [@P1[tuple<!trait.poly<0>>]] {
  func.func @m() -> i64 {
    %a = trait.assume @P1[tuple<!trait.poly<0>>]
    %v = trait.method.call %a @P1[tuple<!trait.poly<0>>]::@m() : () -> i64
    return %v : i64
  }
}
func.func @main() -> i64 {
  %w = trait.allege @P1[i32]
  %v = trait.method.call %w @P1[i32]::@m() : () -> i64
  return %v : i64
}
