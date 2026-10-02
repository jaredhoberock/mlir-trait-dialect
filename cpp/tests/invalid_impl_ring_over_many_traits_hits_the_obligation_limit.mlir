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

trait.trait private @P1(%self: !trait.claim<@P1[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P2(%self: !trait.claim<@P2[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P3(%self: !trait.claim<@P3[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P4(%self: !trait.claim<@P4[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P5(%self: !trait.claim<@P5[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P6(%self: !trait.claim<@P6[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P7(%self: !trait.claim<@P7[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P8(%self: !trait.claim<@P8[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P9(%self: !trait.claim<@P9[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P10(%self: !trait.claim<@P10[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P11(%self: !trait.claim<@P11[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P12(%self: !trait.claim<@P12[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P13(%self: !trait.claim<@P13[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P14(%self: !trait.claim<@P14[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P15(%self: !trait.claim<@P15[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P16(%self: !trait.claim<@P16[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P17(%self: !trait.claim<@P17[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P18(%self: !trait.claim<@P18[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P19(%self: !trait.claim<@P19[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P20(%self: !trait.claim<@P20[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P21(%self: !trait.claim<@P21[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P22(%self: !trait.claim<@P22[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P23(%self: !trait.claim<@P23[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P24(%self: !trait.claim<@P24[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P25(%self: !trait.claim<@P25[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P26(%self: !trait.claim<@P26[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P27(%self: !trait.claim<@P27[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P28(%self: !trait.claim<@P28[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P29(%self: !trait.claim<@P29[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P30(%self: !trait.claim<@P30[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P31(%self: !trait.claim<@P31[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P32(%self: !trait.claim<@P32[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P33(%self: !trait.claim<@P33[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P34(%self: !trait.claim<@P34[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P35(%self: !trait.claim<@P35[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P36(%self: !trait.claim<@P36[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P37(%self: !trait.claim<@P37[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P38(%self: !trait.claim<@P38[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P39(%self: !trait.claim<@P39[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P40(%self: !trait.claim<@P40[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P41(%self: !trait.claim<@P41[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P42(%self: !trait.claim<@P42[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P43(%self: !trait.claim<@P43[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P44(%self: !trait.claim<@P44[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P45(%self: !trait.claim<@P45[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P46(%self: !trait.claim<@P46[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P47(%self: !trait.claim<@P47[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P48(%self: !trait.claim<@P48[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P49(%self: !trait.claim<@P49[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P50(%self: !trait.claim<@P50[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P51(%self: !trait.claim<@P51[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P52(%self: !trait.claim<@P52[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P53(%self: !trait.claim<@P53[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P54(%self: !trait.claim<@P54[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P55(%self: !trait.claim<@P55[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P56(%self: !trait.claim<@P56[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P57(%self: !trait.claim<@P57[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P58(%self: !trait.claim<@P58[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P59(%self: !trait.claim<@P59[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P60(%self: !trait.claim<@P60[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P61(%self: !trait.claim<@P61[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P62(%self: !trait.claim<@P62[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P63(%self: !trait.claim<@P63[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.trait private @P64(%self: !trait.claim<@P64[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @P1_all(%self: !trait.claim<@P1[!trait.poly<0>]>, %p2: !trait.claim<@P2[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p2 @P2[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P2_all(%self: !trait.claim<@P2[!trait.poly<0>]>, %p3: !trait.claim<@P3[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p3 @P3[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P3_all(%self: !trait.claim<@P3[!trait.poly<0>]>, %p4: !trait.claim<@P4[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p4 @P4[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P4_all(%self: !trait.claim<@P4[!trait.poly<0>]>, %p5: !trait.claim<@P5[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p5 @P5[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P5_all(%self: !trait.claim<@P5[!trait.poly<0>]>, %p6: !trait.claim<@P6[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p6 @P6[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P6_all(%self: !trait.claim<@P6[!trait.poly<0>]>, %p7: !trait.claim<@P7[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p7 @P7[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P7_all(%self: !trait.claim<@P7[!trait.poly<0>]>, %p8: !trait.claim<@P8[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p8 @P8[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P8_all(%self: !trait.claim<@P8[!trait.poly<0>]>, %p9: !trait.claim<@P9[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p9 @P9[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P9_all(%self: !trait.claim<@P9[!trait.poly<0>]>, %p10: !trait.claim<@P10[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p10 @P10[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P10_all(%self: !trait.claim<@P10[!trait.poly<0>]>, %p11: !trait.claim<@P11[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p11 @P11[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P11_all(%self: !trait.claim<@P11[!trait.poly<0>]>, %p12: !trait.claim<@P12[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p12 @P12[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P12_all(%self: !trait.claim<@P12[!trait.poly<0>]>, %p13: !trait.claim<@P13[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p13 @P13[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P13_all(%self: !trait.claim<@P13[!trait.poly<0>]>, %p14: !trait.claim<@P14[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p14 @P14[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P14_all(%self: !trait.claim<@P14[!trait.poly<0>]>, %p15: !trait.claim<@P15[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p15 @P15[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P15_all(%self: !trait.claim<@P15[!trait.poly<0>]>, %p16: !trait.claim<@P16[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p16 @P16[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P16_all(%self: !trait.claim<@P16[!trait.poly<0>]>, %p17: !trait.claim<@P17[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p17 @P17[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P17_all(%self: !trait.claim<@P17[!trait.poly<0>]>, %p18: !trait.claim<@P18[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p18 @P18[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P18_all(%self: !trait.claim<@P18[!trait.poly<0>]>, %p19: !trait.claim<@P19[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p19 @P19[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P19_all(%self: !trait.claim<@P19[!trait.poly<0>]>, %p20: !trait.claim<@P20[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p20 @P20[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P20_all(%self: !trait.claim<@P20[!trait.poly<0>]>, %p21: !trait.claim<@P21[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p21 @P21[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P21_all(%self: !trait.claim<@P21[!trait.poly<0>]>, %p22: !trait.claim<@P22[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p22 @P22[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P22_all(%self: !trait.claim<@P22[!trait.poly<0>]>, %p23: !trait.claim<@P23[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p23 @P23[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P23_all(%self: !trait.claim<@P23[!trait.poly<0>]>, %p24: !trait.claim<@P24[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p24 @P24[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P24_all(%self: !trait.claim<@P24[!trait.poly<0>]>, %p25: !trait.claim<@P25[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p25 @P25[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P25_all(%self: !trait.claim<@P25[!trait.poly<0>]>, %p26: !trait.claim<@P26[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p26 @P26[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P26_all(%self: !trait.claim<@P26[!trait.poly<0>]>, %p27: !trait.claim<@P27[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p27 @P27[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P27_all(%self: !trait.claim<@P27[!trait.poly<0>]>, %p28: !trait.claim<@P28[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p28 @P28[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P28_all(%self: !trait.claim<@P28[!trait.poly<0>]>, %p29: !trait.claim<@P29[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p29 @P29[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P29_all(%self: !trait.claim<@P29[!trait.poly<0>]>, %p30: !trait.claim<@P30[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p30 @P30[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P30_all(%self: !trait.claim<@P30[!trait.poly<0>]>, %p31: !trait.claim<@P31[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p31 @P31[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P31_all(%self: !trait.claim<@P31[!trait.poly<0>]>, %p32: !trait.claim<@P32[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p32 @P32[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P32_all(%self: !trait.claim<@P32[!trait.poly<0>]>, %p33: !trait.claim<@P33[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p33 @P33[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P33_all(%self: !trait.claim<@P33[!trait.poly<0>]>, %p34: !trait.claim<@P34[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p34 @P34[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P34_all(%self: !trait.claim<@P34[!trait.poly<0>]>, %p35: !trait.claim<@P35[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p35 @P35[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P35_all(%self: !trait.claim<@P35[!trait.poly<0>]>, %p36: !trait.claim<@P36[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p36 @P36[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P36_all(%self: !trait.claim<@P36[!trait.poly<0>]>, %p37: !trait.claim<@P37[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p37 @P37[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P37_all(%self: !trait.claim<@P37[!trait.poly<0>]>, %p38: !trait.claim<@P38[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p38 @P38[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P38_all(%self: !trait.claim<@P38[!trait.poly<0>]>, %p39: !trait.claim<@P39[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p39 @P39[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P39_all(%self: !trait.claim<@P39[!trait.poly<0>]>, %p40: !trait.claim<@P40[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p40 @P40[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P40_all(%self: !trait.claim<@P40[!trait.poly<0>]>, %p41: !trait.claim<@P41[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p41 @P41[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P41_all(%self: !trait.claim<@P41[!trait.poly<0>]>, %p42: !trait.claim<@P42[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p42 @P42[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P42_all(%self: !trait.claim<@P42[!trait.poly<0>]>, %p43: !trait.claim<@P43[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p43 @P43[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P43_all(%self: !trait.claim<@P43[!trait.poly<0>]>, %p44: !trait.claim<@P44[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p44 @P44[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P44_all(%self: !trait.claim<@P44[!trait.poly<0>]>, %p45: !trait.claim<@P45[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p45 @P45[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P45_all(%self: !trait.claim<@P45[!trait.poly<0>]>, %p46: !trait.claim<@P46[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p46 @P46[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P46_all(%self: !trait.claim<@P46[!trait.poly<0>]>, %p47: !trait.claim<@P47[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p47 @P47[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P47_all(%self: !trait.claim<@P47[!trait.poly<0>]>, %p48: !trait.claim<@P48[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p48 @P48[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P48_all(%self: !trait.claim<@P48[!trait.poly<0>]>, %p49: !trait.claim<@P49[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p49 @P49[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P49_all(%self: !trait.claim<@P49[!trait.poly<0>]>, %p50: !trait.claim<@P50[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p50 @P50[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P50_all(%self: !trait.claim<@P50[!trait.poly<0>]>, %p51: !trait.claim<@P51[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p51 @P51[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P51_all(%self: !trait.claim<@P51[!trait.poly<0>]>, %p52: !trait.claim<@P52[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p52 @P52[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P52_all(%self: !trait.claim<@P52[!trait.poly<0>]>, %p53: !trait.claim<@P53[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p53 @P53[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P53_all(%self: !trait.claim<@P53[!trait.poly<0>]>, %p54: !trait.claim<@P54[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p54 @P54[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P54_all(%self: !trait.claim<@P54[!trait.poly<0>]>, %p55: !trait.claim<@P55[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p55 @P55[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P55_all(%self: !trait.claim<@P55[!trait.poly<0>]>, %p56: !trait.claim<@P56[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p56 @P56[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P56_all(%self: !trait.claim<@P56[!trait.poly<0>]>, %p57: !trait.claim<@P57[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p57 @P57[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P57_all(%self: !trait.claim<@P57[!trait.poly<0>]>, %p58: !trait.claim<@P58[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p58 @P58[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P58_all(%self: !trait.claim<@P58[!trait.poly<0>]>, %p59: !trait.claim<@P59[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p59 @P59[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P59_all(%self: !trait.claim<@P59[!trait.poly<0>]>, %p60: !trait.claim<@P60[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p60 @P60[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P60_all(%self: !trait.claim<@P60[!trait.poly<0>]>, %p61: !trait.claim<@P61[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p61 @P61[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P61_all(%self: !trait.claim<@P61[!trait.poly<0>]>, %p62: !trait.claim<@P62[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p62 @P62[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P62_all(%self: !trait.claim<@P62[!trait.poly<0>]>, %p63: !trait.claim<@P63[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p63 @P63[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P63_all(%self: !trait.claim<@P63[!trait.poly<0>]>, %p64: !trait.claim<@P64[!trait.poly<0>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p64 @P64[!trait.poly<0>]::@m() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @P64_all(%self: !trait.claim<@P64[!trait.poly<0>]>, %p1: !trait.claim<@P1[tuple<!trait.poly<0>>]>) {
  trait.method @m() -> i64 {
    %v = trait.method.call %p1 @P1[tuple<!trait.poly<0>>]::@m() : () -> i64
    trait.return %v : i64
  }
}
func.func @main() -> i64 {
  %w = trait.allege @P1[i32]
  %v = trait.method.call %w @P1[i32]::@m() : () -> i64
  return %v : i64
}
