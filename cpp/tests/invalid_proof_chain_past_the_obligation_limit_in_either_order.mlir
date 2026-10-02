// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A finite chain of proofs one hundred and twenty-nine deep: @p0 derives
// @P0[i32] over @p1's claim, @p1 over @p2's, and so on to @P129, which its
// impl proves alone. The derivation stands deeper than the obligation limit,
// and is refused before any instance is cut. Here the proofs stand from the
// top of the chain down.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @P0(%self: !trait.claim<@P0[!T]>) {}
trait.trait private @P1(%self: !trait.claim<@P1[!T]>) {}
trait.trait private @P2(%self: !trait.claim<@P2[!T]>) {}
trait.trait private @P3(%self: !trait.claim<@P3[!T]>) {}
trait.trait private @P4(%self: !trait.claim<@P4[!T]>) {}
trait.trait private @P5(%self: !trait.claim<@P5[!T]>) {}
trait.trait private @P6(%self: !trait.claim<@P6[!T]>) {}
trait.trait private @P7(%self: !trait.claim<@P7[!T]>) {}
trait.trait private @P8(%self: !trait.claim<@P8[!T]>) {}
trait.trait private @P9(%self: !trait.claim<@P9[!T]>) {}
trait.trait private @P10(%self: !trait.claim<@P10[!T]>) {}
trait.trait private @P11(%self: !trait.claim<@P11[!T]>) {}
trait.trait private @P12(%self: !trait.claim<@P12[!T]>) {}
trait.trait private @P13(%self: !trait.claim<@P13[!T]>) {}
trait.trait private @P14(%self: !trait.claim<@P14[!T]>) {}
trait.trait private @P15(%self: !trait.claim<@P15[!T]>) {}
trait.trait private @P16(%self: !trait.claim<@P16[!T]>) {}
trait.trait private @P17(%self: !trait.claim<@P17[!T]>) {}
trait.trait private @P18(%self: !trait.claim<@P18[!T]>) {}
trait.trait private @P19(%self: !trait.claim<@P19[!T]>) {}
trait.trait private @P20(%self: !trait.claim<@P20[!T]>) {}
trait.trait private @P21(%self: !trait.claim<@P21[!T]>) {}
trait.trait private @P22(%self: !trait.claim<@P22[!T]>) {}
trait.trait private @P23(%self: !trait.claim<@P23[!T]>) {}
trait.trait private @P24(%self: !trait.claim<@P24[!T]>) {}
trait.trait private @P25(%self: !trait.claim<@P25[!T]>) {}
trait.trait private @P26(%self: !trait.claim<@P26[!T]>) {}
trait.trait private @P27(%self: !trait.claim<@P27[!T]>) {}
trait.trait private @P28(%self: !trait.claim<@P28[!T]>) {}
trait.trait private @P29(%self: !trait.claim<@P29[!T]>) {}
trait.trait private @P30(%self: !trait.claim<@P30[!T]>) {}
trait.trait private @P31(%self: !trait.claim<@P31[!T]>) {}
trait.trait private @P32(%self: !trait.claim<@P32[!T]>) {}
trait.trait private @P33(%self: !trait.claim<@P33[!T]>) {}
trait.trait private @P34(%self: !trait.claim<@P34[!T]>) {}
trait.trait private @P35(%self: !trait.claim<@P35[!T]>) {}
trait.trait private @P36(%self: !trait.claim<@P36[!T]>) {}
trait.trait private @P37(%self: !trait.claim<@P37[!T]>) {}
trait.trait private @P38(%self: !trait.claim<@P38[!T]>) {}
trait.trait private @P39(%self: !trait.claim<@P39[!T]>) {}
trait.trait private @P40(%self: !trait.claim<@P40[!T]>) {}
trait.trait private @P41(%self: !trait.claim<@P41[!T]>) {}
trait.trait private @P42(%self: !trait.claim<@P42[!T]>) {}
trait.trait private @P43(%self: !trait.claim<@P43[!T]>) {}
trait.trait private @P44(%self: !trait.claim<@P44[!T]>) {}
trait.trait private @P45(%self: !trait.claim<@P45[!T]>) {}
trait.trait private @P46(%self: !trait.claim<@P46[!T]>) {}
trait.trait private @P47(%self: !trait.claim<@P47[!T]>) {}
trait.trait private @P48(%self: !trait.claim<@P48[!T]>) {}
trait.trait private @P49(%self: !trait.claim<@P49[!T]>) {}
trait.trait private @P50(%self: !trait.claim<@P50[!T]>) {}
trait.trait private @P51(%self: !trait.claim<@P51[!T]>) {}
trait.trait private @P52(%self: !trait.claim<@P52[!T]>) {}
trait.trait private @P53(%self: !trait.claim<@P53[!T]>) {}
trait.trait private @P54(%self: !trait.claim<@P54[!T]>) {}
trait.trait private @P55(%self: !trait.claim<@P55[!T]>) {}
trait.trait private @P56(%self: !trait.claim<@P56[!T]>) {}
trait.trait private @P57(%self: !trait.claim<@P57[!T]>) {}
trait.trait private @P58(%self: !trait.claim<@P58[!T]>) {}
trait.trait private @P59(%self: !trait.claim<@P59[!T]>) {}
trait.trait private @P60(%self: !trait.claim<@P60[!T]>) {}
trait.trait private @P61(%self: !trait.claim<@P61[!T]>) {}
trait.trait private @P62(%self: !trait.claim<@P62[!T]>) {}
trait.trait private @P63(%self: !trait.claim<@P63[!T]>) {}
trait.trait private @P64(%self: !trait.claim<@P64[!T]>) {}
trait.trait private @P65(%self: !trait.claim<@P65[!T]>) {}
trait.trait private @P66(%self: !trait.claim<@P66[!T]>) {}
trait.trait private @P67(%self: !trait.claim<@P67[!T]>) {}
trait.trait private @P68(%self: !trait.claim<@P68[!T]>) {}
trait.trait private @P69(%self: !trait.claim<@P69[!T]>) {}
trait.trait private @P70(%self: !trait.claim<@P70[!T]>) {}
trait.trait private @P71(%self: !trait.claim<@P71[!T]>) {}
trait.trait private @P72(%self: !trait.claim<@P72[!T]>) {}
trait.trait private @P73(%self: !trait.claim<@P73[!T]>) {}
trait.trait private @P74(%self: !trait.claim<@P74[!T]>) {}
trait.trait private @P75(%self: !trait.claim<@P75[!T]>) {}
trait.trait private @P76(%self: !trait.claim<@P76[!T]>) {}
trait.trait private @P77(%self: !trait.claim<@P77[!T]>) {}
trait.trait private @P78(%self: !trait.claim<@P78[!T]>) {}
trait.trait private @P79(%self: !trait.claim<@P79[!T]>) {}
trait.trait private @P80(%self: !trait.claim<@P80[!T]>) {}
trait.trait private @P81(%self: !trait.claim<@P81[!T]>) {}
trait.trait private @P82(%self: !trait.claim<@P82[!T]>) {}
trait.trait private @P83(%self: !trait.claim<@P83[!T]>) {}
trait.trait private @P84(%self: !trait.claim<@P84[!T]>) {}
trait.trait private @P85(%self: !trait.claim<@P85[!T]>) {}
trait.trait private @P86(%self: !trait.claim<@P86[!T]>) {}
trait.trait private @P87(%self: !trait.claim<@P87[!T]>) {}
trait.trait private @P88(%self: !trait.claim<@P88[!T]>) {}
trait.trait private @P89(%self: !trait.claim<@P89[!T]>) {}
trait.trait private @P90(%self: !trait.claim<@P90[!T]>) {}
trait.trait private @P91(%self: !trait.claim<@P91[!T]>) {}
trait.trait private @P92(%self: !trait.claim<@P92[!T]>) {}
trait.trait private @P93(%self: !trait.claim<@P93[!T]>) {}
trait.trait private @P94(%self: !trait.claim<@P94[!T]>) {}
trait.trait private @P95(%self: !trait.claim<@P95[!T]>) {}
trait.trait private @P96(%self: !trait.claim<@P96[!T]>) {}
trait.trait private @P97(%self: !trait.claim<@P97[!T]>) {}
trait.trait private @P98(%self: !trait.claim<@P98[!T]>) {}
trait.trait private @P99(%self: !trait.claim<@P99[!T]>) {}
trait.trait private @P100(%self: !trait.claim<@P100[!T]>) {}
trait.trait private @P101(%self: !trait.claim<@P101[!T]>) {}
trait.trait private @P102(%self: !trait.claim<@P102[!T]>) {}
trait.trait private @P103(%self: !trait.claim<@P103[!T]>) {}
trait.trait private @P104(%self: !trait.claim<@P104[!T]>) {}
trait.trait private @P105(%self: !trait.claim<@P105[!T]>) {}
trait.trait private @P106(%self: !trait.claim<@P106[!T]>) {}
trait.trait private @P107(%self: !trait.claim<@P107[!T]>) {}
trait.trait private @P108(%self: !trait.claim<@P108[!T]>) {}
trait.trait private @P109(%self: !trait.claim<@P109[!T]>) {}
trait.trait private @P110(%self: !trait.claim<@P110[!T]>) {}
trait.trait private @P111(%self: !trait.claim<@P111[!T]>) {}
trait.trait private @P112(%self: !trait.claim<@P112[!T]>) {}
trait.trait private @P113(%self: !trait.claim<@P113[!T]>) {}
trait.trait private @P114(%self: !trait.claim<@P114[!T]>) {}
trait.trait private @P115(%self: !trait.claim<@P115[!T]>) {}
trait.trait private @P116(%self: !trait.claim<@P116[!T]>) {}
trait.trait private @P117(%self: !trait.claim<@P117[!T]>) {}
trait.trait private @P118(%self: !trait.claim<@P118[!T]>) {}
trait.trait private @P119(%self: !trait.claim<@P119[!T]>) {}
trait.trait private @P120(%self: !trait.claim<@P120[!T]>) {}
trait.trait private @P121(%self: !trait.claim<@P121[!T]>) {}
trait.trait private @P122(%self: !trait.claim<@P122[!T]>) {}
trait.trait private @P123(%self: !trait.claim<@P123[!T]>) {}
trait.trait private @P124(%self: !trait.claim<@P124[!T]>) {}
trait.trait private @P125(%self: !trait.claim<@P125[!T]>) {}
trait.trait private @P126(%self: !trait.claim<@P126[!T]>) {}
trait.trait private @P127(%self: !trait.claim<@P127[!T]>) {}
trait.trait private @P128(%self: !trait.claim<@P128[!T]>) {}
trait.trait private @P129(%self: !trait.claim<@P129[!T]>) {}
trait.impl private @P0_i32(%self: !trait.claim<@P0[i32]>, %n: !trait.claim<@P1[i32]>) {}
trait.impl private @P1_i32(%self: !trait.claim<@P1[i32]>, %n: !trait.claim<@P2[i32]>) {}
trait.impl private @P2_i32(%self: !trait.claim<@P2[i32]>, %n: !trait.claim<@P3[i32]>) {}
trait.impl private @P3_i32(%self: !trait.claim<@P3[i32]>, %n: !trait.claim<@P4[i32]>) {}
trait.impl private @P4_i32(%self: !trait.claim<@P4[i32]>, %n: !trait.claim<@P5[i32]>) {}
trait.impl private @P5_i32(%self: !trait.claim<@P5[i32]>, %n: !trait.claim<@P6[i32]>) {}
trait.impl private @P6_i32(%self: !trait.claim<@P6[i32]>, %n: !trait.claim<@P7[i32]>) {}
trait.impl private @P7_i32(%self: !trait.claim<@P7[i32]>, %n: !trait.claim<@P8[i32]>) {}
trait.impl private @P8_i32(%self: !trait.claim<@P8[i32]>, %n: !trait.claim<@P9[i32]>) {}
trait.impl private @P9_i32(%self: !trait.claim<@P9[i32]>, %n: !trait.claim<@P10[i32]>) {}
trait.impl private @P10_i32(%self: !trait.claim<@P10[i32]>, %n: !trait.claim<@P11[i32]>) {}
trait.impl private @P11_i32(%self: !trait.claim<@P11[i32]>, %n: !trait.claim<@P12[i32]>) {}
trait.impl private @P12_i32(%self: !trait.claim<@P12[i32]>, %n: !trait.claim<@P13[i32]>) {}
trait.impl private @P13_i32(%self: !trait.claim<@P13[i32]>, %n: !trait.claim<@P14[i32]>) {}
trait.impl private @P14_i32(%self: !trait.claim<@P14[i32]>, %n: !trait.claim<@P15[i32]>) {}
trait.impl private @P15_i32(%self: !trait.claim<@P15[i32]>, %n: !trait.claim<@P16[i32]>) {}
trait.impl private @P16_i32(%self: !trait.claim<@P16[i32]>, %n: !trait.claim<@P17[i32]>) {}
trait.impl private @P17_i32(%self: !trait.claim<@P17[i32]>, %n: !trait.claim<@P18[i32]>) {}
trait.impl private @P18_i32(%self: !trait.claim<@P18[i32]>, %n: !trait.claim<@P19[i32]>) {}
trait.impl private @P19_i32(%self: !trait.claim<@P19[i32]>, %n: !trait.claim<@P20[i32]>) {}
trait.impl private @P20_i32(%self: !trait.claim<@P20[i32]>, %n: !trait.claim<@P21[i32]>) {}
trait.impl private @P21_i32(%self: !trait.claim<@P21[i32]>, %n: !trait.claim<@P22[i32]>) {}
trait.impl private @P22_i32(%self: !trait.claim<@P22[i32]>, %n: !trait.claim<@P23[i32]>) {}
trait.impl private @P23_i32(%self: !trait.claim<@P23[i32]>, %n: !trait.claim<@P24[i32]>) {}
trait.impl private @P24_i32(%self: !trait.claim<@P24[i32]>, %n: !trait.claim<@P25[i32]>) {}
trait.impl private @P25_i32(%self: !trait.claim<@P25[i32]>, %n: !trait.claim<@P26[i32]>) {}
trait.impl private @P26_i32(%self: !trait.claim<@P26[i32]>, %n: !trait.claim<@P27[i32]>) {}
trait.impl private @P27_i32(%self: !trait.claim<@P27[i32]>, %n: !trait.claim<@P28[i32]>) {}
trait.impl private @P28_i32(%self: !trait.claim<@P28[i32]>, %n: !trait.claim<@P29[i32]>) {}
trait.impl private @P29_i32(%self: !trait.claim<@P29[i32]>, %n: !trait.claim<@P30[i32]>) {}
trait.impl private @P30_i32(%self: !trait.claim<@P30[i32]>, %n: !trait.claim<@P31[i32]>) {}
trait.impl private @P31_i32(%self: !trait.claim<@P31[i32]>, %n: !trait.claim<@P32[i32]>) {}
trait.impl private @P32_i32(%self: !trait.claim<@P32[i32]>, %n: !trait.claim<@P33[i32]>) {}
trait.impl private @P33_i32(%self: !trait.claim<@P33[i32]>, %n: !trait.claim<@P34[i32]>) {}
trait.impl private @P34_i32(%self: !trait.claim<@P34[i32]>, %n: !trait.claim<@P35[i32]>) {}
trait.impl private @P35_i32(%self: !trait.claim<@P35[i32]>, %n: !trait.claim<@P36[i32]>) {}
trait.impl private @P36_i32(%self: !trait.claim<@P36[i32]>, %n: !trait.claim<@P37[i32]>) {}
trait.impl private @P37_i32(%self: !trait.claim<@P37[i32]>, %n: !trait.claim<@P38[i32]>) {}
trait.impl private @P38_i32(%self: !trait.claim<@P38[i32]>, %n: !trait.claim<@P39[i32]>) {}
trait.impl private @P39_i32(%self: !trait.claim<@P39[i32]>, %n: !trait.claim<@P40[i32]>) {}
trait.impl private @P40_i32(%self: !trait.claim<@P40[i32]>, %n: !trait.claim<@P41[i32]>) {}
trait.impl private @P41_i32(%self: !trait.claim<@P41[i32]>, %n: !trait.claim<@P42[i32]>) {}
trait.impl private @P42_i32(%self: !trait.claim<@P42[i32]>, %n: !trait.claim<@P43[i32]>) {}
trait.impl private @P43_i32(%self: !trait.claim<@P43[i32]>, %n: !trait.claim<@P44[i32]>) {}
trait.impl private @P44_i32(%self: !trait.claim<@P44[i32]>, %n: !trait.claim<@P45[i32]>) {}
trait.impl private @P45_i32(%self: !trait.claim<@P45[i32]>, %n: !trait.claim<@P46[i32]>) {}
trait.impl private @P46_i32(%self: !trait.claim<@P46[i32]>, %n: !trait.claim<@P47[i32]>) {}
trait.impl private @P47_i32(%self: !trait.claim<@P47[i32]>, %n: !trait.claim<@P48[i32]>) {}
trait.impl private @P48_i32(%self: !trait.claim<@P48[i32]>, %n: !trait.claim<@P49[i32]>) {}
trait.impl private @P49_i32(%self: !trait.claim<@P49[i32]>, %n: !trait.claim<@P50[i32]>) {}
trait.impl private @P50_i32(%self: !trait.claim<@P50[i32]>, %n: !trait.claim<@P51[i32]>) {}
trait.impl private @P51_i32(%self: !trait.claim<@P51[i32]>, %n: !trait.claim<@P52[i32]>) {}
trait.impl private @P52_i32(%self: !trait.claim<@P52[i32]>, %n: !trait.claim<@P53[i32]>) {}
trait.impl private @P53_i32(%self: !trait.claim<@P53[i32]>, %n: !trait.claim<@P54[i32]>) {}
trait.impl private @P54_i32(%self: !trait.claim<@P54[i32]>, %n: !trait.claim<@P55[i32]>) {}
trait.impl private @P55_i32(%self: !trait.claim<@P55[i32]>, %n: !trait.claim<@P56[i32]>) {}
trait.impl private @P56_i32(%self: !trait.claim<@P56[i32]>, %n: !trait.claim<@P57[i32]>) {}
trait.impl private @P57_i32(%self: !trait.claim<@P57[i32]>, %n: !trait.claim<@P58[i32]>) {}
trait.impl private @P58_i32(%self: !trait.claim<@P58[i32]>, %n: !trait.claim<@P59[i32]>) {}
trait.impl private @P59_i32(%self: !trait.claim<@P59[i32]>, %n: !trait.claim<@P60[i32]>) {}
trait.impl private @P60_i32(%self: !trait.claim<@P60[i32]>, %n: !trait.claim<@P61[i32]>) {}
trait.impl private @P61_i32(%self: !trait.claim<@P61[i32]>, %n: !trait.claim<@P62[i32]>) {}
trait.impl private @P62_i32(%self: !trait.claim<@P62[i32]>, %n: !trait.claim<@P63[i32]>) {}
trait.impl private @P63_i32(%self: !trait.claim<@P63[i32]>, %n: !trait.claim<@P64[i32]>) {}
trait.impl private @P64_i32(%self: !trait.claim<@P64[i32]>, %n: !trait.claim<@P65[i32]>) {}
trait.impl private @P65_i32(%self: !trait.claim<@P65[i32]>, %n: !trait.claim<@P66[i32]>) {}
trait.impl private @P66_i32(%self: !trait.claim<@P66[i32]>, %n: !trait.claim<@P67[i32]>) {}
trait.impl private @P67_i32(%self: !trait.claim<@P67[i32]>, %n: !trait.claim<@P68[i32]>) {}
trait.impl private @P68_i32(%self: !trait.claim<@P68[i32]>, %n: !trait.claim<@P69[i32]>) {}
trait.impl private @P69_i32(%self: !trait.claim<@P69[i32]>, %n: !trait.claim<@P70[i32]>) {}
trait.impl private @P70_i32(%self: !trait.claim<@P70[i32]>, %n: !trait.claim<@P71[i32]>) {}
trait.impl private @P71_i32(%self: !trait.claim<@P71[i32]>, %n: !trait.claim<@P72[i32]>) {}
trait.impl private @P72_i32(%self: !trait.claim<@P72[i32]>, %n: !trait.claim<@P73[i32]>) {}
trait.impl private @P73_i32(%self: !trait.claim<@P73[i32]>, %n: !trait.claim<@P74[i32]>) {}
trait.impl private @P74_i32(%self: !trait.claim<@P74[i32]>, %n: !trait.claim<@P75[i32]>) {}
trait.impl private @P75_i32(%self: !trait.claim<@P75[i32]>, %n: !trait.claim<@P76[i32]>) {}
trait.impl private @P76_i32(%self: !trait.claim<@P76[i32]>, %n: !trait.claim<@P77[i32]>) {}
trait.impl private @P77_i32(%self: !trait.claim<@P77[i32]>, %n: !trait.claim<@P78[i32]>) {}
trait.impl private @P78_i32(%self: !trait.claim<@P78[i32]>, %n: !trait.claim<@P79[i32]>) {}
trait.impl private @P79_i32(%self: !trait.claim<@P79[i32]>, %n: !trait.claim<@P80[i32]>) {}
trait.impl private @P80_i32(%self: !trait.claim<@P80[i32]>, %n: !trait.claim<@P81[i32]>) {}
trait.impl private @P81_i32(%self: !trait.claim<@P81[i32]>, %n: !trait.claim<@P82[i32]>) {}
trait.impl private @P82_i32(%self: !trait.claim<@P82[i32]>, %n: !trait.claim<@P83[i32]>) {}
trait.impl private @P83_i32(%self: !trait.claim<@P83[i32]>, %n: !trait.claim<@P84[i32]>) {}
trait.impl private @P84_i32(%self: !trait.claim<@P84[i32]>, %n: !trait.claim<@P85[i32]>) {}
trait.impl private @P85_i32(%self: !trait.claim<@P85[i32]>, %n: !trait.claim<@P86[i32]>) {}
trait.impl private @P86_i32(%self: !trait.claim<@P86[i32]>, %n: !trait.claim<@P87[i32]>) {}
trait.impl private @P87_i32(%self: !trait.claim<@P87[i32]>, %n: !trait.claim<@P88[i32]>) {}
trait.impl private @P88_i32(%self: !trait.claim<@P88[i32]>, %n: !trait.claim<@P89[i32]>) {}
trait.impl private @P89_i32(%self: !trait.claim<@P89[i32]>, %n: !trait.claim<@P90[i32]>) {}
trait.impl private @P90_i32(%self: !trait.claim<@P90[i32]>, %n: !trait.claim<@P91[i32]>) {}
trait.impl private @P91_i32(%self: !trait.claim<@P91[i32]>, %n: !trait.claim<@P92[i32]>) {}
trait.impl private @P92_i32(%self: !trait.claim<@P92[i32]>, %n: !trait.claim<@P93[i32]>) {}
trait.impl private @P93_i32(%self: !trait.claim<@P93[i32]>, %n: !trait.claim<@P94[i32]>) {}
trait.impl private @P94_i32(%self: !trait.claim<@P94[i32]>, %n: !trait.claim<@P95[i32]>) {}
trait.impl private @P95_i32(%self: !trait.claim<@P95[i32]>, %n: !trait.claim<@P96[i32]>) {}
trait.impl private @P96_i32(%self: !trait.claim<@P96[i32]>, %n: !trait.claim<@P97[i32]>) {}
trait.impl private @P97_i32(%self: !trait.claim<@P97[i32]>, %n: !trait.claim<@P98[i32]>) {}
trait.impl private @P98_i32(%self: !trait.claim<@P98[i32]>, %n: !trait.claim<@P99[i32]>) {}
trait.impl private @P99_i32(%self: !trait.claim<@P99[i32]>, %n: !trait.claim<@P100[i32]>) {}
trait.impl private @P100_i32(%self: !trait.claim<@P100[i32]>, %n: !trait.claim<@P101[i32]>) {}
trait.impl private @P101_i32(%self: !trait.claim<@P101[i32]>, %n: !trait.claim<@P102[i32]>) {}
trait.impl private @P102_i32(%self: !trait.claim<@P102[i32]>, %n: !trait.claim<@P103[i32]>) {}
trait.impl private @P103_i32(%self: !trait.claim<@P103[i32]>, %n: !trait.claim<@P104[i32]>) {}
trait.impl private @P104_i32(%self: !trait.claim<@P104[i32]>, %n: !trait.claim<@P105[i32]>) {}
trait.impl private @P105_i32(%self: !trait.claim<@P105[i32]>, %n: !trait.claim<@P106[i32]>) {}
trait.impl private @P106_i32(%self: !trait.claim<@P106[i32]>, %n: !trait.claim<@P107[i32]>) {}
trait.impl private @P107_i32(%self: !trait.claim<@P107[i32]>, %n: !trait.claim<@P108[i32]>) {}
trait.impl private @P108_i32(%self: !trait.claim<@P108[i32]>, %n: !trait.claim<@P109[i32]>) {}
trait.impl private @P109_i32(%self: !trait.claim<@P109[i32]>, %n: !trait.claim<@P110[i32]>) {}
trait.impl private @P110_i32(%self: !trait.claim<@P110[i32]>, %n: !trait.claim<@P111[i32]>) {}
trait.impl private @P111_i32(%self: !trait.claim<@P111[i32]>, %n: !trait.claim<@P112[i32]>) {}
trait.impl private @P112_i32(%self: !trait.claim<@P112[i32]>, %n: !trait.claim<@P113[i32]>) {}
trait.impl private @P113_i32(%self: !trait.claim<@P113[i32]>, %n: !trait.claim<@P114[i32]>) {}
trait.impl private @P114_i32(%self: !trait.claim<@P114[i32]>, %n: !trait.claim<@P115[i32]>) {}
trait.impl private @P115_i32(%self: !trait.claim<@P115[i32]>, %n: !trait.claim<@P116[i32]>) {}
trait.impl private @P116_i32(%self: !trait.claim<@P116[i32]>, %n: !trait.claim<@P117[i32]>) {}
trait.impl private @P117_i32(%self: !trait.claim<@P117[i32]>, %n: !trait.claim<@P118[i32]>) {}
trait.impl private @P118_i32(%self: !trait.claim<@P118[i32]>, %n: !trait.claim<@P119[i32]>) {}
trait.impl private @P119_i32(%self: !trait.claim<@P119[i32]>, %n: !trait.claim<@P120[i32]>) {}
trait.impl private @P120_i32(%self: !trait.claim<@P120[i32]>, %n: !trait.claim<@P121[i32]>) {}
trait.impl private @P121_i32(%self: !trait.claim<@P121[i32]>, %n: !trait.claim<@P122[i32]>) {}
trait.impl private @P122_i32(%self: !trait.claim<@P122[i32]>, %n: !trait.claim<@P123[i32]>) {}
trait.impl private @P123_i32(%self: !trait.claim<@P123[i32]>, %n: !trait.claim<@P124[i32]>) {}
trait.impl private @P124_i32(%self: !trait.claim<@P124[i32]>, %n: !trait.claim<@P125[i32]>) {}
trait.impl private @P125_i32(%self: !trait.claim<@P125[i32]>, %n: !trait.claim<@P126[i32]>) {}
trait.impl private @P126_i32(%self: !trait.claim<@P126[i32]>, %n: !trait.claim<@P127[i32]>) {}
trait.impl private @P127_i32(%self: !trait.claim<@P127[i32]>, %n: !trait.claim<@P128[i32]>) {}
trait.impl private @P128_i32(%self: !trait.claim<@P128[i32]>, %n: !trait.claim<@P129[i32]>) {}
trait.impl private @P129_i32(%self: !trait.claim<@P129[i32]>) {}
trait.proof private @p0 {
  %n = trait.witness @p1 for @P1[i32]
  %d = trait.derive @P0[i32] from @P0_i32 given(%n) : (!trait.claim<@P1[i32] by @p1>)
  trait.return %d : !trait.claim<@P0[i32]>
}
trait.proof private @p1 {
  %n = trait.witness @p2 for @P2[i32]
  %d = trait.derive @P1[i32] from @P1_i32 given(%n) : (!trait.claim<@P2[i32] by @p2>)
  trait.return %d : !trait.claim<@P1[i32]>
}
trait.proof private @p2 {
  %n = trait.witness @p3 for @P3[i32]
  %d = trait.derive @P2[i32] from @P2_i32 given(%n) : (!trait.claim<@P3[i32] by @p3>)
  trait.return %d : !trait.claim<@P2[i32]>
}
trait.proof private @p3 {
  %n = trait.witness @p4 for @P4[i32]
  %d = trait.derive @P3[i32] from @P3_i32 given(%n) : (!trait.claim<@P4[i32] by @p4>)
  trait.return %d : !trait.claim<@P3[i32]>
}
trait.proof private @p4 {
  %n = trait.witness @p5 for @P5[i32]
  %d = trait.derive @P4[i32] from @P4_i32 given(%n) : (!trait.claim<@P5[i32] by @p5>)
  trait.return %d : !trait.claim<@P4[i32]>
}
trait.proof private @p5 {
  %n = trait.witness @p6 for @P6[i32]
  %d = trait.derive @P5[i32] from @P5_i32 given(%n) : (!trait.claim<@P6[i32] by @p6>)
  trait.return %d : !trait.claim<@P5[i32]>
}
trait.proof private @p6 {
  %n = trait.witness @p7 for @P7[i32]
  %d = trait.derive @P6[i32] from @P6_i32 given(%n) : (!trait.claim<@P7[i32] by @p7>)
  trait.return %d : !trait.claim<@P6[i32]>
}
trait.proof private @p7 {
  %n = trait.witness @p8 for @P8[i32]
  %d = trait.derive @P7[i32] from @P7_i32 given(%n) : (!trait.claim<@P8[i32] by @p8>)
  trait.return %d : !trait.claim<@P7[i32]>
}
trait.proof private @p8 {
  %n = trait.witness @p9 for @P9[i32]
  %d = trait.derive @P8[i32] from @P8_i32 given(%n) : (!trait.claim<@P9[i32] by @p9>)
  trait.return %d : !trait.claim<@P8[i32]>
}
trait.proof private @p9 {
  %n = trait.witness @p10 for @P10[i32]
  %d = trait.derive @P9[i32] from @P9_i32 given(%n) : (!trait.claim<@P10[i32] by @p10>)
  trait.return %d : !trait.claim<@P9[i32]>
}
trait.proof private @p10 {
  %n = trait.witness @p11 for @P11[i32]
  %d = trait.derive @P10[i32] from @P10_i32 given(%n) : (!trait.claim<@P11[i32] by @p11>)
  trait.return %d : !trait.claim<@P10[i32]>
}
trait.proof private @p11 {
  %n = trait.witness @p12 for @P12[i32]
  %d = trait.derive @P11[i32] from @P11_i32 given(%n) : (!trait.claim<@P12[i32] by @p12>)
  trait.return %d : !trait.claim<@P11[i32]>
}
trait.proof private @p12 {
  %n = trait.witness @p13 for @P13[i32]
  %d = trait.derive @P12[i32] from @P12_i32 given(%n) : (!trait.claim<@P13[i32] by @p13>)
  trait.return %d : !trait.claim<@P12[i32]>
}
trait.proof private @p13 {
  %n = trait.witness @p14 for @P14[i32]
  %d = trait.derive @P13[i32] from @P13_i32 given(%n) : (!trait.claim<@P14[i32] by @p14>)
  trait.return %d : !trait.claim<@P13[i32]>
}
trait.proof private @p14 {
  %n = trait.witness @p15 for @P15[i32]
  %d = trait.derive @P14[i32] from @P14_i32 given(%n) : (!trait.claim<@P15[i32] by @p15>)
  trait.return %d : !trait.claim<@P14[i32]>
}
trait.proof private @p15 {
  %n = trait.witness @p16 for @P16[i32]
  %d = trait.derive @P15[i32] from @P15_i32 given(%n) : (!trait.claim<@P16[i32] by @p16>)
  trait.return %d : !trait.claim<@P15[i32]>
}
trait.proof private @p16 {
  %n = trait.witness @p17 for @P17[i32]
  %d = trait.derive @P16[i32] from @P16_i32 given(%n) : (!trait.claim<@P17[i32] by @p17>)
  trait.return %d : !trait.claim<@P16[i32]>
}
trait.proof private @p17 {
  %n = trait.witness @p18 for @P18[i32]
  %d = trait.derive @P17[i32] from @P17_i32 given(%n) : (!trait.claim<@P18[i32] by @p18>)
  trait.return %d : !trait.claim<@P17[i32]>
}
trait.proof private @p18 {
  %n = trait.witness @p19 for @P19[i32]
  %d = trait.derive @P18[i32] from @P18_i32 given(%n) : (!trait.claim<@P19[i32] by @p19>)
  trait.return %d : !trait.claim<@P18[i32]>
}
trait.proof private @p19 {
  %n = trait.witness @p20 for @P20[i32]
  %d = trait.derive @P19[i32] from @P19_i32 given(%n) : (!trait.claim<@P20[i32] by @p20>)
  trait.return %d : !trait.claim<@P19[i32]>
}
trait.proof private @p20 {
  %n = trait.witness @p21 for @P21[i32]
  %d = trait.derive @P20[i32] from @P20_i32 given(%n) : (!trait.claim<@P21[i32] by @p21>)
  trait.return %d : !trait.claim<@P20[i32]>
}
trait.proof private @p21 {
  %n = trait.witness @p22 for @P22[i32]
  %d = trait.derive @P21[i32] from @P21_i32 given(%n) : (!trait.claim<@P22[i32] by @p22>)
  trait.return %d : !trait.claim<@P21[i32]>
}
trait.proof private @p22 {
  %n = trait.witness @p23 for @P23[i32]
  %d = trait.derive @P22[i32] from @P22_i32 given(%n) : (!trait.claim<@P23[i32] by @p23>)
  trait.return %d : !trait.claim<@P22[i32]>
}
trait.proof private @p23 {
  %n = trait.witness @p24 for @P24[i32]
  %d = trait.derive @P23[i32] from @P23_i32 given(%n) : (!trait.claim<@P24[i32] by @p24>)
  trait.return %d : !trait.claim<@P23[i32]>
}
trait.proof private @p24 {
  %n = trait.witness @p25 for @P25[i32]
  %d = trait.derive @P24[i32] from @P24_i32 given(%n) : (!trait.claim<@P25[i32] by @p25>)
  trait.return %d : !trait.claim<@P24[i32]>
}
trait.proof private @p25 {
  %n = trait.witness @p26 for @P26[i32]
  %d = trait.derive @P25[i32] from @P25_i32 given(%n) : (!trait.claim<@P26[i32] by @p26>)
  trait.return %d : !trait.claim<@P25[i32]>
}
trait.proof private @p26 {
  %n = trait.witness @p27 for @P27[i32]
  %d = trait.derive @P26[i32] from @P26_i32 given(%n) : (!trait.claim<@P27[i32] by @p27>)
  trait.return %d : !trait.claim<@P26[i32]>
}
trait.proof private @p27 {
  %n = trait.witness @p28 for @P28[i32]
  %d = trait.derive @P27[i32] from @P27_i32 given(%n) : (!trait.claim<@P28[i32] by @p28>)
  trait.return %d : !trait.claim<@P27[i32]>
}
trait.proof private @p28 {
  %n = trait.witness @p29 for @P29[i32]
  %d = trait.derive @P28[i32] from @P28_i32 given(%n) : (!trait.claim<@P29[i32] by @p29>)
  trait.return %d : !trait.claim<@P28[i32]>
}
trait.proof private @p29 {
  %n = trait.witness @p30 for @P30[i32]
  %d = trait.derive @P29[i32] from @P29_i32 given(%n) : (!trait.claim<@P30[i32] by @p30>)
  trait.return %d : !trait.claim<@P29[i32]>
}
trait.proof private @p30 {
  %n = trait.witness @p31 for @P31[i32]
  %d = trait.derive @P30[i32] from @P30_i32 given(%n) : (!trait.claim<@P31[i32] by @p31>)
  trait.return %d : !trait.claim<@P30[i32]>
}
trait.proof private @p31 {
  %n = trait.witness @p32 for @P32[i32]
  %d = trait.derive @P31[i32] from @P31_i32 given(%n) : (!trait.claim<@P32[i32] by @p32>)
  trait.return %d : !trait.claim<@P31[i32]>
}
trait.proof private @p32 {
  %n = trait.witness @p33 for @P33[i32]
  %d = trait.derive @P32[i32] from @P32_i32 given(%n) : (!trait.claim<@P33[i32] by @p33>)
  trait.return %d : !trait.claim<@P32[i32]>
}
trait.proof private @p33 {
  %n = trait.witness @p34 for @P34[i32]
  %d = trait.derive @P33[i32] from @P33_i32 given(%n) : (!trait.claim<@P34[i32] by @p34>)
  trait.return %d : !trait.claim<@P33[i32]>
}
trait.proof private @p34 {
  %n = trait.witness @p35 for @P35[i32]
  %d = trait.derive @P34[i32] from @P34_i32 given(%n) : (!trait.claim<@P35[i32] by @p35>)
  trait.return %d : !trait.claim<@P34[i32]>
}
trait.proof private @p35 {
  %n = trait.witness @p36 for @P36[i32]
  %d = trait.derive @P35[i32] from @P35_i32 given(%n) : (!trait.claim<@P36[i32] by @p36>)
  trait.return %d : !trait.claim<@P35[i32]>
}
trait.proof private @p36 {
  %n = trait.witness @p37 for @P37[i32]
  %d = trait.derive @P36[i32] from @P36_i32 given(%n) : (!trait.claim<@P37[i32] by @p37>)
  trait.return %d : !trait.claim<@P36[i32]>
}
trait.proof private @p37 {
  %n = trait.witness @p38 for @P38[i32]
  %d = trait.derive @P37[i32] from @P37_i32 given(%n) : (!trait.claim<@P38[i32] by @p38>)
  trait.return %d : !trait.claim<@P37[i32]>
}
trait.proof private @p38 {
  %n = trait.witness @p39 for @P39[i32]
  %d = trait.derive @P38[i32] from @P38_i32 given(%n) : (!trait.claim<@P39[i32] by @p39>)
  trait.return %d : !trait.claim<@P38[i32]>
}
trait.proof private @p39 {
  %n = trait.witness @p40 for @P40[i32]
  %d = trait.derive @P39[i32] from @P39_i32 given(%n) : (!trait.claim<@P40[i32] by @p40>)
  trait.return %d : !trait.claim<@P39[i32]>
}
trait.proof private @p40 {
  %n = trait.witness @p41 for @P41[i32]
  %d = trait.derive @P40[i32] from @P40_i32 given(%n) : (!trait.claim<@P41[i32] by @p41>)
  trait.return %d : !trait.claim<@P40[i32]>
}
trait.proof private @p41 {
  %n = trait.witness @p42 for @P42[i32]
  %d = trait.derive @P41[i32] from @P41_i32 given(%n) : (!trait.claim<@P42[i32] by @p42>)
  trait.return %d : !trait.claim<@P41[i32]>
}
trait.proof private @p42 {
  %n = trait.witness @p43 for @P43[i32]
  %d = trait.derive @P42[i32] from @P42_i32 given(%n) : (!trait.claim<@P43[i32] by @p43>)
  trait.return %d : !trait.claim<@P42[i32]>
}
trait.proof private @p43 {
  %n = trait.witness @p44 for @P44[i32]
  %d = trait.derive @P43[i32] from @P43_i32 given(%n) : (!trait.claim<@P44[i32] by @p44>)
  trait.return %d : !trait.claim<@P43[i32]>
}
trait.proof private @p44 {
  %n = trait.witness @p45 for @P45[i32]
  %d = trait.derive @P44[i32] from @P44_i32 given(%n) : (!trait.claim<@P45[i32] by @p45>)
  trait.return %d : !trait.claim<@P44[i32]>
}
trait.proof private @p45 {
  %n = trait.witness @p46 for @P46[i32]
  %d = trait.derive @P45[i32] from @P45_i32 given(%n) : (!trait.claim<@P46[i32] by @p46>)
  trait.return %d : !trait.claim<@P45[i32]>
}
trait.proof private @p46 {
  %n = trait.witness @p47 for @P47[i32]
  %d = trait.derive @P46[i32] from @P46_i32 given(%n) : (!trait.claim<@P47[i32] by @p47>)
  trait.return %d : !trait.claim<@P46[i32]>
}
trait.proof private @p47 {
  %n = trait.witness @p48 for @P48[i32]
  %d = trait.derive @P47[i32] from @P47_i32 given(%n) : (!trait.claim<@P48[i32] by @p48>)
  trait.return %d : !trait.claim<@P47[i32]>
}
trait.proof private @p48 {
  %n = trait.witness @p49 for @P49[i32]
  %d = trait.derive @P48[i32] from @P48_i32 given(%n) : (!trait.claim<@P49[i32] by @p49>)
  trait.return %d : !trait.claim<@P48[i32]>
}
trait.proof private @p49 {
  %n = trait.witness @p50 for @P50[i32]
  %d = trait.derive @P49[i32] from @P49_i32 given(%n) : (!trait.claim<@P50[i32] by @p50>)
  trait.return %d : !trait.claim<@P49[i32]>
}
trait.proof private @p50 {
  %n = trait.witness @p51 for @P51[i32]
  %d = trait.derive @P50[i32] from @P50_i32 given(%n) : (!trait.claim<@P51[i32] by @p51>)
  trait.return %d : !trait.claim<@P50[i32]>
}
trait.proof private @p51 {
  %n = trait.witness @p52 for @P52[i32]
  %d = trait.derive @P51[i32] from @P51_i32 given(%n) : (!trait.claim<@P52[i32] by @p52>)
  trait.return %d : !trait.claim<@P51[i32]>
}
trait.proof private @p52 {
  %n = trait.witness @p53 for @P53[i32]
  %d = trait.derive @P52[i32] from @P52_i32 given(%n) : (!trait.claim<@P53[i32] by @p53>)
  trait.return %d : !trait.claim<@P52[i32]>
}
trait.proof private @p53 {
  %n = trait.witness @p54 for @P54[i32]
  %d = trait.derive @P53[i32] from @P53_i32 given(%n) : (!trait.claim<@P54[i32] by @p54>)
  trait.return %d : !trait.claim<@P53[i32]>
}
trait.proof private @p54 {
  %n = trait.witness @p55 for @P55[i32]
  %d = trait.derive @P54[i32] from @P54_i32 given(%n) : (!trait.claim<@P55[i32] by @p55>)
  trait.return %d : !trait.claim<@P54[i32]>
}
trait.proof private @p55 {
  %n = trait.witness @p56 for @P56[i32]
  %d = trait.derive @P55[i32] from @P55_i32 given(%n) : (!trait.claim<@P56[i32] by @p56>)
  trait.return %d : !trait.claim<@P55[i32]>
}
trait.proof private @p56 {
  %n = trait.witness @p57 for @P57[i32]
  %d = trait.derive @P56[i32] from @P56_i32 given(%n) : (!trait.claim<@P57[i32] by @p57>)
  trait.return %d : !trait.claim<@P56[i32]>
}
trait.proof private @p57 {
  %n = trait.witness @p58 for @P58[i32]
  %d = trait.derive @P57[i32] from @P57_i32 given(%n) : (!trait.claim<@P58[i32] by @p58>)
  trait.return %d : !trait.claim<@P57[i32]>
}
trait.proof private @p58 {
  %n = trait.witness @p59 for @P59[i32]
  %d = trait.derive @P58[i32] from @P58_i32 given(%n) : (!trait.claim<@P59[i32] by @p59>)
  trait.return %d : !trait.claim<@P58[i32]>
}
trait.proof private @p59 {
  %n = trait.witness @p60 for @P60[i32]
  %d = trait.derive @P59[i32] from @P59_i32 given(%n) : (!trait.claim<@P60[i32] by @p60>)
  trait.return %d : !trait.claim<@P59[i32]>
}
trait.proof private @p60 {
  %n = trait.witness @p61 for @P61[i32]
  %d = trait.derive @P60[i32] from @P60_i32 given(%n) : (!trait.claim<@P61[i32] by @p61>)
  trait.return %d : !trait.claim<@P60[i32]>
}
trait.proof private @p61 {
  %n = trait.witness @p62 for @P62[i32]
  %d = trait.derive @P61[i32] from @P61_i32 given(%n) : (!trait.claim<@P62[i32] by @p62>)
  trait.return %d : !trait.claim<@P61[i32]>
}
trait.proof private @p62 {
  %n = trait.witness @p63 for @P63[i32]
  %d = trait.derive @P62[i32] from @P62_i32 given(%n) : (!trait.claim<@P63[i32] by @p63>)
  trait.return %d : !trait.claim<@P62[i32]>
}
trait.proof private @p63 {
  %n = trait.witness @p64 for @P64[i32]
  %d = trait.derive @P63[i32] from @P63_i32 given(%n) : (!trait.claim<@P64[i32] by @p64>)
  trait.return %d : !trait.claim<@P63[i32]>
}
trait.proof private @p64 {
  %n = trait.witness @p65 for @P65[i32]
  %d = trait.derive @P64[i32] from @P64_i32 given(%n) : (!trait.claim<@P65[i32] by @p65>)
  trait.return %d : !trait.claim<@P64[i32]>
}
trait.proof private @p65 {
  %n = trait.witness @p66 for @P66[i32]
  %d = trait.derive @P65[i32] from @P65_i32 given(%n) : (!trait.claim<@P66[i32] by @p66>)
  trait.return %d : !trait.claim<@P65[i32]>
}
trait.proof private @p66 {
  %n = trait.witness @p67 for @P67[i32]
  %d = trait.derive @P66[i32] from @P66_i32 given(%n) : (!trait.claim<@P67[i32] by @p67>)
  trait.return %d : !trait.claim<@P66[i32]>
}
trait.proof private @p67 {
  %n = trait.witness @p68 for @P68[i32]
  %d = trait.derive @P67[i32] from @P67_i32 given(%n) : (!trait.claim<@P68[i32] by @p68>)
  trait.return %d : !trait.claim<@P67[i32]>
}
trait.proof private @p68 {
  %n = trait.witness @p69 for @P69[i32]
  %d = trait.derive @P68[i32] from @P68_i32 given(%n) : (!trait.claim<@P69[i32] by @p69>)
  trait.return %d : !trait.claim<@P68[i32]>
}
trait.proof private @p69 {
  %n = trait.witness @p70 for @P70[i32]
  %d = trait.derive @P69[i32] from @P69_i32 given(%n) : (!trait.claim<@P70[i32] by @p70>)
  trait.return %d : !trait.claim<@P69[i32]>
}
trait.proof private @p70 {
  %n = trait.witness @p71 for @P71[i32]
  %d = trait.derive @P70[i32] from @P70_i32 given(%n) : (!trait.claim<@P71[i32] by @p71>)
  trait.return %d : !trait.claim<@P70[i32]>
}
trait.proof private @p71 {
  %n = trait.witness @p72 for @P72[i32]
  %d = trait.derive @P71[i32] from @P71_i32 given(%n) : (!trait.claim<@P72[i32] by @p72>)
  trait.return %d : !trait.claim<@P71[i32]>
}
trait.proof private @p72 {
  %n = trait.witness @p73 for @P73[i32]
  %d = trait.derive @P72[i32] from @P72_i32 given(%n) : (!trait.claim<@P73[i32] by @p73>)
  trait.return %d : !trait.claim<@P72[i32]>
}
trait.proof private @p73 {
  %n = trait.witness @p74 for @P74[i32]
  %d = trait.derive @P73[i32] from @P73_i32 given(%n) : (!trait.claim<@P74[i32] by @p74>)
  trait.return %d : !trait.claim<@P73[i32]>
}
trait.proof private @p74 {
  %n = trait.witness @p75 for @P75[i32]
  %d = trait.derive @P74[i32] from @P74_i32 given(%n) : (!trait.claim<@P75[i32] by @p75>)
  trait.return %d : !trait.claim<@P74[i32]>
}
trait.proof private @p75 {
  %n = trait.witness @p76 for @P76[i32]
  %d = trait.derive @P75[i32] from @P75_i32 given(%n) : (!trait.claim<@P76[i32] by @p76>)
  trait.return %d : !trait.claim<@P75[i32]>
}
trait.proof private @p76 {
  %n = trait.witness @p77 for @P77[i32]
  %d = trait.derive @P76[i32] from @P76_i32 given(%n) : (!trait.claim<@P77[i32] by @p77>)
  trait.return %d : !trait.claim<@P76[i32]>
}
trait.proof private @p77 {
  %n = trait.witness @p78 for @P78[i32]
  %d = trait.derive @P77[i32] from @P77_i32 given(%n) : (!trait.claim<@P78[i32] by @p78>)
  trait.return %d : !trait.claim<@P77[i32]>
}
trait.proof private @p78 {
  %n = trait.witness @p79 for @P79[i32]
  %d = trait.derive @P78[i32] from @P78_i32 given(%n) : (!trait.claim<@P79[i32] by @p79>)
  trait.return %d : !trait.claim<@P78[i32]>
}
trait.proof private @p79 {
  %n = trait.witness @p80 for @P80[i32]
  %d = trait.derive @P79[i32] from @P79_i32 given(%n) : (!trait.claim<@P80[i32] by @p80>)
  trait.return %d : !trait.claim<@P79[i32]>
}
trait.proof private @p80 {
  %n = trait.witness @p81 for @P81[i32]
  %d = trait.derive @P80[i32] from @P80_i32 given(%n) : (!trait.claim<@P81[i32] by @p81>)
  trait.return %d : !trait.claim<@P80[i32]>
}
trait.proof private @p81 {
  %n = trait.witness @p82 for @P82[i32]
  %d = trait.derive @P81[i32] from @P81_i32 given(%n) : (!trait.claim<@P82[i32] by @p82>)
  trait.return %d : !trait.claim<@P81[i32]>
}
trait.proof private @p82 {
  %n = trait.witness @p83 for @P83[i32]
  %d = trait.derive @P82[i32] from @P82_i32 given(%n) : (!trait.claim<@P83[i32] by @p83>)
  trait.return %d : !trait.claim<@P82[i32]>
}
trait.proof private @p83 {
  %n = trait.witness @p84 for @P84[i32]
  %d = trait.derive @P83[i32] from @P83_i32 given(%n) : (!trait.claim<@P84[i32] by @p84>)
  trait.return %d : !trait.claim<@P83[i32]>
}
trait.proof private @p84 {
  %n = trait.witness @p85 for @P85[i32]
  %d = trait.derive @P84[i32] from @P84_i32 given(%n) : (!trait.claim<@P85[i32] by @p85>)
  trait.return %d : !trait.claim<@P84[i32]>
}
trait.proof private @p85 {
  %n = trait.witness @p86 for @P86[i32]
  %d = trait.derive @P85[i32] from @P85_i32 given(%n) : (!trait.claim<@P86[i32] by @p86>)
  trait.return %d : !trait.claim<@P85[i32]>
}
trait.proof private @p86 {
  %n = trait.witness @p87 for @P87[i32]
  %d = trait.derive @P86[i32] from @P86_i32 given(%n) : (!trait.claim<@P87[i32] by @p87>)
  trait.return %d : !trait.claim<@P86[i32]>
}
trait.proof private @p87 {
  %n = trait.witness @p88 for @P88[i32]
  %d = trait.derive @P87[i32] from @P87_i32 given(%n) : (!trait.claim<@P88[i32] by @p88>)
  trait.return %d : !trait.claim<@P87[i32]>
}
trait.proof private @p88 {
  %n = trait.witness @p89 for @P89[i32]
  %d = trait.derive @P88[i32] from @P88_i32 given(%n) : (!trait.claim<@P89[i32] by @p89>)
  trait.return %d : !trait.claim<@P88[i32]>
}
trait.proof private @p89 {
  %n = trait.witness @p90 for @P90[i32]
  %d = trait.derive @P89[i32] from @P89_i32 given(%n) : (!trait.claim<@P90[i32] by @p90>)
  trait.return %d : !trait.claim<@P89[i32]>
}
trait.proof private @p90 {
  %n = trait.witness @p91 for @P91[i32]
  %d = trait.derive @P90[i32] from @P90_i32 given(%n) : (!trait.claim<@P91[i32] by @p91>)
  trait.return %d : !trait.claim<@P90[i32]>
}
trait.proof private @p91 {
  %n = trait.witness @p92 for @P92[i32]
  %d = trait.derive @P91[i32] from @P91_i32 given(%n) : (!trait.claim<@P92[i32] by @p92>)
  trait.return %d : !trait.claim<@P91[i32]>
}
trait.proof private @p92 {
  %n = trait.witness @p93 for @P93[i32]
  %d = trait.derive @P92[i32] from @P92_i32 given(%n) : (!trait.claim<@P93[i32] by @p93>)
  trait.return %d : !trait.claim<@P92[i32]>
}
trait.proof private @p93 {
  %n = trait.witness @p94 for @P94[i32]
  %d = trait.derive @P93[i32] from @P93_i32 given(%n) : (!trait.claim<@P94[i32] by @p94>)
  trait.return %d : !trait.claim<@P93[i32]>
}
trait.proof private @p94 {
  %n = trait.witness @p95 for @P95[i32]
  %d = trait.derive @P94[i32] from @P94_i32 given(%n) : (!trait.claim<@P95[i32] by @p95>)
  trait.return %d : !trait.claim<@P94[i32]>
}
trait.proof private @p95 {
  %n = trait.witness @p96 for @P96[i32]
  %d = trait.derive @P95[i32] from @P95_i32 given(%n) : (!trait.claim<@P96[i32] by @p96>)
  trait.return %d : !trait.claim<@P95[i32]>
}
trait.proof private @p96 {
  %n = trait.witness @p97 for @P97[i32]
  %d = trait.derive @P96[i32] from @P96_i32 given(%n) : (!trait.claim<@P97[i32] by @p97>)
  trait.return %d : !trait.claim<@P96[i32]>
}
trait.proof private @p97 {
  %n = trait.witness @p98 for @P98[i32]
  %d = trait.derive @P97[i32] from @P97_i32 given(%n) : (!trait.claim<@P98[i32] by @p98>)
  trait.return %d : !trait.claim<@P97[i32]>
}
trait.proof private @p98 {
  %n = trait.witness @p99 for @P99[i32]
  %d = trait.derive @P98[i32] from @P98_i32 given(%n) : (!trait.claim<@P99[i32] by @p99>)
  trait.return %d : !trait.claim<@P98[i32]>
}
trait.proof private @p99 {
  %n = trait.witness @p100 for @P100[i32]
  %d = trait.derive @P99[i32] from @P99_i32 given(%n) : (!trait.claim<@P100[i32] by @p100>)
  trait.return %d : !trait.claim<@P99[i32]>
}
trait.proof private @p100 {
  %n = trait.witness @p101 for @P101[i32]
  %d = trait.derive @P100[i32] from @P100_i32 given(%n) : (!trait.claim<@P101[i32] by @p101>)
  trait.return %d : !trait.claim<@P100[i32]>
}
trait.proof private @p101 {
  %n = trait.witness @p102 for @P102[i32]
  %d = trait.derive @P101[i32] from @P101_i32 given(%n) : (!trait.claim<@P102[i32] by @p102>)
  trait.return %d : !trait.claim<@P101[i32]>
}
trait.proof private @p102 {
  %n = trait.witness @p103 for @P103[i32]
  %d = trait.derive @P102[i32] from @P102_i32 given(%n) : (!trait.claim<@P103[i32] by @p103>)
  trait.return %d : !trait.claim<@P102[i32]>
}
trait.proof private @p103 {
  %n = trait.witness @p104 for @P104[i32]
  %d = trait.derive @P103[i32] from @P103_i32 given(%n) : (!trait.claim<@P104[i32] by @p104>)
  trait.return %d : !trait.claim<@P103[i32]>
}
trait.proof private @p104 {
  %n = trait.witness @p105 for @P105[i32]
  %d = trait.derive @P104[i32] from @P104_i32 given(%n) : (!trait.claim<@P105[i32] by @p105>)
  trait.return %d : !trait.claim<@P104[i32]>
}
trait.proof private @p105 {
  %n = trait.witness @p106 for @P106[i32]
  %d = trait.derive @P105[i32] from @P105_i32 given(%n) : (!trait.claim<@P106[i32] by @p106>)
  trait.return %d : !trait.claim<@P105[i32]>
}
trait.proof private @p106 {
  %n = trait.witness @p107 for @P107[i32]
  %d = trait.derive @P106[i32] from @P106_i32 given(%n) : (!trait.claim<@P107[i32] by @p107>)
  trait.return %d : !trait.claim<@P106[i32]>
}
trait.proof private @p107 {
  %n = trait.witness @p108 for @P108[i32]
  %d = trait.derive @P107[i32] from @P107_i32 given(%n) : (!trait.claim<@P108[i32] by @p108>)
  trait.return %d : !trait.claim<@P107[i32]>
}
trait.proof private @p108 {
  %n = trait.witness @p109 for @P109[i32]
  %d = trait.derive @P108[i32] from @P108_i32 given(%n) : (!trait.claim<@P109[i32] by @p109>)
  trait.return %d : !trait.claim<@P108[i32]>
}
trait.proof private @p109 {
  %n = trait.witness @p110 for @P110[i32]
  %d = trait.derive @P109[i32] from @P109_i32 given(%n) : (!trait.claim<@P110[i32] by @p110>)
  trait.return %d : !trait.claim<@P109[i32]>
}
trait.proof private @p110 {
  %n = trait.witness @p111 for @P111[i32]
  %d = trait.derive @P110[i32] from @P110_i32 given(%n) : (!trait.claim<@P111[i32] by @p111>)
  trait.return %d : !trait.claim<@P110[i32]>
}
trait.proof private @p111 {
  %n = trait.witness @p112 for @P112[i32]
  %d = trait.derive @P111[i32] from @P111_i32 given(%n) : (!trait.claim<@P112[i32] by @p112>)
  trait.return %d : !trait.claim<@P111[i32]>
}
trait.proof private @p112 {
  %n = trait.witness @p113 for @P113[i32]
  %d = trait.derive @P112[i32] from @P112_i32 given(%n) : (!trait.claim<@P113[i32] by @p113>)
  trait.return %d : !trait.claim<@P112[i32]>
}
trait.proof private @p113 {
  %n = trait.witness @p114 for @P114[i32]
  %d = trait.derive @P113[i32] from @P113_i32 given(%n) : (!trait.claim<@P114[i32] by @p114>)
  trait.return %d : !trait.claim<@P113[i32]>
}
trait.proof private @p114 {
  %n = trait.witness @p115 for @P115[i32]
  %d = trait.derive @P114[i32] from @P114_i32 given(%n) : (!trait.claim<@P115[i32] by @p115>)
  trait.return %d : !trait.claim<@P114[i32]>
}
trait.proof private @p115 {
  %n = trait.witness @p116 for @P116[i32]
  %d = trait.derive @P115[i32] from @P115_i32 given(%n) : (!trait.claim<@P116[i32] by @p116>)
  trait.return %d : !trait.claim<@P115[i32]>
}
trait.proof private @p116 {
  %n = trait.witness @p117 for @P117[i32]
  %d = trait.derive @P116[i32] from @P116_i32 given(%n) : (!trait.claim<@P117[i32] by @p117>)
  trait.return %d : !trait.claim<@P116[i32]>
}
trait.proof private @p117 {
  %n = trait.witness @p118 for @P118[i32]
  %d = trait.derive @P117[i32] from @P117_i32 given(%n) : (!trait.claim<@P118[i32] by @p118>)
  trait.return %d : !trait.claim<@P117[i32]>
}
trait.proof private @p118 {
  %n = trait.witness @p119 for @P119[i32]
  %d = trait.derive @P118[i32] from @P118_i32 given(%n) : (!trait.claim<@P119[i32] by @p119>)
  trait.return %d : !trait.claim<@P118[i32]>
}
trait.proof private @p119 {
  %n = trait.witness @p120 for @P120[i32]
  %d = trait.derive @P119[i32] from @P119_i32 given(%n) : (!trait.claim<@P120[i32] by @p120>)
  trait.return %d : !trait.claim<@P119[i32]>
}
trait.proof private @p120 {
  %n = trait.witness @p121 for @P121[i32]
  %d = trait.derive @P120[i32] from @P120_i32 given(%n) : (!trait.claim<@P121[i32] by @p121>)
  trait.return %d : !trait.claim<@P120[i32]>
}
trait.proof private @p121 {
  %n = trait.witness @p122 for @P122[i32]
  %d = trait.derive @P121[i32] from @P121_i32 given(%n) : (!trait.claim<@P122[i32] by @p122>)
  trait.return %d : !trait.claim<@P121[i32]>
}
trait.proof private @p122 {
  %n = trait.witness @p123 for @P123[i32]
  %d = trait.derive @P122[i32] from @P122_i32 given(%n) : (!trait.claim<@P123[i32] by @p123>)
  trait.return %d : !trait.claim<@P122[i32]>
}
trait.proof private @p123 {
  %n = trait.witness @p124 for @P124[i32]
  %d = trait.derive @P123[i32] from @P123_i32 given(%n) : (!trait.claim<@P124[i32] by @p124>)
  trait.return %d : !trait.claim<@P123[i32]>
}
trait.proof private @p124 {
  %n = trait.witness @p125 for @P125[i32]
  %d = trait.derive @P124[i32] from @P124_i32 given(%n) : (!trait.claim<@P125[i32] by @p125>)
  trait.return %d : !trait.claim<@P124[i32]>
}
trait.proof private @p125 {
  %n = trait.witness @p126 for @P126[i32]
  %d = trait.derive @P125[i32] from @P125_i32 given(%n) : (!trait.claim<@P126[i32] by @p126>)
  trait.return %d : !trait.claim<@P125[i32]>
}
trait.proof private @p126 {
  %n = trait.witness @p127 for @P127[i32]
  %d = trait.derive @P126[i32] from @P126_i32 given(%n) : (!trait.claim<@P127[i32] by @p127>)
  trait.return %d : !trait.claim<@P126[i32]>
}
trait.proof private @p127 {
  %n = trait.witness @p128 for @P128[i32]
  %d = trait.derive @P127[i32] from @P127_i32 given(%n) : (!trait.claim<@P128[i32] by @p128>)
  trait.return %d : !trait.claim<@P127[i32]>
}
trait.proof private @p128 {
  %n = trait.witness @P129_i32 for @P129[i32]
  %d = trait.derive @P128[i32] from @P128_i32 given(%n) : (!trait.claim<@P129[i32] by @P129_i32>)
  trait.return %d : !trait.claim<@P128[i32]>
}

// -----

// The same chain with its proofs standing from the bottom up. The height of
// the derivation below each proof is read wherever that proof is reached
// deeper, so the order proofs stand in does not change the verdict.

// CHECK: error: overflow evaluating the requirement {{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @P0(%self: !trait.claim<@P0[!T]>) {}
trait.trait private @P1(%self: !trait.claim<@P1[!T]>) {}
trait.trait private @P2(%self: !trait.claim<@P2[!T]>) {}
trait.trait private @P3(%self: !trait.claim<@P3[!T]>) {}
trait.trait private @P4(%self: !trait.claim<@P4[!T]>) {}
trait.trait private @P5(%self: !trait.claim<@P5[!T]>) {}
trait.trait private @P6(%self: !trait.claim<@P6[!T]>) {}
trait.trait private @P7(%self: !trait.claim<@P7[!T]>) {}
trait.trait private @P8(%self: !trait.claim<@P8[!T]>) {}
trait.trait private @P9(%self: !trait.claim<@P9[!T]>) {}
trait.trait private @P10(%self: !trait.claim<@P10[!T]>) {}
trait.trait private @P11(%self: !trait.claim<@P11[!T]>) {}
trait.trait private @P12(%self: !trait.claim<@P12[!T]>) {}
trait.trait private @P13(%self: !trait.claim<@P13[!T]>) {}
trait.trait private @P14(%self: !trait.claim<@P14[!T]>) {}
trait.trait private @P15(%self: !trait.claim<@P15[!T]>) {}
trait.trait private @P16(%self: !trait.claim<@P16[!T]>) {}
trait.trait private @P17(%self: !trait.claim<@P17[!T]>) {}
trait.trait private @P18(%self: !trait.claim<@P18[!T]>) {}
trait.trait private @P19(%self: !trait.claim<@P19[!T]>) {}
trait.trait private @P20(%self: !trait.claim<@P20[!T]>) {}
trait.trait private @P21(%self: !trait.claim<@P21[!T]>) {}
trait.trait private @P22(%self: !trait.claim<@P22[!T]>) {}
trait.trait private @P23(%self: !trait.claim<@P23[!T]>) {}
trait.trait private @P24(%self: !trait.claim<@P24[!T]>) {}
trait.trait private @P25(%self: !trait.claim<@P25[!T]>) {}
trait.trait private @P26(%self: !trait.claim<@P26[!T]>) {}
trait.trait private @P27(%self: !trait.claim<@P27[!T]>) {}
trait.trait private @P28(%self: !trait.claim<@P28[!T]>) {}
trait.trait private @P29(%self: !trait.claim<@P29[!T]>) {}
trait.trait private @P30(%self: !trait.claim<@P30[!T]>) {}
trait.trait private @P31(%self: !trait.claim<@P31[!T]>) {}
trait.trait private @P32(%self: !trait.claim<@P32[!T]>) {}
trait.trait private @P33(%self: !trait.claim<@P33[!T]>) {}
trait.trait private @P34(%self: !trait.claim<@P34[!T]>) {}
trait.trait private @P35(%self: !trait.claim<@P35[!T]>) {}
trait.trait private @P36(%self: !trait.claim<@P36[!T]>) {}
trait.trait private @P37(%self: !trait.claim<@P37[!T]>) {}
trait.trait private @P38(%self: !trait.claim<@P38[!T]>) {}
trait.trait private @P39(%self: !trait.claim<@P39[!T]>) {}
trait.trait private @P40(%self: !trait.claim<@P40[!T]>) {}
trait.trait private @P41(%self: !trait.claim<@P41[!T]>) {}
trait.trait private @P42(%self: !trait.claim<@P42[!T]>) {}
trait.trait private @P43(%self: !trait.claim<@P43[!T]>) {}
trait.trait private @P44(%self: !trait.claim<@P44[!T]>) {}
trait.trait private @P45(%self: !trait.claim<@P45[!T]>) {}
trait.trait private @P46(%self: !trait.claim<@P46[!T]>) {}
trait.trait private @P47(%self: !trait.claim<@P47[!T]>) {}
trait.trait private @P48(%self: !trait.claim<@P48[!T]>) {}
trait.trait private @P49(%self: !trait.claim<@P49[!T]>) {}
trait.trait private @P50(%self: !trait.claim<@P50[!T]>) {}
trait.trait private @P51(%self: !trait.claim<@P51[!T]>) {}
trait.trait private @P52(%self: !trait.claim<@P52[!T]>) {}
trait.trait private @P53(%self: !trait.claim<@P53[!T]>) {}
trait.trait private @P54(%self: !trait.claim<@P54[!T]>) {}
trait.trait private @P55(%self: !trait.claim<@P55[!T]>) {}
trait.trait private @P56(%self: !trait.claim<@P56[!T]>) {}
trait.trait private @P57(%self: !trait.claim<@P57[!T]>) {}
trait.trait private @P58(%self: !trait.claim<@P58[!T]>) {}
trait.trait private @P59(%self: !trait.claim<@P59[!T]>) {}
trait.trait private @P60(%self: !trait.claim<@P60[!T]>) {}
trait.trait private @P61(%self: !trait.claim<@P61[!T]>) {}
trait.trait private @P62(%self: !trait.claim<@P62[!T]>) {}
trait.trait private @P63(%self: !trait.claim<@P63[!T]>) {}
trait.trait private @P64(%self: !trait.claim<@P64[!T]>) {}
trait.trait private @P65(%self: !trait.claim<@P65[!T]>) {}
trait.trait private @P66(%self: !trait.claim<@P66[!T]>) {}
trait.trait private @P67(%self: !trait.claim<@P67[!T]>) {}
trait.trait private @P68(%self: !trait.claim<@P68[!T]>) {}
trait.trait private @P69(%self: !trait.claim<@P69[!T]>) {}
trait.trait private @P70(%self: !trait.claim<@P70[!T]>) {}
trait.trait private @P71(%self: !trait.claim<@P71[!T]>) {}
trait.trait private @P72(%self: !trait.claim<@P72[!T]>) {}
trait.trait private @P73(%self: !trait.claim<@P73[!T]>) {}
trait.trait private @P74(%self: !trait.claim<@P74[!T]>) {}
trait.trait private @P75(%self: !trait.claim<@P75[!T]>) {}
trait.trait private @P76(%self: !trait.claim<@P76[!T]>) {}
trait.trait private @P77(%self: !trait.claim<@P77[!T]>) {}
trait.trait private @P78(%self: !trait.claim<@P78[!T]>) {}
trait.trait private @P79(%self: !trait.claim<@P79[!T]>) {}
trait.trait private @P80(%self: !trait.claim<@P80[!T]>) {}
trait.trait private @P81(%self: !trait.claim<@P81[!T]>) {}
trait.trait private @P82(%self: !trait.claim<@P82[!T]>) {}
trait.trait private @P83(%self: !trait.claim<@P83[!T]>) {}
trait.trait private @P84(%self: !trait.claim<@P84[!T]>) {}
trait.trait private @P85(%self: !trait.claim<@P85[!T]>) {}
trait.trait private @P86(%self: !trait.claim<@P86[!T]>) {}
trait.trait private @P87(%self: !trait.claim<@P87[!T]>) {}
trait.trait private @P88(%self: !trait.claim<@P88[!T]>) {}
trait.trait private @P89(%self: !trait.claim<@P89[!T]>) {}
trait.trait private @P90(%self: !trait.claim<@P90[!T]>) {}
trait.trait private @P91(%self: !trait.claim<@P91[!T]>) {}
trait.trait private @P92(%self: !trait.claim<@P92[!T]>) {}
trait.trait private @P93(%self: !trait.claim<@P93[!T]>) {}
trait.trait private @P94(%self: !trait.claim<@P94[!T]>) {}
trait.trait private @P95(%self: !trait.claim<@P95[!T]>) {}
trait.trait private @P96(%self: !trait.claim<@P96[!T]>) {}
trait.trait private @P97(%self: !trait.claim<@P97[!T]>) {}
trait.trait private @P98(%self: !trait.claim<@P98[!T]>) {}
trait.trait private @P99(%self: !trait.claim<@P99[!T]>) {}
trait.trait private @P100(%self: !trait.claim<@P100[!T]>) {}
trait.trait private @P101(%self: !trait.claim<@P101[!T]>) {}
trait.trait private @P102(%self: !trait.claim<@P102[!T]>) {}
trait.trait private @P103(%self: !trait.claim<@P103[!T]>) {}
trait.trait private @P104(%self: !trait.claim<@P104[!T]>) {}
trait.trait private @P105(%self: !trait.claim<@P105[!T]>) {}
trait.trait private @P106(%self: !trait.claim<@P106[!T]>) {}
trait.trait private @P107(%self: !trait.claim<@P107[!T]>) {}
trait.trait private @P108(%self: !trait.claim<@P108[!T]>) {}
trait.trait private @P109(%self: !trait.claim<@P109[!T]>) {}
trait.trait private @P110(%self: !trait.claim<@P110[!T]>) {}
trait.trait private @P111(%self: !trait.claim<@P111[!T]>) {}
trait.trait private @P112(%self: !trait.claim<@P112[!T]>) {}
trait.trait private @P113(%self: !trait.claim<@P113[!T]>) {}
trait.trait private @P114(%self: !trait.claim<@P114[!T]>) {}
trait.trait private @P115(%self: !trait.claim<@P115[!T]>) {}
trait.trait private @P116(%self: !trait.claim<@P116[!T]>) {}
trait.trait private @P117(%self: !trait.claim<@P117[!T]>) {}
trait.trait private @P118(%self: !trait.claim<@P118[!T]>) {}
trait.trait private @P119(%self: !trait.claim<@P119[!T]>) {}
trait.trait private @P120(%self: !trait.claim<@P120[!T]>) {}
trait.trait private @P121(%self: !trait.claim<@P121[!T]>) {}
trait.trait private @P122(%self: !trait.claim<@P122[!T]>) {}
trait.trait private @P123(%self: !trait.claim<@P123[!T]>) {}
trait.trait private @P124(%self: !trait.claim<@P124[!T]>) {}
trait.trait private @P125(%self: !trait.claim<@P125[!T]>) {}
trait.trait private @P126(%self: !trait.claim<@P126[!T]>) {}
trait.trait private @P127(%self: !trait.claim<@P127[!T]>) {}
trait.trait private @P128(%self: !trait.claim<@P128[!T]>) {}
trait.trait private @P129(%self: !trait.claim<@P129[!T]>) {}
trait.impl private @P0_i32(%self: !trait.claim<@P0[i32]>, %n: !trait.claim<@P1[i32]>) {}
trait.impl private @P1_i32(%self: !trait.claim<@P1[i32]>, %n: !trait.claim<@P2[i32]>) {}
trait.impl private @P2_i32(%self: !trait.claim<@P2[i32]>, %n: !trait.claim<@P3[i32]>) {}
trait.impl private @P3_i32(%self: !trait.claim<@P3[i32]>, %n: !trait.claim<@P4[i32]>) {}
trait.impl private @P4_i32(%self: !trait.claim<@P4[i32]>, %n: !trait.claim<@P5[i32]>) {}
trait.impl private @P5_i32(%self: !trait.claim<@P5[i32]>, %n: !trait.claim<@P6[i32]>) {}
trait.impl private @P6_i32(%self: !trait.claim<@P6[i32]>, %n: !trait.claim<@P7[i32]>) {}
trait.impl private @P7_i32(%self: !trait.claim<@P7[i32]>, %n: !trait.claim<@P8[i32]>) {}
trait.impl private @P8_i32(%self: !trait.claim<@P8[i32]>, %n: !trait.claim<@P9[i32]>) {}
trait.impl private @P9_i32(%self: !trait.claim<@P9[i32]>, %n: !trait.claim<@P10[i32]>) {}
trait.impl private @P10_i32(%self: !trait.claim<@P10[i32]>, %n: !trait.claim<@P11[i32]>) {}
trait.impl private @P11_i32(%self: !trait.claim<@P11[i32]>, %n: !trait.claim<@P12[i32]>) {}
trait.impl private @P12_i32(%self: !trait.claim<@P12[i32]>, %n: !trait.claim<@P13[i32]>) {}
trait.impl private @P13_i32(%self: !trait.claim<@P13[i32]>, %n: !trait.claim<@P14[i32]>) {}
trait.impl private @P14_i32(%self: !trait.claim<@P14[i32]>, %n: !trait.claim<@P15[i32]>) {}
trait.impl private @P15_i32(%self: !trait.claim<@P15[i32]>, %n: !trait.claim<@P16[i32]>) {}
trait.impl private @P16_i32(%self: !trait.claim<@P16[i32]>, %n: !trait.claim<@P17[i32]>) {}
trait.impl private @P17_i32(%self: !trait.claim<@P17[i32]>, %n: !trait.claim<@P18[i32]>) {}
trait.impl private @P18_i32(%self: !trait.claim<@P18[i32]>, %n: !trait.claim<@P19[i32]>) {}
trait.impl private @P19_i32(%self: !trait.claim<@P19[i32]>, %n: !trait.claim<@P20[i32]>) {}
trait.impl private @P20_i32(%self: !trait.claim<@P20[i32]>, %n: !trait.claim<@P21[i32]>) {}
trait.impl private @P21_i32(%self: !trait.claim<@P21[i32]>, %n: !trait.claim<@P22[i32]>) {}
trait.impl private @P22_i32(%self: !trait.claim<@P22[i32]>, %n: !trait.claim<@P23[i32]>) {}
trait.impl private @P23_i32(%self: !trait.claim<@P23[i32]>, %n: !trait.claim<@P24[i32]>) {}
trait.impl private @P24_i32(%self: !trait.claim<@P24[i32]>, %n: !trait.claim<@P25[i32]>) {}
trait.impl private @P25_i32(%self: !trait.claim<@P25[i32]>, %n: !trait.claim<@P26[i32]>) {}
trait.impl private @P26_i32(%self: !trait.claim<@P26[i32]>, %n: !trait.claim<@P27[i32]>) {}
trait.impl private @P27_i32(%self: !trait.claim<@P27[i32]>, %n: !trait.claim<@P28[i32]>) {}
trait.impl private @P28_i32(%self: !trait.claim<@P28[i32]>, %n: !trait.claim<@P29[i32]>) {}
trait.impl private @P29_i32(%self: !trait.claim<@P29[i32]>, %n: !trait.claim<@P30[i32]>) {}
trait.impl private @P30_i32(%self: !trait.claim<@P30[i32]>, %n: !trait.claim<@P31[i32]>) {}
trait.impl private @P31_i32(%self: !trait.claim<@P31[i32]>, %n: !trait.claim<@P32[i32]>) {}
trait.impl private @P32_i32(%self: !trait.claim<@P32[i32]>, %n: !trait.claim<@P33[i32]>) {}
trait.impl private @P33_i32(%self: !trait.claim<@P33[i32]>, %n: !trait.claim<@P34[i32]>) {}
trait.impl private @P34_i32(%self: !trait.claim<@P34[i32]>, %n: !trait.claim<@P35[i32]>) {}
trait.impl private @P35_i32(%self: !trait.claim<@P35[i32]>, %n: !trait.claim<@P36[i32]>) {}
trait.impl private @P36_i32(%self: !trait.claim<@P36[i32]>, %n: !trait.claim<@P37[i32]>) {}
trait.impl private @P37_i32(%self: !trait.claim<@P37[i32]>, %n: !trait.claim<@P38[i32]>) {}
trait.impl private @P38_i32(%self: !trait.claim<@P38[i32]>, %n: !trait.claim<@P39[i32]>) {}
trait.impl private @P39_i32(%self: !trait.claim<@P39[i32]>, %n: !trait.claim<@P40[i32]>) {}
trait.impl private @P40_i32(%self: !trait.claim<@P40[i32]>, %n: !trait.claim<@P41[i32]>) {}
trait.impl private @P41_i32(%self: !trait.claim<@P41[i32]>, %n: !trait.claim<@P42[i32]>) {}
trait.impl private @P42_i32(%self: !trait.claim<@P42[i32]>, %n: !trait.claim<@P43[i32]>) {}
trait.impl private @P43_i32(%self: !trait.claim<@P43[i32]>, %n: !trait.claim<@P44[i32]>) {}
trait.impl private @P44_i32(%self: !trait.claim<@P44[i32]>, %n: !trait.claim<@P45[i32]>) {}
trait.impl private @P45_i32(%self: !trait.claim<@P45[i32]>, %n: !trait.claim<@P46[i32]>) {}
trait.impl private @P46_i32(%self: !trait.claim<@P46[i32]>, %n: !trait.claim<@P47[i32]>) {}
trait.impl private @P47_i32(%self: !trait.claim<@P47[i32]>, %n: !trait.claim<@P48[i32]>) {}
trait.impl private @P48_i32(%self: !trait.claim<@P48[i32]>, %n: !trait.claim<@P49[i32]>) {}
trait.impl private @P49_i32(%self: !trait.claim<@P49[i32]>, %n: !trait.claim<@P50[i32]>) {}
trait.impl private @P50_i32(%self: !trait.claim<@P50[i32]>, %n: !trait.claim<@P51[i32]>) {}
trait.impl private @P51_i32(%self: !trait.claim<@P51[i32]>, %n: !trait.claim<@P52[i32]>) {}
trait.impl private @P52_i32(%self: !trait.claim<@P52[i32]>, %n: !trait.claim<@P53[i32]>) {}
trait.impl private @P53_i32(%self: !trait.claim<@P53[i32]>, %n: !trait.claim<@P54[i32]>) {}
trait.impl private @P54_i32(%self: !trait.claim<@P54[i32]>, %n: !trait.claim<@P55[i32]>) {}
trait.impl private @P55_i32(%self: !trait.claim<@P55[i32]>, %n: !trait.claim<@P56[i32]>) {}
trait.impl private @P56_i32(%self: !trait.claim<@P56[i32]>, %n: !trait.claim<@P57[i32]>) {}
trait.impl private @P57_i32(%self: !trait.claim<@P57[i32]>, %n: !trait.claim<@P58[i32]>) {}
trait.impl private @P58_i32(%self: !trait.claim<@P58[i32]>, %n: !trait.claim<@P59[i32]>) {}
trait.impl private @P59_i32(%self: !trait.claim<@P59[i32]>, %n: !trait.claim<@P60[i32]>) {}
trait.impl private @P60_i32(%self: !trait.claim<@P60[i32]>, %n: !trait.claim<@P61[i32]>) {}
trait.impl private @P61_i32(%self: !trait.claim<@P61[i32]>, %n: !trait.claim<@P62[i32]>) {}
trait.impl private @P62_i32(%self: !trait.claim<@P62[i32]>, %n: !trait.claim<@P63[i32]>) {}
trait.impl private @P63_i32(%self: !trait.claim<@P63[i32]>, %n: !trait.claim<@P64[i32]>) {}
trait.impl private @P64_i32(%self: !trait.claim<@P64[i32]>, %n: !trait.claim<@P65[i32]>) {}
trait.impl private @P65_i32(%self: !trait.claim<@P65[i32]>, %n: !trait.claim<@P66[i32]>) {}
trait.impl private @P66_i32(%self: !trait.claim<@P66[i32]>, %n: !trait.claim<@P67[i32]>) {}
trait.impl private @P67_i32(%self: !trait.claim<@P67[i32]>, %n: !trait.claim<@P68[i32]>) {}
trait.impl private @P68_i32(%self: !trait.claim<@P68[i32]>, %n: !trait.claim<@P69[i32]>) {}
trait.impl private @P69_i32(%self: !trait.claim<@P69[i32]>, %n: !trait.claim<@P70[i32]>) {}
trait.impl private @P70_i32(%self: !trait.claim<@P70[i32]>, %n: !trait.claim<@P71[i32]>) {}
trait.impl private @P71_i32(%self: !trait.claim<@P71[i32]>, %n: !trait.claim<@P72[i32]>) {}
trait.impl private @P72_i32(%self: !trait.claim<@P72[i32]>, %n: !trait.claim<@P73[i32]>) {}
trait.impl private @P73_i32(%self: !trait.claim<@P73[i32]>, %n: !trait.claim<@P74[i32]>) {}
trait.impl private @P74_i32(%self: !trait.claim<@P74[i32]>, %n: !trait.claim<@P75[i32]>) {}
trait.impl private @P75_i32(%self: !trait.claim<@P75[i32]>, %n: !trait.claim<@P76[i32]>) {}
trait.impl private @P76_i32(%self: !trait.claim<@P76[i32]>, %n: !trait.claim<@P77[i32]>) {}
trait.impl private @P77_i32(%self: !trait.claim<@P77[i32]>, %n: !trait.claim<@P78[i32]>) {}
trait.impl private @P78_i32(%self: !trait.claim<@P78[i32]>, %n: !trait.claim<@P79[i32]>) {}
trait.impl private @P79_i32(%self: !trait.claim<@P79[i32]>, %n: !trait.claim<@P80[i32]>) {}
trait.impl private @P80_i32(%self: !trait.claim<@P80[i32]>, %n: !trait.claim<@P81[i32]>) {}
trait.impl private @P81_i32(%self: !trait.claim<@P81[i32]>, %n: !trait.claim<@P82[i32]>) {}
trait.impl private @P82_i32(%self: !trait.claim<@P82[i32]>, %n: !trait.claim<@P83[i32]>) {}
trait.impl private @P83_i32(%self: !trait.claim<@P83[i32]>, %n: !trait.claim<@P84[i32]>) {}
trait.impl private @P84_i32(%self: !trait.claim<@P84[i32]>, %n: !trait.claim<@P85[i32]>) {}
trait.impl private @P85_i32(%self: !trait.claim<@P85[i32]>, %n: !trait.claim<@P86[i32]>) {}
trait.impl private @P86_i32(%self: !trait.claim<@P86[i32]>, %n: !trait.claim<@P87[i32]>) {}
trait.impl private @P87_i32(%self: !trait.claim<@P87[i32]>, %n: !trait.claim<@P88[i32]>) {}
trait.impl private @P88_i32(%self: !trait.claim<@P88[i32]>, %n: !trait.claim<@P89[i32]>) {}
trait.impl private @P89_i32(%self: !trait.claim<@P89[i32]>, %n: !trait.claim<@P90[i32]>) {}
trait.impl private @P90_i32(%self: !trait.claim<@P90[i32]>, %n: !trait.claim<@P91[i32]>) {}
trait.impl private @P91_i32(%self: !trait.claim<@P91[i32]>, %n: !trait.claim<@P92[i32]>) {}
trait.impl private @P92_i32(%self: !trait.claim<@P92[i32]>, %n: !trait.claim<@P93[i32]>) {}
trait.impl private @P93_i32(%self: !trait.claim<@P93[i32]>, %n: !trait.claim<@P94[i32]>) {}
trait.impl private @P94_i32(%self: !trait.claim<@P94[i32]>, %n: !trait.claim<@P95[i32]>) {}
trait.impl private @P95_i32(%self: !trait.claim<@P95[i32]>, %n: !trait.claim<@P96[i32]>) {}
trait.impl private @P96_i32(%self: !trait.claim<@P96[i32]>, %n: !trait.claim<@P97[i32]>) {}
trait.impl private @P97_i32(%self: !trait.claim<@P97[i32]>, %n: !trait.claim<@P98[i32]>) {}
trait.impl private @P98_i32(%self: !trait.claim<@P98[i32]>, %n: !trait.claim<@P99[i32]>) {}
trait.impl private @P99_i32(%self: !trait.claim<@P99[i32]>, %n: !trait.claim<@P100[i32]>) {}
trait.impl private @P100_i32(%self: !trait.claim<@P100[i32]>, %n: !trait.claim<@P101[i32]>) {}
trait.impl private @P101_i32(%self: !trait.claim<@P101[i32]>, %n: !trait.claim<@P102[i32]>) {}
trait.impl private @P102_i32(%self: !trait.claim<@P102[i32]>, %n: !trait.claim<@P103[i32]>) {}
trait.impl private @P103_i32(%self: !trait.claim<@P103[i32]>, %n: !trait.claim<@P104[i32]>) {}
trait.impl private @P104_i32(%self: !trait.claim<@P104[i32]>, %n: !trait.claim<@P105[i32]>) {}
trait.impl private @P105_i32(%self: !trait.claim<@P105[i32]>, %n: !trait.claim<@P106[i32]>) {}
trait.impl private @P106_i32(%self: !trait.claim<@P106[i32]>, %n: !trait.claim<@P107[i32]>) {}
trait.impl private @P107_i32(%self: !trait.claim<@P107[i32]>, %n: !trait.claim<@P108[i32]>) {}
trait.impl private @P108_i32(%self: !trait.claim<@P108[i32]>, %n: !trait.claim<@P109[i32]>) {}
trait.impl private @P109_i32(%self: !trait.claim<@P109[i32]>, %n: !trait.claim<@P110[i32]>) {}
trait.impl private @P110_i32(%self: !trait.claim<@P110[i32]>, %n: !trait.claim<@P111[i32]>) {}
trait.impl private @P111_i32(%self: !trait.claim<@P111[i32]>, %n: !trait.claim<@P112[i32]>) {}
trait.impl private @P112_i32(%self: !trait.claim<@P112[i32]>, %n: !trait.claim<@P113[i32]>) {}
trait.impl private @P113_i32(%self: !trait.claim<@P113[i32]>, %n: !trait.claim<@P114[i32]>) {}
trait.impl private @P114_i32(%self: !trait.claim<@P114[i32]>, %n: !trait.claim<@P115[i32]>) {}
trait.impl private @P115_i32(%self: !trait.claim<@P115[i32]>, %n: !trait.claim<@P116[i32]>) {}
trait.impl private @P116_i32(%self: !trait.claim<@P116[i32]>, %n: !trait.claim<@P117[i32]>) {}
trait.impl private @P117_i32(%self: !trait.claim<@P117[i32]>, %n: !trait.claim<@P118[i32]>) {}
trait.impl private @P118_i32(%self: !trait.claim<@P118[i32]>, %n: !trait.claim<@P119[i32]>) {}
trait.impl private @P119_i32(%self: !trait.claim<@P119[i32]>, %n: !trait.claim<@P120[i32]>) {}
trait.impl private @P120_i32(%self: !trait.claim<@P120[i32]>, %n: !trait.claim<@P121[i32]>) {}
trait.impl private @P121_i32(%self: !trait.claim<@P121[i32]>, %n: !trait.claim<@P122[i32]>) {}
trait.impl private @P122_i32(%self: !trait.claim<@P122[i32]>, %n: !trait.claim<@P123[i32]>) {}
trait.impl private @P123_i32(%self: !trait.claim<@P123[i32]>, %n: !trait.claim<@P124[i32]>) {}
trait.impl private @P124_i32(%self: !trait.claim<@P124[i32]>, %n: !trait.claim<@P125[i32]>) {}
trait.impl private @P125_i32(%self: !trait.claim<@P125[i32]>, %n: !trait.claim<@P126[i32]>) {}
trait.impl private @P126_i32(%self: !trait.claim<@P126[i32]>, %n: !trait.claim<@P127[i32]>) {}
trait.impl private @P127_i32(%self: !trait.claim<@P127[i32]>, %n: !trait.claim<@P128[i32]>) {}
trait.impl private @P128_i32(%self: !trait.claim<@P128[i32]>, %n: !trait.claim<@P129[i32]>) {}
trait.impl private @P129_i32(%self: !trait.claim<@P129[i32]>) {}
trait.proof private @p128 {
  %n = trait.witness @P129_i32 for @P129[i32]
  %d = trait.derive @P128[i32] from @P128_i32 given(%n) : (!trait.claim<@P129[i32] by @P129_i32>)
  trait.return %d : !trait.claim<@P128[i32]>
}
trait.proof private @p127 {
  %n = trait.witness @p128 for @P128[i32]
  %d = trait.derive @P127[i32] from @P127_i32 given(%n) : (!trait.claim<@P128[i32] by @p128>)
  trait.return %d : !trait.claim<@P127[i32]>
}
trait.proof private @p126 {
  %n = trait.witness @p127 for @P127[i32]
  %d = trait.derive @P126[i32] from @P126_i32 given(%n) : (!trait.claim<@P127[i32] by @p127>)
  trait.return %d : !trait.claim<@P126[i32]>
}
trait.proof private @p125 {
  %n = trait.witness @p126 for @P126[i32]
  %d = trait.derive @P125[i32] from @P125_i32 given(%n) : (!trait.claim<@P126[i32] by @p126>)
  trait.return %d : !trait.claim<@P125[i32]>
}
trait.proof private @p124 {
  %n = trait.witness @p125 for @P125[i32]
  %d = trait.derive @P124[i32] from @P124_i32 given(%n) : (!trait.claim<@P125[i32] by @p125>)
  trait.return %d : !trait.claim<@P124[i32]>
}
trait.proof private @p123 {
  %n = trait.witness @p124 for @P124[i32]
  %d = trait.derive @P123[i32] from @P123_i32 given(%n) : (!trait.claim<@P124[i32] by @p124>)
  trait.return %d : !trait.claim<@P123[i32]>
}
trait.proof private @p122 {
  %n = trait.witness @p123 for @P123[i32]
  %d = trait.derive @P122[i32] from @P122_i32 given(%n) : (!trait.claim<@P123[i32] by @p123>)
  trait.return %d : !trait.claim<@P122[i32]>
}
trait.proof private @p121 {
  %n = trait.witness @p122 for @P122[i32]
  %d = trait.derive @P121[i32] from @P121_i32 given(%n) : (!trait.claim<@P122[i32] by @p122>)
  trait.return %d : !trait.claim<@P121[i32]>
}
trait.proof private @p120 {
  %n = trait.witness @p121 for @P121[i32]
  %d = trait.derive @P120[i32] from @P120_i32 given(%n) : (!trait.claim<@P121[i32] by @p121>)
  trait.return %d : !trait.claim<@P120[i32]>
}
trait.proof private @p119 {
  %n = trait.witness @p120 for @P120[i32]
  %d = trait.derive @P119[i32] from @P119_i32 given(%n) : (!trait.claim<@P120[i32] by @p120>)
  trait.return %d : !trait.claim<@P119[i32]>
}
trait.proof private @p118 {
  %n = trait.witness @p119 for @P119[i32]
  %d = trait.derive @P118[i32] from @P118_i32 given(%n) : (!trait.claim<@P119[i32] by @p119>)
  trait.return %d : !trait.claim<@P118[i32]>
}
trait.proof private @p117 {
  %n = trait.witness @p118 for @P118[i32]
  %d = trait.derive @P117[i32] from @P117_i32 given(%n) : (!trait.claim<@P118[i32] by @p118>)
  trait.return %d : !trait.claim<@P117[i32]>
}
trait.proof private @p116 {
  %n = trait.witness @p117 for @P117[i32]
  %d = trait.derive @P116[i32] from @P116_i32 given(%n) : (!trait.claim<@P117[i32] by @p117>)
  trait.return %d : !trait.claim<@P116[i32]>
}
trait.proof private @p115 {
  %n = trait.witness @p116 for @P116[i32]
  %d = trait.derive @P115[i32] from @P115_i32 given(%n) : (!trait.claim<@P116[i32] by @p116>)
  trait.return %d : !trait.claim<@P115[i32]>
}
trait.proof private @p114 {
  %n = trait.witness @p115 for @P115[i32]
  %d = trait.derive @P114[i32] from @P114_i32 given(%n) : (!trait.claim<@P115[i32] by @p115>)
  trait.return %d : !trait.claim<@P114[i32]>
}
trait.proof private @p113 {
  %n = trait.witness @p114 for @P114[i32]
  %d = trait.derive @P113[i32] from @P113_i32 given(%n) : (!trait.claim<@P114[i32] by @p114>)
  trait.return %d : !trait.claim<@P113[i32]>
}
trait.proof private @p112 {
  %n = trait.witness @p113 for @P113[i32]
  %d = trait.derive @P112[i32] from @P112_i32 given(%n) : (!trait.claim<@P113[i32] by @p113>)
  trait.return %d : !trait.claim<@P112[i32]>
}
trait.proof private @p111 {
  %n = trait.witness @p112 for @P112[i32]
  %d = trait.derive @P111[i32] from @P111_i32 given(%n) : (!trait.claim<@P112[i32] by @p112>)
  trait.return %d : !trait.claim<@P111[i32]>
}
trait.proof private @p110 {
  %n = trait.witness @p111 for @P111[i32]
  %d = trait.derive @P110[i32] from @P110_i32 given(%n) : (!trait.claim<@P111[i32] by @p111>)
  trait.return %d : !trait.claim<@P110[i32]>
}
trait.proof private @p109 {
  %n = trait.witness @p110 for @P110[i32]
  %d = trait.derive @P109[i32] from @P109_i32 given(%n) : (!trait.claim<@P110[i32] by @p110>)
  trait.return %d : !trait.claim<@P109[i32]>
}
trait.proof private @p108 {
  %n = trait.witness @p109 for @P109[i32]
  %d = trait.derive @P108[i32] from @P108_i32 given(%n) : (!trait.claim<@P109[i32] by @p109>)
  trait.return %d : !trait.claim<@P108[i32]>
}
trait.proof private @p107 {
  %n = trait.witness @p108 for @P108[i32]
  %d = trait.derive @P107[i32] from @P107_i32 given(%n) : (!trait.claim<@P108[i32] by @p108>)
  trait.return %d : !trait.claim<@P107[i32]>
}
trait.proof private @p106 {
  %n = trait.witness @p107 for @P107[i32]
  %d = trait.derive @P106[i32] from @P106_i32 given(%n) : (!trait.claim<@P107[i32] by @p107>)
  trait.return %d : !trait.claim<@P106[i32]>
}
trait.proof private @p105 {
  %n = trait.witness @p106 for @P106[i32]
  %d = trait.derive @P105[i32] from @P105_i32 given(%n) : (!trait.claim<@P106[i32] by @p106>)
  trait.return %d : !trait.claim<@P105[i32]>
}
trait.proof private @p104 {
  %n = trait.witness @p105 for @P105[i32]
  %d = trait.derive @P104[i32] from @P104_i32 given(%n) : (!trait.claim<@P105[i32] by @p105>)
  trait.return %d : !trait.claim<@P104[i32]>
}
trait.proof private @p103 {
  %n = trait.witness @p104 for @P104[i32]
  %d = trait.derive @P103[i32] from @P103_i32 given(%n) : (!trait.claim<@P104[i32] by @p104>)
  trait.return %d : !trait.claim<@P103[i32]>
}
trait.proof private @p102 {
  %n = trait.witness @p103 for @P103[i32]
  %d = trait.derive @P102[i32] from @P102_i32 given(%n) : (!trait.claim<@P103[i32] by @p103>)
  trait.return %d : !trait.claim<@P102[i32]>
}
trait.proof private @p101 {
  %n = trait.witness @p102 for @P102[i32]
  %d = trait.derive @P101[i32] from @P101_i32 given(%n) : (!trait.claim<@P102[i32] by @p102>)
  trait.return %d : !trait.claim<@P101[i32]>
}
trait.proof private @p100 {
  %n = trait.witness @p101 for @P101[i32]
  %d = trait.derive @P100[i32] from @P100_i32 given(%n) : (!trait.claim<@P101[i32] by @p101>)
  trait.return %d : !trait.claim<@P100[i32]>
}
trait.proof private @p99 {
  %n = trait.witness @p100 for @P100[i32]
  %d = trait.derive @P99[i32] from @P99_i32 given(%n) : (!trait.claim<@P100[i32] by @p100>)
  trait.return %d : !trait.claim<@P99[i32]>
}
trait.proof private @p98 {
  %n = trait.witness @p99 for @P99[i32]
  %d = trait.derive @P98[i32] from @P98_i32 given(%n) : (!trait.claim<@P99[i32] by @p99>)
  trait.return %d : !trait.claim<@P98[i32]>
}
trait.proof private @p97 {
  %n = trait.witness @p98 for @P98[i32]
  %d = trait.derive @P97[i32] from @P97_i32 given(%n) : (!trait.claim<@P98[i32] by @p98>)
  trait.return %d : !trait.claim<@P97[i32]>
}
trait.proof private @p96 {
  %n = trait.witness @p97 for @P97[i32]
  %d = trait.derive @P96[i32] from @P96_i32 given(%n) : (!trait.claim<@P97[i32] by @p97>)
  trait.return %d : !trait.claim<@P96[i32]>
}
trait.proof private @p95 {
  %n = trait.witness @p96 for @P96[i32]
  %d = trait.derive @P95[i32] from @P95_i32 given(%n) : (!trait.claim<@P96[i32] by @p96>)
  trait.return %d : !trait.claim<@P95[i32]>
}
trait.proof private @p94 {
  %n = trait.witness @p95 for @P95[i32]
  %d = trait.derive @P94[i32] from @P94_i32 given(%n) : (!trait.claim<@P95[i32] by @p95>)
  trait.return %d : !trait.claim<@P94[i32]>
}
trait.proof private @p93 {
  %n = trait.witness @p94 for @P94[i32]
  %d = trait.derive @P93[i32] from @P93_i32 given(%n) : (!trait.claim<@P94[i32] by @p94>)
  trait.return %d : !trait.claim<@P93[i32]>
}
trait.proof private @p92 {
  %n = trait.witness @p93 for @P93[i32]
  %d = trait.derive @P92[i32] from @P92_i32 given(%n) : (!trait.claim<@P93[i32] by @p93>)
  trait.return %d : !trait.claim<@P92[i32]>
}
trait.proof private @p91 {
  %n = trait.witness @p92 for @P92[i32]
  %d = trait.derive @P91[i32] from @P91_i32 given(%n) : (!trait.claim<@P92[i32] by @p92>)
  trait.return %d : !trait.claim<@P91[i32]>
}
trait.proof private @p90 {
  %n = trait.witness @p91 for @P91[i32]
  %d = trait.derive @P90[i32] from @P90_i32 given(%n) : (!trait.claim<@P91[i32] by @p91>)
  trait.return %d : !trait.claim<@P90[i32]>
}
trait.proof private @p89 {
  %n = trait.witness @p90 for @P90[i32]
  %d = trait.derive @P89[i32] from @P89_i32 given(%n) : (!trait.claim<@P90[i32] by @p90>)
  trait.return %d : !trait.claim<@P89[i32]>
}
trait.proof private @p88 {
  %n = trait.witness @p89 for @P89[i32]
  %d = trait.derive @P88[i32] from @P88_i32 given(%n) : (!trait.claim<@P89[i32] by @p89>)
  trait.return %d : !trait.claim<@P88[i32]>
}
trait.proof private @p87 {
  %n = trait.witness @p88 for @P88[i32]
  %d = trait.derive @P87[i32] from @P87_i32 given(%n) : (!trait.claim<@P88[i32] by @p88>)
  trait.return %d : !trait.claim<@P87[i32]>
}
trait.proof private @p86 {
  %n = trait.witness @p87 for @P87[i32]
  %d = trait.derive @P86[i32] from @P86_i32 given(%n) : (!trait.claim<@P87[i32] by @p87>)
  trait.return %d : !trait.claim<@P86[i32]>
}
trait.proof private @p85 {
  %n = trait.witness @p86 for @P86[i32]
  %d = trait.derive @P85[i32] from @P85_i32 given(%n) : (!trait.claim<@P86[i32] by @p86>)
  trait.return %d : !trait.claim<@P85[i32]>
}
trait.proof private @p84 {
  %n = trait.witness @p85 for @P85[i32]
  %d = trait.derive @P84[i32] from @P84_i32 given(%n) : (!trait.claim<@P85[i32] by @p85>)
  trait.return %d : !trait.claim<@P84[i32]>
}
trait.proof private @p83 {
  %n = trait.witness @p84 for @P84[i32]
  %d = trait.derive @P83[i32] from @P83_i32 given(%n) : (!trait.claim<@P84[i32] by @p84>)
  trait.return %d : !trait.claim<@P83[i32]>
}
trait.proof private @p82 {
  %n = trait.witness @p83 for @P83[i32]
  %d = trait.derive @P82[i32] from @P82_i32 given(%n) : (!trait.claim<@P83[i32] by @p83>)
  trait.return %d : !trait.claim<@P82[i32]>
}
trait.proof private @p81 {
  %n = trait.witness @p82 for @P82[i32]
  %d = trait.derive @P81[i32] from @P81_i32 given(%n) : (!trait.claim<@P82[i32] by @p82>)
  trait.return %d : !trait.claim<@P81[i32]>
}
trait.proof private @p80 {
  %n = trait.witness @p81 for @P81[i32]
  %d = trait.derive @P80[i32] from @P80_i32 given(%n) : (!trait.claim<@P81[i32] by @p81>)
  trait.return %d : !trait.claim<@P80[i32]>
}
trait.proof private @p79 {
  %n = trait.witness @p80 for @P80[i32]
  %d = trait.derive @P79[i32] from @P79_i32 given(%n) : (!trait.claim<@P80[i32] by @p80>)
  trait.return %d : !trait.claim<@P79[i32]>
}
trait.proof private @p78 {
  %n = trait.witness @p79 for @P79[i32]
  %d = trait.derive @P78[i32] from @P78_i32 given(%n) : (!trait.claim<@P79[i32] by @p79>)
  trait.return %d : !trait.claim<@P78[i32]>
}
trait.proof private @p77 {
  %n = trait.witness @p78 for @P78[i32]
  %d = trait.derive @P77[i32] from @P77_i32 given(%n) : (!trait.claim<@P78[i32] by @p78>)
  trait.return %d : !trait.claim<@P77[i32]>
}
trait.proof private @p76 {
  %n = trait.witness @p77 for @P77[i32]
  %d = trait.derive @P76[i32] from @P76_i32 given(%n) : (!trait.claim<@P77[i32] by @p77>)
  trait.return %d : !trait.claim<@P76[i32]>
}
trait.proof private @p75 {
  %n = trait.witness @p76 for @P76[i32]
  %d = trait.derive @P75[i32] from @P75_i32 given(%n) : (!trait.claim<@P76[i32] by @p76>)
  trait.return %d : !trait.claim<@P75[i32]>
}
trait.proof private @p74 {
  %n = trait.witness @p75 for @P75[i32]
  %d = trait.derive @P74[i32] from @P74_i32 given(%n) : (!trait.claim<@P75[i32] by @p75>)
  trait.return %d : !trait.claim<@P74[i32]>
}
trait.proof private @p73 {
  %n = trait.witness @p74 for @P74[i32]
  %d = trait.derive @P73[i32] from @P73_i32 given(%n) : (!trait.claim<@P74[i32] by @p74>)
  trait.return %d : !trait.claim<@P73[i32]>
}
trait.proof private @p72 {
  %n = trait.witness @p73 for @P73[i32]
  %d = trait.derive @P72[i32] from @P72_i32 given(%n) : (!trait.claim<@P73[i32] by @p73>)
  trait.return %d : !trait.claim<@P72[i32]>
}
trait.proof private @p71 {
  %n = trait.witness @p72 for @P72[i32]
  %d = trait.derive @P71[i32] from @P71_i32 given(%n) : (!trait.claim<@P72[i32] by @p72>)
  trait.return %d : !trait.claim<@P71[i32]>
}
trait.proof private @p70 {
  %n = trait.witness @p71 for @P71[i32]
  %d = trait.derive @P70[i32] from @P70_i32 given(%n) : (!trait.claim<@P71[i32] by @p71>)
  trait.return %d : !trait.claim<@P70[i32]>
}
trait.proof private @p69 {
  %n = trait.witness @p70 for @P70[i32]
  %d = trait.derive @P69[i32] from @P69_i32 given(%n) : (!trait.claim<@P70[i32] by @p70>)
  trait.return %d : !trait.claim<@P69[i32]>
}
trait.proof private @p68 {
  %n = trait.witness @p69 for @P69[i32]
  %d = trait.derive @P68[i32] from @P68_i32 given(%n) : (!trait.claim<@P69[i32] by @p69>)
  trait.return %d : !trait.claim<@P68[i32]>
}
trait.proof private @p67 {
  %n = trait.witness @p68 for @P68[i32]
  %d = trait.derive @P67[i32] from @P67_i32 given(%n) : (!trait.claim<@P68[i32] by @p68>)
  trait.return %d : !trait.claim<@P67[i32]>
}
trait.proof private @p66 {
  %n = trait.witness @p67 for @P67[i32]
  %d = trait.derive @P66[i32] from @P66_i32 given(%n) : (!trait.claim<@P67[i32] by @p67>)
  trait.return %d : !trait.claim<@P66[i32]>
}
trait.proof private @p65 {
  %n = trait.witness @p66 for @P66[i32]
  %d = trait.derive @P65[i32] from @P65_i32 given(%n) : (!trait.claim<@P66[i32] by @p66>)
  trait.return %d : !trait.claim<@P65[i32]>
}
trait.proof private @p64 {
  %n = trait.witness @p65 for @P65[i32]
  %d = trait.derive @P64[i32] from @P64_i32 given(%n) : (!trait.claim<@P65[i32] by @p65>)
  trait.return %d : !trait.claim<@P64[i32]>
}
trait.proof private @p63 {
  %n = trait.witness @p64 for @P64[i32]
  %d = trait.derive @P63[i32] from @P63_i32 given(%n) : (!trait.claim<@P64[i32] by @p64>)
  trait.return %d : !trait.claim<@P63[i32]>
}
trait.proof private @p62 {
  %n = trait.witness @p63 for @P63[i32]
  %d = trait.derive @P62[i32] from @P62_i32 given(%n) : (!trait.claim<@P63[i32] by @p63>)
  trait.return %d : !trait.claim<@P62[i32]>
}
trait.proof private @p61 {
  %n = trait.witness @p62 for @P62[i32]
  %d = trait.derive @P61[i32] from @P61_i32 given(%n) : (!trait.claim<@P62[i32] by @p62>)
  trait.return %d : !trait.claim<@P61[i32]>
}
trait.proof private @p60 {
  %n = trait.witness @p61 for @P61[i32]
  %d = trait.derive @P60[i32] from @P60_i32 given(%n) : (!trait.claim<@P61[i32] by @p61>)
  trait.return %d : !trait.claim<@P60[i32]>
}
trait.proof private @p59 {
  %n = trait.witness @p60 for @P60[i32]
  %d = trait.derive @P59[i32] from @P59_i32 given(%n) : (!trait.claim<@P60[i32] by @p60>)
  trait.return %d : !trait.claim<@P59[i32]>
}
trait.proof private @p58 {
  %n = trait.witness @p59 for @P59[i32]
  %d = trait.derive @P58[i32] from @P58_i32 given(%n) : (!trait.claim<@P59[i32] by @p59>)
  trait.return %d : !trait.claim<@P58[i32]>
}
trait.proof private @p57 {
  %n = trait.witness @p58 for @P58[i32]
  %d = trait.derive @P57[i32] from @P57_i32 given(%n) : (!trait.claim<@P58[i32] by @p58>)
  trait.return %d : !trait.claim<@P57[i32]>
}
trait.proof private @p56 {
  %n = trait.witness @p57 for @P57[i32]
  %d = trait.derive @P56[i32] from @P56_i32 given(%n) : (!trait.claim<@P57[i32] by @p57>)
  trait.return %d : !trait.claim<@P56[i32]>
}
trait.proof private @p55 {
  %n = trait.witness @p56 for @P56[i32]
  %d = trait.derive @P55[i32] from @P55_i32 given(%n) : (!trait.claim<@P56[i32] by @p56>)
  trait.return %d : !trait.claim<@P55[i32]>
}
trait.proof private @p54 {
  %n = trait.witness @p55 for @P55[i32]
  %d = trait.derive @P54[i32] from @P54_i32 given(%n) : (!trait.claim<@P55[i32] by @p55>)
  trait.return %d : !trait.claim<@P54[i32]>
}
trait.proof private @p53 {
  %n = trait.witness @p54 for @P54[i32]
  %d = trait.derive @P53[i32] from @P53_i32 given(%n) : (!trait.claim<@P54[i32] by @p54>)
  trait.return %d : !trait.claim<@P53[i32]>
}
trait.proof private @p52 {
  %n = trait.witness @p53 for @P53[i32]
  %d = trait.derive @P52[i32] from @P52_i32 given(%n) : (!trait.claim<@P53[i32] by @p53>)
  trait.return %d : !trait.claim<@P52[i32]>
}
trait.proof private @p51 {
  %n = trait.witness @p52 for @P52[i32]
  %d = trait.derive @P51[i32] from @P51_i32 given(%n) : (!trait.claim<@P52[i32] by @p52>)
  trait.return %d : !trait.claim<@P51[i32]>
}
trait.proof private @p50 {
  %n = trait.witness @p51 for @P51[i32]
  %d = trait.derive @P50[i32] from @P50_i32 given(%n) : (!trait.claim<@P51[i32] by @p51>)
  trait.return %d : !trait.claim<@P50[i32]>
}
trait.proof private @p49 {
  %n = trait.witness @p50 for @P50[i32]
  %d = trait.derive @P49[i32] from @P49_i32 given(%n) : (!trait.claim<@P50[i32] by @p50>)
  trait.return %d : !trait.claim<@P49[i32]>
}
trait.proof private @p48 {
  %n = trait.witness @p49 for @P49[i32]
  %d = trait.derive @P48[i32] from @P48_i32 given(%n) : (!trait.claim<@P49[i32] by @p49>)
  trait.return %d : !trait.claim<@P48[i32]>
}
trait.proof private @p47 {
  %n = trait.witness @p48 for @P48[i32]
  %d = trait.derive @P47[i32] from @P47_i32 given(%n) : (!trait.claim<@P48[i32] by @p48>)
  trait.return %d : !trait.claim<@P47[i32]>
}
trait.proof private @p46 {
  %n = trait.witness @p47 for @P47[i32]
  %d = trait.derive @P46[i32] from @P46_i32 given(%n) : (!trait.claim<@P47[i32] by @p47>)
  trait.return %d : !trait.claim<@P46[i32]>
}
trait.proof private @p45 {
  %n = trait.witness @p46 for @P46[i32]
  %d = trait.derive @P45[i32] from @P45_i32 given(%n) : (!trait.claim<@P46[i32] by @p46>)
  trait.return %d : !trait.claim<@P45[i32]>
}
trait.proof private @p44 {
  %n = trait.witness @p45 for @P45[i32]
  %d = trait.derive @P44[i32] from @P44_i32 given(%n) : (!trait.claim<@P45[i32] by @p45>)
  trait.return %d : !trait.claim<@P44[i32]>
}
trait.proof private @p43 {
  %n = trait.witness @p44 for @P44[i32]
  %d = trait.derive @P43[i32] from @P43_i32 given(%n) : (!trait.claim<@P44[i32] by @p44>)
  trait.return %d : !trait.claim<@P43[i32]>
}
trait.proof private @p42 {
  %n = trait.witness @p43 for @P43[i32]
  %d = trait.derive @P42[i32] from @P42_i32 given(%n) : (!trait.claim<@P43[i32] by @p43>)
  trait.return %d : !trait.claim<@P42[i32]>
}
trait.proof private @p41 {
  %n = trait.witness @p42 for @P42[i32]
  %d = trait.derive @P41[i32] from @P41_i32 given(%n) : (!trait.claim<@P42[i32] by @p42>)
  trait.return %d : !trait.claim<@P41[i32]>
}
trait.proof private @p40 {
  %n = trait.witness @p41 for @P41[i32]
  %d = trait.derive @P40[i32] from @P40_i32 given(%n) : (!trait.claim<@P41[i32] by @p41>)
  trait.return %d : !trait.claim<@P40[i32]>
}
trait.proof private @p39 {
  %n = trait.witness @p40 for @P40[i32]
  %d = trait.derive @P39[i32] from @P39_i32 given(%n) : (!trait.claim<@P40[i32] by @p40>)
  trait.return %d : !trait.claim<@P39[i32]>
}
trait.proof private @p38 {
  %n = trait.witness @p39 for @P39[i32]
  %d = trait.derive @P38[i32] from @P38_i32 given(%n) : (!trait.claim<@P39[i32] by @p39>)
  trait.return %d : !trait.claim<@P38[i32]>
}
trait.proof private @p37 {
  %n = trait.witness @p38 for @P38[i32]
  %d = trait.derive @P37[i32] from @P37_i32 given(%n) : (!trait.claim<@P38[i32] by @p38>)
  trait.return %d : !trait.claim<@P37[i32]>
}
trait.proof private @p36 {
  %n = trait.witness @p37 for @P37[i32]
  %d = trait.derive @P36[i32] from @P36_i32 given(%n) : (!trait.claim<@P37[i32] by @p37>)
  trait.return %d : !trait.claim<@P36[i32]>
}
trait.proof private @p35 {
  %n = trait.witness @p36 for @P36[i32]
  %d = trait.derive @P35[i32] from @P35_i32 given(%n) : (!trait.claim<@P36[i32] by @p36>)
  trait.return %d : !trait.claim<@P35[i32]>
}
trait.proof private @p34 {
  %n = trait.witness @p35 for @P35[i32]
  %d = trait.derive @P34[i32] from @P34_i32 given(%n) : (!trait.claim<@P35[i32] by @p35>)
  trait.return %d : !trait.claim<@P34[i32]>
}
trait.proof private @p33 {
  %n = trait.witness @p34 for @P34[i32]
  %d = trait.derive @P33[i32] from @P33_i32 given(%n) : (!trait.claim<@P34[i32] by @p34>)
  trait.return %d : !trait.claim<@P33[i32]>
}
trait.proof private @p32 {
  %n = trait.witness @p33 for @P33[i32]
  %d = trait.derive @P32[i32] from @P32_i32 given(%n) : (!trait.claim<@P33[i32] by @p33>)
  trait.return %d : !trait.claim<@P32[i32]>
}
trait.proof private @p31 {
  %n = trait.witness @p32 for @P32[i32]
  %d = trait.derive @P31[i32] from @P31_i32 given(%n) : (!trait.claim<@P32[i32] by @p32>)
  trait.return %d : !trait.claim<@P31[i32]>
}
trait.proof private @p30 {
  %n = trait.witness @p31 for @P31[i32]
  %d = trait.derive @P30[i32] from @P30_i32 given(%n) : (!trait.claim<@P31[i32] by @p31>)
  trait.return %d : !trait.claim<@P30[i32]>
}
trait.proof private @p29 {
  %n = trait.witness @p30 for @P30[i32]
  %d = trait.derive @P29[i32] from @P29_i32 given(%n) : (!trait.claim<@P30[i32] by @p30>)
  trait.return %d : !trait.claim<@P29[i32]>
}
trait.proof private @p28 {
  %n = trait.witness @p29 for @P29[i32]
  %d = trait.derive @P28[i32] from @P28_i32 given(%n) : (!trait.claim<@P29[i32] by @p29>)
  trait.return %d : !trait.claim<@P28[i32]>
}
trait.proof private @p27 {
  %n = trait.witness @p28 for @P28[i32]
  %d = trait.derive @P27[i32] from @P27_i32 given(%n) : (!trait.claim<@P28[i32] by @p28>)
  trait.return %d : !trait.claim<@P27[i32]>
}
trait.proof private @p26 {
  %n = trait.witness @p27 for @P27[i32]
  %d = trait.derive @P26[i32] from @P26_i32 given(%n) : (!trait.claim<@P27[i32] by @p27>)
  trait.return %d : !trait.claim<@P26[i32]>
}
trait.proof private @p25 {
  %n = trait.witness @p26 for @P26[i32]
  %d = trait.derive @P25[i32] from @P25_i32 given(%n) : (!trait.claim<@P26[i32] by @p26>)
  trait.return %d : !trait.claim<@P25[i32]>
}
trait.proof private @p24 {
  %n = trait.witness @p25 for @P25[i32]
  %d = trait.derive @P24[i32] from @P24_i32 given(%n) : (!trait.claim<@P25[i32] by @p25>)
  trait.return %d : !trait.claim<@P24[i32]>
}
trait.proof private @p23 {
  %n = trait.witness @p24 for @P24[i32]
  %d = trait.derive @P23[i32] from @P23_i32 given(%n) : (!trait.claim<@P24[i32] by @p24>)
  trait.return %d : !trait.claim<@P23[i32]>
}
trait.proof private @p22 {
  %n = trait.witness @p23 for @P23[i32]
  %d = trait.derive @P22[i32] from @P22_i32 given(%n) : (!trait.claim<@P23[i32] by @p23>)
  trait.return %d : !trait.claim<@P22[i32]>
}
trait.proof private @p21 {
  %n = trait.witness @p22 for @P22[i32]
  %d = trait.derive @P21[i32] from @P21_i32 given(%n) : (!trait.claim<@P22[i32] by @p22>)
  trait.return %d : !trait.claim<@P21[i32]>
}
trait.proof private @p20 {
  %n = trait.witness @p21 for @P21[i32]
  %d = trait.derive @P20[i32] from @P20_i32 given(%n) : (!trait.claim<@P21[i32] by @p21>)
  trait.return %d : !trait.claim<@P20[i32]>
}
trait.proof private @p19 {
  %n = trait.witness @p20 for @P20[i32]
  %d = trait.derive @P19[i32] from @P19_i32 given(%n) : (!trait.claim<@P20[i32] by @p20>)
  trait.return %d : !trait.claim<@P19[i32]>
}
trait.proof private @p18 {
  %n = trait.witness @p19 for @P19[i32]
  %d = trait.derive @P18[i32] from @P18_i32 given(%n) : (!trait.claim<@P19[i32] by @p19>)
  trait.return %d : !trait.claim<@P18[i32]>
}
trait.proof private @p17 {
  %n = trait.witness @p18 for @P18[i32]
  %d = trait.derive @P17[i32] from @P17_i32 given(%n) : (!trait.claim<@P18[i32] by @p18>)
  trait.return %d : !trait.claim<@P17[i32]>
}
trait.proof private @p16 {
  %n = trait.witness @p17 for @P17[i32]
  %d = trait.derive @P16[i32] from @P16_i32 given(%n) : (!trait.claim<@P17[i32] by @p17>)
  trait.return %d : !trait.claim<@P16[i32]>
}
trait.proof private @p15 {
  %n = trait.witness @p16 for @P16[i32]
  %d = trait.derive @P15[i32] from @P15_i32 given(%n) : (!trait.claim<@P16[i32] by @p16>)
  trait.return %d : !trait.claim<@P15[i32]>
}
trait.proof private @p14 {
  %n = trait.witness @p15 for @P15[i32]
  %d = trait.derive @P14[i32] from @P14_i32 given(%n) : (!trait.claim<@P15[i32] by @p15>)
  trait.return %d : !trait.claim<@P14[i32]>
}
trait.proof private @p13 {
  %n = trait.witness @p14 for @P14[i32]
  %d = trait.derive @P13[i32] from @P13_i32 given(%n) : (!trait.claim<@P14[i32] by @p14>)
  trait.return %d : !trait.claim<@P13[i32]>
}
trait.proof private @p12 {
  %n = trait.witness @p13 for @P13[i32]
  %d = trait.derive @P12[i32] from @P12_i32 given(%n) : (!trait.claim<@P13[i32] by @p13>)
  trait.return %d : !trait.claim<@P12[i32]>
}
trait.proof private @p11 {
  %n = trait.witness @p12 for @P12[i32]
  %d = trait.derive @P11[i32] from @P11_i32 given(%n) : (!trait.claim<@P12[i32] by @p12>)
  trait.return %d : !trait.claim<@P11[i32]>
}
trait.proof private @p10 {
  %n = trait.witness @p11 for @P11[i32]
  %d = trait.derive @P10[i32] from @P10_i32 given(%n) : (!trait.claim<@P11[i32] by @p11>)
  trait.return %d : !trait.claim<@P10[i32]>
}
trait.proof private @p9 {
  %n = trait.witness @p10 for @P10[i32]
  %d = trait.derive @P9[i32] from @P9_i32 given(%n) : (!trait.claim<@P10[i32] by @p10>)
  trait.return %d : !trait.claim<@P9[i32]>
}
trait.proof private @p8 {
  %n = trait.witness @p9 for @P9[i32]
  %d = trait.derive @P8[i32] from @P8_i32 given(%n) : (!trait.claim<@P9[i32] by @p9>)
  trait.return %d : !trait.claim<@P8[i32]>
}
trait.proof private @p7 {
  %n = trait.witness @p8 for @P8[i32]
  %d = trait.derive @P7[i32] from @P7_i32 given(%n) : (!trait.claim<@P8[i32] by @p8>)
  trait.return %d : !trait.claim<@P7[i32]>
}
trait.proof private @p6 {
  %n = trait.witness @p7 for @P7[i32]
  %d = trait.derive @P6[i32] from @P6_i32 given(%n) : (!trait.claim<@P7[i32] by @p7>)
  trait.return %d : !trait.claim<@P6[i32]>
}
trait.proof private @p5 {
  %n = trait.witness @p6 for @P6[i32]
  %d = trait.derive @P5[i32] from @P5_i32 given(%n) : (!trait.claim<@P6[i32] by @p6>)
  trait.return %d : !trait.claim<@P5[i32]>
}
trait.proof private @p4 {
  %n = trait.witness @p5 for @P5[i32]
  %d = trait.derive @P4[i32] from @P4_i32 given(%n) : (!trait.claim<@P5[i32] by @p5>)
  trait.return %d : !trait.claim<@P4[i32]>
}
trait.proof private @p3 {
  %n = trait.witness @p4 for @P4[i32]
  %d = trait.derive @P3[i32] from @P3_i32 given(%n) : (!trait.claim<@P4[i32] by @p4>)
  trait.return %d : !trait.claim<@P3[i32]>
}
trait.proof private @p2 {
  %n = trait.witness @p3 for @P3[i32]
  %d = trait.derive @P2[i32] from @P2_i32 given(%n) : (!trait.claim<@P3[i32] by @p3>)
  trait.return %d : !trait.claim<@P2[i32]>
}
trait.proof private @p1 {
  %n = trait.witness @p2 for @P2[i32]
  %d = trait.derive @P1[i32] from @P1_i32 given(%n) : (!trait.claim<@P2[i32] by @p2>)
  trait.return %d : !trait.claim<@P1[i32]>
}
trait.proof private @p0 {
  %n = trait.witness @p1 for @P1[i32]
  %d = trait.derive @P0[i32] from @P0_i32 given(%n) : (!trait.claim<@P1[i32] by @p1>)
  trait.return %d : !trait.claim<@P0[i32]>
}
