// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @r derives @R[i32] over @X[i32], by @x, and @C1[i32], by @c1; @c1 through
// @c126 chain down to @x, and @x cites @y1, which cites @Y2_i32. Through the
// chain the derivation stands 128 obligations deep at @y1, through @x
// directly only three. The height below a pair is kept wherever it is first
// followed and read again where the pair is reached deeper, so the chain is
// refused whichever premise the derive lists first. Here @X comes first.

// CHECK: error: overflow evaluating the requirement {{.*}}@Y1[i32]{{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @R(%self: !trait.claim<@R[!T]>) {}
trait.trait private @X(%self: !trait.claim<@X[!T]>) {}
trait.trait private @Y1(%self: !trait.claim<@Y1[!T]>) {}
trait.trait private @Y2(%self: !trait.claim<@Y2[!T]>) {}
trait.trait private @C1(%self: !trait.claim<@C1[!T]>) {}
trait.trait private @C2(%self: !trait.claim<@C2[!T]>) {}
trait.trait private @C3(%self: !trait.claim<@C3[!T]>) {}
trait.trait private @C4(%self: !trait.claim<@C4[!T]>) {}
trait.trait private @C5(%self: !trait.claim<@C5[!T]>) {}
trait.trait private @C6(%self: !trait.claim<@C6[!T]>) {}
trait.trait private @C7(%self: !trait.claim<@C7[!T]>) {}
trait.trait private @C8(%self: !trait.claim<@C8[!T]>) {}
trait.trait private @C9(%self: !trait.claim<@C9[!T]>) {}
trait.trait private @C10(%self: !trait.claim<@C10[!T]>) {}
trait.trait private @C11(%self: !trait.claim<@C11[!T]>) {}
trait.trait private @C12(%self: !trait.claim<@C12[!T]>) {}
trait.trait private @C13(%self: !trait.claim<@C13[!T]>) {}
trait.trait private @C14(%self: !trait.claim<@C14[!T]>) {}
trait.trait private @C15(%self: !trait.claim<@C15[!T]>) {}
trait.trait private @C16(%self: !trait.claim<@C16[!T]>) {}
trait.trait private @C17(%self: !trait.claim<@C17[!T]>) {}
trait.trait private @C18(%self: !trait.claim<@C18[!T]>) {}
trait.trait private @C19(%self: !trait.claim<@C19[!T]>) {}
trait.trait private @C20(%self: !trait.claim<@C20[!T]>) {}
trait.trait private @C21(%self: !trait.claim<@C21[!T]>) {}
trait.trait private @C22(%self: !trait.claim<@C22[!T]>) {}
trait.trait private @C23(%self: !trait.claim<@C23[!T]>) {}
trait.trait private @C24(%self: !trait.claim<@C24[!T]>) {}
trait.trait private @C25(%self: !trait.claim<@C25[!T]>) {}
trait.trait private @C26(%self: !trait.claim<@C26[!T]>) {}
trait.trait private @C27(%self: !trait.claim<@C27[!T]>) {}
trait.trait private @C28(%self: !trait.claim<@C28[!T]>) {}
trait.trait private @C29(%self: !trait.claim<@C29[!T]>) {}
trait.trait private @C30(%self: !trait.claim<@C30[!T]>) {}
trait.trait private @C31(%self: !trait.claim<@C31[!T]>) {}
trait.trait private @C32(%self: !trait.claim<@C32[!T]>) {}
trait.trait private @C33(%self: !trait.claim<@C33[!T]>) {}
trait.trait private @C34(%self: !trait.claim<@C34[!T]>) {}
trait.trait private @C35(%self: !trait.claim<@C35[!T]>) {}
trait.trait private @C36(%self: !trait.claim<@C36[!T]>) {}
trait.trait private @C37(%self: !trait.claim<@C37[!T]>) {}
trait.trait private @C38(%self: !trait.claim<@C38[!T]>) {}
trait.trait private @C39(%self: !trait.claim<@C39[!T]>) {}
trait.trait private @C40(%self: !trait.claim<@C40[!T]>) {}
trait.trait private @C41(%self: !trait.claim<@C41[!T]>) {}
trait.trait private @C42(%self: !trait.claim<@C42[!T]>) {}
trait.trait private @C43(%self: !trait.claim<@C43[!T]>) {}
trait.trait private @C44(%self: !trait.claim<@C44[!T]>) {}
trait.trait private @C45(%self: !trait.claim<@C45[!T]>) {}
trait.trait private @C46(%self: !trait.claim<@C46[!T]>) {}
trait.trait private @C47(%self: !trait.claim<@C47[!T]>) {}
trait.trait private @C48(%self: !trait.claim<@C48[!T]>) {}
trait.trait private @C49(%self: !trait.claim<@C49[!T]>) {}
trait.trait private @C50(%self: !trait.claim<@C50[!T]>) {}
trait.trait private @C51(%self: !trait.claim<@C51[!T]>) {}
trait.trait private @C52(%self: !trait.claim<@C52[!T]>) {}
trait.trait private @C53(%self: !trait.claim<@C53[!T]>) {}
trait.trait private @C54(%self: !trait.claim<@C54[!T]>) {}
trait.trait private @C55(%self: !trait.claim<@C55[!T]>) {}
trait.trait private @C56(%self: !trait.claim<@C56[!T]>) {}
trait.trait private @C57(%self: !trait.claim<@C57[!T]>) {}
trait.trait private @C58(%self: !trait.claim<@C58[!T]>) {}
trait.trait private @C59(%self: !trait.claim<@C59[!T]>) {}
trait.trait private @C60(%self: !trait.claim<@C60[!T]>) {}
trait.trait private @C61(%self: !trait.claim<@C61[!T]>) {}
trait.trait private @C62(%self: !trait.claim<@C62[!T]>) {}
trait.trait private @C63(%self: !trait.claim<@C63[!T]>) {}
trait.trait private @C64(%self: !trait.claim<@C64[!T]>) {}
trait.trait private @C65(%self: !trait.claim<@C65[!T]>) {}
trait.trait private @C66(%self: !trait.claim<@C66[!T]>) {}
trait.trait private @C67(%self: !trait.claim<@C67[!T]>) {}
trait.trait private @C68(%self: !trait.claim<@C68[!T]>) {}
trait.trait private @C69(%self: !trait.claim<@C69[!T]>) {}
trait.trait private @C70(%self: !trait.claim<@C70[!T]>) {}
trait.trait private @C71(%self: !trait.claim<@C71[!T]>) {}
trait.trait private @C72(%self: !trait.claim<@C72[!T]>) {}
trait.trait private @C73(%self: !trait.claim<@C73[!T]>) {}
trait.trait private @C74(%self: !trait.claim<@C74[!T]>) {}
trait.trait private @C75(%self: !trait.claim<@C75[!T]>) {}
trait.trait private @C76(%self: !trait.claim<@C76[!T]>) {}
trait.trait private @C77(%self: !trait.claim<@C77[!T]>) {}
trait.trait private @C78(%self: !trait.claim<@C78[!T]>) {}
trait.trait private @C79(%self: !trait.claim<@C79[!T]>) {}
trait.trait private @C80(%self: !trait.claim<@C80[!T]>) {}
trait.trait private @C81(%self: !trait.claim<@C81[!T]>) {}
trait.trait private @C82(%self: !trait.claim<@C82[!T]>) {}
trait.trait private @C83(%self: !trait.claim<@C83[!T]>) {}
trait.trait private @C84(%self: !trait.claim<@C84[!T]>) {}
trait.trait private @C85(%self: !trait.claim<@C85[!T]>) {}
trait.trait private @C86(%self: !trait.claim<@C86[!T]>) {}
trait.trait private @C87(%self: !trait.claim<@C87[!T]>) {}
trait.trait private @C88(%self: !trait.claim<@C88[!T]>) {}
trait.trait private @C89(%self: !trait.claim<@C89[!T]>) {}
trait.trait private @C90(%self: !trait.claim<@C90[!T]>) {}
trait.trait private @C91(%self: !trait.claim<@C91[!T]>) {}
trait.trait private @C92(%self: !trait.claim<@C92[!T]>) {}
trait.trait private @C93(%self: !trait.claim<@C93[!T]>) {}
trait.trait private @C94(%self: !trait.claim<@C94[!T]>) {}
trait.trait private @C95(%self: !trait.claim<@C95[!T]>) {}
trait.trait private @C96(%self: !trait.claim<@C96[!T]>) {}
trait.trait private @C97(%self: !trait.claim<@C97[!T]>) {}
trait.trait private @C98(%self: !trait.claim<@C98[!T]>) {}
trait.trait private @C99(%self: !trait.claim<@C99[!T]>) {}
trait.trait private @C100(%self: !trait.claim<@C100[!T]>) {}
trait.trait private @C101(%self: !trait.claim<@C101[!T]>) {}
trait.trait private @C102(%self: !trait.claim<@C102[!T]>) {}
trait.trait private @C103(%self: !trait.claim<@C103[!T]>) {}
trait.trait private @C104(%self: !trait.claim<@C104[!T]>) {}
trait.trait private @C105(%self: !trait.claim<@C105[!T]>) {}
trait.trait private @C106(%self: !trait.claim<@C106[!T]>) {}
trait.trait private @C107(%self: !trait.claim<@C107[!T]>) {}
trait.trait private @C108(%self: !trait.claim<@C108[!T]>) {}
trait.trait private @C109(%self: !trait.claim<@C109[!T]>) {}
trait.trait private @C110(%self: !trait.claim<@C110[!T]>) {}
trait.trait private @C111(%self: !trait.claim<@C111[!T]>) {}
trait.trait private @C112(%self: !trait.claim<@C112[!T]>) {}
trait.trait private @C113(%self: !trait.claim<@C113[!T]>) {}
trait.trait private @C114(%self: !trait.claim<@C114[!T]>) {}
trait.trait private @C115(%self: !trait.claim<@C115[!T]>) {}
trait.trait private @C116(%self: !trait.claim<@C116[!T]>) {}
trait.trait private @C117(%self: !trait.claim<@C117[!T]>) {}
trait.trait private @C118(%self: !trait.claim<@C118[!T]>) {}
trait.trait private @C119(%self: !trait.claim<@C119[!T]>) {}
trait.trait private @C120(%self: !trait.claim<@C120[!T]>) {}
trait.trait private @C121(%self: !trait.claim<@C121[!T]>) {}
trait.trait private @C122(%self: !trait.claim<@C122[!T]>) {}
trait.trait private @C123(%self: !trait.claim<@C123[!T]>) {}
trait.trait private @C124(%self: !trait.claim<@C124[!T]>) {}
trait.trait private @C125(%self: !trait.claim<@C125[!T]>) {}
trait.trait private @C126(%self: !trait.claim<@C126[!T]>) {}
trait.impl private @R_i32(%self: !trait.claim<@R[i32]>, %x: !trait.claim<@X[i32]>, %c: !trait.claim<@C1[i32]>) {}
trait.impl private @X_i32(%self: !trait.claim<@X[i32]>, %y: !trait.claim<@Y1[i32]>) {}
trait.impl private @Y1_i32(%self: !trait.claim<@Y1[i32]>, %y: !trait.claim<@Y2[i32]>) {}
trait.impl private @Y2_i32(%self: !trait.claim<@Y2[i32]>) {}
trait.impl private @C1_i32(%self: !trait.claim<@C1[i32]>, %n: !trait.claim<@C2[i32]>) {}
trait.impl private @C2_i32(%self: !trait.claim<@C2[i32]>, %n: !trait.claim<@C3[i32]>) {}
trait.impl private @C3_i32(%self: !trait.claim<@C3[i32]>, %n: !trait.claim<@C4[i32]>) {}
trait.impl private @C4_i32(%self: !trait.claim<@C4[i32]>, %n: !trait.claim<@C5[i32]>) {}
trait.impl private @C5_i32(%self: !trait.claim<@C5[i32]>, %n: !trait.claim<@C6[i32]>) {}
trait.impl private @C6_i32(%self: !trait.claim<@C6[i32]>, %n: !trait.claim<@C7[i32]>) {}
trait.impl private @C7_i32(%self: !trait.claim<@C7[i32]>, %n: !trait.claim<@C8[i32]>) {}
trait.impl private @C8_i32(%self: !trait.claim<@C8[i32]>, %n: !trait.claim<@C9[i32]>) {}
trait.impl private @C9_i32(%self: !trait.claim<@C9[i32]>, %n: !trait.claim<@C10[i32]>) {}
trait.impl private @C10_i32(%self: !trait.claim<@C10[i32]>, %n: !trait.claim<@C11[i32]>) {}
trait.impl private @C11_i32(%self: !trait.claim<@C11[i32]>, %n: !trait.claim<@C12[i32]>) {}
trait.impl private @C12_i32(%self: !trait.claim<@C12[i32]>, %n: !trait.claim<@C13[i32]>) {}
trait.impl private @C13_i32(%self: !trait.claim<@C13[i32]>, %n: !trait.claim<@C14[i32]>) {}
trait.impl private @C14_i32(%self: !trait.claim<@C14[i32]>, %n: !trait.claim<@C15[i32]>) {}
trait.impl private @C15_i32(%self: !trait.claim<@C15[i32]>, %n: !trait.claim<@C16[i32]>) {}
trait.impl private @C16_i32(%self: !trait.claim<@C16[i32]>, %n: !trait.claim<@C17[i32]>) {}
trait.impl private @C17_i32(%self: !trait.claim<@C17[i32]>, %n: !trait.claim<@C18[i32]>) {}
trait.impl private @C18_i32(%self: !trait.claim<@C18[i32]>, %n: !trait.claim<@C19[i32]>) {}
trait.impl private @C19_i32(%self: !trait.claim<@C19[i32]>, %n: !trait.claim<@C20[i32]>) {}
trait.impl private @C20_i32(%self: !trait.claim<@C20[i32]>, %n: !trait.claim<@C21[i32]>) {}
trait.impl private @C21_i32(%self: !trait.claim<@C21[i32]>, %n: !trait.claim<@C22[i32]>) {}
trait.impl private @C22_i32(%self: !trait.claim<@C22[i32]>, %n: !trait.claim<@C23[i32]>) {}
trait.impl private @C23_i32(%self: !trait.claim<@C23[i32]>, %n: !trait.claim<@C24[i32]>) {}
trait.impl private @C24_i32(%self: !trait.claim<@C24[i32]>, %n: !trait.claim<@C25[i32]>) {}
trait.impl private @C25_i32(%self: !trait.claim<@C25[i32]>, %n: !trait.claim<@C26[i32]>) {}
trait.impl private @C26_i32(%self: !trait.claim<@C26[i32]>, %n: !trait.claim<@C27[i32]>) {}
trait.impl private @C27_i32(%self: !trait.claim<@C27[i32]>, %n: !trait.claim<@C28[i32]>) {}
trait.impl private @C28_i32(%self: !trait.claim<@C28[i32]>, %n: !trait.claim<@C29[i32]>) {}
trait.impl private @C29_i32(%self: !trait.claim<@C29[i32]>, %n: !trait.claim<@C30[i32]>) {}
trait.impl private @C30_i32(%self: !trait.claim<@C30[i32]>, %n: !trait.claim<@C31[i32]>) {}
trait.impl private @C31_i32(%self: !trait.claim<@C31[i32]>, %n: !trait.claim<@C32[i32]>) {}
trait.impl private @C32_i32(%self: !trait.claim<@C32[i32]>, %n: !trait.claim<@C33[i32]>) {}
trait.impl private @C33_i32(%self: !trait.claim<@C33[i32]>, %n: !trait.claim<@C34[i32]>) {}
trait.impl private @C34_i32(%self: !trait.claim<@C34[i32]>, %n: !trait.claim<@C35[i32]>) {}
trait.impl private @C35_i32(%self: !trait.claim<@C35[i32]>, %n: !trait.claim<@C36[i32]>) {}
trait.impl private @C36_i32(%self: !trait.claim<@C36[i32]>, %n: !trait.claim<@C37[i32]>) {}
trait.impl private @C37_i32(%self: !trait.claim<@C37[i32]>, %n: !trait.claim<@C38[i32]>) {}
trait.impl private @C38_i32(%self: !trait.claim<@C38[i32]>, %n: !trait.claim<@C39[i32]>) {}
trait.impl private @C39_i32(%self: !trait.claim<@C39[i32]>, %n: !trait.claim<@C40[i32]>) {}
trait.impl private @C40_i32(%self: !trait.claim<@C40[i32]>, %n: !trait.claim<@C41[i32]>) {}
trait.impl private @C41_i32(%self: !trait.claim<@C41[i32]>, %n: !trait.claim<@C42[i32]>) {}
trait.impl private @C42_i32(%self: !trait.claim<@C42[i32]>, %n: !trait.claim<@C43[i32]>) {}
trait.impl private @C43_i32(%self: !trait.claim<@C43[i32]>, %n: !trait.claim<@C44[i32]>) {}
trait.impl private @C44_i32(%self: !trait.claim<@C44[i32]>, %n: !trait.claim<@C45[i32]>) {}
trait.impl private @C45_i32(%self: !trait.claim<@C45[i32]>, %n: !trait.claim<@C46[i32]>) {}
trait.impl private @C46_i32(%self: !trait.claim<@C46[i32]>, %n: !trait.claim<@C47[i32]>) {}
trait.impl private @C47_i32(%self: !trait.claim<@C47[i32]>, %n: !trait.claim<@C48[i32]>) {}
trait.impl private @C48_i32(%self: !trait.claim<@C48[i32]>, %n: !trait.claim<@C49[i32]>) {}
trait.impl private @C49_i32(%self: !trait.claim<@C49[i32]>, %n: !trait.claim<@C50[i32]>) {}
trait.impl private @C50_i32(%self: !trait.claim<@C50[i32]>, %n: !trait.claim<@C51[i32]>) {}
trait.impl private @C51_i32(%self: !trait.claim<@C51[i32]>, %n: !trait.claim<@C52[i32]>) {}
trait.impl private @C52_i32(%self: !trait.claim<@C52[i32]>, %n: !trait.claim<@C53[i32]>) {}
trait.impl private @C53_i32(%self: !trait.claim<@C53[i32]>, %n: !trait.claim<@C54[i32]>) {}
trait.impl private @C54_i32(%self: !trait.claim<@C54[i32]>, %n: !trait.claim<@C55[i32]>) {}
trait.impl private @C55_i32(%self: !trait.claim<@C55[i32]>, %n: !trait.claim<@C56[i32]>) {}
trait.impl private @C56_i32(%self: !trait.claim<@C56[i32]>, %n: !trait.claim<@C57[i32]>) {}
trait.impl private @C57_i32(%self: !trait.claim<@C57[i32]>, %n: !trait.claim<@C58[i32]>) {}
trait.impl private @C58_i32(%self: !trait.claim<@C58[i32]>, %n: !trait.claim<@C59[i32]>) {}
trait.impl private @C59_i32(%self: !trait.claim<@C59[i32]>, %n: !trait.claim<@C60[i32]>) {}
trait.impl private @C60_i32(%self: !trait.claim<@C60[i32]>, %n: !trait.claim<@C61[i32]>) {}
trait.impl private @C61_i32(%self: !trait.claim<@C61[i32]>, %n: !trait.claim<@C62[i32]>) {}
trait.impl private @C62_i32(%self: !trait.claim<@C62[i32]>, %n: !trait.claim<@C63[i32]>) {}
trait.impl private @C63_i32(%self: !trait.claim<@C63[i32]>, %n: !trait.claim<@C64[i32]>) {}
trait.impl private @C64_i32(%self: !trait.claim<@C64[i32]>, %n: !trait.claim<@C65[i32]>) {}
trait.impl private @C65_i32(%self: !trait.claim<@C65[i32]>, %n: !trait.claim<@C66[i32]>) {}
trait.impl private @C66_i32(%self: !trait.claim<@C66[i32]>, %n: !trait.claim<@C67[i32]>) {}
trait.impl private @C67_i32(%self: !trait.claim<@C67[i32]>, %n: !trait.claim<@C68[i32]>) {}
trait.impl private @C68_i32(%self: !trait.claim<@C68[i32]>, %n: !trait.claim<@C69[i32]>) {}
trait.impl private @C69_i32(%self: !trait.claim<@C69[i32]>, %n: !trait.claim<@C70[i32]>) {}
trait.impl private @C70_i32(%self: !trait.claim<@C70[i32]>, %n: !trait.claim<@C71[i32]>) {}
trait.impl private @C71_i32(%self: !trait.claim<@C71[i32]>, %n: !trait.claim<@C72[i32]>) {}
trait.impl private @C72_i32(%self: !trait.claim<@C72[i32]>, %n: !trait.claim<@C73[i32]>) {}
trait.impl private @C73_i32(%self: !trait.claim<@C73[i32]>, %n: !trait.claim<@C74[i32]>) {}
trait.impl private @C74_i32(%self: !trait.claim<@C74[i32]>, %n: !trait.claim<@C75[i32]>) {}
trait.impl private @C75_i32(%self: !trait.claim<@C75[i32]>, %n: !trait.claim<@C76[i32]>) {}
trait.impl private @C76_i32(%self: !trait.claim<@C76[i32]>, %n: !trait.claim<@C77[i32]>) {}
trait.impl private @C77_i32(%self: !trait.claim<@C77[i32]>, %n: !trait.claim<@C78[i32]>) {}
trait.impl private @C78_i32(%self: !trait.claim<@C78[i32]>, %n: !trait.claim<@C79[i32]>) {}
trait.impl private @C79_i32(%self: !trait.claim<@C79[i32]>, %n: !trait.claim<@C80[i32]>) {}
trait.impl private @C80_i32(%self: !trait.claim<@C80[i32]>, %n: !trait.claim<@C81[i32]>) {}
trait.impl private @C81_i32(%self: !trait.claim<@C81[i32]>, %n: !trait.claim<@C82[i32]>) {}
trait.impl private @C82_i32(%self: !trait.claim<@C82[i32]>, %n: !trait.claim<@C83[i32]>) {}
trait.impl private @C83_i32(%self: !trait.claim<@C83[i32]>, %n: !trait.claim<@C84[i32]>) {}
trait.impl private @C84_i32(%self: !trait.claim<@C84[i32]>, %n: !trait.claim<@C85[i32]>) {}
trait.impl private @C85_i32(%self: !trait.claim<@C85[i32]>, %n: !trait.claim<@C86[i32]>) {}
trait.impl private @C86_i32(%self: !trait.claim<@C86[i32]>, %n: !trait.claim<@C87[i32]>) {}
trait.impl private @C87_i32(%self: !trait.claim<@C87[i32]>, %n: !trait.claim<@C88[i32]>) {}
trait.impl private @C88_i32(%self: !trait.claim<@C88[i32]>, %n: !trait.claim<@C89[i32]>) {}
trait.impl private @C89_i32(%self: !trait.claim<@C89[i32]>, %n: !trait.claim<@C90[i32]>) {}
trait.impl private @C90_i32(%self: !trait.claim<@C90[i32]>, %n: !trait.claim<@C91[i32]>) {}
trait.impl private @C91_i32(%self: !trait.claim<@C91[i32]>, %n: !trait.claim<@C92[i32]>) {}
trait.impl private @C92_i32(%self: !trait.claim<@C92[i32]>, %n: !trait.claim<@C93[i32]>) {}
trait.impl private @C93_i32(%self: !trait.claim<@C93[i32]>, %n: !trait.claim<@C94[i32]>) {}
trait.impl private @C94_i32(%self: !trait.claim<@C94[i32]>, %n: !trait.claim<@C95[i32]>) {}
trait.impl private @C95_i32(%self: !trait.claim<@C95[i32]>, %n: !trait.claim<@C96[i32]>) {}
trait.impl private @C96_i32(%self: !trait.claim<@C96[i32]>, %n: !trait.claim<@C97[i32]>) {}
trait.impl private @C97_i32(%self: !trait.claim<@C97[i32]>, %n: !trait.claim<@C98[i32]>) {}
trait.impl private @C98_i32(%self: !trait.claim<@C98[i32]>, %n: !trait.claim<@C99[i32]>) {}
trait.impl private @C99_i32(%self: !trait.claim<@C99[i32]>, %n: !trait.claim<@C100[i32]>) {}
trait.impl private @C100_i32(%self: !trait.claim<@C100[i32]>, %n: !trait.claim<@C101[i32]>) {}
trait.impl private @C101_i32(%self: !trait.claim<@C101[i32]>, %n: !trait.claim<@C102[i32]>) {}
trait.impl private @C102_i32(%self: !trait.claim<@C102[i32]>, %n: !trait.claim<@C103[i32]>) {}
trait.impl private @C103_i32(%self: !trait.claim<@C103[i32]>, %n: !trait.claim<@C104[i32]>) {}
trait.impl private @C104_i32(%self: !trait.claim<@C104[i32]>, %n: !trait.claim<@C105[i32]>) {}
trait.impl private @C105_i32(%self: !trait.claim<@C105[i32]>, %n: !trait.claim<@C106[i32]>) {}
trait.impl private @C106_i32(%self: !trait.claim<@C106[i32]>, %n: !trait.claim<@C107[i32]>) {}
trait.impl private @C107_i32(%self: !trait.claim<@C107[i32]>, %n: !trait.claim<@C108[i32]>) {}
trait.impl private @C108_i32(%self: !trait.claim<@C108[i32]>, %n: !trait.claim<@C109[i32]>) {}
trait.impl private @C109_i32(%self: !trait.claim<@C109[i32]>, %n: !trait.claim<@C110[i32]>) {}
trait.impl private @C110_i32(%self: !trait.claim<@C110[i32]>, %n: !trait.claim<@C111[i32]>) {}
trait.impl private @C111_i32(%self: !trait.claim<@C111[i32]>, %n: !trait.claim<@C112[i32]>) {}
trait.impl private @C112_i32(%self: !trait.claim<@C112[i32]>, %n: !trait.claim<@C113[i32]>) {}
trait.impl private @C113_i32(%self: !trait.claim<@C113[i32]>, %n: !trait.claim<@C114[i32]>) {}
trait.impl private @C114_i32(%self: !trait.claim<@C114[i32]>, %n: !trait.claim<@C115[i32]>) {}
trait.impl private @C115_i32(%self: !trait.claim<@C115[i32]>, %n: !trait.claim<@C116[i32]>) {}
trait.impl private @C116_i32(%self: !trait.claim<@C116[i32]>, %n: !trait.claim<@C117[i32]>) {}
trait.impl private @C117_i32(%self: !trait.claim<@C117[i32]>, %n: !trait.claim<@C118[i32]>) {}
trait.impl private @C118_i32(%self: !trait.claim<@C118[i32]>, %n: !trait.claim<@C119[i32]>) {}
trait.impl private @C119_i32(%self: !trait.claim<@C119[i32]>, %n: !trait.claim<@C120[i32]>) {}
trait.impl private @C120_i32(%self: !trait.claim<@C120[i32]>, %n: !trait.claim<@C121[i32]>) {}
trait.impl private @C121_i32(%self: !trait.claim<@C121[i32]>, %n: !trait.claim<@C122[i32]>) {}
trait.impl private @C122_i32(%self: !trait.claim<@C122[i32]>, %n: !trait.claim<@C123[i32]>) {}
trait.impl private @C123_i32(%self: !trait.claim<@C123[i32]>, %n: !trait.claim<@C124[i32]>) {}
trait.impl private @C124_i32(%self: !trait.claim<@C124[i32]>, %n: !trait.claim<@C125[i32]>) {}
trait.impl private @C125_i32(%self: !trait.claim<@C125[i32]>, %n: !trait.claim<@C126[i32]>) {}
trait.impl private @C126_i32(%self: !trait.claim<@C126[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.proof private @r {
  %x = trait.witness @x for @X[i32]
  %c = trait.witness @c1 for @C1[i32]
  %d = trait.derive @R[i32] from @R_i32 given(%x, %c) : (!trait.claim<@X[i32] by @x>, !trait.claim<@C1[i32] by @c1>)
  trait.return %d : !trait.claim<@R[i32]>
}
trait.proof private @x {
  %y = trait.witness @y1 for @Y1[i32]
  %d = trait.derive @X[i32] from @X_i32 given(%y) : (!trait.claim<@Y1[i32] by @y1>)
  trait.return %d : !trait.claim<@X[i32]>
}
trait.proof private @y1 {
  %y = trait.witness @Y2_i32 for @Y2[i32]
  %d = trait.derive @Y1[i32] from @Y1_i32 given(%y) : (!trait.claim<@Y2[i32] by @Y2_i32>)
  trait.return %d : !trait.claim<@Y1[i32]>
}
trait.proof private @c1 {
  %n = trait.witness @c2 for @C2[i32]
  %d = trait.derive @C1[i32] from @C1_i32 given(%n) : (!trait.claim<@C2[i32] by @c2>)
  trait.return %d : !trait.claim<@C1[i32]>
}
trait.proof private @c2 {
  %n = trait.witness @c3 for @C3[i32]
  %d = trait.derive @C2[i32] from @C2_i32 given(%n) : (!trait.claim<@C3[i32] by @c3>)
  trait.return %d : !trait.claim<@C2[i32]>
}
trait.proof private @c3 {
  %n = trait.witness @c4 for @C4[i32]
  %d = trait.derive @C3[i32] from @C3_i32 given(%n) : (!trait.claim<@C4[i32] by @c4>)
  trait.return %d : !trait.claim<@C3[i32]>
}
trait.proof private @c4 {
  %n = trait.witness @c5 for @C5[i32]
  %d = trait.derive @C4[i32] from @C4_i32 given(%n) : (!trait.claim<@C5[i32] by @c5>)
  trait.return %d : !trait.claim<@C4[i32]>
}
trait.proof private @c5 {
  %n = trait.witness @c6 for @C6[i32]
  %d = trait.derive @C5[i32] from @C5_i32 given(%n) : (!trait.claim<@C6[i32] by @c6>)
  trait.return %d : !trait.claim<@C5[i32]>
}
trait.proof private @c6 {
  %n = trait.witness @c7 for @C7[i32]
  %d = trait.derive @C6[i32] from @C6_i32 given(%n) : (!trait.claim<@C7[i32] by @c7>)
  trait.return %d : !trait.claim<@C6[i32]>
}
trait.proof private @c7 {
  %n = trait.witness @c8 for @C8[i32]
  %d = trait.derive @C7[i32] from @C7_i32 given(%n) : (!trait.claim<@C8[i32] by @c8>)
  trait.return %d : !trait.claim<@C7[i32]>
}
trait.proof private @c8 {
  %n = trait.witness @c9 for @C9[i32]
  %d = trait.derive @C8[i32] from @C8_i32 given(%n) : (!trait.claim<@C9[i32] by @c9>)
  trait.return %d : !trait.claim<@C8[i32]>
}
trait.proof private @c9 {
  %n = trait.witness @c10 for @C10[i32]
  %d = trait.derive @C9[i32] from @C9_i32 given(%n) : (!trait.claim<@C10[i32] by @c10>)
  trait.return %d : !trait.claim<@C9[i32]>
}
trait.proof private @c10 {
  %n = trait.witness @c11 for @C11[i32]
  %d = trait.derive @C10[i32] from @C10_i32 given(%n) : (!trait.claim<@C11[i32] by @c11>)
  trait.return %d : !trait.claim<@C10[i32]>
}
trait.proof private @c11 {
  %n = trait.witness @c12 for @C12[i32]
  %d = trait.derive @C11[i32] from @C11_i32 given(%n) : (!trait.claim<@C12[i32] by @c12>)
  trait.return %d : !trait.claim<@C11[i32]>
}
trait.proof private @c12 {
  %n = trait.witness @c13 for @C13[i32]
  %d = trait.derive @C12[i32] from @C12_i32 given(%n) : (!trait.claim<@C13[i32] by @c13>)
  trait.return %d : !trait.claim<@C12[i32]>
}
trait.proof private @c13 {
  %n = trait.witness @c14 for @C14[i32]
  %d = trait.derive @C13[i32] from @C13_i32 given(%n) : (!trait.claim<@C14[i32] by @c14>)
  trait.return %d : !trait.claim<@C13[i32]>
}
trait.proof private @c14 {
  %n = trait.witness @c15 for @C15[i32]
  %d = trait.derive @C14[i32] from @C14_i32 given(%n) : (!trait.claim<@C15[i32] by @c15>)
  trait.return %d : !trait.claim<@C14[i32]>
}
trait.proof private @c15 {
  %n = trait.witness @c16 for @C16[i32]
  %d = trait.derive @C15[i32] from @C15_i32 given(%n) : (!trait.claim<@C16[i32] by @c16>)
  trait.return %d : !trait.claim<@C15[i32]>
}
trait.proof private @c16 {
  %n = trait.witness @c17 for @C17[i32]
  %d = trait.derive @C16[i32] from @C16_i32 given(%n) : (!trait.claim<@C17[i32] by @c17>)
  trait.return %d : !trait.claim<@C16[i32]>
}
trait.proof private @c17 {
  %n = trait.witness @c18 for @C18[i32]
  %d = trait.derive @C17[i32] from @C17_i32 given(%n) : (!trait.claim<@C18[i32] by @c18>)
  trait.return %d : !trait.claim<@C17[i32]>
}
trait.proof private @c18 {
  %n = trait.witness @c19 for @C19[i32]
  %d = trait.derive @C18[i32] from @C18_i32 given(%n) : (!trait.claim<@C19[i32] by @c19>)
  trait.return %d : !trait.claim<@C18[i32]>
}
trait.proof private @c19 {
  %n = trait.witness @c20 for @C20[i32]
  %d = trait.derive @C19[i32] from @C19_i32 given(%n) : (!trait.claim<@C20[i32] by @c20>)
  trait.return %d : !trait.claim<@C19[i32]>
}
trait.proof private @c20 {
  %n = trait.witness @c21 for @C21[i32]
  %d = trait.derive @C20[i32] from @C20_i32 given(%n) : (!trait.claim<@C21[i32] by @c21>)
  trait.return %d : !trait.claim<@C20[i32]>
}
trait.proof private @c21 {
  %n = trait.witness @c22 for @C22[i32]
  %d = trait.derive @C21[i32] from @C21_i32 given(%n) : (!trait.claim<@C22[i32] by @c22>)
  trait.return %d : !trait.claim<@C21[i32]>
}
trait.proof private @c22 {
  %n = trait.witness @c23 for @C23[i32]
  %d = trait.derive @C22[i32] from @C22_i32 given(%n) : (!trait.claim<@C23[i32] by @c23>)
  trait.return %d : !trait.claim<@C22[i32]>
}
trait.proof private @c23 {
  %n = trait.witness @c24 for @C24[i32]
  %d = trait.derive @C23[i32] from @C23_i32 given(%n) : (!trait.claim<@C24[i32] by @c24>)
  trait.return %d : !trait.claim<@C23[i32]>
}
trait.proof private @c24 {
  %n = trait.witness @c25 for @C25[i32]
  %d = trait.derive @C24[i32] from @C24_i32 given(%n) : (!trait.claim<@C25[i32] by @c25>)
  trait.return %d : !trait.claim<@C24[i32]>
}
trait.proof private @c25 {
  %n = trait.witness @c26 for @C26[i32]
  %d = trait.derive @C25[i32] from @C25_i32 given(%n) : (!trait.claim<@C26[i32] by @c26>)
  trait.return %d : !trait.claim<@C25[i32]>
}
trait.proof private @c26 {
  %n = trait.witness @c27 for @C27[i32]
  %d = trait.derive @C26[i32] from @C26_i32 given(%n) : (!trait.claim<@C27[i32] by @c27>)
  trait.return %d : !trait.claim<@C26[i32]>
}
trait.proof private @c27 {
  %n = trait.witness @c28 for @C28[i32]
  %d = trait.derive @C27[i32] from @C27_i32 given(%n) : (!trait.claim<@C28[i32] by @c28>)
  trait.return %d : !trait.claim<@C27[i32]>
}
trait.proof private @c28 {
  %n = trait.witness @c29 for @C29[i32]
  %d = trait.derive @C28[i32] from @C28_i32 given(%n) : (!trait.claim<@C29[i32] by @c29>)
  trait.return %d : !trait.claim<@C28[i32]>
}
trait.proof private @c29 {
  %n = trait.witness @c30 for @C30[i32]
  %d = trait.derive @C29[i32] from @C29_i32 given(%n) : (!trait.claim<@C30[i32] by @c30>)
  trait.return %d : !trait.claim<@C29[i32]>
}
trait.proof private @c30 {
  %n = trait.witness @c31 for @C31[i32]
  %d = trait.derive @C30[i32] from @C30_i32 given(%n) : (!trait.claim<@C31[i32] by @c31>)
  trait.return %d : !trait.claim<@C30[i32]>
}
trait.proof private @c31 {
  %n = trait.witness @c32 for @C32[i32]
  %d = trait.derive @C31[i32] from @C31_i32 given(%n) : (!trait.claim<@C32[i32] by @c32>)
  trait.return %d : !trait.claim<@C31[i32]>
}
trait.proof private @c32 {
  %n = trait.witness @c33 for @C33[i32]
  %d = trait.derive @C32[i32] from @C32_i32 given(%n) : (!trait.claim<@C33[i32] by @c33>)
  trait.return %d : !trait.claim<@C32[i32]>
}
trait.proof private @c33 {
  %n = trait.witness @c34 for @C34[i32]
  %d = trait.derive @C33[i32] from @C33_i32 given(%n) : (!trait.claim<@C34[i32] by @c34>)
  trait.return %d : !trait.claim<@C33[i32]>
}
trait.proof private @c34 {
  %n = trait.witness @c35 for @C35[i32]
  %d = trait.derive @C34[i32] from @C34_i32 given(%n) : (!trait.claim<@C35[i32] by @c35>)
  trait.return %d : !trait.claim<@C34[i32]>
}
trait.proof private @c35 {
  %n = trait.witness @c36 for @C36[i32]
  %d = trait.derive @C35[i32] from @C35_i32 given(%n) : (!trait.claim<@C36[i32] by @c36>)
  trait.return %d : !trait.claim<@C35[i32]>
}
trait.proof private @c36 {
  %n = trait.witness @c37 for @C37[i32]
  %d = trait.derive @C36[i32] from @C36_i32 given(%n) : (!trait.claim<@C37[i32] by @c37>)
  trait.return %d : !trait.claim<@C36[i32]>
}
trait.proof private @c37 {
  %n = trait.witness @c38 for @C38[i32]
  %d = trait.derive @C37[i32] from @C37_i32 given(%n) : (!trait.claim<@C38[i32] by @c38>)
  trait.return %d : !trait.claim<@C37[i32]>
}
trait.proof private @c38 {
  %n = trait.witness @c39 for @C39[i32]
  %d = trait.derive @C38[i32] from @C38_i32 given(%n) : (!trait.claim<@C39[i32] by @c39>)
  trait.return %d : !trait.claim<@C38[i32]>
}
trait.proof private @c39 {
  %n = trait.witness @c40 for @C40[i32]
  %d = trait.derive @C39[i32] from @C39_i32 given(%n) : (!trait.claim<@C40[i32] by @c40>)
  trait.return %d : !trait.claim<@C39[i32]>
}
trait.proof private @c40 {
  %n = trait.witness @c41 for @C41[i32]
  %d = trait.derive @C40[i32] from @C40_i32 given(%n) : (!trait.claim<@C41[i32] by @c41>)
  trait.return %d : !trait.claim<@C40[i32]>
}
trait.proof private @c41 {
  %n = trait.witness @c42 for @C42[i32]
  %d = trait.derive @C41[i32] from @C41_i32 given(%n) : (!trait.claim<@C42[i32] by @c42>)
  trait.return %d : !trait.claim<@C41[i32]>
}
trait.proof private @c42 {
  %n = trait.witness @c43 for @C43[i32]
  %d = trait.derive @C42[i32] from @C42_i32 given(%n) : (!trait.claim<@C43[i32] by @c43>)
  trait.return %d : !trait.claim<@C42[i32]>
}
trait.proof private @c43 {
  %n = trait.witness @c44 for @C44[i32]
  %d = trait.derive @C43[i32] from @C43_i32 given(%n) : (!trait.claim<@C44[i32] by @c44>)
  trait.return %d : !trait.claim<@C43[i32]>
}
trait.proof private @c44 {
  %n = trait.witness @c45 for @C45[i32]
  %d = trait.derive @C44[i32] from @C44_i32 given(%n) : (!trait.claim<@C45[i32] by @c45>)
  trait.return %d : !trait.claim<@C44[i32]>
}
trait.proof private @c45 {
  %n = trait.witness @c46 for @C46[i32]
  %d = trait.derive @C45[i32] from @C45_i32 given(%n) : (!trait.claim<@C46[i32] by @c46>)
  trait.return %d : !trait.claim<@C45[i32]>
}
trait.proof private @c46 {
  %n = trait.witness @c47 for @C47[i32]
  %d = trait.derive @C46[i32] from @C46_i32 given(%n) : (!trait.claim<@C47[i32] by @c47>)
  trait.return %d : !trait.claim<@C46[i32]>
}
trait.proof private @c47 {
  %n = trait.witness @c48 for @C48[i32]
  %d = trait.derive @C47[i32] from @C47_i32 given(%n) : (!trait.claim<@C48[i32] by @c48>)
  trait.return %d : !trait.claim<@C47[i32]>
}
trait.proof private @c48 {
  %n = trait.witness @c49 for @C49[i32]
  %d = trait.derive @C48[i32] from @C48_i32 given(%n) : (!trait.claim<@C49[i32] by @c49>)
  trait.return %d : !trait.claim<@C48[i32]>
}
trait.proof private @c49 {
  %n = trait.witness @c50 for @C50[i32]
  %d = trait.derive @C49[i32] from @C49_i32 given(%n) : (!trait.claim<@C50[i32] by @c50>)
  trait.return %d : !trait.claim<@C49[i32]>
}
trait.proof private @c50 {
  %n = trait.witness @c51 for @C51[i32]
  %d = trait.derive @C50[i32] from @C50_i32 given(%n) : (!trait.claim<@C51[i32] by @c51>)
  trait.return %d : !trait.claim<@C50[i32]>
}
trait.proof private @c51 {
  %n = trait.witness @c52 for @C52[i32]
  %d = trait.derive @C51[i32] from @C51_i32 given(%n) : (!trait.claim<@C52[i32] by @c52>)
  trait.return %d : !trait.claim<@C51[i32]>
}
trait.proof private @c52 {
  %n = trait.witness @c53 for @C53[i32]
  %d = trait.derive @C52[i32] from @C52_i32 given(%n) : (!trait.claim<@C53[i32] by @c53>)
  trait.return %d : !trait.claim<@C52[i32]>
}
trait.proof private @c53 {
  %n = trait.witness @c54 for @C54[i32]
  %d = trait.derive @C53[i32] from @C53_i32 given(%n) : (!trait.claim<@C54[i32] by @c54>)
  trait.return %d : !trait.claim<@C53[i32]>
}
trait.proof private @c54 {
  %n = trait.witness @c55 for @C55[i32]
  %d = trait.derive @C54[i32] from @C54_i32 given(%n) : (!trait.claim<@C55[i32] by @c55>)
  trait.return %d : !trait.claim<@C54[i32]>
}
trait.proof private @c55 {
  %n = trait.witness @c56 for @C56[i32]
  %d = trait.derive @C55[i32] from @C55_i32 given(%n) : (!trait.claim<@C56[i32] by @c56>)
  trait.return %d : !trait.claim<@C55[i32]>
}
trait.proof private @c56 {
  %n = trait.witness @c57 for @C57[i32]
  %d = trait.derive @C56[i32] from @C56_i32 given(%n) : (!trait.claim<@C57[i32] by @c57>)
  trait.return %d : !trait.claim<@C56[i32]>
}
trait.proof private @c57 {
  %n = trait.witness @c58 for @C58[i32]
  %d = trait.derive @C57[i32] from @C57_i32 given(%n) : (!trait.claim<@C58[i32] by @c58>)
  trait.return %d : !trait.claim<@C57[i32]>
}
trait.proof private @c58 {
  %n = trait.witness @c59 for @C59[i32]
  %d = trait.derive @C58[i32] from @C58_i32 given(%n) : (!trait.claim<@C59[i32] by @c59>)
  trait.return %d : !trait.claim<@C58[i32]>
}
trait.proof private @c59 {
  %n = trait.witness @c60 for @C60[i32]
  %d = trait.derive @C59[i32] from @C59_i32 given(%n) : (!trait.claim<@C60[i32] by @c60>)
  trait.return %d : !trait.claim<@C59[i32]>
}
trait.proof private @c60 {
  %n = trait.witness @c61 for @C61[i32]
  %d = trait.derive @C60[i32] from @C60_i32 given(%n) : (!trait.claim<@C61[i32] by @c61>)
  trait.return %d : !trait.claim<@C60[i32]>
}
trait.proof private @c61 {
  %n = trait.witness @c62 for @C62[i32]
  %d = trait.derive @C61[i32] from @C61_i32 given(%n) : (!trait.claim<@C62[i32] by @c62>)
  trait.return %d : !trait.claim<@C61[i32]>
}
trait.proof private @c62 {
  %n = trait.witness @c63 for @C63[i32]
  %d = trait.derive @C62[i32] from @C62_i32 given(%n) : (!trait.claim<@C63[i32] by @c63>)
  trait.return %d : !trait.claim<@C62[i32]>
}
trait.proof private @c63 {
  %n = trait.witness @c64 for @C64[i32]
  %d = trait.derive @C63[i32] from @C63_i32 given(%n) : (!trait.claim<@C64[i32] by @c64>)
  trait.return %d : !trait.claim<@C63[i32]>
}
trait.proof private @c64 {
  %n = trait.witness @c65 for @C65[i32]
  %d = trait.derive @C64[i32] from @C64_i32 given(%n) : (!trait.claim<@C65[i32] by @c65>)
  trait.return %d : !trait.claim<@C64[i32]>
}
trait.proof private @c65 {
  %n = trait.witness @c66 for @C66[i32]
  %d = trait.derive @C65[i32] from @C65_i32 given(%n) : (!trait.claim<@C66[i32] by @c66>)
  trait.return %d : !trait.claim<@C65[i32]>
}
trait.proof private @c66 {
  %n = trait.witness @c67 for @C67[i32]
  %d = trait.derive @C66[i32] from @C66_i32 given(%n) : (!trait.claim<@C67[i32] by @c67>)
  trait.return %d : !trait.claim<@C66[i32]>
}
trait.proof private @c67 {
  %n = trait.witness @c68 for @C68[i32]
  %d = trait.derive @C67[i32] from @C67_i32 given(%n) : (!trait.claim<@C68[i32] by @c68>)
  trait.return %d : !trait.claim<@C67[i32]>
}
trait.proof private @c68 {
  %n = trait.witness @c69 for @C69[i32]
  %d = trait.derive @C68[i32] from @C68_i32 given(%n) : (!trait.claim<@C69[i32] by @c69>)
  trait.return %d : !trait.claim<@C68[i32]>
}
trait.proof private @c69 {
  %n = trait.witness @c70 for @C70[i32]
  %d = trait.derive @C69[i32] from @C69_i32 given(%n) : (!trait.claim<@C70[i32] by @c70>)
  trait.return %d : !trait.claim<@C69[i32]>
}
trait.proof private @c70 {
  %n = trait.witness @c71 for @C71[i32]
  %d = trait.derive @C70[i32] from @C70_i32 given(%n) : (!trait.claim<@C71[i32] by @c71>)
  trait.return %d : !trait.claim<@C70[i32]>
}
trait.proof private @c71 {
  %n = trait.witness @c72 for @C72[i32]
  %d = trait.derive @C71[i32] from @C71_i32 given(%n) : (!trait.claim<@C72[i32] by @c72>)
  trait.return %d : !trait.claim<@C71[i32]>
}
trait.proof private @c72 {
  %n = trait.witness @c73 for @C73[i32]
  %d = trait.derive @C72[i32] from @C72_i32 given(%n) : (!trait.claim<@C73[i32] by @c73>)
  trait.return %d : !trait.claim<@C72[i32]>
}
trait.proof private @c73 {
  %n = trait.witness @c74 for @C74[i32]
  %d = trait.derive @C73[i32] from @C73_i32 given(%n) : (!trait.claim<@C74[i32] by @c74>)
  trait.return %d : !trait.claim<@C73[i32]>
}
trait.proof private @c74 {
  %n = trait.witness @c75 for @C75[i32]
  %d = trait.derive @C74[i32] from @C74_i32 given(%n) : (!trait.claim<@C75[i32] by @c75>)
  trait.return %d : !trait.claim<@C74[i32]>
}
trait.proof private @c75 {
  %n = trait.witness @c76 for @C76[i32]
  %d = trait.derive @C75[i32] from @C75_i32 given(%n) : (!trait.claim<@C76[i32] by @c76>)
  trait.return %d : !trait.claim<@C75[i32]>
}
trait.proof private @c76 {
  %n = trait.witness @c77 for @C77[i32]
  %d = trait.derive @C76[i32] from @C76_i32 given(%n) : (!trait.claim<@C77[i32] by @c77>)
  trait.return %d : !trait.claim<@C76[i32]>
}
trait.proof private @c77 {
  %n = trait.witness @c78 for @C78[i32]
  %d = trait.derive @C77[i32] from @C77_i32 given(%n) : (!trait.claim<@C78[i32] by @c78>)
  trait.return %d : !trait.claim<@C77[i32]>
}
trait.proof private @c78 {
  %n = trait.witness @c79 for @C79[i32]
  %d = trait.derive @C78[i32] from @C78_i32 given(%n) : (!trait.claim<@C79[i32] by @c79>)
  trait.return %d : !trait.claim<@C78[i32]>
}
trait.proof private @c79 {
  %n = trait.witness @c80 for @C80[i32]
  %d = trait.derive @C79[i32] from @C79_i32 given(%n) : (!trait.claim<@C80[i32] by @c80>)
  trait.return %d : !trait.claim<@C79[i32]>
}
trait.proof private @c80 {
  %n = trait.witness @c81 for @C81[i32]
  %d = trait.derive @C80[i32] from @C80_i32 given(%n) : (!trait.claim<@C81[i32] by @c81>)
  trait.return %d : !trait.claim<@C80[i32]>
}
trait.proof private @c81 {
  %n = trait.witness @c82 for @C82[i32]
  %d = trait.derive @C81[i32] from @C81_i32 given(%n) : (!trait.claim<@C82[i32] by @c82>)
  trait.return %d : !trait.claim<@C81[i32]>
}
trait.proof private @c82 {
  %n = trait.witness @c83 for @C83[i32]
  %d = trait.derive @C82[i32] from @C82_i32 given(%n) : (!trait.claim<@C83[i32] by @c83>)
  trait.return %d : !trait.claim<@C82[i32]>
}
trait.proof private @c83 {
  %n = trait.witness @c84 for @C84[i32]
  %d = trait.derive @C83[i32] from @C83_i32 given(%n) : (!trait.claim<@C84[i32] by @c84>)
  trait.return %d : !trait.claim<@C83[i32]>
}
trait.proof private @c84 {
  %n = trait.witness @c85 for @C85[i32]
  %d = trait.derive @C84[i32] from @C84_i32 given(%n) : (!trait.claim<@C85[i32] by @c85>)
  trait.return %d : !trait.claim<@C84[i32]>
}
trait.proof private @c85 {
  %n = trait.witness @c86 for @C86[i32]
  %d = trait.derive @C85[i32] from @C85_i32 given(%n) : (!trait.claim<@C86[i32] by @c86>)
  trait.return %d : !trait.claim<@C85[i32]>
}
trait.proof private @c86 {
  %n = trait.witness @c87 for @C87[i32]
  %d = trait.derive @C86[i32] from @C86_i32 given(%n) : (!trait.claim<@C87[i32] by @c87>)
  trait.return %d : !trait.claim<@C86[i32]>
}
trait.proof private @c87 {
  %n = trait.witness @c88 for @C88[i32]
  %d = trait.derive @C87[i32] from @C87_i32 given(%n) : (!trait.claim<@C88[i32] by @c88>)
  trait.return %d : !trait.claim<@C87[i32]>
}
trait.proof private @c88 {
  %n = trait.witness @c89 for @C89[i32]
  %d = trait.derive @C88[i32] from @C88_i32 given(%n) : (!trait.claim<@C89[i32] by @c89>)
  trait.return %d : !trait.claim<@C88[i32]>
}
trait.proof private @c89 {
  %n = trait.witness @c90 for @C90[i32]
  %d = trait.derive @C89[i32] from @C89_i32 given(%n) : (!trait.claim<@C90[i32] by @c90>)
  trait.return %d : !trait.claim<@C89[i32]>
}
trait.proof private @c90 {
  %n = trait.witness @c91 for @C91[i32]
  %d = trait.derive @C90[i32] from @C90_i32 given(%n) : (!trait.claim<@C91[i32] by @c91>)
  trait.return %d : !trait.claim<@C90[i32]>
}
trait.proof private @c91 {
  %n = trait.witness @c92 for @C92[i32]
  %d = trait.derive @C91[i32] from @C91_i32 given(%n) : (!trait.claim<@C92[i32] by @c92>)
  trait.return %d : !trait.claim<@C91[i32]>
}
trait.proof private @c92 {
  %n = trait.witness @c93 for @C93[i32]
  %d = trait.derive @C92[i32] from @C92_i32 given(%n) : (!trait.claim<@C93[i32] by @c93>)
  trait.return %d : !trait.claim<@C92[i32]>
}
trait.proof private @c93 {
  %n = trait.witness @c94 for @C94[i32]
  %d = trait.derive @C93[i32] from @C93_i32 given(%n) : (!trait.claim<@C94[i32] by @c94>)
  trait.return %d : !trait.claim<@C93[i32]>
}
trait.proof private @c94 {
  %n = trait.witness @c95 for @C95[i32]
  %d = trait.derive @C94[i32] from @C94_i32 given(%n) : (!trait.claim<@C95[i32] by @c95>)
  trait.return %d : !trait.claim<@C94[i32]>
}
trait.proof private @c95 {
  %n = trait.witness @c96 for @C96[i32]
  %d = trait.derive @C95[i32] from @C95_i32 given(%n) : (!trait.claim<@C96[i32] by @c96>)
  trait.return %d : !trait.claim<@C95[i32]>
}
trait.proof private @c96 {
  %n = trait.witness @c97 for @C97[i32]
  %d = trait.derive @C96[i32] from @C96_i32 given(%n) : (!trait.claim<@C97[i32] by @c97>)
  trait.return %d : !trait.claim<@C96[i32]>
}
trait.proof private @c97 {
  %n = trait.witness @c98 for @C98[i32]
  %d = trait.derive @C97[i32] from @C97_i32 given(%n) : (!trait.claim<@C98[i32] by @c98>)
  trait.return %d : !trait.claim<@C97[i32]>
}
trait.proof private @c98 {
  %n = trait.witness @c99 for @C99[i32]
  %d = trait.derive @C98[i32] from @C98_i32 given(%n) : (!trait.claim<@C99[i32] by @c99>)
  trait.return %d : !trait.claim<@C98[i32]>
}
trait.proof private @c99 {
  %n = trait.witness @c100 for @C100[i32]
  %d = trait.derive @C99[i32] from @C99_i32 given(%n) : (!trait.claim<@C100[i32] by @c100>)
  trait.return %d : !trait.claim<@C99[i32]>
}
trait.proof private @c100 {
  %n = trait.witness @c101 for @C101[i32]
  %d = trait.derive @C100[i32] from @C100_i32 given(%n) : (!trait.claim<@C101[i32] by @c101>)
  trait.return %d : !trait.claim<@C100[i32]>
}
trait.proof private @c101 {
  %n = trait.witness @c102 for @C102[i32]
  %d = trait.derive @C101[i32] from @C101_i32 given(%n) : (!trait.claim<@C102[i32] by @c102>)
  trait.return %d : !trait.claim<@C101[i32]>
}
trait.proof private @c102 {
  %n = trait.witness @c103 for @C103[i32]
  %d = trait.derive @C102[i32] from @C102_i32 given(%n) : (!trait.claim<@C103[i32] by @c103>)
  trait.return %d : !trait.claim<@C102[i32]>
}
trait.proof private @c103 {
  %n = trait.witness @c104 for @C104[i32]
  %d = trait.derive @C103[i32] from @C103_i32 given(%n) : (!trait.claim<@C104[i32] by @c104>)
  trait.return %d : !trait.claim<@C103[i32]>
}
trait.proof private @c104 {
  %n = trait.witness @c105 for @C105[i32]
  %d = trait.derive @C104[i32] from @C104_i32 given(%n) : (!trait.claim<@C105[i32] by @c105>)
  trait.return %d : !trait.claim<@C104[i32]>
}
trait.proof private @c105 {
  %n = trait.witness @c106 for @C106[i32]
  %d = trait.derive @C105[i32] from @C105_i32 given(%n) : (!trait.claim<@C106[i32] by @c106>)
  trait.return %d : !trait.claim<@C105[i32]>
}
trait.proof private @c106 {
  %n = trait.witness @c107 for @C107[i32]
  %d = trait.derive @C106[i32] from @C106_i32 given(%n) : (!trait.claim<@C107[i32] by @c107>)
  trait.return %d : !trait.claim<@C106[i32]>
}
trait.proof private @c107 {
  %n = trait.witness @c108 for @C108[i32]
  %d = trait.derive @C107[i32] from @C107_i32 given(%n) : (!trait.claim<@C108[i32] by @c108>)
  trait.return %d : !trait.claim<@C107[i32]>
}
trait.proof private @c108 {
  %n = trait.witness @c109 for @C109[i32]
  %d = trait.derive @C108[i32] from @C108_i32 given(%n) : (!trait.claim<@C109[i32] by @c109>)
  trait.return %d : !trait.claim<@C108[i32]>
}
trait.proof private @c109 {
  %n = trait.witness @c110 for @C110[i32]
  %d = trait.derive @C109[i32] from @C109_i32 given(%n) : (!trait.claim<@C110[i32] by @c110>)
  trait.return %d : !trait.claim<@C109[i32]>
}
trait.proof private @c110 {
  %n = trait.witness @c111 for @C111[i32]
  %d = trait.derive @C110[i32] from @C110_i32 given(%n) : (!trait.claim<@C111[i32] by @c111>)
  trait.return %d : !trait.claim<@C110[i32]>
}
trait.proof private @c111 {
  %n = trait.witness @c112 for @C112[i32]
  %d = trait.derive @C111[i32] from @C111_i32 given(%n) : (!trait.claim<@C112[i32] by @c112>)
  trait.return %d : !trait.claim<@C111[i32]>
}
trait.proof private @c112 {
  %n = trait.witness @c113 for @C113[i32]
  %d = trait.derive @C112[i32] from @C112_i32 given(%n) : (!trait.claim<@C113[i32] by @c113>)
  trait.return %d : !trait.claim<@C112[i32]>
}
trait.proof private @c113 {
  %n = trait.witness @c114 for @C114[i32]
  %d = trait.derive @C113[i32] from @C113_i32 given(%n) : (!trait.claim<@C114[i32] by @c114>)
  trait.return %d : !trait.claim<@C113[i32]>
}
trait.proof private @c114 {
  %n = trait.witness @c115 for @C115[i32]
  %d = trait.derive @C114[i32] from @C114_i32 given(%n) : (!trait.claim<@C115[i32] by @c115>)
  trait.return %d : !trait.claim<@C114[i32]>
}
trait.proof private @c115 {
  %n = trait.witness @c116 for @C116[i32]
  %d = trait.derive @C115[i32] from @C115_i32 given(%n) : (!trait.claim<@C116[i32] by @c116>)
  trait.return %d : !trait.claim<@C115[i32]>
}
trait.proof private @c116 {
  %n = trait.witness @c117 for @C117[i32]
  %d = trait.derive @C116[i32] from @C116_i32 given(%n) : (!trait.claim<@C117[i32] by @c117>)
  trait.return %d : !trait.claim<@C116[i32]>
}
trait.proof private @c117 {
  %n = trait.witness @c118 for @C118[i32]
  %d = trait.derive @C117[i32] from @C117_i32 given(%n) : (!trait.claim<@C118[i32] by @c118>)
  trait.return %d : !trait.claim<@C117[i32]>
}
trait.proof private @c118 {
  %n = trait.witness @c119 for @C119[i32]
  %d = trait.derive @C118[i32] from @C118_i32 given(%n) : (!trait.claim<@C119[i32] by @c119>)
  trait.return %d : !trait.claim<@C118[i32]>
}
trait.proof private @c119 {
  %n = trait.witness @c120 for @C120[i32]
  %d = trait.derive @C119[i32] from @C119_i32 given(%n) : (!trait.claim<@C120[i32] by @c120>)
  trait.return %d : !trait.claim<@C119[i32]>
}
trait.proof private @c120 {
  %n = trait.witness @c121 for @C121[i32]
  %d = trait.derive @C120[i32] from @C120_i32 given(%n) : (!trait.claim<@C121[i32] by @c121>)
  trait.return %d : !trait.claim<@C120[i32]>
}
trait.proof private @c121 {
  %n = trait.witness @c122 for @C122[i32]
  %d = trait.derive @C121[i32] from @C121_i32 given(%n) : (!trait.claim<@C122[i32] by @c122>)
  trait.return %d : !trait.claim<@C121[i32]>
}
trait.proof private @c122 {
  %n = trait.witness @c123 for @C123[i32]
  %d = trait.derive @C122[i32] from @C122_i32 given(%n) : (!trait.claim<@C123[i32] by @c123>)
  trait.return %d : !trait.claim<@C122[i32]>
}
trait.proof private @c123 {
  %n = trait.witness @c124 for @C124[i32]
  %d = trait.derive @C123[i32] from @C123_i32 given(%n) : (!trait.claim<@C124[i32] by @c124>)
  trait.return %d : !trait.claim<@C123[i32]>
}
trait.proof private @c124 {
  %n = trait.witness @c125 for @C125[i32]
  %d = trait.derive @C124[i32] from @C124_i32 given(%n) : (!trait.claim<@C125[i32] by @c125>)
  trait.return %d : !trait.claim<@C124[i32]>
}
trait.proof private @c125 {
  %n = trait.witness @c126 for @C126[i32]
  %d = trait.derive @C125[i32] from @C125_i32 given(%n) : (!trait.claim<@C126[i32] by @c126>)
  trait.return %d : !trait.claim<@C125[i32]>
}
trait.proof private @c126 {
  %x = trait.witness @x for @X[i32]
  %d = trait.derive @C126[i32] from @C126_i32 given(%x) : (!trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@C126[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @r for @R[i32]
  %v = arith.constant 9 : i64
  return %v : i64
}

// -----

// The same derivation with @C1 listed first.

// CHECK: error: overflow evaluating the requirement {{.*}}@Y1[i32]{{.*}}: 128 obligations stand on the chain that reaches it

!T = !trait.poly<0>
trait.trait private @R(%self: !trait.claim<@R[!T]>) {}
trait.trait private @X(%self: !trait.claim<@X[!T]>) {}
trait.trait private @Y1(%self: !trait.claim<@Y1[!T]>) {}
trait.trait private @Y2(%self: !trait.claim<@Y2[!T]>) {}
trait.trait private @C1(%self: !trait.claim<@C1[!T]>) {}
trait.trait private @C2(%self: !trait.claim<@C2[!T]>) {}
trait.trait private @C3(%self: !trait.claim<@C3[!T]>) {}
trait.trait private @C4(%self: !trait.claim<@C4[!T]>) {}
trait.trait private @C5(%self: !trait.claim<@C5[!T]>) {}
trait.trait private @C6(%self: !trait.claim<@C6[!T]>) {}
trait.trait private @C7(%self: !trait.claim<@C7[!T]>) {}
trait.trait private @C8(%self: !trait.claim<@C8[!T]>) {}
trait.trait private @C9(%self: !trait.claim<@C9[!T]>) {}
trait.trait private @C10(%self: !trait.claim<@C10[!T]>) {}
trait.trait private @C11(%self: !trait.claim<@C11[!T]>) {}
trait.trait private @C12(%self: !trait.claim<@C12[!T]>) {}
trait.trait private @C13(%self: !trait.claim<@C13[!T]>) {}
trait.trait private @C14(%self: !trait.claim<@C14[!T]>) {}
trait.trait private @C15(%self: !trait.claim<@C15[!T]>) {}
trait.trait private @C16(%self: !trait.claim<@C16[!T]>) {}
trait.trait private @C17(%self: !trait.claim<@C17[!T]>) {}
trait.trait private @C18(%self: !trait.claim<@C18[!T]>) {}
trait.trait private @C19(%self: !trait.claim<@C19[!T]>) {}
trait.trait private @C20(%self: !trait.claim<@C20[!T]>) {}
trait.trait private @C21(%self: !trait.claim<@C21[!T]>) {}
trait.trait private @C22(%self: !trait.claim<@C22[!T]>) {}
trait.trait private @C23(%self: !trait.claim<@C23[!T]>) {}
trait.trait private @C24(%self: !trait.claim<@C24[!T]>) {}
trait.trait private @C25(%self: !trait.claim<@C25[!T]>) {}
trait.trait private @C26(%self: !trait.claim<@C26[!T]>) {}
trait.trait private @C27(%self: !trait.claim<@C27[!T]>) {}
trait.trait private @C28(%self: !trait.claim<@C28[!T]>) {}
trait.trait private @C29(%self: !trait.claim<@C29[!T]>) {}
trait.trait private @C30(%self: !trait.claim<@C30[!T]>) {}
trait.trait private @C31(%self: !trait.claim<@C31[!T]>) {}
trait.trait private @C32(%self: !trait.claim<@C32[!T]>) {}
trait.trait private @C33(%self: !trait.claim<@C33[!T]>) {}
trait.trait private @C34(%self: !trait.claim<@C34[!T]>) {}
trait.trait private @C35(%self: !trait.claim<@C35[!T]>) {}
trait.trait private @C36(%self: !trait.claim<@C36[!T]>) {}
trait.trait private @C37(%self: !trait.claim<@C37[!T]>) {}
trait.trait private @C38(%self: !trait.claim<@C38[!T]>) {}
trait.trait private @C39(%self: !trait.claim<@C39[!T]>) {}
trait.trait private @C40(%self: !trait.claim<@C40[!T]>) {}
trait.trait private @C41(%self: !trait.claim<@C41[!T]>) {}
trait.trait private @C42(%self: !trait.claim<@C42[!T]>) {}
trait.trait private @C43(%self: !trait.claim<@C43[!T]>) {}
trait.trait private @C44(%self: !trait.claim<@C44[!T]>) {}
trait.trait private @C45(%self: !trait.claim<@C45[!T]>) {}
trait.trait private @C46(%self: !trait.claim<@C46[!T]>) {}
trait.trait private @C47(%self: !trait.claim<@C47[!T]>) {}
trait.trait private @C48(%self: !trait.claim<@C48[!T]>) {}
trait.trait private @C49(%self: !trait.claim<@C49[!T]>) {}
trait.trait private @C50(%self: !trait.claim<@C50[!T]>) {}
trait.trait private @C51(%self: !trait.claim<@C51[!T]>) {}
trait.trait private @C52(%self: !trait.claim<@C52[!T]>) {}
trait.trait private @C53(%self: !trait.claim<@C53[!T]>) {}
trait.trait private @C54(%self: !trait.claim<@C54[!T]>) {}
trait.trait private @C55(%self: !trait.claim<@C55[!T]>) {}
trait.trait private @C56(%self: !trait.claim<@C56[!T]>) {}
trait.trait private @C57(%self: !trait.claim<@C57[!T]>) {}
trait.trait private @C58(%self: !trait.claim<@C58[!T]>) {}
trait.trait private @C59(%self: !trait.claim<@C59[!T]>) {}
trait.trait private @C60(%self: !trait.claim<@C60[!T]>) {}
trait.trait private @C61(%self: !trait.claim<@C61[!T]>) {}
trait.trait private @C62(%self: !trait.claim<@C62[!T]>) {}
trait.trait private @C63(%self: !trait.claim<@C63[!T]>) {}
trait.trait private @C64(%self: !trait.claim<@C64[!T]>) {}
trait.trait private @C65(%self: !trait.claim<@C65[!T]>) {}
trait.trait private @C66(%self: !trait.claim<@C66[!T]>) {}
trait.trait private @C67(%self: !trait.claim<@C67[!T]>) {}
trait.trait private @C68(%self: !trait.claim<@C68[!T]>) {}
trait.trait private @C69(%self: !trait.claim<@C69[!T]>) {}
trait.trait private @C70(%self: !trait.claim<@C70[!T]>) {}
trait.trait private @C71(%self: !trait.claim<@C71[!T]>) {}
trait.trait private @C72(%self: !trait.claim<@C72[!T]>) {}
trait.trait private @C73(%self: !trait.claim<@C73[!T]>) {}
trait.trait private @C74(%self: !trait.claim<@C74[!T]>) {}
trait.trait private @C75(%self: !trait.claim<@C75[!T]>) {}
trait.trait private @C76(%self: !trait.claim<@C76[!T]>) {}
trait.trait private @C77(%self: !trait.claim<@C77[!T]>) {}
trait.trait private @C78(%self: !trait.claim<@C78[!T]>) {}
trait.trait private @C79(%self: !trait.claim<@C79[!T]>) {}
trait.trait private @C80(%self: !trait.claim<@C80[!T]>) {}
trait.trait private @C81(%self: !trait.claim<@C81[!T]>) {}
trait.trait private @C82(%self: !trait.claim<@C82[!T]>) {}
trait.trait private @C83(%self: !trait.claim<@C83[!T]>) {}
trait.trait private @C84(%self: !trait.claim<@C84[!T]>) {}
trait.trait private @C85(%self: !trait.claim<@C85[!T]>) {}
trait.trait private @C86(%self: !trait.claim<@C86[!T]>) {}
trait.trait private @C87(%self: !trait.claim<@C87[!T]>) {}
trait.trait private @C88(%self: !trait.claim<@C88[!T]>) {}
trait.trait private @C89(%self: !trait.claim<@C89[!T]>) {}
trait.trait private @C90(%self: !trait.claim<@C90[!T]>) {}
trait.trait private @C91(%self: !trait.claim<@C91[!T]>) {}
trait.trait private @C92(%self: !trait.claim<@C92[!T]>) {}
trait.trait private @C93(%self: !trait.claim<@C93[!T]>) {}
trait.trait private @C94(%self: !trait.claim<@C94[!T]>) {}
trait.trait private @C95(%self: !trait.claim<@C95[!T]>) {}
trait.trait private @C96(%self: !trait.claim<@C96[!T]>) {}
trait.trait private @C97(%self: !trait.claim<@C97[!T]>) {}
trait.trait private @C98(%self: !trait.claim<@C98[!T]>) {}
trait.trait private @C99(%self: !trait.claim<@C99[!T]>) {}
trait.trait private @C100(%self: !trait.claim<@C100[!T]>) {}
trait.trait private @C101(%self: !trait.claim<@C101[!T]>) {}
trait.trait private @C102(%self: !trait.claim<@C102[!T]>) {}
trait.trait private @C103(%self: !trait.claim<@C103[!T]>) {}
trait.trait private @C104(%self: !trait.claim<@C104[!T]>) {}
trait.trait private @C105(%self: !trait.claim<@C105[!T]>) {}
trait.trait private @C106(%self: !trait.claim<@C106[!T]>) {}
trait.trait private @C107(%self: !trait.claim<@C107[!T]>) {}
trait.trait private @C108(%self: !trait.claim<@C108[!T]>) {}
trait.trait private @C109(%self: !trait.claim<@C109[!T]>) {}
trait.trait private @C110(%self: !trait.claim<@C110[!T]>) {}
trait.trait private @C111(%self: !trait.claim<@C111[!T]>) {}
trait.trait private @C112(%self: !trait.claim<@C112[!T]>) {}
trait.trait private @C113(%self: !trait.claim<@C113[!T]>) {}
trait.trait private @C114(%self: !trait.claim<@C114[!T]>) {}
trait.trait private @C115(%self: !trait.claim<@C115[!T]>) {}
trait.trait private @C116(%self: !trait.claim<@C116[!T]>) {}
trait.trait private @C117(%self: !trait.claim<@C117[!T]>) {}
trait.trait private @C118(%self: !trait.claim<@C118[!T]>) {}
trait.trait private @C119(%self: !trait.claim<@C119[!T]>) {}
trait.trait private @C120(%self: !trait.claim<@C120[!T]>) {}
trait.trait private @C121(%self: !trait.claim<@C121[!T]>) {}
trait.trait private @C122(%self: !trait.claim<@C122[!T]>) {}
trait.trait private @C123(%self: !trait.claim<@C123[!T]>) {}
trait.trait private @C124(%self: !trait.claim<@C124[!T]>) {}
trait.trait private @C125(%self: !trait.claim<@C125[!T]>) {}
trait.trait private @C126(%self: !trait.claim<@C126[!T]>) {}
trait.impl private @R_i32(%self: !trait.claim<@R[i32]>, %c: !trait.claim<@C1[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.impl private @X_i32(%self: !trait.claim<@X[i32]>, %y: !trait.claim<@Y1[i32]>) {}
trait.impl private @Y1_i32(%self: !trait.claim<@Y1[i32]>, %y: !trait.claim<@Y2[i32]>) {}
trait.impl private @Y2_i32(%self: !trait.claim<@Y2[i32]>) {}
trait.impl private @C1_i32(%self: !trait.claim<@C1[i32]>, %n: !trait.claim<@C2[i32]>) {}
trait.impl private @C2_i32(%self: !trait.claim<@C2[i32]>, %n: !trait.claim<@C3[i32]>) {}
trait.impl private @C3_i32(%self: !trait.claim<@C3[i32]>, %n: !trait.claim<@C4[i32]>) {}
trait.impl private @C4_i32(%self: !trait.claim<@C4[i32]>, %n: !trait.claim<@C5[i32]>) {}
trait.impl private @C5_i32(%self: !trait.claim<@C5[i32]>, %n: !trait.claim<@C6[i32]>) {}
trait.impl private @C6_i32(%self: !trait.claim<@C6[i32]>, %n: !trait.claim<@C7[i32]>) {}
trait.impl private @C7_i32(%self: !trait.claim<@C7[i32]>, %n: !trait.claim<@C8[i32]>) {}
trait.impl private @C8_i32(%self: !trait.claim<@C8[i32]>, %n: !trait.claim<@C9[i32]>) {}
trait.impl private @C9_i32(%self: !trait.claim<@C9[i32]>, %n: !trait.claim<@C10[i32]>) {}
trait.impl private @C10_i32(%self: !trait.claim<@C10[i32]>, %n: !trait.claim<@C11[i32]>) {}
trait.impl private @C11_i32(%self: !trait.claim<@C11[i32]>, %n: !trait.claim<@C12[i32]>) {}
trait.impl private @C12_i32(%self: !trait.claim<@C12[i32]>, %n: !trait.claim<@C13[i32]>) {}
trait.impl private @C13_i32(%self: !trait.claim<@C13[i32]>, %n: !trait.claim<@C14[i32]>) {}
trait.impl private @C14_i32(%self: !trait.claim<@C14[i32]>, %n: !trait.claim<@C15[i32]>) {}
trait.impl private @C15_i32(%self: !trait.claim<@C15[i32]>, %n: !trait.claim<@C16[i32]>) {}
trait.impl private @C16_i32(%self: !trait.claim<@C16[i32]>, %n: !trait.claim<@C17[i32]>) {}
trait.impl private @C17_i32(%self: !trait.claim<@C17[i32]>, %n: !trait.claim<@C18[i32]>) {}
trait.impl private @C18_i32(%self: !trait.claim<@C18[i32]>, %n: !trait.claim<@C19[i32]>) {}
trait.impl private @C19_i32(%self: !trait.claim<@C19[i32]>, %n: !trait.claim<@C20[i32]>) {}
trait.impl private @C20_i32(%self: !trait.claim<@C20[i32]>, %n: !trait.claim<@C21[i32]>) {}
trait.impl private @C21_i32(%self: !trait.claim<@C21[i32]>, %n: !trait.claim<@C22[i32]>) {}
trait.impl private @C22_i32(%self: !trait.claim<@C22[i32]>, %n: !trait.claim<@C23[i32]>) {}
trait.impl private @C23_i32(%self: !trait.claim<@C23[i32]>, %n: !trait.claim<@C24[i32]>) {}
trait.impl private @C24_i32(%self: !trait.claim<@C24[i32]>, %n: !trait.claim<@C25[i32]>) {}
trait.impl private @C25_i32(%self: !trait.claim<@C25[i32]>, %n: !trait.claim<@C26[i32]>) {}
trait.impl private @C26_i32(%self: !trait.claim<@C26[i32]>, %n: !trait.claim<@C27[i32]>) {}
trait.impl private @C27_i32(%self: !trait.claim<@C27[i32]>, %n: !trait.claim<@C28[i32]>) {}
trait.impl private @C28_i32(%self: !trait.claim<@C28[i32]>, %n: !trait.claim<@C29[i32]>) {}
trait.impl private @C29_i32(%self: !trait.claim<@C29[i32]>, %n: !trait.claim<@C30[i32]>) {}
trait.impl private @C30_i32(%self: !trait.claim<@C30[i32]>, %n: !trait.claim<@C31[i32]>) {}
trait.impl private @C31_i32(%self: !trait.claim<@C31[i32]>, %n: !trait.claim<@C32[i32]>) {}
trait.impl private @C32_i32(%self: !trait.claim<@C32[i32]>, %n: !trait.claim<@C33[i32]>) {}
trait.impl private @C33_i32(%self: !trait.claim<@C33[i32]>, %n: !trait.claim<@C34[i32]>) {}
trait.impl private @C34_i32(%self: !trait.claim<@C34[i32]>, %n: !trait.claim<@C35[i32]>) {}
trait.impl private @C35_i32(%self: !trait.claim<@C35[i32]>, %n: !trait.claim<@C36[i32]>) {}
trait.impl private @C36_i32(%self: !trait.claim<@C36[i32]>, %n: !trait.claim<@C37[i32]>) {}
trait.impl private @C37_i32(%self: !trait.claim<@C37[i32]>, %n: !trait.claim<@C38[i32]>) {}
trait.impl private @C38_i32(%self: !trait.claim<@C38[i32]>, %n: !trait.claim<@C39[i32]>) {}
trait.impl private @C39_i32(%self: !trait.claim<@C39[i32]>, %n: !trait.claim<@C40[i32]>) {}
trait.impl private @C40_i32(%self: !trait.claim<@C40[i32]>, %n: !trait.claim<@C41[i32]>) {}
trait.impl private @C41_i32(%self: !trait.claim<@C41[i32]>, %n: !trait.claim<@C42[i32]>) {}
trait.impl private @C42_i32(%self: !trait.claim<@C42[i32]>, %n: !trait.claim<@C43[i32]>) {}
trait.impl private @C43_i32(%self: !trait.claim<@C43[i32]>, %n: !trait.claim<@C44[i32]>) {}
trait.impl private @C44_i32(%self: !trait.claim<@C44[i32]>, %n: !trait.claim<@C45[i32]>) {}
trait.impl private @C45_i32(%self: !trait.claim<@C45[i32]>, %n: !trait.claim<@C46[i32]>) {}
trait.impl private @C46_i32(%self: !trait.claim<@C46[i32]>, %n: !trait.claim<@C47[i32]>) {}
trait.impl private @C47_i32(%self: !trait.claim<@C47[i32]>, %n: !trait.claim<@C48[i32]>) {}
trait.impl private @C48_i32(%self: !trait.claim<@C48[i32]>, %n: !trait.claim<@C49[i32]>) {}
trait.impl private @C49_i32(%self: !trait.claim<@C49[i32]>, %n: !trait.claim<@C50[i32]>) {}
trait.impl private @C50_i32(%self: !trait.claim<@C50[i32]>, %n: !trait.claim<@C51[i32]>) {}
trait.impl private @C51_i32(%self: !trait.claim<@C51[i32]>, %n: !trait.claim<@C52[i32]>) {}
trait.impl private @C52_i32(%self: !trait.claim<@C52[i32]>, %n: !trait.claim<@C53[i32]>) {}
trait.impl private @C53_i32(%self: !trait.claim<@C53[i32]>, %n: !trait.claim<@C54[i32]>) {}
trait.impl private @C54_i32(%self: !trait.claim<@C54[i32]>, %n: !trait.claim<@C55[i32]>) {}
trait.impl private @C55_i32(%self: !trait.claim<@C55[i32]>, %n: !trait.claim<@C56[i32]>) {}
trait.impl private @C56_i32(%self: !trait.claim<@C56[i32]>, %n: !trait.claim<@C57[i32]>) {}
trait.impl private @C57_i32(%self: !trait.claim<@C57[i32]>, %n: !trait.claim<@C58[i32]>) {}
trait.impl private @C58_i32(%self: !trait.claim<@C58[i32]>, %n: !trait.claim<@C59[i32]>) {}
trait.impl private @C59_i32(%self: !trait.claim<@C59[i32]>, %n: !trait.claim<@C60[i32]>) {}
trait.impl private @C60_i32(%self: !trait.claim<@C60[i32]>, %n: !trait.claim<@C61[i32]>) {}
trait.impl private @C61_i32(%self: !trait.claim<@C61[i32]>, %n: !trait.claim<@C62[i32]>) {}
trait.impl private @C62_i32(%self: !trait.claim<@C62[i32]>, %n: !trait.claim<@C63[i32]>) {}
trait.impl private @C63_i32(%self: !trait.claim<@C63[i32]>, %n: !trait.claim<@C64[i32]>) {}
trait.impl private @C64_i32(%self: !trait.claim<@C64[i32]>, %n: !trait.claim<@C65[i32]>) {}
trait.impl private @C65_i32(%self: !trait.claim<@C65[i32]>, %n: !trait.claim<@C66[i32]>) {}
trait.impl private @C66_i32(%self: !trait.claim<@C66[i32]>, %n: !trait.claim<@C67[i32]>) {}
trait.impl private @C67_i32(%self: !trait.claim<@C67[i32]>, %n: !trait.claim<@C68[i32]>) {}
trait.impl private @C68_i32(%self: !trait.claim<@C68[i32]>, %n: !trait.claim<@C69[i32]>) {}
trait.impl private @C69_i32(%self: !trait.claim<@C69[i32]>, %n: !trait.claim<@C70[i32]>) {}
trait.impl private @C70_i32(%self: !trait.claim<@C70[i32]>, %n: !trait.claim<@C71[i32]>) {}
trait.impl private @C71_i32(%self: !trait.claim<@C71[i32]>, %n: !trait.claim<@C72[i32]>) {}
trait.impl private @C72_i32(%self: !trait.claim<@C72[i32]>, %n: !trait.claim<@C73[i32]>) {}
trait.impl private @C73_i32(%self: !trait.claim<@C73[i32]>, %n: !trait.claim<@C74[i32]>) {}
trait.impl private @C74_i32(%self: !trait.claim<@C74[i32]>, %n: !trait.claim<@C75[i32]>) {}
trait.impl private @C75_i32(%self: !trait.claim<@C75[i32]>, %n: !trait.claim<@C76[i32]>) {}
trait.impl private @C76_i32(%self: !trait.claim<@C76[i32]>, %n: !trait.claim<@C77[i32]>) {}
trait.impl private @C77_i32(%self: !trait.claim<@C77[i32]>, %n: !trait.claim<@C78[i32]>) {}
trait.impl private @C78_i32(%self: !trait.claim<@C78[i32]>, %n: !trait.claim<@C79[i32]>) {}
trait.impl private @C79_i32(%self: !trait.claim<@C79[i32]>, %n: !trait.claim<@C80[i32]>) {}
trait.impl private @C80_i32(%self: !trait.claim<@C80[i32]>, %n: !trait.claim<@C81[i32]>) {}
trait.impl private @C81_i32(%self: !trait.claim<@C81[i32]>, %n: !trait.claim<@C82[i32]>) {}
trait.impl private @C82_i32(%self: !trait.claim<@C82[i32]>, %n: !trait.claim<@C83[i32]>) {}
trait.impl private @C83_i32(%self: !trait.claim<@C83[i32]>, %n: !trait.claim<@C84[i32]>) {}
trait.impl private @C84_i32(%self: !trait.claim<@C84[i32]>, %n: !trait.claim<@C85[i32]>) {}
trait.impl private @C85_i32(%self: !trait.claim<@C85[i32]>, %n: !trait.claim<@C86[i32]>) {}
trait.impl private @C86_i32(%self: !trait.claim<@C86[i32]>, %n: !trait.claim<@C87[i32]>) {}
trait.impl private @C87_i32(%self: !trait.claim<@C87[i32]>, %n: !trait.claim<@C88[i32]>) {}
trait.impl private @C88_i32(%self: !trait.claim<@C88[i32]>, %n: !trait.claim<@C89[i32]>) {}
trait.impl private @C89_i32(%self: !trait.claim<@C89[i32]>, %n: !trait.claim<@C90[i32]>) {}
trait.impl private @C90_i32(%self: !trait.claim<@C90[i32]>, %n: !trait.claim<@C91[i32]>) {}
trait.impl private @C91_i32(%self: !trait.claim<@C91[i32]>, %n: !trait.claim<@C92[i32]>) {}
trait.impl private @C92_i32(%self: !trait.claim<@C92[i32]>, %n: !trait.claim<@C93[i32]>) {}
trait.impl private @C93_i32(%self: !trait.claim<@C93[i32]>, %n: !trait.claim<@C94[i32]>) {}
trait.impl private @C94_i32(%self: !trait.claim<@C94[i32]>, %n: !trait.claim<@C95[i32]>) {}
trait.impl private @C95_i32(%self: !trait.claim<@C95[i32]>, %n: !trait.claim<@C96[i32]>) {}
trait.impl private @C96_i32(%self: !trait.claim<@C96[i32]>, %n: !trait.claim<@C97[i32]>) {}
trait.impl private @C97_i32(%self: !trait.claim<@C97[i32]>, %n: !trait.claim<@C98[i32]>) {}
trait.impl private @C98_i32(%self: !trait.claim<@C98[i32]>, %n: !trait.claim<@C99[i32]>) {}
trait.impl private @C99_i32(%self: !trait.claim<@C99[i32]>, %n: !trait.claim<@C100[i32]>) {}
trait.impl private @C100_i32(%self: !trait.claim<@C100[i32]>, %n: !trait.claim<@C101[i32]>) {}
trait.impl private @C101_i32(%self: !trait.claim<@C101[i32]>, %n: !trait.claim<@C102[i32]>) {}
trait.impl private @C102_i32(%self: !trait.claim<@C102[i32]>, %n: !trait.claim<@C103[i32]>) {}
trait.impl private @C103_i32(%self: !trait.claim<@C103[i32]>, %n: !trait.claim<@C104[i32]>) {}
trait.impl private @C104_i32(%self: !trait.claim<@C104[i32]>, %n: !trait.claim<@C105[i32]>) {}
trait.impl private @C105_i32(%self: !trait.claim<@C105[i32]>, %n: !trait.claim<@C106[i32]>) {}
trait.impl private @C106_i32(%self: !trait.claim<@C106[i32]>, %n: !trait.claim<@C107[i32]>) {}
trait.impl private @C107_i32(%self: !trait.claim<@C107[i32]>, %n: !trait.claim<@C108[i32]>) {}
trait.impl private @C108_i32(%self: !trait.claim<@C108[i32]>, %n: !trait.claim<@C109[i32]>) {}
trait.impl private @C109_i32(%self: !trait.claim<@C109[i32]>, %n: !trait.claim<@C110[i32]>) {}
trait.impl private @C110_i32(%self: !trait.claim<@C110[i32]>, %n: !trait.claim<@C111[i32]>) {}
trait.impl private @C111_i32(%self: !trait.claim<@C111[i32]>, %n: !trait.claim<@C112[i32]>) {}
trait.impl private @C112_i32(%self: !trait.claim<@C112[i32]>, %n: !trait.claim<@C113[i32]>) {}
trait.impl private @C113_i32(%self: !trait.claim<@C113[i32]>, %n: !trait.claim<@C114[i32]>) {}
trait.impl private @C114_i32(%self: !trait.claim<@C114[i32]>, %n: !trait.claim<@C115[i32]>) {}
trait.impl private @C115_i32(%self: !trait.claim<@C115[i32]>, %n: !trait.claim<@C116[i32]>) {}
trait.impl private @C116_i32(%self: !trait.claim<@C116[i32]>, %n: !trait.claim<@C117[i32]>) {}
trait.impl private @C117_i32(%self: !trait.claim<@C117[i32]>, %n: !trait.claim<@C118[i32]>) {}
trait.impl private @C118_i32(%self: !trait.claim<@C118[i32]>, %n: !trait.claim<@C119[i32]>) {}
trait.impl private @C119_i32(%self: !trait.claim<@C119[i32]>, %n: !trait.claim<@C120[i32]>) {}
trait.impl private @C120_i32(%self: !trait.claim<@C120[i32]>, %n: !trait.claim<@C121[i32]>) {}
trait.impl private @C121_i32(%self: !trait.claim<@C121[i32]>, %n: !trait.claim<@C122[i32]>) {}
trait.impl private @C122_i32(%self: !trait.claim<@C122[i32]>, %n: !trait.claim<@C123[i32]>) {}
trait.impl private @C123_i32(%self: !trait.claim<@C123[i32]>, %n: !trait.claim<@C124[i32]>) {}
trait.impl private @C124_i32(%self: !trait.claim<@C124[i32]>, %n: !trait.claim<@C125[i32]>) {}
trait.impl private @C125_i32(%self: !trait.claim<@C125[i32]>, %n: !trait.claim<@C126[i32]>) {}
trait.impl private @C126_i32(%self: !trait.claim<@C126[i32]>, %x: !trait.claim<@X[i32]>) {}
trait.proof private @r {
  %x = trait.witness @x for @X[i32]
  %c = trait.witness @c1 for @C1[i32]
  %d = trait.derive @R[i32] from @R_i32 given(%c, %x) : (!trait.claim<@C1[i32] by @c1>, !trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@R[i32]>
}
trait.proof private @x {
  %y = trait.witness @y1 for @Y1[i32]
  %d = trait.derive @X[i32] from @X_i32 given(%y) : (!trait.claim<@Y1[i32] by @y1>)
  trait.return %d : !trait.claim<@X[i32]>
}
trait.proof private @y1 {
  %y = trait.witness @Y2_i32 for @Y2[i32]
  %d = trait.derive @Y1[i32] from @Y1_i32 given(%y) : (!trait.claim<@Y2[i32] by @Y2_i32>)
  trait.return %d : !trait.claim<@Y1[i32]>
}
trait.proof private @c1 {
  %n = trait.witness @c2 for @C2[i32]
  %d = trait.derive @C1[i32] from @C1_i32 given(%n) : (!trait.claim<@C2[i32] by @c2>)
  trait.return %d : !trait.claim<@C1[i32]>
}
trait.proof private @c2 {
  %n = trait.witness @c3 for @C3[i32]
  %d = trait.derive @C2[i32] from @C2_i32 given(%n) : (!trait.claim<@C3[i32] by @c3>)
  trait.return %d : !trait.claim<@C2[i32]>
}
trait.proof private @c3 {
  %n = trait.witness @c4 for @C4[i32]
  %d = trait.derive @C3[i32] from @C3_i32 given(%n) : (!trait.claim<@C4[i32] by @c4>)
  trait.return %d : !trait.claim<@C3[i32]>
}
trait.proof private @c4 {
  %n = trait.witness @c5 for @C5[i32]
  %d = trait.derive @C4[i32] from @C4_i32 given(%n) : (!trait.claim<@C5[i32] by @c5>)
  trait.return %d : !trait.claim<@C4[i32]>
}
trait.proof private @c5 {
  %n = trait.witness @c6 for @C6[i32]
  %d = trait.derive @C5[i32] from @C5_i32 given(%n) : (!trait.claim<@C6[i32] by @c6>)
  trait.return %d : !trait.claim<@C5[i32]>
}
trait.proof private @c6 {
  %n = trait.witness @c7 for @C7[i32]
  %d = trait.derive @C6[i32] from @C6_i32 given(%n) : (!trait.claim<@C7[i32] by @c7>)
  trait.return %d : !trait.claim<@C6[i32]>
}
trait.proof private @c7 {
  %n = trait.witness @c8 for @C8[i32]
  %d = trait.derive @C7[i32] from @C7_i32 given(%n) : (!trait.claim<@C8[i32] by @c8>)
  trait.return %d : !trait.claim<@C7[i32]>
}
trait.proof private @c8 {
  %n = trait.witness @c9 for @C9[i32]
  %d = trait.derive @C8[i32] from @C8_i32 given(%n) : (!trait.claim<@C9[i32] by @c9>)
  trait.return %d : !trait.claim<@C8[i32]>
}
trait.proof private @c9 {
  %n = trait.witness @c10 for @C10[i32]
  %d = trait.derive @C9[i32] from @C9_i32 given(%n) : (!trait.claim<@C10[i32] by @c10>)
  trait.return %d : !trait.claim<@C9[i32]>
}
trait.proof private @c10 {
  %n = trait.witness @c11 for @C11[i32]
  %d = trait.derive @C10[i32] from @C10_i32 given(%n) : (!trait.claim<@C11[i32] by @c11>)
  trait.return %d : !trait.claim<@C10[i32]>
}
trait.proof private @c11 {
  %n = trait.witness @c12 for @C12[i32]
  %d = trait.derive @C11[i32] from @C11_i32 given(%n) : (!trait.claim<@C12[i32] by @c12>)
  trait.return %d : !trait.claim<@C11[i32]>
}
trait.proof private @c12 {
  %n = trait.witness @c13 for @C13[i32]
  %d = trait.derive @C12[i32] from @C12_i32 given(%n) : (!trait.claim<@C13[i32] by @c13>)
  trait.return %d : !trait.claim<@C12[i32]>
}
trait.proof private @c13 {
  %n = trait.witness @c14 for @C14[i32]
  %d = trait.derive @C13[i32] from @C13_i32 given(%n) : (!trait.claim<@C14[i32] by @c14>)
  trait.return %d : !trait.claim<@C13[i32]>
}
trait.proof private @c14 {
  %n = trait.witness @c15 for @C15[i32]
  %d = trait.derive @C14[i32] from @C14_i32 given(%n) : (!trait.claim<@C15[i32] by @c15>)
  trait.return %d : !trait.claim<@C14[i32]>
}
trait.proof private @c15 {
  %n = trait.witness @c16 for @C16[i32]
  %d = trait.derive @C15[i32] from @C15_i32 given(%n) : (!trait.claim<@C16[i32] by @c16>)
  trait.return %d : !trait.claim<@C15[i32]>
}
trait.proof private @c16 {
  %n = trait.witness @c17 for @C17[i32]
  %d = trait.derive @C16[i32] from @C16_i32 given(%n) : (!trait.claim<@C17[i32] by @c17>)
  trait.return %d : !trait.claim<@C16[i32]>
}
trait.proof private @c17 {
  %n = trait.witness @c18 for @C18[i32]
  %d = trait.derive @C17[i32] from @C17_i32 given(%n) : (!trait.claim<@C18[i32] by @c18>)
  trait.return %d : !trait.claim<@C17[i32]>
}
trait.proof private @c18 {
  %n = trait.witness @c19 for @C19[i32]
  %d = trait.derive @C18[i32] from @C18_i32 given(%n) : (!trait.claim<@C19[i32] by @c19>)
  trait.return %d : !trait.claim<@C18[i32]>
}
trait.proof private @c19 {
  %n = trait.witness @c20 for @C20[i32]
  %d = trait.derive @C19[i32] from @C19_i32 given(%n) : (!trait.claim<@C20[i32] by @c20>)
  trait.return %d : !trait.claim<@C19[i32]>
}
trait.proof private @c20 {
  %n = trait.witness @c21 for @C21[i32]
  %d = trait.derive @C20[i32] from @C20_i32 given(%n) : (!trait.claim<@C21[i32] by @c21>)
  trait.return %d : !trait.claim<@C20[i32]>
}
trait.proof private @c21 {
  %n = trait.witness @c22 for @C22[i32]
  %d = trait.derive @C21[i32] from @C21_i32 given(%n) : (!trait.claim<@C22[i32] by @c22>)
  trait.return %d : !trait.claim<@C21[i32]>
}
trait.proof private @c22 {
  %n = trait.witness @c23 for @C23[i32]
  %d = trait.derive @C22[i32] from @C22_i32 given(%n) : (!trait.claim<@C23[i32] by @c23>)
  trait.return %d : !trait.claim<@C22[i32]>
}
trait.proof private @c23 {
  %n = trait.witness @c24 for @C24[i32]
  %d = trait.derive @C23[i32] from @C23_i32 given(%n) : (!trait.claim<@C24[i32] by @c24>)
  trait.return %d : !trait.claim<@C23[i32]>
}
trait.proof private @c24 {
  %n = trait.witness @c25 for @C25[i32]
  %d = trait.derive @C24[i32] from @C24_i32 given(%n) : (!trait.claim<@C25[i32] by @c25>)
  trait.return %d : !trait.claim<@C24[i32]>
}
trait.proof private @c25 {
  %n = trait.witness @c26 for @C26[i32]
  %d = trait.derive @C25[i32] from @C25_i32 given(%n) : (!trait.claim<@C26[i32] by @c26>)
  trait.return %d : !trait.claim<@C25[i32]>
}
trait.proof private @c26 {
  %n = trait.witness @c27 for @C27[i32]
  %d = trait.derive @C26[i32] from @C26_i32 given(%n) : (!trait.claim<@C27[i32] by @c27>)
  trait.return %d : !trait.claim<@C26[i32]>
}
trait.proof private @c27 {
  %n = trait.witness @c28 for @C28[i32]
  %d = trait.derive @C27[i32] from @C27_i32 given(%n) : (!trait.claim<@C28[i32] by @c28>)
  trait.return %d : !trait.claim<@C27[i32]>
}
trait.proof private @c28 {
  %n = trait.witness @c29 for @C29[i32]
  %d = trait.derive @C28[i32] from @C28_i32 given(%n) : (!trait.claim<@C29[i32] by @c29>)
  trait.return %d : !trait.claim<@C28[i32]>
}
trait.proof private @c29 {
  %n = trait.witness @c30 for @C30[i32]
  %d = trait.derive @C29[i32] from @C29_i32 given(%n) : (!trait.claim<@C30[i32] by @c30>)
  trait.return %d : !trait.claim<@C29[i32]>
}
trait.proof private @c30 {
  %n = trait.witness @c31 for @C31[i32]
  %d = trait.derive @C30[i32] from @C30_i32 given(%n) : (!trait.claim<@C31[i32] by @c31>)
  trait.return %d : !trait.claim<@C30[i32]>
}
trait.proof private @c31 {
  %n = trait.witness @c32 for @C32[i32]
  %d = trait.derive @C31[i32] from @C31_i32 given(%n) : (!trait.claim<@C32[i32] by @c32>)
  trait.return %d : !trait.claim<@C31[i32]>
}
trait.proof private @c32 {
  %n = trait.witness @c33 for @C33[i32]
  %d = trait.derive @C32[i32] from @C32_i32 given(%n) : (!trait.claim<@C33[i32] by @c33>)
  trait.return %d : !trait.claim<@C32[i32]>
}
trait.proof private @c33 {
  %n = trait.witness @c34 for @C34[i32]
  %d = trait.derive @C33[i32] from @C33_i32 given(%n) : (!trait.claim<@C34[i32] by @c34>)
  trait.return %d : !trait.claim<@C33[i32]>
}
trait.proof private @c34 {
  %n = trait.witness @c35 for @C35[i32]
  %d = trait.derive @C34[i32] from @C34_i32 given(%n) : (!trait.claim<@C35[i32] by @c35>)
  trait.return %d : !trait.claim<@C34[i32]>
}
trait.proof private @c35 {
  %n = trait.witness @c36 for @C36[i32]
  %d = trait.derive @C35[i32] from @C35_i32 given(%n) : (!trait.claim<@C36[i32] by @c36>)
  trait.return %d : !trait.claim<@C35[i32]>
}
trait.proof private @c36 {
  %n = trait.witness @c37 for @C37[i32]
  %d = trait.derive @C36[i32] from @C36_i32 given(%n) : (!trait.claim<@C37[i32] by @c37>)
  trait.return %d : !trait.claim<@C36[i32]>
}
trait.proof private @c37 {
  %n = trait.witness @c38 for @C38[i32]
  %d = trait.derive @C37[i32] from @C37_i32 given(%n) : (!trait.claim<@C38[i32] by @c38>)
  trait.return %d : !trait.claim<@C37[i32]>
}
trait.proof private @c38 {
  %n = trait.witness @c39 for @C39[i32]
  %d = trait.derive @C38[i32] from @C38_i32 given(%n) : (!trait.claim<@C39[i32] by @c39>)
  trait.return %d : !trait.claim<@C38[i32]>
}
trait.proof private @c39 {
  %n = trait.witness @c40 for @C40[i32]
  %d = trait.derive @C39[i32] from @C39_i32 given(%n) : (!trait.claim<@C40[i32] by @c40>)
  trait.return %d : !trait.claim<@C39[i32]>
}
trait.proof private @c40 {
  %n = trait.witness @c41 for @C41[i32]
  %d = trait.derive @C40[i32] from @C40_i32 given(%n) : (!trait.claim<@C41[i32] by @c41>)
  trait.return %d : !trait.claim<@C40[i32]>
}
trait.proof private @c41 {
  %n = trait.witness @c42 for @C42[i32]
  %d = trait.derive @C41[i32] from @C41_i32 given(%n) : (!trait.claim<@C42[i32] by @c42>)
  trait.return %d : !trait.claim<@C41[i32]>
}
trait.proof private @c42 {
  %n = trait.witness @c43 for @C43[i32]
  %d = trait.derive @C42[i32] from @C42_i32 given(%n) : (!trait.claim<@C43[i32] by @c43>)
  trait.return %d : !trait.claim<@C42[i32]>
}
trait.proof private @c43 {
  %n = trait.witness @c44 for @C44[i32]
  %d = trait.derive @C43[i32] from @C43_i32 given(%n) : (!trait.claim<@C44[i32] by @c44>)
  trait.return %d : !trait.claim<@C43[i32]>
}
trait.proof private @c44 {
  %n = trait.witness @c45 for @C45[i32]
  %d = trait.derive @C44[i32] from @C44_i32 given(%n) : (!trait.claim<@C45[i32] by @c45>)
  trait.return %d : !trait.claim<@C44[i32]>
}
trait.proof private @c45 {
  %n = trait.witness @c46 for @C46[i32]
  %d = trait.derive @C45[i32] from @C45_i32 given(%n) : (!trait.claim<@C46[i32] by @c46>)
  trait.return %d : !trait.claim<@C45[i32]>
}
trait.proof private @c46 {
  %n = trait.witness @c47 for @C47[i32]
  %d = trait.derive @C46[i32] from @C46_i32 given(%n) : (!trait.claim<@C47[i32] by @c47>)
  trait.return %d : !trait.claim<@C46[i32]>
}
trait.proof private @c47 {
  %n = trait.witness @c48 for @C48[i32]
  %d = trait.derive @C47[i32] from @C47_i32 given(%n) : (!trait.claim<@C48[i32] by @c48>)
  trait.return %d : !trait.claim<@C47[i32]>
}
trait.proof private @c48 {
  %n = trait.witness @c49 for @C49[i32]
  %d = trait.derive @C48[i32] from @C48_i32 given(%n) : (!trait.claim<@C49[i32] by @c49>)
  trait.return %d : !trait.claim<@C48[i32]>
}
trait.proof private @c49 {
  %n = trait.witness @c50 for @C50[i32]
  %d = trait.derive @C49[i32] from @C49_i32 given(%n) : (!trait.claim<@C50[i32] by @c50>)
  trait.return %d : !trait.claim<@C49[i32]>
}
trait.proof private @c50 {
  %n = trait.witness @c51 for @C51[i32]
  %d = trait.derive @C50[i32] from @C50_i32 given(%n) : (!trait.claim<@C51[i32] by @c51>)
  trait.return %d : !trait.claim<@C50[i32]>
}
trait.proof private @c51 {
  %n = trait.witness @c52 for @C52[i32]
  %d = trait.derive @C51[i32] from @C51_i32 given(%n) : (!trait.claim<@C52[i32] by @c52>)
  trait.return %d : !trait.claim<@C51[i32]>
}
trait.proof private @c52 {
  %n = trait.witness @c53 for @C53[i32]
  %d = trait.derive @C52[i32] from @C52_i32 given(%n) : (!trait.claim<@C53[i32] by @c53>)
  trait.return %d : !trait.claim<@C52[i32]>
}
trait.proof private @c53 {
  %n = trait.witness @c54 for @C54[i32]
  %d = trait.derive @C53[i32] from @C53_i32 given(%n) : (!trait.claim<@C54[i32] by @c54>)
  trait.return %d : !trait.claim<@C53[i32]>
}
trait.proof private @c54 {
  %n = trait.witness @c55 for @C55[i32]
  %d = trait.derive @C54[i32] from @C54_i32 given(%n) : (!trait.claim<@C55[i32] by @c55>)
  trait.return %d : !trait.claim<@C54[i32]>
}
trait.proof private @c55 {
  %n = trait.witness @c56 for @C56[i32]
  %d = trait.derive @C55[i32] from @C55_i32 given(%n) : (!trait.claim<@C56[i32] by @c56>)
  trait.return %d : !trait.claim<@C55[i32]>
}
trait.proof private @c56 {
  %n = trait.witness @c57 for @C57[i32]
  %d = trait.derive @C56[i32] from @C56_i32 given(%n) : (!trait.claim<@C57[i32] by @c57>)
  trait.return %d : !trait.claim<@C56[i32]>
}
trait.proof private @c57 {
  %n = trait.witness @c58 for @C58[i32]
  %d = trait.derive @C57[i32] from @C57_i32 given(%n) : (!trait.claim<@C58[i32] by @c58>)
  trait.return %d : !trait.claim<@C57[i32]>
}
trait.proof private @c58 {
  %n = trait.witness @c59 for @C59[i32]
  %d = trait.derive @C58[i32] from @C58_i32 given(%n) : (!trait.claim<@C59[i32] by @c59>)
  trait.return %d : !trait.claim<@C58[i32]>
}
trait.proof private @c59 {
  %n = trait.witness @c60 for @C60[i32]
  %d = trait.derive @C59[i32] from @C59_i32 given(%n) : (!trait.claim<@C60[i32] by @c60>)
  trait.return %d : !trait.claim<@C59[i32]>
}
trait.proof private @c60 {
  %n = trait.witness @c61 for @C61[i32]
  %d = trait.derive @C60[i32] from @C60_i32 given(%n) : (!trait.claim<@C61[i32] by @c61>)
  trait.return %d : !trait.claim<@C60[i32]>
}
trait.proof private @c61 {
  %n = trait.witness @c62 for @C62[i32]
  %d = trait.derive @C61[i32] from @C61_i32 given(%n) : (!trait.claim<@C62[i32] by @c62>)
  trait.return %d : !trait.claim<@C61[i32]>
}
trait.proof private @c62 {
  %n = trait.witness @c63 for @C63[i32]
  %d = trait.derive @C62[i32] from @C62_i32 given(%n) : (!trait.claim<@C63[i32] by @c63>)
  trait.return %d : !trait.claim<@C62[i32]>
}
trait.proof private @c63 {
  %n = trait.witness @c64 for @C64[i32]
  %d = trait.derive @C63[i32] from @C63_i32 given(%n) : (!trait.claim<@C64[i32] by @c64>)
  trait.return %d : !trait.claim<@C63[i32]>
}
trait.proof private @c64 {
  %n = trait.witness @c65 for @C65[i32]
  %d = trait.derive @C64[i32] from @C64_i32 given(%n) : (!trait.claim<@C65[i32] by @c65>)
  trait.return %d : !trait.claim<@C64[i32]>
}
trait.proof private @c65 {
  %n = trait.witness @c66 for @C66[i32]
  %d = trait.derive @C65[i32] from @C65_i32 given(%n) : (!trait.claim<@C66[i32] by @c66>)
  trait.return %d : !trait.claim<@C65[i32]>
}
trait.proof private @c66 {
  %n = trait.witness @c67 for @C67[i32]
  %d = trait.derive @C66[i32] from @C66_i32 given(%n) : (!trait.claim<@C67[i32] by @c67>)
  trait.return %d : !trait.claim<@C66[i32]>
}
trait.proof private @c67 {
  %n = trait.witness @c68 for @C68[i32]
  %d = trait.derive @C67[i32] from @C67_i32 given(%n) : (!trait.claim<@C68[i32] by @c68>)
  trait.return %d : !trait.claim<@C67[i32]>
}
trait.proof private @c68 {
  %n = trait.witness @c69 for @C69[i32]
  %d = trait.derive @C68[i32] from @C68_i32 given(%n) : (!trait.claim<@C69[i32] by @c69>)
  trait.return %d : !trait.claim<@C68[i32]>
}
trait.proof private @c69 {
  %n = trait.witness @c70 for @C70[i32]
  %d = trait.derive @C69[i32] from @C69_i32 given(%n) : (!trait.claim<@C70[i32] by @c70>)
  trait.return %d : !trait.claim<@C69[i32]>
}
trait.proof private @c70 {
  %n = trait.witness @c71 for @C71[i32]
  %d = trait.derive @C70[i32] from @C70_i32 given(%n) : (!trait.claim<@C71[i32] by @c71>)
  trait.return %d : !trait.claim<@C70[i32]>
}
trait.proof private @c71 {
  %n = trait.witness @c72 for @C72[i32]
  %d = trait.derive @C71[i32] from @C71_i32 given(%n) : (!trait.claim<@C72[i32] by @c72>)
  trait.return %d : !trait.claim<@C71[i32]>
}
trait.proof private @c72 {
  %n = trait.witness @c73 for @C73[i32]
  %d = trait.derive @C72[i32] from @C72_i32 given(%n) : (!trait.claim<@C73[i32] by @c73>)
  trait.return %d : !trait.claim<@C72[i32]>
}
trait.proof private @c73 {
  %n = trait.witness @c74 for @C74[i32]
  %d = trait.derive @C73[i32] from @C73_i32 given(%n) : (!trait.claim<@C74[i32] by @c74>)
  trait.return %d : !trait.claim<@C73[i32]>
}
trait.proof private @c74 {
  %n = trait.witness @c75 for @C75[i32]
  %d = trait.derive @C74[i32] from @C74_i32 given(%n) : (!trait.claim<@C75[i32] by @c75>)
  trait.return %d : !trait.claim<@C74[i32]>
}
trait.proof private @c75 {
  %n = trait.witness @c76 for @C76[i32]
  %d = trait.derive @C75[i32] from @C75_i32 given(%n) : (!trait.claim<@C76[i32] by @c76>)
  trait.return %d : !trait.claim<@C75[i32]>
}
trait.proof private @c76 {
  %n = trait.witness @c77 for @C77[i32]
  %d = trait.derive @C76[i32] from @C76_i32 given(%n) : (!trait.claim<@C77[i32] by @c77>)
  trait.return %d : !trait.claim<@C76[i32]>
}
trait.proof private @c77 {
  %n = trait.witness @c78 for @C78[i32]
  %d = trait.derive @C77[i32] from @C77_i32 given(%n) : (!trait.claim<@C78[i32] by @c78>)
  trait.return %d : !trait.claim<@C77[i32]>
}
trait.proof private @c78 {
  %n = trait.witness @c79 for @C79[i32]
  %d = trait.derive @C78[i32] from @C78_i32 given(%n) : (!trait.claim<@C79[i32] by @c79>)
  trait.return %d : !trait.claim<@C78[i32]>
}
trait.proof private @c79 {
  %n = trait.witness @c80 for @C80[i32]
  %d = trait.derive @C79[i32] from @C79_i32 given(%n) : (!trait.claim<@C80[i32] by @c80>)
  trait.return %d : !trait.claim<@C79[i32]>
}
trait.proof private @c80 {
  %n = trait.witness @c81 for @C81[i32]
  %d = trait.derive @C80[i32] from @C80_i32 given(%n) : (!trait.claim<@C81[i32] by @c81>)
  trait.return %d : !trait.claim<@C80[i32]>
}
trait.proof private @c81 {
  %n = trait.witness @c82 for @C82[i32]
  %d = trait.derive @C81[i32] from @C81_i32 given(%n) : (!trait.claim<@C82[i32] by @c82>)
  trait.return %d : !trait.claim<@C81[i32]>
}
trait.proof private @c82 {
  %n = trait.witness @c83 for @C83[i32]
  %d = trait.derive @C82[i32] from @C82_i32 given(%n) : (!trait.claim<@C83[i32] by @c83>)
  trait.return %d : !trait.claim<@C82[i32]>
}
trait.proof private @c83 {
  %n = trait.witness @c84 for @C84[i32]
  %d = trait.derive @C83[i32] from @C83_i32 given(%n) : (!trait.claim<@C84[i32] by @c84>)
  trait.return %d : !trait.claim<@C83[i32]>
}
trait.proof private @c84 {
  %n = trait.witness @c85 for @C85[i32]
  %d = trait.derive @C84[i32] from @C84_i32 given(%n) : (!trait.claim<@C85[i32] by @c85>)
  trait.return %d : !trait.claim<@C84[i32]>
}
trait.proof private @c85 {
  %n = trait.witness @c86 for @C86[i32]
  %d = trait.derive @C85[i32] from @C85_i32 given(%n) : (!trait.claim<@C86[i32] by @c86>)
  trait.return %d : !trait.claim<@C85[i32]>
}
trait.proof private @c86 {
  %n = trait.witness @c87 for @C87[i32]
  %d = trait.derive @C86[i32] from @C86_i32 given(%n) : (!trait.claim<@C87[i32] by @c87>)
  trait.return %d : !trait.claim<@C86[i32]>
}
trait.proof private @c87 {
  %n = trait.witness @c88 for @C88[i32]
  %d = trait.derive @C87[i32] from @C87_i32 given(%n) : (!trait.claim<@C88[i32] by @c88>)
  trait.return %d : !trait.claim<@C87[i32]>
}
trait.proof private @c88 {
  %n = trait.witness @c89 for @C89[i32]
  %d = trait.derive @C88[i32] from @C88_i32 given(%n) : (!trait.claim<@C89[i32] by @c89>)
  trait.return %d : !trait.claim<@C88[i32]>
}
trait.proof private @c89 {
  %n = trait.witness @c90 for @C90[i32]
  %d = trait.derive @C89[i32] from @C89_i32 given(%n) : (!trait.claim<@C90[i32] by @c90>)
  trait.return %d : !trait.claim<@C89[i32]>
}
trait.proof private @c90 {
  %n = trait.witness @c91 for @C91[i32]
  %d = trait.derive @C90[i32] from @C90_i32 given(%n) : (!trait.claim<@C91[i32] by @c91>)
  trait.return %d : !trait.claim<@C90[i32]>
}
trait.proof private @c91 {
  %n = trait.witness @c92 for @C92[i32]
  %d = trait.derive @C91[i32] from @C91_i32 given(%n) : (!trait.claim<@C92[i32] by @c92>)
  trait.return %d : !trait.claim<@C91[i32]>
}
trait.proof private @c92 {
  %n = trait.witness @c93 for @C93[i32]
  %d = trait.derive @C92[i32] from @C92_i32 given(%n) : (!trait.claim<@C93[i32] by @c93>)
  trait.return %d : !trait.claim<@C92[i32]>
}
trait.proof private @c93 {
  %n = trait.witness @c94 for @C94[i32]
  %d = trait.derive @C93[i32] from @C93_i32 given(%n) : (!trait.claim<@C94[i32] by @c94>)
  trait.return %d : !trait.claim<@C93[i32]>
}
trait.proof private @c94 {
  %n = trait.witness @c95 for @C95[i32]
  %d = trait.derive @C94[i32] from @C94_i32 given(%n) : (!trait.claim<@C95[i32] by @c95>)
  trait.return %d : !trait.claim<@C94[i32]>
}
trait.proof private @c95 {
  %n = trait.witness @c96 for @C96[i32]
  %d = trait.derive @C95[i32] from @C95_i32 given(%n) : (!trait.claim<@C96[i32] by @c96>)
  trait.return %d : !trait.claim<@C95[i32]>
}
trait.proof private @c96 {
  %n = trait.witness @c97 for @C97[i32]
  %d = trait.derive @C96[i32] from @C96_i32 given(%n) : (!trait.claim<@C97[i32] by @c97>)
  trait.return %d : !trait.claim<@C96[i32]>
}
trait.proof private @c97 {
  %n = trait.witness @c98 for @C98[i32]
  %d = trait.derive @C97[i32] from @C97_i32 given(%n) : (!trait.claim<@C98[i32] by @c98>)
  trait.return %d : !trait.claim<@C97[i32]>
}
trait.proof private @c98 {
  %n = trait.witness @c99 for @C99[i32]
  %d = trait.derive @C98[i32] from @C98_i32 given(%n) : (!trait.claim<@C99[i32] by @c99>)
  trait.return %d : !trait.claim<@C98[i32]>
}
trait.proof private @c99 {
  %n = trait.witness @c100 for @C100[i32]
  %d = trait.derive @C99[i32] from @C99_i32 given(%n) : (!trait.claim<@C100[i32] by @c100>)
  trait.return %d : !trait.claim<@C99[i32]>
}
trait.proof private @c100 {
  %n = trait.witness @c101 for @C101[i32]
  %d = trait.derive @C100[i32] from @C100_i32 given(%n) : (!trait.claim<@C101[i32] by @c101>)
  trait.return %d : !trait.claim<@C100[i32]>
}
trait.proof private @c101 {
  %n = trait.witness @c102 for @C102[i32]
  %d = trait.derive @C101[i32] from @C101_i32 given(%n) : (!trait.claim<@C102[i32] by @c102>)
  trait.return %d : !trait.claim<@C101[i32]>
}
trait.proof private @c102 {
  %n = trait.witness @c103 for @C103[i32]
  %d = trait.derive @C102[i32] from @C102_i32 given(%n) : (!trait.claim<@C103[i32] by @c103>)
  trait.return %d : !trait.claim<@C102[i32]>
}
trait.proof private @c103 {
  %n = trait.witness @c104 for @C104[i32]
  %d = trait.derive @C103[i32] from @C103_i32 given(%n) : (!trait.claim<@C104[i32] by @c104>)
  trait.return %d : !trait.claim<@C103[i32]>
}
trait.proof private @c104 {
  %n = trait.witness @c105 for @C105[i32]
  %d = trait.derive @C104[i32] from @C104_i32 given(%n) : (!trait.claim<@C105[i32] by @c105>)
  trait.return %d : !trait.claim<@C104[i32]>
}
trait.proof private @c105 {
  %n = trait.witness @c106 for @C106[i32]
  %d = trait.derive @C105[i32] from @C105_i32 given(%n) : (!trait.claim<@C106[i32] by @c106>)
  trait.return %d : !trait.claim<@C105[i32]>
}
trait.proof private @c106 {
  %n = trait.witness @c107 for @C107[i32]
  %d = trait.derive @C106[i32] from @C106_i32 given(%n) : (!trait.claim<@C107[i32] by @c107>)
  trait.return %d : !trait.claim<@C106[i32]>
}
trait.proof private @c107 {
  %n = trait.witness @c108 for @C108[i32]
  %d = trait.derive @C107[i32] from @C107_i32 given(%n) : (!trait.claim<@C108[i32] by @c108>)
  trait.return %d : !trait.claim<@C107[i32]>
}
trait.proof private @c108 {
  %n = trait.witness @c109 for @C109[i32]
  %d = trait.derive @C108[i32] from @C108_i32 given(%n) : (!trait.claim<@C109[i32] by @c109>)
  trait.return %d : !trait.claim<@C108[i32]>
}
trait.proof private @c109 {
  %n = trait.witness @c110 for @C110[i32]
  %d = trait.derive @C109[i32] from @C109_i32 given(%n) : (!trait.claim<@C110[i32] by @c110>)
  trait.return %d : !trait.claim<@C109[i32]>
}
trait.proof private @c110 {
  %n = trait.witness @c111 for @C111[i32]
  %d = trait.derive @C110[i32] from @C110_i32 given(%n) : (!trait.claim<@C111[i32] by @c111>)
  trait.return %d : !trait.claim<@C110[i32]>
}
trait.proof private @c111 {
  %n = trait.witness @c112 for @C112[i32]
  %d = trait.derive @C111[i32] from @C111_i32 given(%n) : (!trait.claim<@C112[i32] by @c112>)
  trait.return %d : !trait.claim<@C111[i32]>
}
trait.proof private @c112 {
  %n = trait.witness @c113 for @C113[i32]
  %d = trait.derive @C112[i32] from @C112_i32 given(%n) : (!trait.claim<@C113[i32] by @c113>)
  trait.return %d : !trait.claim<@C112[i32]>
}
trait.proof private @c113 {
  %n = trait.witness @c114 for @C114[i32]
  %d = trait.derive @C113[i32] from @C113_i32 given(%n) : (!trait.claim<@C114[i32] by @c114>)
  trait.return %d : !trait.claim<@C113[i32]>
}
trait.proof private @c114 {
  %n = trait.witness @c115 for @C115[i32]
  %d = trait.derive @C114[i32] from @C114_i32 given(%n) : (!trait.claim<@C115[i32] by @c115>)
  trait.return %d : !trait.claim<@C114[i32]>
}
trait.proof private @c115 {
  %n = trait.witness @c116 for @C116[i32]
  %d = trait.derive @C115[i32] from @C115_i32 given(%n) : (!trait.claim<@C116[i32] by @c116>)
  trait.return %d : !trait.claim<@C115[i32]>
}
trait.proof private @c116 {
  %n = trait.witness @c117 for @C117[i32]
  %d = trait.derive @C116[i32] from @C116_i32 given(%n) : (!trait.claim<@C117[i32] by @c117>)
  trait.return %d : !trait.claim<@C116[i32]>
}
trait.proof private @c117 {
  %n = trait.witness @c118 for @C118[i32]
  %d = trait.derive @C117[i32] from @C117_i32 given(%n) : (!trait.claim<@C118[i32] by @c118>)
  trait.return %d : !trait.claim<@C117[i32]>
}
trait.proof private @c118 {
  %n = trait.witness @c119 for @C119[i32]
  %d = trait.derive @C118[i32] from @C118_i32 given(%n) : (!trait.claim<@C119[i32] by @c119>)
  trait.return %d : !trait.claim<@C118[i32]>
}
trait.proof private @c119 {
  %n = trait.witness @c120 for @C120[i32]
  %d = trait.derive @C119[i32] from @C119_i32 given(%n) : (!trait.claim<@C120[i32] by @c120>)
  trait.return %d : !trait.claim<@C119[i32]>
}
trait.proof private @c120 {
  %n = trait.witness @c121 for @C121[i32]
  %d = trait.derive @C120[i32] from @C120_i32 given(%n) : (!trait.claim<@C121[i32] by @c121>)
  trait.return %d : !trait.claim<@C120[i32]>
}
trait.proof private @c121 {
  %n = trait.witness @c122 for @C122[i32]
  %d = trait.derive @C121[i32] from @C121_i32 given(%n) : (!trait.claim<@C122[i32] by @c122>)
  trait.return %d : !trait.claim<@C121[i32]>
}
trait.proof private @c122 {
  %n = trait.witness @c123 for @C123[i32]
  %d = trait.derive @C122[i32] from @C122_i32 given(%n) : (!trait.claim<@C123[i32] by @c123>)
  trait.return %d : !trait.claim<@C122[i32]>
}
trait.proof private @c123 {
  %n = trait.witness @c124 for @C124[i32]
  %d = trait.derive @C123[i32] from @C123_i32 given(%n) : (!trait.claim<@C124[i32] by @c124>)
  trait.return %d : !trait.claim<@C123[i32]>
}
trait.proof private @c124 {
  %n = trait.witness @c125 for @C125[i32]
  %d = trait.derive @C124[i32] from @C124_i32 given(%n) : (!trait.claim<@C125[i32] by @c125>)
  trait.return %d : !trait.claim<@C124[i32]>
}
trait.proof private @c125 {
  %n = trait.witness @c126 for @C126[i32]
  %d = trait.derive @C125[i32] from @C125_i32 given(%n) : (!trait.claim<@C126[i32] by @c126>)
  trait.return %d : !trait.claim<@C125[i32]>
}
trait.proof private @c126 {
  %x = trait.witness @x for @X[i32]
  %d = trait.derive @C126[i32] from @C126_i32 given(%x) : (!trait.claim<@X[i32] by @x>)
  trait.return %d : !trait.claim<@C126[i32]>
}
func.func @main() -> i64 {
  %a = trait.witness @r for @R[i32]
  %v = arith.constant 9 : i64
  return %v : i64
}
