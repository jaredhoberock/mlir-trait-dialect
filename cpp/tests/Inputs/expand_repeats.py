# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Prints a fixture with each of its repeat lines written out.

A repeat line is `// REPEAT <from> <to>: <text>`. It stands for <text> once per
k from <from> to <to>, in that order (counting down when <to> is smaller), with
`{k}` written as k and `{k+1}` as k + 1. A row whose module needs a long run of
declarations of one shape spells the shape once and the run's bounds, so the
depth it tests is a number in the row rather than a listing."""

import re
import sys

REPEAT = re.compile(r"^// REPEAT (\d+) (\d+): (.*)$")

with open(sys.argv[1]) as fixture:
    for line in fixture:
        repeat = REPEAT.match(line.rstrip("\n"))
        if not repeat:
            sys.stdout.write(line)
            continue
        first, last, text = int(repeat[1]), int(repeat[2]), repeat[3]
        step = 1 if first <= last else -1
        for k in range(first, last + step, step):
            print(text.replace("{k+1}", str(k + 1)).replace("{k}", str(k)))
