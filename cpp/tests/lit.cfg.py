# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import os
import lit.formats

config.name = "Trait Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

plugin_path = os.path.join(os.path.dirname(__file__), '..', 'libtrait_dialect.so')

llvm_bin = os.path.join(os.path.expanduser('~'), 'dev/git/llvm-project-22/build/bin')
mlir_opt = os.path.join(llvm_bin, 'mlir-opt')
config.substitutions.append(('mlir-opt', f'{mlir_opt} --load-dialect-plugin={plugin_path}'))
