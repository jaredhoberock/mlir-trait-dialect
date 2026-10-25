# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "Trait Dialect Tests"
config.test_format = lit.formats.ShTest()
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

# Every tool and plugin path comes from the site configuration build.rs writes
# into the build's output directory (lit.site.cfg.py), which loads this file.
# The toolchain's bin leads a row's path, so a row inverting an exit status
# writes `not` and gets LLVM's.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.use_default_substitutions()
# The dialect plugins in the order the site configuration records them: each
# after the plugins whose symbols it references, which must be loaded first.
dialect_plugins = [f'--load-dialect-plugin={plugin}' for plugin in config.dialect_plugins]
llvm_config.add_tool_substitutions(
    [ToolSubst('mlir-opt', extra_args=dialect_plugins), 'mlir-runner'],
    [config.llvm_tools_dir])
# A row whose module repeats one declaration shape many times is written out by
# Inputs/expand_repeats.py before mlir-opt reads it; lit.llvm's default
# substitutions name the running Python as %python.
