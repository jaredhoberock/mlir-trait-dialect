import os
import lit.formats

config.name = "Trait Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

llvm_bin_dir = '/home/jhoberock/dev/git/llvm-project-20/build/bin'
plugin_path = os.path.join(os.path.dirname(__file__), '..', 'libtrait_dialect.so')

config.substitutions.append(('opt', f'{os.path.join(llvm_bin_dir, "mlir-opt")} --load-dialect-plugin={plugin_path}'))
config.substitutions.append(('FileCheck', os.path.join(llvm_bin_dir, 'FileCheck')))
