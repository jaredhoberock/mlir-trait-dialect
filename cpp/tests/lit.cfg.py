import os
import lit.formats

config.name = "Trait Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

plugin_path = os.path.join(os.path.dirname(__file__), '..', 'libtrait_dialect.so')

config.substitutions.append(('mlir-opt', f'mlir-opt --load-dialect-plugin={plugin_path}'))
