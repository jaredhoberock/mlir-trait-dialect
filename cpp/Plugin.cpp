#include "Dialect.hpp"
#include "Passes.hpp"
#include <mlir/Pass/PassManager.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>
#include <mlir/Tools/Plugins/PassPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::trait::TraitDialect>();
  ::mlir::PassRegistration<::mlir::trait::MonomorphizePass>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "TraitDialectPlugin", 
    "v0.1", 
    registerPlugin
  };
}
