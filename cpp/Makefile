LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-20/build/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = $(HOME)/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources
DIALECT_SOURCES := c_api.cpp Dialect.cpp Instantiation.cpp Ops.cpp Passes.cpp Types.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Dialect.hpp.inc Dialect.cpp.inc Ops.hpp.inc Ops.cpp.inc Types.hpp.inc Types.cpp.inc TypeInterfaces.hpp.inc TypeInterfaces.cpp.inc

.PHONY: all clean

all: libtrait_dialect.a libtrait_dialect.so

# TableGen rules
Dialect.hpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Dialect.cpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

Ops.hpp.inc: Ops.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

Ops.cpp.inc: Ops.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

Types.hpp.inc: Types.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

Types.cpp.inc: Types.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

TypeInterfaces.hpp.inc: TypeInterfaces.td
	$(TBLGEN) --gen-type-interface-decls $(INCLUDES) $< -o $@

TypeInterfaces.cpp.inc: TypeInterfaces.td
	$(TBLGEN) --gen-type-interface-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

libtrait_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libtrait_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared -undefined dynamic_lookup $^ -o $@

.PHONY: test
test: libtrait_dialect.so
	@echo "Running trait dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
