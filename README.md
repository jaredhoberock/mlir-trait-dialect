# mlir-trait-dialect

An **MLIR** dialect that models *Rust‑style traits* (interfaces) and their implementations so that **polymorphic, generic algorithms** can live directly in intermediate representation.

---
## Why?
Modern C++/Rust libraries such as **Thrust**, **ranges**, or **CUTLASS** express powerful, reusable algorithms (e.g. `thrust::reduce`, `std::sort`) in terms of *type‑parametric* functions constrained by small interfaces—“the type `T` must support `Add`, must be totally ordered”, and so on.

To make such libraries *portable across tool‑chains* and *optimisation passes* we would like to lower them to MLIR **before** choosing a concrete type.  Encoding them as ordinary `func.func` would lose the information that they are *still polymorphic*.

`mlir-trait-dialect` solves this by:
1. Introducing a `trait.trait` operation that declares a trait and its required methods.
2. Introducing a `trait.impl` operation that provides an implementation for a concrete type.
3. Letting you write generic functions that call trait methods through `trait.method.call` (and friends).
4. Providing a lowering that *monomorphises* those calls on demand, ultimately emitting plain `LLVM` dialect with no dynamic dispatch.

---
## Tiny example
The snippet below defines `PartialEq`, gives `i32` an implementation, and defines a generic function `foo`.  After running the *Monomorphisation + LLVM lowering* pipeline the IR no longer contains trait ops—only concrete `llvm.func`s remain.

<details>
<summary>Click to expand MLIR</summary>

```mlir
// ----- Generic IR using trait dialect -----
module {
  // 1. Declare a trait
  trait.trait @PartialEq {
    // a required method
    func.func private @eq(!trait.self, !trait.self) -> i1

    // an optional method
    func.func @neq(%self: !trait.self, %other: !trait.self) -> i1 {
      %equal = trait.method.call @PartialEq<!trait.self>::@eq(%self, %other) : (!trait.self, !trait.self) -> i1
      %true = arith.constant 1 : i1
      %result = arith.xori %equal, %true : i1
      return %result : i1
    }
  }

  // 2. Implementation for i32
  trait.impl @PartialEq<i32> {
    func.func @eq(%self: i32, %other: i32) -> i1 {
      %result = arith.cmpi eq, %self, %other : i32
      return %result : i1
    }
  }

  // 3. Generic functions: rely on the trait, not a concrete type
  !T = !trait.poly<0,[@PartialEq]>
  func.func @foo(%a : !T, %b : !T) -> i1 {
    %0 = trait.method.call @PartialEq<!T>::@eq(%a, %b) : (!T, !T) -> i1
    return %0 : i1
  }

  // 4. Concrete functions: call a trait-bounded polymorphic function with a concrete type
  func.func @baz(%a : i32, %b : i32) -> i1 {
    %result = trait.func.call @foo(%a, %b) : (i32,i32) -> i1
    return %result : i1
  }
}
```
</details>

<details>
<summary>Lowered LLVM dialect (excerpt)</summary>

```mlir
module {
  llvm.func @__trait_PartialEq_impl_i32_eq(%arg0: i32, %arg1: i32) -> i1 {
    %0 = llvm.icmp "eq" %arg0, %arg1 : i32
    llvm.return %0 : i1
  }
  llvm.func @__trait_PartialEq_impl_i32_neq(%arg0: i32, %arg1: i32) -> i1 {
    %0 = llvm.call @__trait_PartialEq_impl_i32_eq(%arg0, %arg1) : (i32, i32) -> i1
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.xor %0, %1 : i1
    llvm.return %2 : i1
  }
  llvm.func @foo_i32(%arg0: i32, %arg1: i32) -> i1 {
    %0 = llvm.call @__trait_PartialEq_impl_i32_eq(%arg0, %arg1) : (i32, i32) -> i1
    llvm.return %0 : i1
  }
  llvm.func @baz(%arg0: i32, %arg1: i32) -> i1 {
    %0 = llvm.call @foo_i32(%arg0, %arg1) : (i32, i32) -> i1
    llvm.return %0 : i1
  }
}
```
</details>
