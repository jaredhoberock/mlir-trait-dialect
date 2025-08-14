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
// 1. Declare a trait
!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait @PartialEq[!S,!O] {
  // a required method
  func.func private @eq(!S, !O) -> i1

  // an optional method with default implementation
  func.func @ne(%self: !S, %other: !O) -> i1 {
    // get a claim value for this trait
    %partial_eq = trait.assume @PartialEq[!S,!O]

    // call a method using the claim
    %equal = trait.method.call %partial_eq @PartialEq[!S,!O]::@eq(%self, %other)
      :  (!S, !O) -> i1
      as (!S, !O) -> i1

    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    return %res : i1
  }
}

// 2. Implementation for i32
trait.impl for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %result = arith.cmpi eq, %self, %other : i32
    return %result : i1
  }
}

// 3. Generic functions: rely on the trait, not a concrete type
!T = !trait.poly<2>
!C = !trait.claim<@PartialEq[!T,!T]>
func.func @foo(%a : !T, %b : !T, %c: !C) -> i1 {
  // use our polymorphic claim value to call @eq
  %res = trait.method.call %c @PartialEq[!T,!T]::@eq(%a, %b)
    :  (!S, !O) -> i1
    as (!T, !T) -> i1

  return %res : i1
}

// 4. Concrete functions: call a trait-bounded polymorphic function with a concrete type
func.func @baz(%a : i32, %b : i32) -> i1 {
  // get a monomorphic claim for @PartialEq[i32,i32]
  %c = trait.allege @PartialEq[i32,i32]

  // call polymorphic @foo using our claim
  %res = trait.func.call @foo(%a, %b, %c)
    :  (!T,!T,!C) -> i1
    as (i32,i32, !trait.claim<@PartialEq[i32,i32]>) -> i1

  return %res : i1
}
```
</details>

<details>
<summary>Lowered to LLVM dialect</summary>

```mlir
llvm.func @PartialEq_impl_i32_i32_eq(%arg0: i32, %arg1: i32) -> i1 {
  %0 = llvm.icmp "eq" %arg0, %arg1 : i32
  llvm.return %0 : i1
}
llvm.func @foo_i32(%arg0: i32, %arg1: i32) -> i1 {
  %0 = llvm.call @PartialEq_impl_i32_i32_eq(%arg0, %arg1) : (i32, i32) -> i1
  llvm.return %0 : i1
}
llvm.func @baz(%arg0: i32, %arg1: i32) -> i1 {
  %0 = llvm.call @foo_i32(%arg0, %arg1) : (i32, i32) -> i1
  llvm.return %0 : i1
}
```
</details>
