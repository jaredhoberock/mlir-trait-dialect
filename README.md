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
// 1. Declare a trait. Its one block argument is the claim of its own
//    application.
!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @PartialEq(%self: !trait.claim<@PartialEq[!S, !O]>) {
  // a required method
  trait.method @eq(!S, !O) -> i1

  // an optional method with a default implementation
  trait.method @ne(%x: !S, %y: !O) -> i1 {
    // call a method through the trait's own claim
    %equal = trait.method.call %self @PartialEq[!S, !O]::@eq(%x, %y) : (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    trait.return %res : i1
  }
}

// 2. An implementation for i32. Its block arguments are the claim it
//    implements and one claim per where-clause entry (none here).
trait.impl private @PartialEq_i32(%self: !trait.claim<@PartialEq[i32, i32]>) {
  trait.method @eq(%x: i32, %y: i32) -> i1 {
    %result = arith.cmpi eq, %x, %y : i32
    trait.return %result : i1
  }
}

// 3. A generic function relies on the trait through a claim parameter, not on
//    a concrete type.
!T = !trait.poly<2>
func.func private @foo(%c: !trait.claim<@PartialEq[!T, !T]>, %a: !T, %b: !T) -> i1 {
  %res = trait.method.call %c @PartialEq[!T, !T]::@ne(%a, %b) : (!T, !T) -> i1
  return %res : i1
}

// 4. A concrete function calls it with a claim of the concrete application;
//    monomorphization proves the claim and instantiates the calls.
func.func @baz(%a: i32, %b: i32) -> i1 {
  %c = trait.allege @PartialEq[i32, i32]
  %res = trait.func.call @foo(%c, %a, %b) : (!trait.claim<@PartialEq[i32, i32]>, i32, i32) -> i1
  return %res : i1
}
```
</details>

<details>
<summary>Lowered to LLVM dialect</summary>

```mlir
llvm.func @PartialEq_hc1ee045a1cf60171_ne(%arg0: i32, %arg1: i32) -> i1 attributes {sym_visibility = "private"} {
  %0 = llvm.call @PartialEq_i32_hc2dcd4c42b04f237_eq(%arg0, %arg1) : (i32, i32) -> i1
  %1 = llvm.mlir.constant(true) : i1
  %2 = llvm.xor %0, %1 : i1
  llvm.return %2 : i1
}
llvm.func @PartialEq_i32_hc2dcd4c42b04f237_eq(%arg0: i32, %arg1: i32) -> i1 attributes {sym_visibility = "private"} {
  %0 = llvm.icmp "eq" %arg0, %arg1 : i32
  llvm.return %0 : i1
}
llvm.func @foo_hde61e1ab813906bd(%arg0: i32, %arg1: i32) -> i1 attributes {sym_visibility = "private"} {
  %0 = llvm.call @PartialEq_hc1ee045a1cf60171_ne(%arg0, %arg1) : (i32, i32) -> i1
  llvm.return %0 : i1
}
llvm.func @baz(%arg0: i32, %arg1: i32) -> i1 {
  %0 = llvm.call @foo_hde61e1ab813906bd(%arg0, %arg1) : (i32, i32) -> i1
  llvm.return %0 : i1
}
```
</details>
