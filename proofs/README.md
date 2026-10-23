# The evidence calculus of mlir-trait-dialect, in Lean 4

What the evidence operations of this dialect mean, and why each one is correct and needed, as machine-checked
theorems. The specification is `docs/plans/EVIDENCE_CALCULUS_SPEC.md` in the Screech repository, where the
estate keeps its documents: one card per operation (`trait.allege`, `trait.derive`,
`trait.project`, `trait.witness`, `trait.coerce`, `trait.proof`) with its typing rule, its soundness case and its
independence lemma, each card naming the Lean theorem that proves it. The object is the operations as their
descriptions and verifiers define them, not the C++.

Build: Lean 4.35 and Lake 5 (`elan` installs both from `lean-toolchain`); standard library only.

    lake build

The files, in import order: `Defs.lean` (types, signature, evidence terms), `Logic.lean` (the trait logic and the
denotation), `Calculus.lean` (the typing rules, one per operation, and the reading of a term to its root),
then one file per theorem family: `Soundness.lean` (theorem 3), `Completeness.lean` (theorems 4 and 4b: `trait.allege`
is a cut), `Coherence.lean` (theorem 5), `Minimality.lean` (theorem 6: the independence lemmas, `witness compose`
carried by `coerce`, `trait.proof` as the `let`), `Erasure.lean` (theorem 7), `Findings.lean` (machine-checked
counterexamples to the statements that had to be weakened).

Proved as the specification states them: soundness, completeness and the cut, the five independence lemmas, the
`let` theorem, the inversion theorem for a projection past the trait's requirements. Proved weakened: erasure and
coherence hold at the limit of the resolution fuel, for primitive derivations whose projections resolve, under an
oracle that is sound and complete and a constructor law that forgets no argument; `Findings.lean` and
`Erasure.lean` refute the stated forms without those hypotheses. One lemma stands unproved and unused,
`Trait.req_admissible` in `Logic.lean` (the specification's (REQ-ADMISSIBLE)); no theorem depends on it.
`#print axioms` on every theorem lists only `propext`, `Classical.choice` and `Quot.sound`.
