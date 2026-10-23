import Calculus
/-!
# Soundness: theorem 3 (SPEC.md 6.1)

An allege-free evidence term's claim is valid in the trait logic under the predicates of its
hypotheses. Hypotheses: H1 (built into `Ty`), H2 (`ProofsWF`), G-WF (`GammaWF`).

Proof plan from SPEC.md 6.1: induction on the typing derivation. HYP by (HYP). DERIVE by (IMPL), each
operand's predicate by induction. PROJECT-REQ by (REQ). PROJECT-WHERE by inversion on the cited proof's
derive (H2). REQ-CALL by (REQ-A). WIT-APP by H2 and weakening (`Deriv.weakenNil`), or (IMPL) with no
premises. WIT-PROJ by (BIND). WIT-REFL by (REFL). WIT-COMPOSE and the equality arm of COERCE by the
closure lemma. COERCE on an application claim by (CONV) with the closure lemma at each argument.
ALLEGE has no case.

The induction is `hasType_valid` in Calculus.lean, the shared form of theorem 3 for any impl
restriction `Ok` (Coherence.lean needs it at `DenL`); `soundness` is its instance, relaxed to `AnyImpl`.
Where the proof departs from the plan:
* WIT-APP of a proof is not a sub-derivation of the term's typing: the proof body is typed on its own
  in the proof table. The proof therefore runs an outer strong induction on a bound on the proofs
  cited (H2's clause "p cites q only for q < p" makes every citation chain descend) and an inner
  induction on the typing derivation, carrying the fact that every proof citation on a claim of the
  derivation is below the bound, which PROJECT-WHERE needs.
* Neither `DeclWF` nor G-WF is used. Soundness speaks of the predicate of a claim, never of its
  citation: PROJECT-WHERE concludes the operand's own closed type, valid whatever the citation names,
  and a proven hypothesis is used by (HYP) alone. G-WF matters where the root of a term is read
  (theorem 5). This is recorded as a finding in the report.
* `entails_of_equational` is false as the skeleton stated it: its hypothesis `hEq` (every derivation
  under the seeds is primitive) does not exclude (BIND), whose equality the closure never knows, and a
  signature with one unconditional impl binding an associated type satisfies `hEq`. The theorem is
  weakened to the docstring's own words, a derivation by the equality rules alone (`Deriv.Equational`),
  and the refutation of the original statement is `entails_of_equational_as_stated_fails`.
-/

namespace Trait

/-! ## Theorem 3 -/

set_option linter.unusedVariables false in
/-- Theorem 3, SOUNDNESS: `H1, H2, G-WF. e allege-free, M; G |- e : C => S; pred(G) |- |C|`.
`hS` and `hΓ` are the specification's hypotheses; the proof needs neither (see the header). -/
theorem soundness (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim} {e : Ev}
    {c : Claim} (hS : M.sig.DeclWF) (hP : ProofsWF L M Ok) (hΓ : GammaWF M Γ)
    (he : e.allegeFree) (ht : HasType L M Ok Γ e c) : Valid L M.sig (Γ.map Claim.pred) c.pred :=
  valid_relax (hasType_valid L M Ok hS hP hΓ he ht)

/-! ## The converse of the closure lemma -/

/-- A derivation by the equality rules alone: (HYP) on a seed, (REFL), (SYM), (TRANS), (CONG),
(CONG-P), (NORM). (IMPL), (REQ), (REQ-A), (BIND) and (CONV) read the signature or derive an
application and are not equality rules. -/
def Deriv.Equational : Deriv L S Ok Γ P -> Prop
  | .hyp _ _ => True
  | .impl _ _ _ _ _ _ => False
  | .req _ _ _ => False
  | .reqA _ _ _ _ _ _ => False
  | .bind _ _ _ _ _ _ _ _ => False
  | .refl => True
  | .symm d => d.Equational
  | .trans d₁ d₂ => d₁.Equational ∧ d₂.Equational
  | .cong _ ds => ∀ j hj, (ds j hj).Equational
  | .congP _ _ ds es => (∀ j hj, (ds j hj).Equational) ∧ (∀ j hj, (es j hj).Equational)
  | .norm _ => True
  | .conv _ _ _ => False

/-- The closure lemma's converse, weakened from the skeleton's statement (SPEC.md 6.1, "the closure
decides this fragment"): an equality derived from the seeds by the equality rules alone is a closure
fact. The hypotheses of the derivation are exactly the seeds. The skeleton asked instead that every
derivation under the seeds be primitive, which (BIND) satisfies while the closure reads no binding;
`entails_of_equational_as_stated_fails` is that refutation. -/
theorem entails_of_equational (L : ConLaw) (S : Sig) (E : List (Ty × Ty)) {a b : Ty}
    (d : Deriv L S AnyImpl (E.map fun lr => Pred.eq lr.1 lr.2) (.eq a b))
    (hd : d.Equational) : Entails L E a b := by
  suffices h : ∀ {P} (d : Deriv L S AnyImpl (E.map fun lr => Pred.eq lr.1 lr.2) P), d.Equational →
      ∀ a b, P = .eq a b → Entails L E a b from h d hd a b rfl
  intro P d
  induction d with
  | hyp k hk =>
    intro _ a b hP
    simp only [List.getElem_map] at hP
    cases hP
    exact Entails.ax (List.getElem_mem (by simpa using hk))
  | refl => intro _ a b hP; cases hP; exact .refl
  | symm _ ih => intro hd a b hP; cases hP; exact .symm (ih hd _ _ rfl)
  | trans _ _ ih₁ ih₂ => intro hd a b hP; cases hP; exact .trans (ih₁ hd.1 _ _ rfl) (ih₂ hd.2 _ _ rfl)
  | cong hlen _ ih => intro hd a b hP; cases hP; exact .con hlen fun i hi => ih i hi (hd i hi) _ _ rfl
  | congP hlen hlen' _ _ ih₁ ih₂ =>
    intro hd a b hP; cases hP
    exact .proj hlen hlen' (fun i hi => ih₁ i hi (hd.1 i hi) _ _ rfl) (fun i hi => ih₂ i hi (hd.2 i hi) _ _ rfl)
  | norm h => intro _ a b hP; cases hP; exact .law h
  | impl => intro hd; exact hd.elim
  | req => intro hd; exact hd.elim
  | reqA => intro hd; exact hd.elim
  | bind => intro hd; exact hd.elim
  | conv => intro hd; exact hd.elim

/-! ### The refutation of the statement as the skeleton gave it

The signature: one trait with no plain requirements, no binders and one associated type; one impl of
it, unconditional, binding that type to another constructor. No derivation under it reads a
requirement, so every derivation is primitive; (BIND) derives `T[K0]::N = K1` closed, and the closure
with no seeds and the free algebra relates nothing to anything but itself. The dialect's program is
card 4's WIT-PROJ independence case: `impl A for i64 { Item = i32 }` with `G = []` derives
`A[i64]::Item = i32`, which `compose` and `coerce` cannot (Q/p26, Q/p30, Q/p34 refused). -/

/-- One trait: arity 1, no requirements, no binders, one associated type with no parameters. -/
def refuteTrait : TraitDecl := ⟨1, [], [], [[]]⟩

/-- One impl: unconditional, header `T[K0]`, binding the associated type to `K1`. -/
def refuteImpl : ImplDecl := ⟨0, 0, [.con 0 []], [], [some ([], .con 1 [])], [], []⟩

def refuteSig : Sig := ⟨[refuteTrait], [refuteImpl]⟩

/-- Under `refuteSig` every closed derivation is primitive: no (REQ) or (REQ-A) can fire, the trait
having no requirements and no binders. -/
theorem refuteSig_all_primitive {L : ConLaw} {Ok : ImplId -> Subst -> Prop} :
    ∀ {P} (d : Deriv L refuteSig Ok [] P), d.Primitive := by
  intro P d
  induction d with
  | hyp k hk => exact absurd hk (Nat.not_lt_zero k)
  | impl _ _ _ _ _ _ ih => exact ih
  | req _ ht hk _ => exfalso; simp [refuteSig, refuteTrait] at hk
  | reqA _ ht hb _ _ _ _ _ => exfalso; simp [refuteSig, refuteTrait] at hb
  | bind _ _ _ _ _ _ _ _ ih => exact ih
  | refl => trivial
  | symm _ ih => exact ih
  | trans _ _ ih₁ ih₂ => exact ⟨ih₁, ih₂⟩
  | cong _ _ ih => exact ih
  | congP _ _ _ _ ih₁ ih₂ => exact ⟨ih₁, ih₂⟩
  | norm _ => trivial
  | conv _ _ _ ih ihs => exact ⟨ih, ihs⟩

/-- With no seeds and the free algebra, the closure is identity. -/
theorem entails_nil_free {a b : Ty} (h : Entails freeAlgebra [] a b) : a = b := by
  induction h with
  | ax h => nomatch h
  | refl => rfl
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | con hlen _ ih =>
    congr 1
    exact List.ext_getElem hlen fun i h₁ h₂ => by
      have := ih i h₁
      rwa [getD_eq_getElem_of_lt _ h₁, getD_eq_getElem_of_lt _ h₂] at this
  | proj hlen hlen' _ _ ih₁ ih₂ =>
    congr 1
    · exact List.ext_getElem hlen fun i h₁ h₂ => by
        have := ih₁ i h₁
        rwa [getD_eq_getElem_of_lt _ h₁, getD_eq_getElem_of_lt _ h₂] at this
    · exact List.ext_getElem hlen' fun i h₁ h₂ => by
        have := ih₂ i h₁
        rwa [getD_eq_getElem_of_lt _ h₁, getD_eq_getElem_of_lt _ h₂] at this
  | law h => obtain ⟨c, as, rfl, rfl⟩ := h; rfl

/-- The skeleton's `entails_of_equational` is false: (BIND) under `refuteSig` derives
`T[K0]::N = K1` from no seeds, every derivation there is primitive, and the closure from no seeds
relates `T[K0]::N` to nothing but itself. -/
theorem entails_of_equational_as_stated_fails :
    ¬ ∀ (L : ConLaw) (S : Sig) (E : List (Ty × Ty)) {a b : Ty}
        (_d : Deriv L S AnyImpl (E.map fun lr => Pred.eq lr.1 lr.2) (.eq a b))
        (_hEq : ∀ {P} (d' : Deriv L S AnyImpl (E.map fun lr => Pred.eq lr.1 lr.2) P), d'.Primitive),
        Entails L E a b := by
  intro H
  have d : Deriv freeAlgebra refuteSig AnyImpl (([] : List (Ty × Ty)).map fun lr => Pred.eq lr.1 lr.2)
      (.eq (.proj 0 [.con 0 []] 0 []) (.con 1 [])) :=
    Deriv.bind (L := freeAlgebra) (S := refuteSig) (Ok := AnyImpl) (a := 0) (labels := []) (B := .con 1 [])
      (gs := []) 0 (by decide) [] rfl trivial (fun j hj => absurd hj (Nat.not_lt_zero j)) rfl rfl
  have h := H freeAlgebra refuteSig [] d (fun {_} d' => refuteSig_all_primitive d')
  exact absurd (entails_nil_free h) (by simp)

end Trait
