import Calculus
/-!
# Erasure: theorem 7 (SPEC.md 6.6)

A closed valid equality relates two written forms of one type, so a `trait.coerce` on a value changes
nothing the value carries, and every claim value carries nothing.

Proof plan from SPEC.md 6.6: induction on the derivation. (REFL)(SYM)(TRANS) are equality's;
(CONG)(CONG-P) because `denote` is a congruence; (BIND) is the definition's projection clause with
uniqueness of the server (`Coherent`, `SelSound`; H4 gives the server exists); (NORM) because the laws
are context-free; (HYP) vacuous at `[]`; (REQ) absent under H3.

The induction is `erasure_equiv` in Logic.lean, with the denotation at the limit of the fuel (`Dn`:
some fuel reaches the answer, and then every larger fuel does, `denote_mono`) and the oracle notions
built on it; `erasure_core` there is theorem 7's shared statement and `erasure` below its instance.
The theorems are proved under four hypotheses section 7 did not state. The first three are each forced by a
refutation of section 7's statement without them, written out at the end of this file
(`ErasureAsStated` is that statement, in the skeleton's fixed-fuel vocabulary, kept here for that
purpose alone); the fourth by the instances its bullet names. The report records them as findings.
* The denotation at one fixed fuel is not a congruence: an impl argument that denotes at fuel `f` sits
  deeper inside the impl's binding than at the application, so `Coherent` at fuel `f` says nothing
  about it there (`fixed_fuel_erasure_fails`). The limit denotation is what SPEC.md 6.3 describes
  ("to a fixed point"); the stage's bound of 128 is a stage fact (F16).
* `SelSound` constrains the oracle's answers only; an oracle that declines a served application leaves
  `[[T[i32]::Out]]` undefined while (BIND) proves it equal to `i32` (`erasure_needs_sel_complete`).
  `SelCompleteL` is the clause H4 lacks: the oracle answers every served ground application, which
  `resolveImplFor` does (D/ImplResolution.cpp:172-401) unless it refuses the program.
* (REQ) reads an impl's requirement off its header, and H3 discharges that requirement with a
  derivation in the full signature (`ValidP` at `AnyImpl`), which may cite impls whose headers do not
  denote; the (REQ-ADMISSIBLE) argument of SPEC.md 4.3 therefore does not stay inside the denotable
  fragment, and the equality a (REQ) concludes can be false under `[[.]]` (`erasure_needs_primitive`).
  The theorems take a primitive derivation (`Deriv.Primitive`), where (REQ) and (REQ-A) have no
  instance at `G = []`.
* `[[.]]` resolves ground projections only and fails where a GAT argument or a law's argument does not
  denote, while (BIND) and (NORM) prove equalities through such types. Three instances, each a valid
  closed equality in the denotable fragment with one side denoting and the other not or differently:
  `T[a_3]::Out = a_3` by (BIND) through a generic impl binding `Out = Self` at the variable `a_3`, where
  `[[T[a_3]::Out]]` is the standing projection; `T[i32]::Out<u> = i64` by (BIND) through a binding
  `Out<X> = i64` at a GAT argument `u` no impl serves, where `[[i64]]` is defined and the left side is
  not; `K(u) = i32` by (NORM) under a law `K(x) = i32` for every `x`, at the same `u`. `Deriv.Resolvable`
  keeps every (BIND) at a ground application whose GAT arguments denote, and `ConLaw.KeepsArgs` says a
  law forgets no argument. These are what the exit walk enforces on the program (D/Passes.cpp:1962-1990,
  standing ground projections refused), stated on the derivation and the law.

`Sig.DeclWF` and `Sig.ReqDuty` are not needed once the derivation is primitive; (REQ-ADMISSIBLE) was
H3's only role in theorem 7.
-/

namespace Trait

/-! ## Groundness -/

/-- The substitution that closes every variable. -/
def closeAll : Subst := fun _ => some (.con 0 [])

mutual
  theorem Ty.subst_closeAll_ground : ∀ t : Ty, (t.subst closeAll).ground = true
    | .var _ => by simp [Ty.subst, closeAll, Ty.ground, groundList]
    | .con c as => by simp [Ty.subst, Ty.ground, substList_closeAll_ground as]
    | .proj t as a gs => by
      simp [Ty.subst, Ty.ground, substList_closeAll_ground as, substList_closeAll_ground gs]
  theorem substList_closeAll_ground : ∀ as : List Ty, groundList (substList closeAll as) = true
    | [] => rfl
    | a :: as => by simp [substList, groundList, Ty.subst_closeAll_ground a, substList_closeAll_ground as]
end

/-- A constructor law applied to ground arguments is ground: stability under substitution (H5) fixes
the rebuilt type under the substitution that closes every variable. -/
theorem ConLaw.rebuild_ground (L : ConLaw) (c : ConId) (as : List Ty) (h : groundList as = true) :
    (L.rebuild c as).ground = true := by
  have h1 := L.substOk closeAll c as
  rw [substList_of_ground closeAll as h] at h1
  rw [← h1]
  exact Ty.subst_closeAll_ground _

/-! ## The theorems -/

section Theorems
variable {sel : Sel} {L : ConLaw} {S : Sig}

/-- Theorem 7, ERASURE: `Coherent(S), H1, H3, H4, H5. S; [] |- a = b in the denotable fragment, [[a]]
defined => [[b]] defined and [[a]] = [[b]]`, stated for the denotation at the limit of the fuel, a
primitive derivation whose (BIND)s `[[.]]` resolves, an oracle that is sound and complete, and a
constructor law that forgets no argument (the file's header explains each). `Sig.DeclWF` and
`Sig.ReqDuty` are not needed. The instance of `erasure_core`. -/
theorem erasure (sel : Sel) (L : ConLaw) (S : Sig) {a b : Ty} (hL : L.KeepsArgs sel S)
    (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S) (hC : S.CoherentL sel L)
    (h : ValidGround sel L S (.eq a b)) (d : Ty) (ha : Dn sel L S a d) : Dn sel L S b d :=
  (erasure_core sel L S hL hSel hSelC hC h d).mp ha


/-- The closure lemma under the denotation: a closure fact over closed valid equalities relates
types that denote alike. -/
theorem entails_dnEquiv (hL : L.KeepsArgs sel S) (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S)
    (hC : S.CoherentL sel L) {E : List (Ty × Ty)} (hE : ∀ l r, (l, r) ∈ E -> ValidGround sel L S (.eq l r))
    {a b : Ty} (h : Entails L E a b) : DnEquiv sel L S a b := by
  induction h with
  | ax hab =>
    obtain ⟨der, hp, hr⟩ := hE _ _ hab
    exact erasure_equiv hL hSel hSelC hC der hp hr _ _ rfl
  | refl => exact DnEquiv.refl _
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | con hl _ ih => exact DnEquiv_con (All2.of_getD hl ih)
  | proj hl₁ hl₂ _ _ ih₁ ih₂ => exact DnEquiv_proj (All2.of_getD hl₁ ih₁) (All2.of_getD hl₂ ih₂)
  | law hlaw => exact DnEquiv_law hL hlaw

/-- The value arm of `trait.coerce` erases: a closure fact over closed valid equalities relates two
written forms of one type, so the barrier may forward the input (D/Passes.cpp:2108-2160;
T/erase_coerce_barrier.mlir). Hypotheses as in `erasure`. -/
theorem coerce_value_erases (sel : Sel) (L : ConLaw) (S : Sig) {a b : Ty} (hL : L.KeepsArgs sel S)
    (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S) (hC : S.CoherentL sel L) (E : List (Ty × Ty))
    (hE : ∀ l r, (l, r) ∈ E -> ValidGround sel L S (.eq l r)) (h : Entails L E a b)
    (d : Ty) (ha : Dn sel L S a d) : Dn sel L S b d :=
  (entails_dnEquiv hL hSel hSelC hC hE h d).mp ha

/-- Every claim value carries nothing: the erasure of `witness`, `derive`, `project` and `proof`
(D/Passes.cpp:2091-2106) is sound because the typing judgment speaks of claims alone. Stated as the
corollary that two applications whose arguments are closed-valid equal denote alike, so there is no
value to forward and no instance to tell apart. Hypotheses as in `erasure`; the original's one-sided
`isSome` premise is not needed. -/
theorem claim_denotes (sel : Sel) (L : ConLaw) (S : Sig) {t : TraitId} {as bs : List Ty}
    (hL : L.KeepsArgs sel S) (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S) (hC : S.CoherentL sel L)
    (hab : as.length = bs.length)
    (h : ∀ i, i < as.length -> ValidGround sel L S (.eq (as.getD i default) (bs.getD i default))) :
    ∀ Q, DnP sel L S (.app t bs) Q ↔ DnP sel L S (.app t as) Q := by
  intro Q
  have hall : All2 (DnEquiv sel L S) as bs := All2.of_getD hab fun i hi =>
    let ⟨der, hp, hr⟩ := h i hi
    erasure_equiv hL hSel hSelC hC der hp hr _ _ rfl
  rw [DnP_app_iff, DnP_app_iff]
  constructor
  · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr hall cs).mpr hcs, rfl⟩
  · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr hall cs).mp hcs, rfl⟩

end Theorems

/-! ## Refutations of the statement of section 7 at a fixed fuel

Each takes section 7's statement of `erasure` with its hypotheses and instantiates it at a signature
and an oracle that satisfy them (`Sig.DeclWF`, `Sig.ReqDuty`, `SelSound`, `Sig.Coherent` are proved for
the instance; H1 is the grammar; H5 is `freeAlgebra`), where the conclusion is false. -/

/-! ### The fixed-fuel vocabulary of the section 7 skeleton

The skeleton indexed the oracle notions by one fuel. They are stated here, and nowhere else, so that
the refuted proposition can be written as section 7 wrote it. -/

/-- H4 at one fuel: an impl citation whose header denotes at that fuel. -/
def Den (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) : ImplId -> Subst -> Prop :=
  fun i σ => ∃ hi : i < S.impls.length, (denotePred sel L S fuel (S.impls[i].header σ)).isSome

/-- `S; G |- P` in the fixed-fuel denotable fragment. -/
abbrev ValidD (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) (Γ : List Pred) (P : Pred) : Prop :=
  Nonempty (Deriv L S (Den sel L S fuel) Γ P)

/-- `ServedBy` at one fuel. -/
def ServedBy (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) (P : Pred) (i : ImplId) (σ : List Ty) : Prop :=
  ∃ hi : i < S.impls.length, σ.length = S.impls[i].arity ∧ (denotePred sel L S fuel P).isSome ∧
    denotePred sel L S fuel (S.impls[i].header (Subst.ofList σ)) = denotePred sel L S fuel P ∧
    (∀ W ∈ S.impls[i].wheres, ValidD sel L S fuel [] (W.subst (Subst.ofList σ)))

/-- The oracle answers only with a server, at one fuel. -/
def SelSound (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) : Prop :=
  ∀ P i σ, sel P = some (i, σ) -> ServedBy sel L S fuel P i σ

/-- `Coherent(S)` at one fuel. -/
def Sig.Coherent (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) : Prop :=
  ∀ t as, as.all Ty.ground -> (denotePred sel L S fuel (.app t as)).isSome ->
    ∀ i σ j σ', ServedBy sel L S fuel (.app t as) i σ -> ServedBy sel L S fuel (.app t as) j σ' ->
      i = j ∧ σ.map (denote sel L S fuel) = σ'.map (denote sel L S fuel)

/-- Section 7's statement of theorem 7, as a proposition. -/
def ErasureAsStated : Prop :=
  ∀ (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) (a b : Ty), S.DeclWF -> S.ReqDuty L ->
    SelSound sel L S fuel -> S.Coherent sel L fuel -> ValidD sel L S fuel [] (.eq a b) ->
    ∀ d, denote sel L S fuel a = some d -> denote sel L S fuel b = some d

/-- One trait `T` (0) with one associated type `Out` (0) and no requirement; one impl of it at `i32`
(`con 0`), binding `Out` to `i32`. -/
def oneImplSig : Sig :=
  { traits := [{ arity := 1, reqs := [], binders := [], assocs := [[]] }],
    impls := [{ arity := 0, trait := 0, head := [.con 0 []], wheres := [], bindings := [some ([], .con 0 [])],
                returns := [], binderReturns := [] }] }

theorem oneImplSig_declWF : oneImplSig.DeclWF := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t T hT r hr
    cases t with
    | zero => simp [oneImplSig] at hT; subst hT; simp at hr
    | succ t => simp [oneImplSig] at hT
  · intro T hT B hB; simp [oneImplSig] at hT; subst hT; simp at hB
  · intro T hT X hX; simp [oneImplSig] at hT; subst hT; simp at hX; subst hX; simp
  · intro I hI; simp [oneImplSig] at hI; subst hI; exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  · intro I hI; simp [oneImplSig] at hI; subst hI
    refine ⟨?_, ?_, ?_⟩
    · intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩
    · intro W hW; simp at hW
    · intro T hT a labels B hb
      cases a with
      | zero =>
        simp at hb
        obtain ⟨rfl, rfl⟩ := hb
        simp [oneImplSig] at hT; subst hT
        exact ⟨[], rfl, rfl, List.nodup_nil, by simp, by decide, by decide⟩
      | succ a => simp at hb
  · intro I hI v hv; simp [oneImplSig] at hI; subst hI; simp at hv

theorem oneImplSig_reqDuty : oneImplSig.ReqDuty freeAlgebra := by
  refine ⟨?_, ?_⟩
  · intro i hi ht k hk
    have : i = 0 := by simp [oneImplSig] at hi; omega
    subst this
    simp [oneImplSig] at hk
  · intro i hi ht b hb
    have : i = 0 := by simp [oneImplSig] at hi; omega
    subst this
    simp [oneImplSig] at hb

/-- The oracle that declines everything satisfies `SelSound`. -/
theorem declining_sound (L : ConLaw) (S : Sig) (fuel : Nat) : SelSound (fun _ => none) L S fuel := by
  intro P i σ h; cases h

theorem oneImplSig_coherent : oneImplSig.Coherent (fun _ => none) freeAlgebra 2 := by
  intro t as _ _ i σ j σ' ⟨hi, hσ, _⟩ ⟨hj, hσ', _⟩
  have hi0 : i = 0 := by simp [oneImplSig] at hi; omega
  have hj0 : j = 0 := by simp [oneImplSig] at hj; omega
  subst hi0; subst hj0
  simp [oneImplSig] at hσ hσ'
  subst hσ; subst hσ'
  exact ⟨rfl, rfl⟩

/-- `i32 = T[i32]::Out`, by (SYM) of (BIND) through the one impl; its header denotes at fuel 2. -/
def oneImplDeriv : Deriv freeAlgebra oneImplSig (Den (fun _ => none) freeAlgebra oneImplSig 2) []
    (.eq (.con 0 []) (.proj 0 [.con 0 []] 0 [])) :=
  .symm (.bind (a := 0) (labels := []) (B := .con 0 []) (gs := []) 0 (by decide) [] rfl
    ⟨by decide, by decide⟩ (fun j hj => absurd hj (by simp [oneImplSig])) rfl rfl)

/-- Under an oracle that declines `T[i32]`, `[[i32]]` is defined and `[[T[i32]::Out]]` is not, while
`i32 = T[i32]::Out` is valid closed in the denotable fragment: section 7's statement needs the oracle
to answer every served ground application (`SelCompleteL`). -/
theorem erasure_needs_sel_complete : ¬ ErasureAsStated := by
  intro H
  have h := H (fun _ => none) freeAlgebra oneImplSig 2 (.con 0 []) (.proj 0 [.con 0 []] 0 [])
    oneImplSig_declWF oneImplSig_reqDuty (declining_sound _ _ _) oneImplSig_coherent ⟨oneImplDeriv⟩
    (.con 0 []) rfl
  have hnone : denote (fun _ => none) freeAlgebra oneImplSig 2 (.proj 0 [.con 0 []] 0 []) = none := rfl
  rw [hnone] at h
  cases h

/-- Two traits: `T` (0) with `Out` (0), `U` (1) with `Item` (0). A generic impl of `T` binding
`Out = K(K(Self))` (`K` is `con 1`); impls of `U` at `i64` (`con 2`) and at `i32` (`con 0`), both
binding `Item = i32`. -/
def fuelSig : Sig :=
  { traits := [{ arity := 1, reqs := [], binders := [], assocs := [[]] },
               { arity := 1, reqs := [], binders := [], assocs := [[]] }],
    impls := [{ arity := 1, trait := 0, head := [.var 0], wheres := [],
                bindings := [some ([], .con 1 [.con 1 [.var 0]])], returns := [], binderReturns := [] },
              { arity := 0, trait := 1, head := [.con 2 []], wheres := [], bindings := [some ([], .con 0 [])],
                returns := [], binderReturns := [] },
              { arity := 0, trait := 1, head := [.con 0 []], wheres := [], bindings := [some ([], .con 0 [])],
                returns := [], binderReturns := [] }] }

/-- The oracle that answers the three applications the refutation reaches with their one server. -/
def fuelSel : Sel
  | .app 0 [.con 0 []] => some (0, [.con 0 []])
  | .app 1 [.con 2 []] => some (1, [])
  | .app 1 [.con 0 []] => some (2, [])
  | _ => none

/-- `U[U[i64]::Item]::Item`: denotes to `i32` at fuel 3 and at no smaller fuel. -/
def deepTy : Ty := .proj 1 [.proj 1 [.con 2 []] 0 []] 0 []

theorem fuelSig_declWF : fuelSig.DeclWF := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t T hT r hr
    match t, hT with
    | 0, hT => simp [fuelSig] at hT; subst hT; simp at hr
    | 1, hT => simp [fuelSig] at hT; subst hT; simp at hr
    | t+2, hT => simp [fuelSig] at hT
  · intro T hT B hB; simp [fuelSig] at hT; rcases hT with rfl | rfl <;> simp at hB
  · intro T hT X hX; simp [fuelSig] at hT; rcases hT with rfl | rfl <;> simp at hX <;> subst hX <;> simp
  · intro I hI; simp [fuelSig] at hI; rcases hI with rfl | rfl | rfl <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  · intro I hI; simp [fuelSig] at hI
    rcases hI with rfl | rfl | rfl <;> refine ⟨?_, ?_, ?_⟩
    all_goals first
      | (intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; done)
      | (intro T hT a labels B hb
         cases a with
         | zero =>
           simp at hb
           obtain ⟨rfl, rfl⟩ := hb
           simp [fuelSig] at hT; subst hT
           exact ⟨[], rfl, rfl, List.nodup_nil, by simp, by decide, by decide⟩
         | succ a => simp at hb)
  · intro I hI v hv; simp [fuelSig] at hI
    rcases hI with rfl | rfl | rfl
    · simp at hv; subst hv; left; decide
    · simp at hv
    · simp at hv

theorem fuelSig_reqDuty : fuelSig.ReqDuty freeAlgebra := by
  refine ⟨?_, ?_⟩
  · intro i hi ht k hk
    have : i = 0 ∨ i = 1 ∨ i = 2 := by simp [fuelSig] at hi; omega
    rcases this with rfl | rfl | rfl <;> simp [fuelSig] at hk
  · intro i hi ht b hb
    have : i = 0 ∨ i = 1 ∨ i = 2 := by simp [fuelSig] at hi; omega
    rcases this with rfl | rfl | rfl <;> simp [fuelSig] at hb

theorem fuelSel_sound : SelSound fuelSel freeAlgebra fuelSig 4 := by
  intro P i σ h
  unfold fuelSel at h
  split at h <;> cases h
  · exact ⟨by decide, by decide, by decide, by decide, fun W hW => by simp [fuelSig] at hW⟩
  · exact ⟨by decide, by decide, by decide, by decide, fun W hW => by simp [fuelSig] at hW⟩
  · exact ⟨by decide, by decide, by decide, by decide, fun W hW => by simp [fuelSig] at hW⟩

theorem fuelSig_coherent : fuelSig.Coherent fuelSel freeAlgebra 4 := by
  intro t as _ _ i σ j σ' ⟨hi, hσ, _, heq, _⟩ ⟨hj, hσ', _, heq', _⟩
  have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by
    have : i < 3 := by simpa [fuelSig] using hi
    match i, this with
    | 0, _ => simp
    | 1, _ => simp
    | 2, _ => simp
  have hj3 : j = 0 ∨ j = 1 ∨ j = 2 := by
    have : j < 3 := by simpa [fuelSig] using hj
    match j, this with
    | 0, _ => simp
    | 1, _ => simp
    | 2, _ => simp
  -- a nullary impl's header denotes to itself; the generic impl's header at `[s]` denotes as `s` does
  have h1 : ∀ τ : List Ty, τ.length = 0 ->
      denotePred fuelSel freeAlgebra fuelSig 4 (fuelSig.impls[1].header (Subst.ofList τ)) = some (.app 1 [.con 2 []]) := by
    intro τ hτ; rw [List.length_eq_zero_iff] at hτ; subst hτ; decide
  have h2 : ∀ τ : List Ty, τ.length = 0 ->
      denotePred fuelSel freeAlgebra fuelSig 4 (fuelSig.impls[2].header (Subst.ofList τ)) = some (.app 1 [.con 0 []]) := by
    intro τ hτ; rw [List.length_eq_zero_iff] at hτ; subst hτ; decide
  have h0 : ∀ τ : List Ty, τ.length = 1 -> ∃ s, τ = [s] ∧
      denotePred fuelSel freeAlgebra fuelSig 4 (fuelSig.impls[0].header (Subst.ofList τ)) =
        (denote fuelSel freeAlgebra fuelSig 4 s).map (fun y => .app 0 [y]) := by
    intro τ hτ
    match τ, hτ with
    | [s], _ =>
      refine ⟨s, rfl, ?_⟩
      show ([s].mapM (denote fuelSel freeAlgebra fuelSig 4)).map (Pred.app 0) = _
      rw [List.mapM_cons, List.mapM_nil]
      cases denote fuelSel freeAlgebra fuelSig 4 s <;> rfl
  rcases hi3 with rfl | rfl | rfl <;> rcases hj3 with rfl | rfl | rfl
  · obtain ⟨s, rfl, e⟩ := h0 σ hσ
    obtain ⟨s', rfl, e'⟩ := h0 σ' hσ'
    refine ⟨rfl, ?_⟩
    rw [e, ← heq'] at heq
    rw [e'] at heq
    cases hs : denote fuelSel freeAlgebra fuelSig 4 s <;> cases hs' : denote fuelSel freeAlgebra fuelSig 4 s' <;> simp_all
  · obtain ⟨s, rfl, e⟩ := h0 σ hσ
    rw [e, ← heq', h1 σ' hσ'] at heq
    cases denote fuelSel freeAlgebra fuelSig 4 s <;> simp at heq
  · obtain ⟨s, rfl, e⟩ := h0 σ hσ
    rw [e, ← heq', h2 σ' hσ'] at heq
    cases denote fuelSel freeAlgebra fuelSig 4 s <;> simp at heq
  · obtain ⟨s, rfl, e⟩ := h0 σ' hσ'
    rw [e, ← heq, h1 σ hσ] at heq'
    cases denote fuelSel freeAlgebra fuelSig 4 s <;> simp at heq'
  · obtain rfl := List.length_eq_zero_iff.mp hσ
    obtain rfl := List.length_eq_zero_iff.mp hσ'
    exact ⟨rfl, rfl⟩
  · rw [h1 σ hσ, ← heq', h2 σ' hσ'] at heq; simp at heq
  · obtain ⟨s, rfl, e⟩ := h0 σ' hσ'
    rw [e, ← heq, h2 σ hσ] at heq'
    cases denote fuelSel freeAlgebra fuelSig 4 s <;> simp at heq'
  · rw [h2 σ hσ, ← heq', h1 σ' hσ'] at heq; simp at heq
  · obtain rfl := List.length_eq_zero_iff.mp hσ
    obtain rfl := List.length_eq_zero_iff.mp hσ'
    exact ⟨rfl, rfl⟩

/-- `T[deepTy]::Out = K(K(deepTy))` by (BIND) through the generic impl at `[deepTy]`; its header
denotes at fuel 4. -/
def fuelDeriv : Deriv freeAlgebra fuelSig (Den fuelSel freeAlgebra fuelSig 4) []
    (.eq (.proj 0 [deepTy] 0 []) (.con 1 [.con 1 [deepTy]])) :=
  .bind (a := 0) (labels := []) (B := .con 1 [.con 1 [.var 0]]) (gs := []) 0 (by decide) [deepTy] rfl
    ⟨by decide, by decide⟩ (fun j hj => absurd hj (by simp [fuelSig])) rfl rfl

/-- At fuel 4, `[[T[deepTy]::Out]] = K(K(i32))` (the oracle's argument `i32` is shallow) while
`[[K(K(deepTy))]]` is undefined (`deepTy` sits two constructors deep and runs out of fuel), under a
sound oracle that answers every served application and a coherent signature: the denotation at one
fixed fuel is not a congruence, so section 7's statement is false at any fixed fuel and theorem 7 is
about the limit. -/
theorem fixed_fuel_erasure_fails : ¬ ErasureAsStated := by
  intro H
  have h := H fuelSel freeAlgebra fuelSig 4 (.proj 0 [deepTy] 0 []) (.con 1 [.con 1 [deepTy]])
    fuelSig_declWF fuelSig_reqDuty fuelSel_sound fuelSig_coherent ⟨fuelDeriv⟩
    (.con 1 [.con 1 [.con 0 []]]) (by decide)
  have hnone : denote fuelSel freeAlgebra fuelSig 4 (.con 1 [.con 1 [deepTy]]) = none := by decide
  rw [hnone] at h
  cases h

/-- Two traits: `Has` (0) with `Out` (0) and the requirement `Has[Self]::Out = i64` (`i64` is `con 2`);
`U` (1) with `Item` (0) and no requirement. Impls: `Has_i32` (0) at `i32` (`con 0`) binding `Out = i32`;
`Has_bad` (1) at the projection `U[i64]::Item` binding `Out = i64`; `U_bad2` (2) at `i64` binding
`Item = i32`. Every impl discharges its requirement in the full signature (`reqSig_reqDuty`): `Has_i32`
does so through `U_bad2` and `Has_bad`, whose header does not denote under an oracle that declines
`U[i64]`. -/
def reqSig : Sig :=
  { traits := [{ arity := 1, reqs := [.eq (.proj 0 [.var 0] 0 []) (.con 2 [])], binders := [], assocs := [[]] },
               { arity := 1, reqs := [], binders := [], assocs := [[]] }],
    impls := [{ arity := 0, trait := 0, head := [.con 0 []], wheres := [], bindings := [some ([], .con 0 [])],
                returns := [.hyp 0], binderReturns := [] },
              { arity := 0, trait := 0, head := [.proj 1 [.con 2 []] 0 []], wheres := [],
                bindings := [some ([], .con 2 [])], returns := [.hyp 0], binderReturns := [] },
              { arity := 0, trait := 1, head := [.con 2 []], wheres := [], bindings := [some ([], .con 0 [])],
                returns := [], binderReturns := [] }] }

/-- The oracle that serves `Has[i32]` with `Has_i32` and declines everything else, `U[i64]` included. -/
def reqSel : Sel
  | .app 0 [.con 0 []] => some (0, [])
  | _ => none

theorem reqSig_declWF : reqSig.DeclWF := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t T hT r hr
    match t, hT with
    | 0, hT =>
      simp [reqSig] at hT; subst hT; simp at hr; subst hr
      exact ⟨by decide, by decide, fun as h => by cases h⟩
    | 1, hT => simp [reqSig] at hT; subst hT; simp at hr
    | t+2, hT => simp [reqSig] at hT
  · intro T hT B hB; simp [reqSig] at hT; rcases hT with rfl | rfl <;> simp at hB
  · intro T hT X hX; simp [reqSig] at hT; rcases hT with rfl | rfl <;> simp at hX <;> subst hX <;> simp
  · intro I hI; simp [reqSig] at hI; rcases hI with rfl | rfl | rfl <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  · intro I hI; simp [reqSig] at hI
    rcases hI with rfl | rfl | rfl <;> refine ⟨?_, ?_, ?_⟩
    all_goals first
      | (intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; done)
      | (intro T hT a labels B hb
         cases a with
         | zero =>
           simp at hb
           obtain ⟨rfl, rfl⟩ := hb
           simp [reqSig] at hT; subst hT
           exact ⟨[], rfl, rfl, List.nodup_nil, by simp, by decide, by decide⟩
         | succ a => simp at hb)
  · intro I hI v hv; simp [reqSig] at hI; rcases hI with rfl | rfl | rfl <;> simp at hv

/-- `U[i64]::Item = i32`, (BIND) through `U_bad2`. -/
def reqD1 : Deriv freeAlgebra reqSig AnyImpl [] (.eq (.proj 1 [.con 2 []] 0 []) (.con 0 [])) :=
  .bind (a := 0) (labels := []) (B := .con 0 []) (gs := []) 2 (by decide) [] rfl trivial
    (fun j hj => absurd hj (by simp [reqSig])) rfl rfl

/-- `Has[i32]::Out = Has[U[i64]::Item]::Out`, (CONG-P) over (SYM) of `reqD1`. -/
def reqD3 : Deriv freeAlgebra reqSig AnyImpl []
    (.eq (.proj 0 [.con 0 []] 0 []) (.proj 0 [.proj 1 [.con 2 []] 0 []] 0 [])) :=
  .congP (t := 0) (as := [.con 0 []]) (bs := [.proj 1 [.con 2 []] 0 []]) (a := 0) (gs := []) (hs := []) rfl rfl
    (fun j hj => match j, hj with
      | 0, _ => .symm reqD1
      | k+1, h => absurd h (by simp))
    (fun j hj => absurd hj (by simp))

/-- `Has[U[i64]::Item]::Out = i64`, (BIND) through `Has_bad`. -/
def reqD4 : Deriv freeAlgebra reqSig AnyImpl [] (.eq (.proj 0 [.proj 1 [.con 2 []] 0 []] 0 []) (.con 2 [])) :=
  .bind (a := 0) (labels := []) (B := .con 2 []) (gs := []) 1 (by decide) [] rfl trivial
    (fun j hj => absurd hj (by simp [reqSig])) rfl rfl

theorem reqD4_primitive : reqD4.Primitive := by
  simp only [reqD4, Deriv.Primitive]
  intro j hj; simp [reqSig] at hj

/-- `Has[i32]::Out = i64`, the requirement of `Has_i32`, (TRANS) of `reqD3` and `reqD4`: primitive, in
the full signature. -/
def reqD5 : Deriv freeAlgebra reqSig AnyImpl [] (.eq (.proj 0 [.con 0 []] 0 []) (.con 2 [])) := .trans reqD3 reqD4

theorem reqD5_primitive : reqD5.Primitive := by
  refine ⟨?_, reqD4_primitive⟩
  simp only [reqD3, Deriv.Primitive]
  refine ⟨fun j hj => ?_, fun j hj => by simp at hj⟩
  match j, hj with
  | 0, _ =>
    simp only [reqD1, Deriv.Primitive]
    intro j hj; simp [reqSig] at hj
  | k+1, h => simp at h

theorem reqSig_reqDuty : reqSig.ReqDuty freeAlgebra := by
  refine ⟨?_, ?_⟩
  · intro i hi ht k hk
    have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by
      have : i < 3 := by simpa [reqSig] using hi
      match i, this with
      | 0, _ => simp
      | 1, _ => simp
      | 2, _ => simp
    rcases hi3 with rfl | rfl | rfl
    · match k, hk with
      | 0, _ => exact ⟨reqD5, reqD5_primitive⟩
      | k+1, hk => simp [reqSig] at hk
    · match k, hk with
      | 0, _ => exact ⟨reqD4, reqD4_primitive⟩
      | k+1, hk => simp [reqSig] at hk
    · simp [reqSig] at hk
  · intro i hi ht b hb
    have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by
      have : i < 3 := by simpa [reqSig] using hi
      match i, this with
      | 0, _ => simp
      | 1, _ => simp
      | 2, _ => simp
    rcases hi3 with rfl | rfl | rfl <;> simp [reqSig] at hb

theorem reqSel_sound : SelSound reqSel freeAlgebra reqSig 3 := by
  intro P i σ h
  unfold reqSel at h
  split at h <;> cases h
  exact ⟨by decide, by decide, by decide, by decide, fun W hW => by simp [reqSig] at hW⟩

theorem reqSig_coherent : reqSig.Coherent reqSel freeAlgebra 3 := by
  intro t as _ hsome i σ j σ' ⟨hi, hσ, _, heq, _⟩ ⟨hj, hσ', _, heq', _⟩
  have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by
    have : i < 3 := by simpa [reqSig] using hi
    match i, this with
    | 0, _ => simp
    | 1, _ => simp
    | 2, _ => simp
  have hj3 : j = 0 ∨ j = 1 ∨ j = 2 := by
    have : j < 3 := by simpa [reqSig] using hj
    match j, this with
    | 0, _ => simp
    | 1, _ => simp
    | 2, _ => simp
  have e0 : denotePred reqSel freeAlgebra reqSig 3 (reqSig.impls[0].header (Subst.ofList [])) = some (.app 0 [.con 0 []]) := by decide
  have e1 : denotePred reqSel freeAlgebra reqSig 3 (reqSig.impls[1].header (Subst.ofList [])) = none := by decide
  have e2 : denotePred reqSel freeAlgebra reqSig 3 (reqSig.impls[2].header (Subst.ofList [])) = some (.app 1 [.con 2 []]) := by decide
  rcases hi3 with rfl | rfl | rfl <;> rcases hj3 with rfl | rfl | rfl <;>
    obtain rfl := List.length_eq_zero_iff.mp hσ <;> obtain rfl := List.length_eq_zero_iff.mp hσ'
  all_goals first
    | exact ⟨rfl, rfl⟩
    | (rw [e1] at heq; rw [← heq] at hsome; cases hsome)
    | (rw [e1] at heq'; rw [← heq'] at hsome; cases hsome)
    | (rw [e0, ← heq'] at heq; rw [e2] at heq; cases heq)
    | (rw [e2, ← heq'] at heq; rw [e0] at heq; cases heq)

/-- `Has[i32]::Out = i64`, (REQ) off `Has[i32]` derived through `Has_i32`, whose header denotes. -/
def reqDeriv : Deriv freeAlgebra reqSig (Den reqSel freeAlgebra reqSig 3) [] (.eq (.proj 0 [.con 0 []] 0 []) (.con 2 [])) :=
  .req (t := 0) (as := [.con 0 []]) (k := 0)
    (.impl 0 (by decide) [] rfl ⟨by decide, by decide⟩ (fun j hj => absurd hj (by simp [reqSig])))
    (by decide) (by decide)

/-- `[[Has[i32]::Out]] = i32` while `Has[i32]::Out = i64` is valid closed in the denotable fragment by
(REQ): H3 discharges `Has_i32`'s requirement in the full signature, through `Has_bad`, whose header
does not denote, so the (REQ-ADMISSIBLE) argument does not stay in the denotable fragment and section 7's
statement is false with (REQ) admitted. The theorems above take a primitive derivation. -/
theorem erasure_needs_primitive : ¬ ErasureAsStated := by
  intro H
  have h := H reqSel freeAlgebra reqSig 3 (.proj 0 [.con 0 []] 0 []) (.con 2 [])
    reqSig_declWF reqSig_reqDuty reqSel_sound reqSig_coherent ⟨reqDeriv⟩ (.con 0 []) (by decide)
  have hval : denote reqSel freeAlgebra reqSig 3 (.con 2 []) = some (.con 2 []) := by decide
  rw [hval] at h
  cases h

end Trait
