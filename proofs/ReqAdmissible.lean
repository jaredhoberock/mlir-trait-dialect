import Calculus
/-!
# ReqAdmissible: (REQ-ADMISSIBLE) is false as `Sig.ReqDuty` states H3

What the lemma claimed. SPEC.md 4.3 states (REQ-ADMISSIBLE): under well-formed declarations and H3 (every
impl discharges its requirements primitively under its where entries, `Sig.ReqDuty`), whatever the full
logic proves the primitive fragment proves: `S; G |- P` implies `S; G |-p P`. The section 7 skeleton stated
it as Logic.lean's `req_admissible`; `ReqAdmissible` below is that statement, `req_admissible_fails` its
refutation, and `Deriv.substTy_exists` the half of the intended proof that holds (every derivation
instantiates under a type substitution, primitive to primitive: the logic's form of the clone rule of
SPEC.md 6.5).

The cycle that refutes it (`loopSig`). Trait `A` requires `C[Self]`, trait `B` requires `A[Self]`, trait
`C` requires nothing and has no impl. `I_A` is `impl A for a_0 where B[a_0]`; its duty reads `C[a_0]` off
the where entry as the requirement of its requirement, (REQ) on (REQ) on (HYP), an `Env(G)` member.
`I_B0` is `impl B for i32`; its duty derives `A[i32]` by (IMPL) at `I_A` given (IMPL) at `I_B0`. Both duties
are primitive, so H3 holds; the declarations are well formed; `C[i32]` is valid under no hypotheses, by
(REQ) on the (IMPL) node at `I_A`; and no primitive derivation concludes any application of `C`, since
`C` has no impl and under `G = []` nothing is an `Env` member (`Deriv.not_inEnv_nil`). The intended proof
replaces the (REQ) by `I_A`'s duty with the where entry replaced by the premise, the (IMPL) node at
`I_B0`; the duty's inner (REQ) now stands on that node and is replaced by `I_B0`'s duty, the (IMPL) node
at `I_A`; the outer (REQ) stands on it again. The replacement never ends, and there is nothing else.

The missing hypothesis. The replacement terminates only if the duties' mutual reading is well founded:
no chain of duties reads requirements off where entries whose impls' duties lead back to the start. The
stage checks exactly this on each instance and refuses the program ("has no base", `readEvidence`, X/x3,
X/x10, SPEC.md F13); the verifier does not, except in the one-impl form, an impl reading its own
application through its where entry's witness (`projectsOwnApplication`,
T/invalid_impl_return_reads_its_own_requirement_through_its_witness.mlir), which `loopSig` is. H3 as the
dialect means it includes the base condition; `Sig.ReqDuty` does not state it.

Two repairs, neither made here. (1) State the base condition on the signature: every reading of every
return at every instance reaches a base. As a hypothesis this is the termination of the very replacement
the proof performs, so it is the theorem's own content. (2) Require the duty derivations to contain no
(REQ) and no (REQ-A) node at all. Then replacing where entries by premises never puts a (REQ) on an
(IMPL) node, the cut lemma preserves primitiveness outright, and the theorem follows by structural
induction from `Deriv.substTy_exists`, a cut lemma, and the commutation of (REQ) through (CONV). Its cost:
a return may no longer read a requirement off a where entry (`trait.project ^j[k]`), which SPEC.md card
3 allows and the verifier admits.

Imported by nothing.
-/namespace Trait

/-- Under well-formed declarations and hypotheses at their traits' arity, every derived application is at
its trait's arity: (IMPL) states the header at the trait's arity, (REQ) and (REQ-A) instantiate well-formed
requirements, (CONV) keeps the length. -/
theorem Deriv.atArity {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} {Γ : List Pred} (hS : S.DeclWF)
    (hΓ : ∀ Q ∈ Γ, Q.AtArity S) : ∀ {P : Pred}, Deriv L S Ok Γ P -> P.AtArity S := by
  intro P d
  induction d with
  | hyp k hk => exact hΓ _ (List.getElem_mem hk)
  | impl i hi σ hσ _ _ _ =>
    obtain ⟨T, hT, hlen, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
    obtain ⟨ht, hT'⟩ := List.getElem?_eq_some_iff.mp hT
    exact ⟨ht, by simp only [substList_length, hT', hlen]⟩
  | req d ht hk _ =>
    exact Pred.atArity_subst _ (Pred.atArity_of_wf
      (hS.trait_reqs _ _ (List.getElem?_eq_getElem ht) _ (List.getElem_mem hk)).1)
  | reqA d ht hb us hus prem _ _ =>
    exact Pred.atArity_subst _ (Pred.atArity_of_wf
      (hS.trait_binders _ (List.getElem_mem ht) _ (List.getElem_mem hb)).2.2.2.1)
  | bind => trivial
  | refl => trivial
  | symm => trivial
  | trans => trivial
  | cong => trivial
  | congP => trivial
  | norm => trivial
  | conv d hlen _ ih _ =>
    obtain ⟨ht, hlen'⟩ := ih
    exact ⟨ht, hlen ▸ hlen'⟩

/-- Substituting into a list and reading by position below the length is reading and then substituting. -/
theorem substList_getD (σ : Subst) (as : List Ty) {i : Nat} (hi : i < as.length) :
    (substList σ as).getD i default = (as.getD i default).subst σ := by
  simp [List.getD_eq_getElem?_getD, substList_getElem?, List.getElem?_eq_getElem hi]

/-- A derivation moved to an equal conclusion. -/
def Deriv.cast {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} {Γ : List Pred} {P Q : Pred}
    (h : P = Q) (d : Deriv L S Ok Γ P) : Deriv L S Ok Γ Q := h ▸ d

theorem Deriv.cast_primitive {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} {Γ : List Pred} {P Q : Pred}
    (h : P = Q) (d : Deriv L S Ok Γ P) : (d.cast h).Primitive ↔ d.Primitive := by
  subst h; exact Iff.rfl

theorem Deriv.cast_inEnv {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} {Γ : List Pred} {P Q : Pred}
    (h : P = Q) (d : Deriv L S Ok Γ P) : (d.cast h).InEnv ↔ d.InEnv := by
  subst h; exact Iff.rfl

/-- Type substitution through a derivation (the clone rule of SPEC.md 6.5 at the level of the logic): a
derivation under `Γ` instantiates to one under `Γ[σ]` of `P[σ]`, primitive when the original is, and an
`Env(G)` member when the original is. Requires the hypotheses at their traits' arity, so that a (REQ) or
(REQ-A) node's positional substitution covers every position its requirement mentions. -/
theorem Deriv.substTy_exists {L : ConLaw} {S : Sig} {Γ : List Pred} (hS : S.DeclWF)
    (hΓ : ∀ Q ∈ Γ, Q.AtArity S) (σ : Subst) :
    ∀ {P : Pred} (d : Deriv L S AnyImpl Γ P),
      ∃ d' : Deriv L S AnyImpl (Γ.map (Pred.subst σ)) (P.subst σ),
        (d.Primitive -> d'.Primitive) ∧ (d.InEnv -> d'.InEnv) := by
  intro P d
  induction d with
  | hyp k hk =>
    have hk' : k < (Γ.map (Pred.subst σ)).length := by simpa using hk
    refine ⟨(Deriv.hyp k hk').cast (by simp), fun _ => ?_, fun _ => ?_⟩
    · exact (Deriv.cast_primitive _ _).mpr trivial
    · exact (Deriv.cast_inEnv _ _).mpr trivial
  | impl i hi τ hτ _ ds ih =>
    obtain ⟨hhead, hwh, _⟩ := hS.impl_wf _ (List.getElem_mem hi)
    have hhv : allVarsList (fun v => decide (v < τ.length)) S.impls[i].head = true := by
      rw [hτ]; exact allVarsList_of_forall _ _ (fun h hh => (hhead h hh).2)
    refine ⟨(Deriv.impl i hi (substList σ τ) (by rw [substList_length]; exact hτ) trivial
      (fun j hj => (Classical.choose (ih j hj)).cast
        (Pred.subst_ofList_subst σ τ _ (by rw [hτ]; exact (hwh _ (getD_mem_of_lt _ hj)).2)))).cast
      (by simp only [ImplDecl.header, Pred.subst]; rw [substList_ofList_subst σ τ _ hhv]),
      fun h => ?_, fun h => by simp [Deriv.InEnv] at h⟩
    rw [Deriv.cast_primitive]
    simp only [Deriv.Primitive] at h ⊢
    intro j hj
    exact (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ih j hj)).1 (h j hj))
  | req d ht hk ih =>
    rename_i t as k
    obtain ⟨_, hlen⟩ := Deriv.atArity hS hΓ d
    obtain ⟨d', hp, he⟩ := ih
    refine ⟨(Deriv.req (as := substList σ as) d' ht hk).cast
      (Pred.subst_ofList_subst σ as _ (by
        rw [hlen]; exact (hS.trait_reqs _ _ (List.getElem?_eq_getElem ht) _ (List.getElem_mem hk)).2.1)).symm,
      fun h => ?_, fun h => ?_⟩
    · rw [Deriv.cast_primitive]; simp only [Deriv.Primitive] at h ⊢; exact he h
    · rw [Deriv.cast_inEnv]; simp only [Deriv.InEnv] at h ⊢; exact he h
  | reqA d ht hb us hus prem ih ihs =>
    rename_i t as b
    obtain ⟨_, hlen⟩ := Deriv.atArity hS hΓ d
    have hB := hS.trait_binders _ (List.getElem_mem ht) _ (List.getElem_mem hb)
    obtain ⟨d', hp, he⟩ := ih
    refine ⟨(Deriv.reqA (as := substList σ as) d' ht hb (substList σ us) (by rw [substList_length]; exact hus)
      (fun j hj => (Classical.choose (ihs j hj)).cast (by
        simp only [Binder.callSubst]
        exact Pred.subst_extend_subst σ as us _ hus _
          (by rw [hlen]; exact (hB.2.2.1 _ (getD_mem_of_lt _ hj)).2)))).cast (by
      simp only [Binder.callSubst]
      exact (Pred.subst_extend_subst σ as us _ hus _ (by rw [hlen]; exact hB.2.2.2.2)).symm),
      fun h => ?_, fun h => ?_⟩
    · refine (Deriv.cast_primitive _ _).mpr ?_; simp only [Deriv.Primitive] at h ⊢
      exact ⟨he h.1, fun j hj => (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ihs j hj)).1 (h.2 j hj))⟩
    · refine (Deriv.cast_inEnv _ _).mpr ?_; simp only [Deriv.InEnv] at h ⊢
      exact ⟨he h.1, fun j hj => (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ihs j hj)).1 (h.2 j hj))⟩
  | bind i hi τ hτ _ ds hb hg ih =>
    rename_i a labels B gs hok
    obtain ⟨hhead, hwh, hbind⟩ := hS.impl_wf _ (List.getElem_mem hi)
    obtain ⟨T, hT, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
    obtain ⟨X, hX, hlab, _, _, _, hBv⟩ := hbind T hT _ _ _ hb
    have hhv : allVarsList (fun v => decide (v < τ.length)) S.impls[i].head = true := by
      rw [hτ]; exact allVarsList_of_forall _ _ (fun h hh => (hhead h hh).2)
    refine ⟨(Deriv.bind (gs := substList σ gs) i hi (substList σ τ) (by rw [substList_length]; exact hτ) trivial
      (fun j hj => (Classical.choose (ih j hj)).cast
        (Pred.subst_ofList_subst σ τ _ (by rw [hτ]; exact (hwh _ (getD_mem_of_lt _ hj)).2)))
      hb (by rw [substList_length, hg])).cast ?_, fun h => ?_, fun h => by simp [Deriv.InEnv] at h⟩
    · simp only [Pred.subst, Ty.subst]
      rw [substList_ofList_subst σ τ _ hhv, Ty.subst_extend_subst σ τ gs labels hg _ (by rw [hτ]; exact hBv)]
    · rw [Deriv.cast_primitive]
      simp only [Deriv.Primitive] at h ⊢
      intro j hj
      exact (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ih j hj)).1 (h j hj))
  | refl => exact ⟨.refl, fun _ => trivial, fun h => by simp [Deriv.InEnv] at h⟩
  | symm d ih =>
    obtain ⟨d', hp, _⟩ := ih
    exact ⟨.symm d', fun h => hp h, fun h => by simp [Deriv.InEnv] at h⟩
  | trans d₁ d₂ ih₁ ih₂ =>
    obtain ⟨d₁', hp₁, _⟩ := ih₁
    obtain ⟨d₂', hp₂, _⟩ := ih₂
    exact ⟨.trans d₁' d₂', fun h => ⟨hp₁ h.1, hp₂ h.2⟩, fun h => by simp [Deriv.InEnv] at h⟩
  | @cong c as bs hlen ds ih =>
    refine ⟨.cong (as := substList σ as) (bs := substList σ bs) (by simp [substList_length, hlen])
      (fun i hi => (Classical.choose (ih i (by simpa [substList_length] using hi))).cast
        (by rw [substList_length] at hi
            simp only [Pred.subst, substList_getD σ as hi, substList_getD σ bs (hlen ▸ hi)])),
      fun h => ?_, fun h => by simp [Deriv.InEnv] at h⟩
    simp only [Deriv.Primitive] at h ⊢
    intro i hi
    exact (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ih i _)).1 (h i _))
  | congP hlen hlen' ds es ih ihs =>
    rename_i t as a gs bs hs
    refine ⟨.congP (as := substList σ as) (bs := substList σ bs) (gs := substList σ gs) (hs := substList σ hs)
      (by simp [substList_length, hlen]) (by simp [substList_length, hlen'])
      (fun i hi => (Classical.choose (ih i (by simpa [substList_length] using hi))).cast
        (by rw [substList_length] at hi
            simp only [Pred.subst, substList_getD σ as hi, substList_getD σ bs (hlen ▸ hi)]))
      (fun i hi => (Classical.choose (ihs i (by simpa [substList_length] using hi))).cast
        (by rw [substList_length] at hi
            simp only [Pred.subst, substList_getD σ gs hi, substList_getD σ hs (hlen' ▸ hi)])),
      fun h => ?_, fun h => by simp [Deriv.InEnv] at h⟩
    simp only [Deriv.Primitive] at h ⊢
    exact ⟨fun i hi => (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ih i _)).1 (h.1 i _)),
      fun i hi => (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ihs i _)).1 (h.2 i _))⟩
  | norm h => exact ⟨.norm (L.law_subst σ h), fun _ => trivial, fun h => by simp [Deriv.InEnv] at h⟩
  | conv d hlen ds ih ihs =>
    rename_i t as bs
    obtain ⟨d', hp, _⟩ := ih
    refine ⟨.conv (as := substList σ as) (bs := substList σ bs) d' (by simp [substList_length, hlen])
      (fun i hi => (Classical.choose (ihs i (by simpa [substList_length] using hi))).cast
        (by rw [substList_length] at hi
            simp only [Pred.subst, substList_getD σ as hi, substList_getD σ bs (hlen ▸ hi)])),
      fun h => ?_, fun h => by simp [Deriv.InEnv] at h⟩
    simp only [Deriv.Primitive] at h ⊢
    exact ⟨hp h.1, fun i hi => (Deriv.cast_primitive _ _).mpr ((Classical.choose_spec (ihs i _)).1 (h.2 i _))⟩


/-! ## The refutation -/

/-- Three traits of arity 1: `A` (0) with the requirement `C[Self]`, `B` (1) with the requirement `A[Self]`,
`C` (2) with none and no impl. Two impls: `I_A` (0), `impl A for a_0 where B[a_0]`, whose return reads
`C[a_0]` as the requirement of the requirement of its where entry; `I_B0` (1), `impl B for i32` (`i32` is
`con 0`), whose return derives `A[i32]` through `I_A` given itself. -/
def loopSig : Sig :=
  { traits := [{ arity := 1, reqs := [.app 2 [.var 0]], binders := [], assocs := [] },
               { arity := 1, reqs := [.app 0 [.var 0]], binders := [], assocs := [] },
               { arity := 1, reqs := [], binders := [], assocs := [] }],
    impls := [{ arity := 1, trait := 0, head := [.var 0], wheres := [.app 1 [.var 0]], bindings := [],
                returns := [.project (.project (.hyp 0) 0) 0], binderReturns := [] },
              { arity := 0, trait := 1, head := [.con 0 []], wheres := [], bindings := [],
                returns := [.derive 0 [.con 0 []] [.derive 1 [] []]], binderReturns := [] }] }

theorem loopSig_declWF : loopSig.DeclWF := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t T hT r hr
    match t, hT with
    | 0, hT =>
      simp [loopSig] at hT; subst hT; simp at hr; subst hr
      exact ⟨by decide, by decide, fun as h => by cases h⟩
    | 1, hT =>
      simp [loopSig] at hT; subst hT; simp at hr; subst hr
      exact ⟨by decide, by decide, fun as h => by cases h⟩
    | 2, hT => simp [loopSig] at hT; subst hT; simp at hr
    | t+3, hT => simp [loopSig] at hT
  · intro T hT B hB; simp [loopSig] at hT; rcases hT with rfl | rfl | rfl <;> simp at hB
  · intro T hT X hX; simp [loopSig] at hT; rcases hT with rfl | rfl | rfl <;> simp at hX
  · intro I hI; simp [loopSig] at hI; rcases hI with rfl | rfl <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  · intro I hI; simp [loopSig] at hI
    rcases hI with rfl | rfl <;> refine ⟨?_, ?_, ?_⟩
    all_goals first
      | (intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; subst hW; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; done)
      | (intro T hT a labels B hb; simp at hb; done)
  · intro I hI v hv; simp [loopSig] at hI
    rcases hI with rfl | rfl <;> simp at hv
    subst hv; decide

/-- `I_A`'s duty: `C[a_0]` under `B[a_0]`, the requirement (`A[a_0]`) of the requirement of the where
entry, read off the hypothesis: an `Env(G)` member, primitive. -/
def loopDutyA (L : ConLaw) : Deriv L loopSig AnyImpl [.app 1 [.var 0]] (.app 2 [.var 0]) :=
  .req (t := 0) (as := [.var 0]) (k := 0)
    (.req (t := 1) (as := [.var 0]) (k := 0) (.hyp 0 (by decide)) (by decide) (by decide)) (by decide) (by decide)

theorem loopDutyA_primitive (L : ConLaw) : (loopDutyA L).Primitive := by
  simp only [loopDutyA, Deriv.Primitive, Deriv.InEnv]

/-- `A[i32]` under no hypotheses: (IMPL) at `I_A`, its where entry `B[i32]` by (IMPL) at `I_B0`. -/
def loopImplA (L : ConLaw) : Deriv L loopSig AnyImpl [] (.app 0 [.con 0 []]) :=
  .impl 0 (by decide) [.con 0 []] rfl trivial
    (fun j hj => match j, hj with
      | 0, _ => .impl 1 (by decide) [] rfl trivial (fun j hj => absurd hj (by simp [loopSig]))
      | k+1, h => absurd h (by simp [loopSig]))

theorem loopImplA_primitive (L : ConLaw) : (loopImplA L).Primitive := by
  simp only [loopImplA, Deriv.Primitive]
  intro j hj
  match j, hj with
  | 0, _ => simp only [Deriv.Primitive]; intro j hj; simp [loopSig] at hj
  | k+1, h => simp [loopSig] at h

theorem loopSig_reqDuty (L : ConLaw) : loopSig.ReqDuty L := by
  refine ⟨?_, ?_⟩
  · intro i hi ht k hk
    have hi2 : i = 0 ∨ i = 1 := by
      have : i < 2 := by simpa [loopSig] using hi
      match i, this with
      | 0, _ => simp
      | 1, _ => simp
    rcases hi2 with rfl | rfl
    · match k, hk with
      | 0, _ => exact ⟨loopDutyA L, loopDutyA_primitive L⟩
      | k+1, hk => simp [loopSig] at hk
    · match k, hk with
      | 0, _ => exact ⟨loopImplA L, loopImplA_primitive L⟩
      | k+1, hk => simp [loopSig] at hk
  · intro i hi ht b hb
    have hi2 : i = 0 ∨ i = 1 := by
      have : i < 2 := by simpa [loopSig] using hi
      match i, this with
      | 0, _ => simp
      | 1, _ => simp
    rcases hi2 with rfl | rfl <;> simp [loopSig] at hb

/-- `C[i32]` in the full logic: (REQ) on the (IMPL)-derived `A[i32]`. -/
def loopDeriv (L : ConLaw) : Deriv L loopSig AnyImpl [] (.app 2 [.con 0 []]) :=
  .req (t := 0) (as := [.con 0 []]) (k := 0) (loopImplA L) (by decide) (by decide)

/-- No primitive derivation under no hypotheses concludes an application of `C`: no impl is of `C`,
(REQ) and (REQ-A) have no `Env([])` member to read, and (CONV) needs one. -/
theorem loopSig_no_C (L : ConLaw) : ∀ {P : Pred} (d : Deriv L loopSig AnyImpl [] P), d.Primitive ->
    ∀ as, P ≠ .app 2 as := by
  intro P d
  induction d with
  | hyp k hk => exact absurd hk (by simp)
  | impl i hi σ hσ _ ds _ =>
    intro _ as h
    have hi2 : i = 0 ∨ i = 1 := by
      have : i < 2 := by simpa [loopSig] using hi
      match i, this with
      | 0, _ => simp
      | 1, _ => simp
    rcases hi2 with rfl | rfl <;> simp [loopSig, ImplDecl.header] at h
  | req d _ _ _ => intro hp; exact absurd hp (Deriv.not_inEnv_nil d)
  | reqA d _ _ _ _ _ _ _ => intro hp; exact absurd hp.1 (Deriv.not_inEnv_nil d)
  | conv d hlen ds ih _ => intro hp as h; cases h; exact ih hp.1 _ rfl
  | bind => intro _ as h; cases h
  | refl => intro _ as h; cases h
  | symm => intro _ as h; cases h
  | trans => intro _ as h; cases h
  | cong => intro _ as h; cases h
  | congP => intro _ as h; cases h
  | norm => intro _ as h; cases h

/-- (REQ-ADMISSIBLE), SPEC.md 4.3, as the section 7 skeleton stated it: under well-formed declarations and
H3, whatever the full logic proves under `G` the primitive fragment proves under `G`. -/
def ReqAdmissible (L : ConLaw) (S : Sig) : Prop :=
  ∀ (Γ : List Pred) (P : Pred), S.DeclWF -> S.ReqDuty L -> Valid L S Γ P -> ValidP L S Γ P

/-- (REQ-ADMISSIBLE) is false: `loopSig` is well formed and satisfies H3, `C[i32]` is valid under no
hypotheses, and no primitive derivation concludes it. The free algebra is the constructor law; the
derivations use no (NORM), so any other law refutes it too. -/
theorem req_admissible_fails : ¬ ReqAdmissible freeAlgebra loopSig := by
  intro h
  obtain ⟨d, hd⟩ := h [] (.app 2 [.con 0 []]) loopSig_declWF (loopSig_reqDuty _) ⟨loopDeriv freeAlgebra⟩
  exact loopSig_no_C freeAlgebra d hd _ rfl

end Trait
