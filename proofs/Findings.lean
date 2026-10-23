import Calculus
/-!
# Findings: the refutations of SPEC.md section 8, as examples

Each example states the counterexample its finding names, in the calculus's vocabulary. Findings that
are observations (F5, F9, F11, F14 to F17, F19, F20), that concern shapes the grammar cannot write (F10:
no claim inside an equality endpoint, `Ty` has no claim constructor; F12: a value feeding itself, terms
are trees) or a rule the logic does not state (F4, F6), or the stage alone (F1's dispatch on an
annotation: `root` reads the term), have no example here.

How the refutations are proved. A positive clause (a declaration is well formed, a term has a type) is
the concrete signature or module written out and the typing rule applied; the closed computations
(substitution, groundness, allege-freedom) reduce by `rfl` or `decide`. A negative clause (a judgment
is not derivable) is a model: a property of predicates that every rule of the logic preserves under the
signature in question, which the judgment in question lacks. F2's model is "no application at all is
derivable" (the one impl's where entry is its own header); F13's is "both sides of a derivable equality
agree on whether `i64` occurs in them", which the binding `Out = i32` keeps and the requirement
`Has[i32]::Out = i64` breaks.

Two further refutations are of the proof files' statements rather than of section 8's findings: the
inversion theorem of Minimality.lean needs H2 (the projection index is read against the trait of the
cited proof's derive, the written trait's only when the proof's body is typed at the application it
proves), and the skeleton's `witness_is_let` for any `Γ` fails at a proven hypothesis.
-/

namespace Trait

/-! ## Two distinct ground types for the counterexamples -/

/-- `i32`: constructor `0` with no children. -/
private abbrev i32 : Ty := .con 0 []
/-- `i64`: constructor `1` with no children. -/
private abbrev i64 : Ty := .con 1 []

/-! ## Helpers on the judgments -/

/-- Inversion of (HYP): the term `^k` types only at the `k`-th hypothesis itself, its citation included. -/
private theorem hasType_hyp_inv {L : ConLaw} {M : Module} {Ok : ImplId -> Subst -> Prop} {Γ : List Claim}
    {k : Nat} {c : Claim} (h : HasType L M Ok Γ (.hyp k) c) : ∃ hk : k < Γ.length, c = Γ[k] := by
  cases h with
  | hyp hk => exact ⟨hk, rfl⟩

/-- Under no hypotheses nothing is an `Env(G)` member: `Env([])` is empty, since every member is a
hypothesis or a requirement read off a member. -/
private theorem not_inEnv_nil {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} :
    ∀ {P : Pred} (d : Deriv L S Ok [] P), ¬ d.InEnv := by
  intro P d
  induction d with
  | hyp k hk => exact absurd hk (Nat.not_lt_zero k)
  | req d ht hk ih => simpa [Deriv.InEnv] using ih
  | reqA d ht hb us hus prem ih _ =>
    intro h
    simp only [Deriv.InEnv] at h
    exact ih h.1
  | _ => simp [Deriv.InEnv]

/-! ## F2: soundness fails without H2's well-founded citation clause (R/p14) -/

/-- R/p14's signature: one trait `Foo` of arity 1 with no requirements, and the blanket impl
`impl<T: Foo> Foo for T`: header `Foo[a0]`, one where entry `Foo[a0]`, nothing to return. -/
private def S_F2 : Sig :=
  ⟨[⟨1, [], [], []⟩], [⟨1, 0, [.var 0], [.app 0 [.var 0]], [], [], []⟩]⟩

/-- R/p14's module: the proof `@p` of `Foo[i32]` whose body derives `Foo[i32]` from `@Foo_all[i32]`
given a witness of `@p` itself. -/
private def M_F2 : Module :=
  ⟨S_F2, [(.app 0 [i32], .derive 0 [i32] [.witness (.proof 0) (.app 0 [i32])])]⟩

/-- `S_F2` passes every check `TraitOp::verify` and `ImplOp::verify` make. -/
private theorem S_F2_wf : S_F2.DeclWF where
  trait_reqs := by
    intro t T hT r hr
    cases t with
    | zero => simp [S_F2] at hT; subst hT; simp at hr
    | succ t => simp [S_F2] at hT
  trait_binders := by
    intro T hT B hB
    simp [S_F2] at hT; subst hT; simp at hB
  trait_assocs := by
    intro T hT X hX
    simp [S_F2] at hT; subst hT; simp at hX
  impl_trait := by
    intro I hI
    simp [S_F2] at hI; subst hI
    exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  impl_wf := by
    intro I hI
    simp [S_F2] at hI; subst hI
    refine ⟨?_, ?_, ?_⟩
    · intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩
    · intro W hW; simp at hW; subst hW; exact ⟨by decide, by decide⟩
    · intro T hT a labels B hB; simp at hB
  impl_params_constrained := by
    intro I hI v hv
    simp [S_F2] at hI; subst hI
    simp at hv; subst hv
    exact Or.inl (by decide)

/-- The proof's body, the closed allege-free term for `Foo[i32]`: (DERIVE) over (WIT-APP) of the
proof itself. -/
private theorem M_F2_body_typed :
    HasType freeAlgebra M_F2 AnyImpl [] (.derive 0 [i32] [.witness (.proof 0) (.app 0 [i32])])
      ⟨.app 0 [i32], none⟩ :=
  HasType.derive (i := 0) (by decide) rfl trivial rfl (fun _ => some (.proof 0)) (fun j hj => by
    cases j with
    | zero => exact HasType.witProof (by decide) rfl
    | succ j => simp at hj)

/-- The proof table of `M_F2` satisfies every clause of H2 but the well-founded citation one. -/
private theorem M_F2_proofs : ∀ p (hp : p < M_F2.proofs.length),
    (∃ π : Option Cite, HasType freeAlgebra M_F2 AnyImpl [] M_F2.proofs[p].2 ⟨M_F2.proofs[p].1, π⟩) ∧
    (M_F2.proofs[p].1).ground = true ∧ (M_F2.proofs[p].2).allegeFree := by
  intro p hp
  cases p with
  | zero => exact ⟨⟨none, M_F2_body_typed⟩, rfl, rfl⟩
  | succ p => simp [M_F2] at hp

/-- The model for F2: a predicate that is not an application. -/
private def NoApp : Pred -> Prop
  | .app _ _ => False
  | .eq _ _ => True

/-- No application is derivable in `S_F2` under no hypotheses: (IMPL) needs its own conclusion as its
premise, the trait has no requirement or binder to read, and (CONV) moves between applications only. -/
private theorem S_F2_noApp : ∀ {P : Pred}, Deriv freeAlgebra S_F2 AnyImpl [] P -> NoApp P := by
  intro P d
  induction d with
  | hyp k hk => exact absurd hk (Nat.not_lt_zero k)
  | impl i hi σ hσ ok ds ih =>
    have hi0 : i = 0 := by simp [S_F2] at hi; omega
    subst hi0
    exact False.elim (ih 0 (by simp [S_F2]))
  | @req t as k _ ht hk _ =>
    have ht0 : t = 0 := by simp [S_F2] at ht; omega
    subst ht0
    simp [S_F2] at hk
  | @reqA t as b _ ht hb _ _ _ _ _ =>
    have ht0 : t = 0 := by simp [S_F2] at ht; omega
    subst ht0
    simp [S_F2] at hb
  | conv d h ds ih _ => exact ih
  | _ => trivial

/-- F2 (high): without H2's well-founded citation clause, a proof citing itself through a where entry
(R/p14: `impl<T: Foo> Foo for T`, a proof deriving `Foo[i32]` from a witness of itself) types a closed
allege-free term for an application the logic does not derive. Every other hypothesis of theorem 3
is satisfied. -/
example : ∃ (M : Module) (e : Ev) (t : TraitId) (as : List Ty),
    M.sig.DeclWF ∧
    (∀ p (hp : p < M.proofs.length),
      (∃ π : Option Cite, HasType freeAlgebra M AnyImpl [] M.proofs[p].2 ⟨M.proofs[p].1, π⟩) ∧
      (M.proofs[p].1).ground = true ∧ (M.proofs[p].2).allegeFree) ∧
    e.allegeFree ∧ HasType freeAlgebra M AnyImpl [] e ⟨.app t as, none⟩ ∧
    ¬ Valid freeAlgebra M.sig [] (.app t as) :=
  ⟨M_F2, .derive 0 [i32] [.witness (.proof 0) (.app 0 [i32])], 0, [i32],
   S_F2_wf, M_F2_proofs, rfl, M_F2_body_typed, fun h => h.elim fun d => S_F2_noApp d⟩

/-! ## F3: the converse of `compose_derivable` fails (X/x1a, X/x1b) -/

/-- F3 (medium): the converse of `compose_derivable` fails (X/x1a accepted, X/x1b refused): a coerce
of a `refl` with no equality operand types `l = l`, which `compose` with no operand does not. -/
example : ∃ (M : Module) (l : Ty),
    HasType freeAlgebra M AnyImpl [] (.coerce (.refl l) [] ⟨.eq l l, none⟩) ⟨.eq l l, none⟩ ∧
    ¬ HasType freeAlgebra M AnyImpl [] (.compose [] l l) ⟨.eq l l, none⟩ :=
  ⟨⟨⟨[], []⟩, []⟩, i32,
   HasType.coerce (E := []) HasType.refl rfl (fun j hj => absurd hj (Nat.not_lt_zero j)) (Or.inl rfl)
     trivial (PredEntails.eq Entails.refl Entails.refl),
   fun h => by cases h with | compose hne _ _ _ => exact hne rfl⟩

/-! ## F7: a coerce carries its citation to another application (Q/p20) -/

/-- Q/p20's module: one trait `T` of arity 1 and one proof of `T[i32]` (its body is not read here). -/
private def M_F7 : Module :=
  ⟨⟨[⟨1, [], [], []⟩], []⟩, [(.app 0 [i32], .allege ⟨.app 0 [i32], none⟩)]⟩

/-- F7 (low): a coerce may carry its input's citation to another application (Q/p20: `<@T[i32] by
@T_i32>` to `<@T[i64] by @T_i32>` via `i32 = i64`); the result names a proof of a different predicate. -/
example : ∃ (M : Module) (Γ : List Claim) (e : Ev) (P : Pred) (p : ProofId) (hp : p < M.proofs.length),
    HasType freeAlgebra M AnyImpl Γ e ⟨P, some (.proof p)⟩ ∧ M.proofs[p].1 ≠ P :=
  ⟨M_F7, [⟨.eq i32 i64, none⟩],
   .coerce (.witness (.proof 0) (.app 0 [i32])) [.hyp 0] ⟨.app 0 [i64], some (.proof 0)⟩,
   .app 0 [i64], 0, by decide,
   HasType.coerce (E := [(i32, i64)]) (HasType.witProof (by decide) rfl) rfl
     (fun j hj => by
       cases j with
       | zero => exact HasType.hyp (Γ := [⟨.eq i32 i64, none⟩]) (k := 0) (by decide)
       | succ j => simp at hj)
     (Or.inr rfl) trivial
     (PredEntails.app rfl (fun i hi => by
       cases i with
       | zero => exact Entails.ax (List.Mem.head _)
       | succ i => simp at hi)),
   by decide⟩

/-! ## F13: DeclWF does not give H3 (X/x9) -/

/-- X/x9's signature: trait `Has` of arity 1 with one associated type `Out` (no parameters of its own)
and the plain requirement `Has[a0]::Out = i64`; `impl Has for i32 { Out = i32 }`, returning for the
requirement the allegation X/x9 writes. -/
private def S_F13 : Sig :=
  ⟨[⟨1, [.eq (.proj 0 [.var 0] 0 []) i64], [], [[]]⟩],
   [⟨0, 0, [i32], [], [some ([], i32)], [.allege ⟨.eq (.proj 0 [i32] 0 []) i64, none⟩], []⟩]⟩

/-- `S_F13` passes every check `TraitOp::verify` and `ImplOp::verify` make. -/
private theorem S_F13_wf : S_F13.DeclWF where
  trait_reqs := by
    intro t T hT r hr
    cases t with
    | zero =>
      simp [S_F13] at hT; subst hT
      simp at hr; subst hr
      exact ⟨by decide, by decide, fun as h => by cases h⟩
    | succ t => simp [S_F13] at hT
  trait_binders := by
    intro T hT B hB
    simp [S_F13] at hT; subst hT; simp at hB
  trait_assocs := by
    intro T hT X hX
    simp [S_F13] at hT; subst hT
    simp at hX; subst hX
    exact ⟨List.nodup_nil, by simp⟩
  impl_trait := by
    intro I hI
    simp [S_F13] at hI; subst hI
    exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  impl_wf := by
    intro I hI
    simp [S_F13] at hI; subst hI
    refine ⟨?_, ?_, ?_⟩
    · intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩
    · intro W hW; simp at hW
    · intro T hT a labels B hB
      simp [S_F13] at hT; subst hT
      cases a with
      | zero =>
        simp at hB
        obtain ⟨rfl, rfl⟩ := hB
        exact ⟨[], rfl, rfl, List.nodup_nil, by simp, by decide, by decide⟩
      | succ a => simp at hB
  impl_params_constrained := by
    intro I hI v hv
    simp [S_F13] at hI; subst hI
    simp at hv

mutual
  /-- The model for F13: whether `i64` (constructor `1`) occurs in a type. -/
  private def hasI64 : Ty -> Bool
    | .var _ => false
    | .con c as => c == 1 || hasI64List as
    | .proj _ as _ gs => hasI64List as || hasI64List gs
  private def hasI64List : List Ty -> Bool
    | [] => false
    | a :: as => hasI64 a || hasI64List as
end

/-- Two lists of one length whose members agree pairwise on `hasI64` agree on `hasI64List`. -/
private theorem hasI64List_congr : ∀ (as bs : List Ty), as.length = bs.length ->
    (∀ i, i < as.length -> hasI64 (as.getD i default) = hasI64 (bs.getD i default)) ->
    hasI64List as = hasI64List bs
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: as, b :: bs, h, hi => by
    have h0 := hi 0 (by simp)
    simp only [List.getD_cons_zero] at h0
    have htl : hasI64List as = hasI64List bs :=
      hasI64List_congr as bs (by simpa using h) (fun i hi' => by
        have := hi (i+1) (by simpa using hi')
        simpa only [List.getD_cons_succ] using this)
    simp only [hasI64List, h0, htl]

/-- What the model says of a predicate: the two sides of an equality agree on `hasI64`; an application
says nothing. -/
private def ModelOk : Pred -> Prop
  | .app _ _ => True
  | .eq a b => hasI64 a = hasI64 b

/-- Every primitive derivation under no hypotheses in `S_F13` respects the model: (BIND) relates
`Has[i32]::Out` to `i32`, neither mentioning `i64`; the equality rules preserve agreement; (REQ) and
(REQ-A) have no `Env([])` member to read. -/
private theorem S_F13_model : ∀ {P : Pred} (d : Deriv freeAlgebra S_F13 AnyImpl [] P),
    d.Primitive -> ModelOk P := by
  intro P d
  induction d with
  | hyp k hk => exact absurd hk (Nat.not_lt_zero k)
  | impl i hi σ hσ ok ds ih => intro _; trivial
  | req d ht hk ih =>
    intro h
    simp only [Deriv.Primitive] at h
    exact absurd h (not_inEnv_nil d)
  | reqA d ht hb us hus prem ih _ =>
    intro h
    simp only [Deriv.Primitive] at h
    exact absurd h.1 (not_inEnv_nil d)
  | @bind a labels B gs i hi σ hσ ok ds hb hg ih =>
    intro _
    have hi0 : i = 0 := by simp [S_F13] at hi; omega
    subst hi0
    have hσ0 : σ = [] := by simpa [S_F13] using hσ
    subst hσ0
    cases a with
    | zero =>
      simp [S_F13] at hb
      obtain ⟨rfl, rfl⟩ := hb
      have hg0 : gs = [] := by simpa using hg
      subst hg0
      exact rfl
    | succ a => simp [S_F13] at hb
  | refl => intro _; exact rfl
  | symm d ih =>
    intro h
    simp only [Deriv.Primitive] at h
    exact Eq.symm (ih h)
  | trans d₁ d₂ ih₁ ih₂ =>
    intro h
    simp only [Deriv.Primitive] at h
    exact Eq.trans (ih₁ h.1) (ih₂ h.2)
  | @cong c as bs hl ds ih =>
    intro h
    simp only [Deriv.Primitive] at h
    show hasI64 (.con c as) = hasI64 (.con c bs)
    simp only [hasI64, hasI64List_congr as bs hl (fun j hj => ih j hj (h j hj))]
  | @congP t as a gs bs hs hl₁ hl₂ ds es ih₁ ih₂ =>
    intro h
    simp only [Deriv.Primitive] at h
    show hasI64 (.proj t as a gs) = hasI64 (.proj t bs a hs)
    simp only [hasI64, hasI64List_congr as bs hl₁ (fun j hj => ih₁ j hj (h.1 j hj)),
      hasI64List_congr gs hs hl₂ (fun j hj => ih₂ j hj (h.2 j hj))]
  | norm hl =>
    intro _
    obtain ⟨c, as, rfl, rfl⟩ := hl
    exact rfl
  | conv d h ds ih _ => intro _; trivial

/-- H3 fails in `S_F13`: the impl's one requirement, `Has[i32]::Out = i64`, has no primitive derivation
from its (empty) where entries. -/
private theorem S_F13_not_reqDuty : ¬ S_F13.ReqDuty freeAlgebra := by
  intro hD
  obtain ⟨d, hd⟩ := hD.1 0 (by decide) (by decide) 0 (by decide)
  have h : hasI64 (.proj 0 [i32] 0 []) = hasI64 i64 := S_F13_model d hd
  exact absurd h (by decide)

/-- F13 (medium): the declaration well-formedness the verifier enforces does not give H3 (X/x9: trait
`Has` requires `Has[Self]::Out = i64`; `impl Has for i32 { Out = i32 }` is a legal declaration whose
requirement `i32 = i64` the logic does not derive). -/
example : ∃ S : Sig, S.DeclWF ∧ ¬ S.ReqDuty freeAlgebra := ⟨S_F13, S_F13_wf, S_F13_not_reqDuty⟩

/-! ## F18: two closed terms root at two impls (one_callee_called_with_two_impls_of_one_claim_runs_each.mlir) -/

/-- The lit test's signature: one trait `Tr` of arity 1 and two unconditional impls `@One`, `@Two`,
both with header `Tr[i32]`. -/
private def S_F18 : Sig :=
  ⟨[⟨1, [], [], []⟩], [⟨0, 0, [i32], [], [], [], []⟩, ⟨0, 0, [i32], [], [], [], []⟩]⟩

/-- `S_F18` passes every check `TraitOp::verify` and `ImplOp::verify` make: coherence is not among them. -/
private theorem S_F18_wf : S_F18.DeclWF where
  trait_reqs := by
    intro t T hT r hr
    cases t with
    | zero => simp [S_F18] at hT; subst hT; simp at hr
    | succ t => simp [S_F18] at hT
  trait_binders := by
    intro T hT B hB
    simp [S_F18] at hT; subst hT; simp at hB
  trait_assocs := by
    intro T hT X hX
    simp [S_F18] at hT; subst hT; simp at hX
  impl_trait := by
    intro I hI
    simp [S_F18] at hI; subst hI
    exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  impl_wf := by
    intro I hI
    simp [S_F18] at hI; subst hI
    refine ⟨?_, ?_, ?_⟩
    · intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩
    · intro W hW; simp at hW
    · intro T hT a labels B hB; simp at hB
  impl_params_constrained := by
    intro I hI v hv
    simp [S_F18] at hI; subst hI
    simp at hv

/-- F18 (observation): coherence is not a verifier invariant. Two impls of one ground application are
legal declarations, and two closed terms root at the two of them
(T/one_callee_called_with_two_impls_of_one_claim_runs_each.mlir). -/
example : ∃ (M : Module) (e₁ e₂ : Ev) (t : TraitId) (as : List Ty) (i₁ i₂ : ImplId) (σ₁ σ₂ : List Ty),
    M.sig.DeclWF ∧
    e₁.allegeFree ∧ HasType freeAlgebra M AnyImpl [] e₁ ⟨.app t as, none⟩ ∧
    e₂.allegeFree ∧ HasType freeAlgebra M AnyImpl [] e₂ ⟨.app t as, none⟩ ∧
    root M e₁ i₁ σ₁ ∧ root M e₂ i₂ σ₂ ∧ i₁ ≠ i₂ :=
  ⟨⟨S_F18, []⟩, .derive 0 [] [], .derive 1 [] [], 0, [i32], 0, 1, [], [],
   S_F18_wf,
   rfl, HasType.derive (by decide) rfl trivial rfl (fun _ => none) (fun j hj => absurd hj (Nat.not_lt_zero j)),
   rfl, HasType.derive (by decide) rfl trivial rfl (fun _ => none) (fun j hj => absurd hj (Nat.not_lt_zero j)),
   ⟨[], Root.derive⟩, ⟨[], Root.derive⟩, by decide⟩

/-! ## The inversion theorem needs H2: the index is read against the derive's trait -/

/-- Two traits: `T0` (0) with no requirement and `T1` (1) with the requirement `T0[a0]`. Impl `I0` (0)
of `T1` at `i32` with the where entry `T0[i32]`, returning a `refl` for the requirement (a return no
reading reaches a base through); impl `I1` (1) of `T0` at `i32`, unconditional. -/
private def S_inv : Sig :=
  ⟨[⟨1, [], [], []⟩, ⟨1, [.app 0 [.var 0]], [], []⟩],
   [⟨0, 1, [i32], [.app 0 [i32]], [], [.refl i32], []⟩, ⟨0, 0, [i32], [], [], [], []⟩]⟩

/-- `S_inv` passes every check `TraitOp::verify` and `ImplOp::verify` make. -/
private theorem S_inv_wf : S_inv.DeclWF where
  trait_reqs := by
    intro t T hT r hr
    match t, hT with
    | 0, hT => simp [S_inv] at hT; subst hT; simp at hr
    | 1, hT =>
      simp [S_inv] at hT; subst hT; simp at hr; subst hr
      exact ⟨by decide, by decide, fun as h => by cases h⟩
    | t+2, hT => simp [S_inv] at hT
  trait_binders := by
    intro T hT B hB
    simp [S_inv] at hT; rcases hT with rfl | rfl <;> simp at hB
  trait_assocs := by
    intro T hT X hX
    simp [S_inv] at hT; rcases hT with rfl | rfl <;> simp at hX
  impl_trait := by
    intro I hI
    simp [S_inv] at hI; rcases hI with rfl | rfl <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl⟩
  impl_wf := by
    intro I hI
    simp [S_inv] at hI
    rcases hI with rfl | rfl <;> refine ⟨?_, ?_, ?_⟩
    all_goals first
      | (intro h hh; simp at hh; subst hh; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; subst hW; exact ⟨by decide, by decide⟩)
      | (intro W hW; simp at hW; done)
      | (intro T hT a labels B hB; simp at hB)
  impl_params_constrained := by
    intro I hI v hv
    simp [S_inv] at hI; rcases hI with rfl | rfl <;> simp at hv

/-- The module whose proof table says `@p` proves `T0[i32]` while its body derives `T1[i32]` from
`I0` given a derive of `I1`: H2 fails, the body being typed at another application than the table
states. -/
private def M_inv : Module := ⟨S_inv, [(.app 0 [i32], .derive 0 [] [.derive 1 [] []])]⟩

/-- Without H2 the inversion `project_where_is_inversion` states fails: the projection's index
`|R_T0| + 0 = 0` is read by `Root` against the derive's trait `T1`, whose one plain requirement makes
it a PROJECT-REQ reading through `I0`'s return (a `refl`, no base), while the operand it names by
inversion, the derive of `I1`, reads to `I1`. -/
example : ∃ (M : Module) (e : Ev) (t : TraitId) (as : List Ty) (p : ProofId) (i : ImplId) (σ : List Ty)
    (ds : List Ev) (j : Nat) (c : Claim) (ht : t < M.sig.traits.length),
    M.sig.DeclWF ∧
    HasType freeAlgebra M AnyImpl [] e ⟨.app t as, some (.proof p)⟩ ∧
    M.proofDerive p = some (i, σ, ds) ∧ j < ds.length ∧
    HasType freeAlgebra M AnyImpl [] (ds.getD j default) c ∧
    Root M e (i, σ, ds) ∧
    ¬ (∀ r, Root M (.project e (M.sig.traits[t].reqs.length + j)) r ↔ Root M (ds.getD j default) r) :=
  ⟨M_inv, .witness (.proof 0) (.app 0 [i32]), 0, [i32], 0, 0, [], [.derive 1 [] []], 0, ⟨.app 0 [i32], none⟩,
   by decide, S_inv_wf, HasType.witProof (by decide) rfl, rfl, by decide,
   HasType.derive (by decide) rfl trivial rfl (fun _ => none) (fun j hj => absurd hj (Nat.not_lt_zero j)),
   Root.witProof rfl Root.derive,
   fun h => by
     have h1 := (h (1, [], [])).mpr Root.derive
     have hroot : Root M_inv (.witness (.proof 0) (.app 0 [i32])) (0, [], [.derive 1 [] []]) :=
       Root.witProof rfl Root.derive
     cases h1 with
     | projReq hb hI hT hk hret =>
       cases root_unique M_inv hb hroot
       simp [M_inv, S_inv] at hI; subst hI
       simp [M_inv, S_inv] at hT; subst hT
       simp [M_inv, S_inv, Ev.substTy, Ev.substHyp, Ty.subst, substList] at hret
       cases hret
     | projWhere hb hI hT hk _ =>
       cases root_unique M_inv hb hroot
       simp [M_inv, S_inv] at hI; subst hI
       simp [M_inv, S_inv] at hT; subst hT
       simp [M_inv, S_inv] at hk⟩

/-! ## Scaffold observation: `witness_is_let` for any `Γ` fails at a proven hypothesis -/

/-- A module with one unconditional impl `@I` of `Tr[i32]` and the proof `@p` whose body is its derive. -/
private def M_W : Module :=
  ⟨⟨[⟨1, [], [], []⟩], [⟨0, 0, [i32], [], [], [], []⟩]⟩, [(.app 0 [i32], .derive 0 [] [])]⟩

/-- The one hypothesis in scope: `Tr[i32] by @p`. -/
private def Γ_W : List Claim := [⟨.app 0 [i32], some (.proof 0)⟩]

/-- `M_W` satisfies H2: its one proof is closed, allege-free, ground and cites nothing. -/
private theorem M_W_proofsWF : ProofsWF freeAlgebra M_W AnyImpl := by
  intro p hp
  cases p with
  | zero =>
    refine ⟨⟨none, ?_⟩, rfl, rfl, ?_⟩
    · exact HasType.derive (by decide) rfl trivial rfl (fun _ => none) (fun j hj => absurd hj (Nat.not_lt_zero j))
    · intro q hq; simp [M_W, Ev.cites, citesList] at hq
  | succ p => simp [M_W] at hp

/-- `Γ_W` satisfies G-WF: its hypothesis cites the proof of its own predicate. -/
private theorem M_W_gammaWF : GammaWF M_W Γ_W := by
  intro c hc
  simp [Γ_W] at hc; subst hc
  refine ⟨fun p hp => ?_, fun i hi => ?_⟩
  · simp at hp; subst hp; exact ⟨by decide, rfl⟩
  · simp at hi

/-- Scaffold observation, not in SPEC.md section 8: `witness_is_let` as the section 7 skeleton stated
it, for any `Γ`, fails at a proven hypothesis: `^0 : <P by @p>` is its own inlining and keeps its
citation, so its claim is not `c.strip`. The theorem is stated closed in Minimality.lean, as SPEC.md
6.4 states it. -/
example : ∃ (M : Module) (Γ : List Claim) (e : Ev) (c : Claim),
    ProofsWF freeAlgebra M AnyImpl ∧ GammaWF M Γ ∧ e.allegeFree ∧
    HasType freeAlgebra M AnyImpl Γ e c ∧
    ¬ HasType freeAlgebra M AnyImpl Γ (e.inlineProofs M) c.strip :=
  ⟨M_W, Γ_W, .hyp 0, ⟨.app 0 [i32], some (.proof 0)⟩, M_W_proofsWF, M_W_gammaWF, rfl,
   HasType.hyp (Γ := Γ_W) (k := 0) (by decide),
   fun h => by
     have h' : HasType freeAlgebra M_W AnyImpl Γ_W (.hyp 0) ⟨.app 0 [i32], none⟩ := by
       simpa [Ev.inlineProofs, inlineAux, Claim.strip] using h
     obtain ⟨hk, hc⟩ := hasType_hyp_inv h'
     simp [Γ_W] at hc⟩

end Trait
