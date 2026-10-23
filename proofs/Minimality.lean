import Calculus
/-!
# Minimality: theorem 6 (SPEC.md 6.4)

Independence: for each of `trait.derive`, `trait.project` (a plain requirement), `trait.witness
proj_resolve`, `trait.witness refl` and `trait.coerce`, a signature, a context and a valid judgment
every allege-free term for which uses the operation (cards 2, 3, 4, 4, 5), each by induction on the
typing derivation: without the operation, every claim a term can state in the witness module has a
shape the goal does not have. The witnesses SPEC.md names:
* derive (card 2): trait `Tr`; `impl<U> Tr for tuple<U> where Tr[U]`; `G = [Tr[a0]]`; goal `Tr[tuple<a0>]`.
* project (card 3): traits `Eq`, `Ord` with `R_Ord = [Eq[a0]]`, no impls; `G = [Ord[i32]]`; goal `Eq[i32]`.
* proj_resolve (card 4): trait `A` with `Item`, `impl A for i64 { Item = i32 }`; `G = []`; goal `A[i64]::Item = i32`.
* refl (card 4): `G = []`; goal `i32 = i32`.
* coerce (card 5): trait `Tr` with no impl for `i32`; `G = [Tr[a0], a0 = i32]`; goal `Tr[i32]`.

Redundancy of `trait.witness compose` (D1, F3): one direction; the converse fails (Findings.lean).
Let: a closed allege-free term with a witness of `@p` has the term with `p`'s body in its place, same
type up to the citation, same root (`witness_is_let`). Inversion (D2): a projection past the plain
requirements is the cited derive's operand in its place (`project_where_is_inversion`).

The root half of the let theorem goes through a relation `Sim` between a term and its inlining at any
fuel, with two lemmas: a reading of the term is a reading of its inlining to a related base, and
back. The type half carries, through the induction, what the inlining of a proof-cited term returns:
the cited proof's derive with its operands inlined at a fuel above every proof they cite
(`InlinedCite`), which is what a projection past the plain requirements reads. The let theorem is
proved as SPEC.md 6.4 states it, closed; the inversion theorem takes H2, which the skeleton omitted
(Findings.lean has the refutation without it).
-/
namespace Trait


private theorem list_eq_of_getD {α : Type} [Inhabited α] : ∀ (as bs : List α), as.length = bs.length ->
    (∀ i, i < as.length -> as.getD i default = bs.getD i default) -> as = bs
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: as, b :: bs, h, hi => by
    have h0 := hi 0 (Nat.zero_lt_succ _)
    simp [List.getD] at h0
    have := list_eq_of_getD as bs (by simpa using h) (fun i hi' => by
      have := hi (i+1) (by simpa using Nat.succ_lt_succ hi')
      simpa [List.getD] using this)
    rw [h0, this]

private theorem entails_trivial {E : List (Ty × Ty)} (hE : ∀ l r, (l, r) ∈ E -> l = r) {a b : Ty}
    (h : Entails freeAlgebra E a b) : a = b := by
  induction h with
  | ax h => exact hE _ _ h
  | refl => rfl
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | con hl _ ih => rw [list_eq_of_getD _ _ hl ih]
  | proj hl hg _ _ ih₁ ih₂ => rw [list_eq_of_getD _ _ hl ih₁, list_eq_of_getD _ _ hg ih₂]
  | law h => obtain ⟨c, as, ha, hb⟩ := h; subst ha; exact hb.symm

private theorem predEntails_trivial {E : List (Ty × Ty)} (hE : ∀ l r, (l, r) ∈ E -> l = r) {P Q : Pred}
    (h : PredEntails freeAlgebra E P Q) : P = Q := by
  cases h with
  | app hl h => rw [list_eq_of_getD _ _ hl (fun i hi => entails_trivial hE (h i hi))]
  | eq h₁ h₂ => rw [entails_trivial hE h₁, entails_trivial hE h₂]

private theorem getD_mem {α : Type} [Inhabited α] (as : List α) (j : Nat) (hj : j < as.length) :
    as.getD j default ∈ as := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]; exact List.getElem_mem hj

private theorem anyList_false {f : Ev -> Bool} : ∀ {es : List Ev}, anyList f es = false -> ∀ e ∈ es, e.any f = false
  | [], _, _, h => by simp at h
  | e :: es, h, e', he' => by
    simp only [anyList, Bool.or_eq_false_iff] at h
    cases List.mem_cons.mp he' with
    | inl h' => subst h'; exact h.1
    | inr h' => exact anyList_false h.2 e' h'

private theorem hasAllegeList_false : ∀ {es : List Ev}, hasAllegeList es = false -> ∀ e ∈ es, e.hasAllege = false
  | [], _, _, h => by simp at h
  | e :: es, h, e', he' => by
    simp only [hasAllegeList, Bool.or_eq_false_iff] at h
    cases List.mem_cons.mp he' with
    | inl h' => subst h'; exact h.1
    | inr h' => exact hasAllegeList_false h.2 e' h'

/-- The seeds a list of equality operands proves are trivial when each operand's claim is. -/
private theorem seeds_trivial {E : List (Ty × Ty)} {es : List Ev} (hE : E.length = es.length)
    (h : ∀ j, j < es.length -> (E.getD j default).1 = (E.getD j default).2) :
    ∀ l r, (l, r) ∈ E -> l = r := by
  intro l r hlr
  obtain ⟨j, hj, hget⟩ := List.getElem_of_mem hlr
  have := h j (hE ▸ hj)
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget] at this
  exact this

/-! ## Card 2: `trait.derive` -/

/-- Trait `Tr`, one parameter; `impl<U> Tr for tuple<U> where Tr[U]`; no proofs. `tuple` is constructor 0. -/
private def M_derive : Module :=
  ⟨⟨[⟨1, [], [], []⟩], [⟨1, 0, [.con 0 [.var 0]], [.app 0 [.var 0]], [], [], []⟩]⟩, []⟩

/-- What a derive-free term can state in `M_derive` under `G = [Tr[a0]]`: the hypothesis itself, or a
trivial equality. -/
private def GoodD (c : Claim) : Prop :=
  match c.pred with
  | .app _ bs => bs = [.var 0]
  | .eq l r => l = r

private theorem goodD_of_hasType {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType freeAlgebra M_derive AnyImpl Γ e c) (hΓ : ∀ c ∈ Γ, GoodD c)
    (he : e.hasAllege = false) (hd : e.any Ev.isDerive = false) : GoodD c := by
  induction ht with
  | hyp hk => exact hΓ _ (List.getElem_mem hk)
  | derive => simp [Ev.any, Ev.isDerive] at hd
  | projReq _ ht hk => simp [M_derive] at ht; subst ht; simp [M_derive] at hk
  | projWhere _ _ hder => simp [M_derive, Module.proofDerive, Module.proofDeriveAux] at hder
  | reqCall _ ht hb => simp [M_derive] at ht; subst ht; simp [M_derive] at hb
  | witProof hp => simp [M_derive] at hp
  | witImpl hi hu => simp [M_derive] at hi; subst hi; simp [M_derive, ImplDecl.unconditional] at hu
  | projRes hi _ _ _ _ _ hb => simp [M_derive] at hi; subst hi; simp [M_derive] at hb
  | refl => simp [GoodD]
  | compose _ hE _ hent ih =>
    simp only [Ev.hasAllege] at he
    simp only [Ev.any, Ev.isDerive, Bool.false_or] at hd
    have hseeds := seeds_trivial hE (fun j hj => ih j hj hΓ
      (hasAllegeList_false he _ (getD_mem _ j hj)) (anyList_false hd _ (getD_mem _ j hj)))
    simp only [GoodD]; exact entails_trivial hseeds hent
  | coerce _ hE _ _ _ hpe ih ihs =>
    simp only [Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.any, Ev.isDerive, Bool.false_or, Bool.or_eq_false_iff] at hd
    have hseeds := seeds_trivial hE (fun j hj => ihs j hj hΓ
      (hasAllegeList_false he.2 _ (getD_mem _ j hj)) (anyList_false hd.2 _ (getD_mem _ j hj)))
    have := predEntails_trivial hseeds hpe
    subst this
    exact ih hΓ he.1 hd.1

theorem independence_derive : ∃ (M : Module) (Γ : List Claim) (P : Pred),
    Valid freeAlgebra M.sig (Γ.map Claim.pred) P ∧
    ∀ e, e.allegeFree -> HasType freeAlgebra M AnyImpl Γ e ⟨P, none⟩ -> e.uses Ev.isDerive := by
  refine ⟨M_derive, [⟨.app 0 [.var 0], none⟩], .app 0 [.con 0 [.var 0]], ?_, ?_⟩
  · refine ⟨.impl 0 (by decide) [.var 0] rfl trivial (fun j hj => ?_)⟩
    have hj0 : j = 0 := by simp [M_derive] at hj; omega
    subst hj0
    exact .hyp 0 (by decide)
  · intro e he ht
    show e.any Ev.isDerive = true
    cases hd : e.any Ev.isDerive with
    | true => rfl
    | false =>
      have := goodD_of_hasType ht (by simp [GoodD]) he hd
      simp [GoodD] at this

/-! ## Card 3: `trait.project` at a plain requirement -/

/-- Traits `Eq` (0) and `Ord` (1) with `R_Ord = [Eq[a0]]`; no impls, no proofs. `i32` is constructor 1. -/
private def M_project : Module :=
  ⟨⟨[⟨1, [], [], []⟩, ⟨1, [.app 0 [.var 0]], [], []⟩], []⟩, []⟩

/-- Neither trait of `M_project` has a binder. -/
private theorem M_project_no_binders : ∀ t (ht : t < M_project.sig.traits.length),
    M_project.sig.traits[t].binders = [] := by
  intro t ht
  match t, ht with
  | 0, _ => rfl
  | 1, _ => rfl

/-- What a project-free term can state in `M_project` under `G = [Ord[i32]]`. -/
private def GoodP (c : Claim) : Prop :=
  match c.pred with
  | .app t bs => t = 1 ∧ bs = [.con 1 []]
  | .eq l r => l = r

private theorem goodP_of_hasType {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType freeAlgebra M_project AnyImpl Γ e c) (hΓ : ∀ c ∈ Γ, GoodP c)
    (he : e.hasAllege = false) (hd : e.any Ev.isProject = false) : GoodP c := by
  induction ht with
  | hyp hk => exact hΓ _ (List.getElem_mem hk)
  | derive hi => simp [M_project] at hi
  | projReq => simp [Ev.any, Ev.isProject] at hd
  | projWhere => simp [Ev.any, Ev.isProject] at hd
  | reqCall _ ht hb => rw [M_project_no_binders _ ht] at hb; simp at hb
  | witProof hp => simp [M_project] at hp
  | witImpl hi => simp [M_project] at hi
  | projRes hi => simp [M_project] at hi
  | refl => simp [GoodP]
  | compose _ hE _ hent ih =>
    simp only [Ev.hasAllege] at he
    simp only [Ev.any, Ev.isProject, Bool.false_or] at hd
    have hseeds := seeds_trivial hE (fun j hj => ih j hj hΓ
      (hasAllegeList_false he _ (getD_mem _ j hj)) (anyList_false hd _ (getD_mem _ j hj)))
    simp only [GoodP]; exact entails_trivial hseeds hent
  | coerce _ hE _ _ _ hpe ih ihs =>
    simp only [Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.any, Ev.isProject, Bool.false_or, Bool.or_eq_false_iff] at hd
    have hseeds := seeds_trivial hE (fun j hj => ihs j hj hΓ
      (hasAllegeList_false he.2 _ (getD_mem _ j hj)) (anyList_false hd.2 _ (getD_mem _ j hj)))
    have := predEntails_trivial hseeds hpe
    subst this
    exact ih hΓ he.1 hd.1

/-- Card 3: `trait.project` at a plain requirement is independent. -/
theorem independence_project : ∃ (M : Module) (Γ : List Claim) (P : Pred),
    Valid freeAlgebra M.sig (Γ.map Claim.pred) P ∧
    ∀ e, e.allegeFree -> HasType freeAlgebra M AnyImpl Γ e ⟨P, none⟩ -> e.uses Ev.isProject := by
  refine ⟨M_project, [⟨.app 1 [.con 1 []], none⟩], .app 0 [.con 1 []], ?_, ?_⟩
  · have d0 : Deriv freeAlgebra M_project.sig AnyImpl ([⟨.app 1 [.con 1 []], none⟩].map Claim.pred)
        (.app 1 [.con 1 []]) := .hyp 0 (by decide)
    exact ⟨.req (k := 0) d0 (by decide) (by decide)⟩
  · intro e he ht
    show e.any Ev.isProject = true
    cases hd : e.any Ev.isProject with
    | true => rfl
    | false =>
      have := goodP_of_hasType ht (by simp [GoodP]) he hd
      simp [GoodP] at this

/-! ## Card 4: `trait.witness proj_resolve` -/

/-- Trait `A` (0) with one associated type `Item` of no parameters; `impl A for i64 { Item = i32 }`,
unconditional; no proofs. `i32` is constructor 1, `i64` constructor 2. -/
private def M_projRes : Module :=
  ⟨⟨[⟨1, [], [], [[]]⟩], [⟨0, 0, [.con 2 []], [], [some ([], .con 1 [])], [], []⟩]⟩, []⟩

/-- What a proj_resolve-free term can state in `M_projRes`: any application, a trivial equality. -/
private def GoodR (c : Claim) : Prop :=
  match c.pred with
  | .app _ _ => True
  | .eq l r => l = r

private theorem goodR_of_hasType {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType freeAlgebra M_projRes AnyImpl Γ e c) (hΓ : ∀ c ∈ Γ, GoodR c)
    (he : e.hasAllege = false) (hd : e.any Ev.isProjRes = false) : GoodR c := by
  induction ht with
  | hyp hk => exact hΓ _ (List.getElem_mem hk)
  | derive => simp [GoodR, ImplDecl.header]
  | projReq _ ht hk => simp [M_projRes] at ht; subst ht; simp [M_projRes] at hk
  | projWhere _ _ hder => simp [M_projRes, Module.proofDerive, Module.proofDeriveAux] at hder
  | reqCall _ ht hb => simp [M_projRes] at ht; subst ht; simp [M_projRes] at hb
  | witProof hp => simp [M_projRes] at hp
  | witImpl => simp [GoodR, ImplDecl.header]
  | projRes => simp [Ev.any, Ev.isProjRes] at hd
  | refl => simp [GoodR]
  | compose _ hE _ hent ih =>
    simp only [Ev.hasAllege] at he
    simp only [Ev.any, Ev.isProjRes, Bool.false_or] at hd
    have hseeds := seeds_trivial hE (fun j hj => ih j hj hΓ
      (hasAllegeList_false he _ (getD_mem _ j hj)) (anyList_false hd _ (getD_mem _ j hj)))
    simp only [GoodR]; exact entails_trivial hseeds hent
  | coerce _ hE _ _ _ hpe ih ihs =>
    simp only [Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.any, Ev.isProjRes, Bool.false_or, Bool.or_eq_false_iff] at hd
    have hseeds := seeds_trivial hE (fun j hj => ihs j hj hΓ
      (hasAllegeList_false he.2 _ (getD_mem _ j hj)) (anyList_false hd.2 _ (getD_mem _ j hj)))
    have := predEntails_trivial hseeds hpe
    subst this
    exact ih hΓ he.1 hd.1

/-- Card 4: `trait.witness proj_resolve` is independent. -/
theorem independence_projRes : ∃ (M : Module) (P : Pred), Valid freeAlgebra M.sig [] P ∧
    ∀ e, e.allegeFree -> HasType freeAlgebra M AnyImpl [] e ⟨P, none⟩ -> e.uses Ev.isProjRes := by
  refine ⟨M_projRes, .eq (.proj 0 [.con 2 []] 0 []) (.con 1 []), ?_, ?_⟩
  · exact ⟨.bind (a := 0) (labels := []) (B := .con 1 []) (gs := []) 0 (by decide) [] rfl trivial
      (fun j hj => by simp [M_projRes] at hj) (by decide) rfl⟩
  · intro e he ht
    show e.any Ev.isProjRes = true
    cases hd : e.any Ev.isProjRes with
    | true => rfl
    | false =>
      have := goodR_of_hasType ht (by simp) he hd
      simp [GoodR] at this

/-! ## Card 4: `trait.witness refl` -/

/-- The empty module: no traits, no impls, no proofs. -/
private def M_refl : Module := ⟨⟨[], []⟩, []⟩

/-- What a refl-free closed term can state in `M_refl`: an application, never an equality. -/
private def GoodF (c : Claim) : Prop :=
  match c.pred with
  | .app _ _ => True
  | .eq _ _ => False

private theorem goodF_of_hasType {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType freeAlgebra M_refl AnyImpl Γ e c) (hΓ : ∀ c ∈ Γ, GoodF c)
    (he : e.hasAllege = false) (hd : e.any Ev.isRefl = false) : GoodF c := by
  induction ht with
  | hyp hk => exact hΓ _ (List.getElem_mem hk)
  | derive hi => simp [M_refl] at hi
  | projReq _ ht => simp [M_refl] at ht
  | projWhere _ ht => simp [M_refl] at ht
  | reqCall _ ht => simp [M_refl] at ht
  | witProof hp => simp [M_refl] at hp
  | witImpl hi => simp [M_refl] at hi
  | projRes hi => simp [M_refl] at hi
  | refl => simp [Ev.any, Ev.isRefl] at hd
  | compose hne hE _ _ ih =>
    simp only [Ev.hasAllege] at he
    simp only [Ev.any, Ev.isRefl, Bool.false_or] at hd
    have h0 : 0 < _ := List.length_pos_iff.mpr hne
    have := ih 0 h0 hΓ (hasAllegeList_false he _ (getD_mem _ 0 h0)) (anyList_false hd _ (getD_mem _ 0 h0))
    simp [GoodF] at this
  | coerce _ hE _ _ _ hpe ih ihs =>
    simp only [Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.any, Ev.isRefl, Bool.false_or, Bool.or_eq_false_iff] at hd
    have hin := ih hΓ he.1 hd.1
    cases hpe with
    | app => simp [GoodF]
    | eq => simp [GoodF] at hin

/-- Card 4: `trait.witness refl` is independent. -/
theorem independence_refl : ∃ (M : Module) (P : Pred), Valid freeAlgebra M.sig [] P ∧
    ∀ e, e.allegeFree -> HasType freeAlgebra M AnyImpl [] e ⟨P, none⟩ -> e.uses Ev.isRefl := by
  refine ⟨M_refl, .eq (.con 1 []) (.con 1 []), ⟨.refl⟩, ?_⟩
  intro e he ht
  show e.any Ev.isRefl = true
  cases hd : e.any Ev.isRefl with
  | true => rfl
  | false =>
    have := goodF_of_hasType ht (by simp) he hd
    simp [GoodF] at this

/-! ## Card 5: `trait.coerce` on a claim -/

/-- Trait `Tr` (0) with no impls and no proofs. -/
private def M_coerce : Module := ⟨⟨[⟨1, [], [], []⟩], []⟩, []⟩

/-- What a coerce-free term can state in `M_coerce` under `G = [Tr[a0], a0 = i32]`: `Tr[a0]` is the
only application. -/
private def GoodC (c : Claim) : Prop :=
  match c.pred with
  | .app _ bs => bs = [.var 0]
  | .eq _ _ => True

private theorem goodC_of_hasType {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType freeAlgebra M_coerce AnyImpl Γ e c) (hΓ : ∀ c ∈ Γ, GoodC c)
    (he : e.hasAllege = false) (hd : e.any Ev.isCoerce = false) : GoodC c := by
  induction ht with
  | hyp hk => exact hΓ _ (List.getElem_mem hk)
  | derive hi => simp [M_coerce] at hi
  | projReq _ ht hk => simp [M_coerce] at ht; subst ht; simp [M_coerce] at hk
  | projWhere _ _ hder => simp [M_coerce, Module.proofDerive, Module.proofDeriveAux] at hder
  | reqCall _ ht hb => simp [M_coerce] at ht; subst ht; simp [M_coerce] at hb
  | witProof hp => simp [M_coerce] at hp
  | witImpl hi => simp [M_coerce] at hi
  | projRes hi => simp [M_coerce] at hi
  | refl => simp [GoodC]
  | compose => simp [GoodC]
  | coerce => simp [Ev.any, Ev.isCoerce] at hd

/-- Card 5: `trait.coerce` on a claim is independent. -/
theorem independence_coerce : ∃ (M : Module) (Γ : List Claim) (P : Pred),
    Valid freeAlgebra M.sig (Γ.map Claim.pred) P ∧
    ∀ e, e.allegeFree -> HasType freeAlgebra M AnyImpl Γ e ⟨P, none⟩ -> e.uses Ev.isCoerce := by
  refine ⟨M_coerce, [⟨.app 0 [.var 0], none⟩, ⟨.eq (.var 0) (.con 1 []), none⟩], .app 0 [.con 1 []], ?_, ?_⟩
  · refine ⟨.conv (as := [.var 0]) (.hyp 0 (by decide)) rfl (fun i hi => ?_)⟩
    have hi0 : i = 0 := by simp at hi; omega
    subst hi0
    exact .hyp 1 (by decide)
  · intro e he ht
    show e.any Ev.isCoerce = true
    cases hd : e.any Ev.isCoerce with
    | true => rfl
    | false =>
      have := goodC_of_hasType ht (by simp [GoodC]) he hd
      simp [GoodC] at this

/-! ## D1: `trait.witness compose` is carried by `trait.coerce` -/

/-- D1, one direction (F3): every accepted `compose(es) : l = r` has the accepted
`coerce (refl : l = l) to (l = r) via (es)`. The converse fails: Findings.lean. -/
theorem compose_derivable (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim}
    {es : List Ev} {l r : Ty} :
    HasType L M Ok Γ (.compose es l r) ⟨.eq l r, none⟩ ->
    HasType L M Ok Γ (.coerce (.refl l) es ⟨.eq l r, none⟩) ⟨.eq l r, none⟩ := by
  intro h
  cases h with
  | compose _ hE hes hent =>
    exact .coerce .refl hE hes (Or.inl rfl) (by simp [Claim.WF]) (.eq .refl hent)





/-! ## Lists of citations -/

/-- A proof cited by a list is cited by one of its members. -/
private theorem mem_citesList_inv : ∀ {es : List Ev} {q : ProofId}, q ∈ citesList es -> ∃ e ∈ es, q ∈ e.cites
  | [], _, h => by simp [citesList] at h
  | e :: es, q, h => by
    simp only [citesList, List.mem_append] at h
    rcases h with h | h
    · exact ⟨e, List.mem_cons_self .., h⟩
    · obtain ⟨e', he', hq⟩ := mem_citesList_inv h
      exact ⟨e', List.mem_cons_of_mem _ he', hq⟩

/-! ## What a proof body returns (`Ev.returned`), read against its typing and its root -/

/-- A body that returns a derive reads to it. -/
private theorem root_of_returned_inl (M : Module) : ∀ {e : Ev} {d}, e.returned = some (.inl d) -> Root M e d
  | .derive i σ es, d, h => by simp [Ev.returned] at h; subst h; exact .derive
  | .witness (.impl i) P, d, h => by simp [Ev.returned] at h; subst h; exact .witImpl
  | .witness (.proof q) P, d, h => by simp [Ev.returned] at h
  | .coerce e es c, d, h => .coerce (root_of_returned_inl M (by simpa [Ev.returned] using h))
  | .hyp _, _, h | .allege _, _, h | .project _ _, _, h | .reqCall _ _ _ _, _, h
  | .projRes _ _ _ _ _, _, h | .refl _, _, h | .compose _ _ _, _, h => by simp [Ev.returned] at h

/-- A body that returns a witness of another proof reads as that proof does. -/
private theorem root_of_returned_inr (M : Module) : ∀ {e : Ev} {q : ProofId} {Q : Pred} {r},
    e.returned = some (.inr q) -> Root M (.witness (.proof q) Q) r -> Root M e r
  | .witness (.proof q') P, q, Q, r, h, hr => by
    simp [Ev.returned] at h; subst h
    cases hr with
    | witProof hp hb => exact .witProof hp hb
  | .coerce e es c, q, Q, r, h, hr => .coerce (root_of_returned_inr M (by simpa [Ev.returned] using h) hr)
  | .derive _ _ _, _, _, _, h, _ | .witness (.impl _) _, _, _, _, h, _
  | .hyp _, _, _, _, h, _ | .allege _, _, _, _, h, _ | .project _ _, _, _, _, h, _ | .reqCall _ _ _ _, _, _, _, h, _
  | .projRes _ _ _ _ _, _, _, _, h, _ | .refl _, _, _, _, h, _ | .compose _ _ _, _, _, _, h, _ => by simp [Ev.returned] at h

/-- A body returning a witness cites its proof. -/
private theorem cites_of_returned_inr : ∀ {e : Ev} {q : ProofId}, e.returned = some (.inr q) -> q ∈ e.cites
  | .witness (.proof q') P, q, h => by simp [Ev.returned] at h; subst h; simp [Ev.cites]
  | .coerce e es c, q, h => by
    simp only [Ev.cites, List.mem_append]; exact Or.inl (cites_of_returned_inr (by simpa [Ev.returned] using h))
  | .derive _ _ _, _, h | .witness (.impl _) _, _, h
  | .hyp _, _, h | .allege _, _, h | .project _ _, _, h | .reqCall _ _ _ _, _, h
  | .projRes _ _ _ _ _, _, h | .refl _, _, h | .compose _ _ _, _, h => by simp [Ev.returned] at h

/-- The operands of the derive a body returns cite only what the body cites. -/
private theorem cites_of_returned_inl : ∀ {e : Ev} {i σ} {ds : List Ev}, e.returned = some (.inl (i, σ, ds)) ->
    ∀ q ∈ citesList ds, q ∈ e.cites
  | .derive i' σ' es, i, σ, ds, h, q, hq => by
    simp [Ev.returned] at h; obtain ⟨_, _, h⟩ := h; subst h; simpa [Ev.cites] using hq
  | .witness (.impl i') P, i, σ, ds, h, q, hq => by
    simp [Ev.returned] at h; obtain ⟨_, _, h⟩ := h; subst h; simp [citesList] at hq
  | .witness (.proof q') P, _, _, _, h, _, _ => by simp [Ev.returned] at h
  | .coerce e es c, i, σ, ds, h, q, hq => by
    simp only [Ev.cites, List.mem_append]
    exact Or.inl (cites_of_returned_inl (by simpa [Ev.returned] using h) q hq)
  | .hyp _, _, _, _, h, _, _ | .allege _, _, _, _, h, _, _ | .project _ _, _, _, _, h, _, _
  | .reqCall _ _ _ _, _, _, _, h, _, _ | .projRes _ _ _ _ _, _, _, _, h, _, _ | .refl _, _, _, _, h, _, _
  | .compose _ _ _, _, _, _, h, _, _ => by simp [Ev.returned] at h

/-- The derive a well-typed body returns is of an impl of the body's trait. -/
private theorem trait_of_returned_inl (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim}
    {e : Ev} {t : TraitId} {as : List Ty} {π : Option Cite} {i : ImplId} {σ : List Ty} {ds : List Ev}
    (ht : HasType L M Ok Γ e ⟨.app t as, π⟩) (hr : e.returned = some (.inl (i, σ, ds))) :
    ∃ hi : i < M.sig.impls.length, M.sig.impls[i].trait = t := by
  generalize hc : (⟨.app t as, π⟩ : Claim) = c at ht
  induction ht generalizing t as π with
  | derive hi =>
    simp [Ev.returned] at hr; obtain ⟨rfl, rfl, rfl⟩ := hr
    simp [ImplDecl.header] at hc; exact ⟨hi, hc.1.1.symm⟩
  | witImpl hi =>
    simp [Ev.returned] at hr; obtain ⟨rfl, rfl, rfl⟩ := hr
    simp [ImplDecl.header] at hc; exact ⟨hi, hc.1.1.symm⟩
  | coerce _ _ _ _ _ hpe ih =>
    simp only [Ev.returned] at hr
    cases hc
    cases hpe with
    | app => exact ih hr rfl
  | hyp | projReq | projWhere | reqCall | witProof | projRes | refl | compose => simp [Ev.returned] at hr

/-- The proof a well-typed body's cast cites proves an application of the body's trait. -/
private theorem pred_of_returned_inr (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim}
    {e : Ev} {t : TraitId} {as : List Ty} {π : Option Cite} {q : ProofId}
    (ht : HasType L M Ok Γ e ⟨.app t as, π⟩) (hr : e.returned = some (.inr q)) :
    ∃ hq : q < M.proofs.length, ∃ as', M.proofs[q].1 = .app t as' := by
  generalize hc : (⟨.app t as, π⟩ : Claim) = c at ht
  induction ht generalizing t as π with
  | witProof hp hP =>
    simp [Ev.returned] at hr; subst hr
    cases hc; exact ⟨hp, as, hP⟩
  | coerce _ _ _ _ _ hpe ih =>
    simp only [Ev.returned] at hr
    cases hc
    cases hpe with
    | app => exact ih hr rfl
  | hyp | derive | projReq | projWhere | reqCall | witImpl | projRes | refl | compose => simp [Ev.returned] at hr

/-! ## The proof table's chain of casts (`Module.proofDerive`) -/

/-- A proof whose chain ends at a derive reads to that derive. -/
private theorem root_of_proofDeriveAux (M : Module) : ∀ (f : Nat) {p : ProofId} {d} {Q : Pred},
    M.proofDeriveAux f p = some d -> Root M (.witness (.proof p) Q) d
  | 0, p, d, Q, h => by simp [Module.proofDeriveAux] at h
  | f+1, p, d, Q, h => by
    simp only [Module.proofDeriveAux] at h
    split at h
    · simp at h
    · rename_i P body hp
      split at h
      · rename_i d' hret
        cases h
        exact .witProof hp (root_of_returned_inl M hret)
      · rename_i q hret
        exact .witProof hp (root_of_returned_inr M (Q := Q) hret (root_of_proofDeriveAux M f (Q := Q) h))
      · simp at h

/-- Under H2 the derive a proof's chain ends at is of an impl of the proof's trait. -/
private theorem trait_of_proofDeriveAux (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop}
    (hP : ProofsWF L M Ok) : ∀ (f : Nat) {p : ProofId} {t : TraitId} {as : List Ty} {i σ ds},
    M.proofDeriveAux f p = some (i, σ, ds) -> (hp : p < M.proofs.length) -> M.proofs[p].1 = .app t as ->
    ∃ hi : i < M.sig.impls.length, M.sig.impls[i].trait = t
  | 0, p, t, as, i, σ, ds, h, _, _ => by simp [Module.proofDeriveAux] at h
  | f+1, p, t, as, i, σ, ds, h, hp, hpred => by
    simp only [Module.proofDeriveAux, List.getElem?_eq_getElem hp] at h
    obtain ⟨⟨π, hty⟩, _, _, _⟩ := hP p hp
    rw [hpred] at hty
    split at h
    · rename_i d' hret
      cases h
      exact trait_of_returned_inl L M hty hret
    · rename_i q hret
      obtain ⟨hq, as', hq'⟩ := pred_of_returned_inr L M hty hret
      exact trait_of_proofDeriveAux L M hP f h hq hq'
    · simp at h


/-- A term with a root whose claim cites a proof (under any hypotheses: a hypothesis has no root): it
reads to what the proof's witness reads to, and the proof proves an application of the claim's trait. -/
private theorem proofCite_rooted (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hP : ProofsWF L M Ok)
    {Γ : List Claim} {e : Ev} {c : Claim} (ht : HasType L M Ok Γ e c) :
    ∀ {p : ProofId} {r}, c.cite = some (.proof p) -> Root M e r ->
      (∀ Q : Pred, Root M (.witness (.proof p) Q) r) ∧
      (∀ t as, c.pred = .app t as -> ∃ hp : p < M.proofs.length, ∃ as', M.proofs[p].1 = .app t as') := by
  induction ht with
  | hyp hk => intro p r hc hr; cases hr
  | witProof hp hpred =>
    intro p' r hc hr; cases hc
    refine ⟨fun Q => ?_, fun t as hpred' => ⟨hp, as, by simp only at hpred'; rw [hpred, hpred']⟩⟩
    cases hr with | witProof hp hb => exact .witProof hp hb
  | coerce _ _ _ hπ _ hpe ih =>
    intro p r hc hr
    simp only at hc
    rcases hπ with h | h
    · rw [h] at hc; cases hc
    · subst h
      cases hr with
      | coerce hr =>
        obtain ⟨iha, ihb⟩ := ih hc hr
        refine ⟨iha, fun t as hpred => ?_⟩
        simp only at hpred; subst hpred
        cases hpe with
        | app => exact ihb _ _ rfl
  | projWhere _ ht hder hj _ ih₁ ih₂ =>
    intro p r hc hr
    -- The projection's reading passes through its operand's base, which is the cited derive.
    cases hr with
    | projReq hb hI hT hk _ =>
      obtain ⟨iha₁, ihb₁⟩ := ih₁ rfl hb
      obtain ⟨hp₀, as', hpred₀⟩ := ihb₁ _ _ rfl
      obtain ⟨hi, htrait⟩ := trait_of_proofDeriveAux L M hP _ hder hp₀ hpred₀
      cases root_unique M (iha₁ default) (root_of_proofDeriveAux M _ (Q := default) hder)
      rw [List.getElem?_eq_getElem hi] at hI; cases hI
      rw [htrait, List.getElem?_eq_getElem ht] at hT; cases hT
      exact absurd hk (by omega)
    | projWhere hb hI hT hk hop =>
      obtain ⟨iha₁, ihb₁⟩ := ih₁ rfl hb
      obtain ⟨hp₀, as', hpred₀⟩ := ihb₁ _ _ rfl
      obtain ⟨hi, htrait⟩ := trait_of_proofDeriveAux L M hP _ hder hp₀ hpred₀
      cases root_unique M (iha₁ default) (root_of_proofDeriveAux M _ (Q := default) hder)
      rw [List.getElem?_eq_getElem hi] at hI; cases hI
      rw [htrait, List.getElem?_eq_getElem ht] at hT; cases hT
      rw [Nat.add_sub_cancel_left] at hop
      exact ih₂ hc hop
  | derive | projReq | reqCall | witImpl | projRes | refl | compose => intro p r hc; cases hc



/-! ## Terms related by inlining, and their roots -/

mutual
  /-- `Sim M e e'`: `e'` is `e` with some witnesses of proofs replaced by the proofs' bodies, some
  witnesses of unconditional impls by their derives, some projections past the plain requirements by
  the cited derive's operand (D2), and some coerce claims rewritten, at any depth. These are the shapes
  `inlineAux` produces at any fuel. The D2 step is stated through the operand's root, not through the
  proof table, so that the root lemmas need no typing. -/
  inductive Sim (M : Module) : Ev -> Ev -> Prop where
    | top : SimTop M e e' -> Sim M e e'
    | wit : M.proofs[p]? = some (Q, body) -> Sim M body e' -> Sim M (.witness (.proof p) P) e'
    | projWhere : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
        T.reqs.length ≤ k -> Sim M (ds.getD (k - T.reqs.length) default) e' -> Sim M (.project e k) e'
  /-- The congruence steps of `Sim`: the term's own constructor kept, the children related. -/
  inductive SimTop (M : Module) : Ev -> Ev -> Prop where
    | refl : SimTop M e e
    | derive : es.length = es'.length -> (∀ j, j < es.length -> Sim M (es.getD j default) (es'.getD j default)) ->
        SimTop M (.derive i σ es) (.derive i σ es')
    | project : Sim M e e' -> SimTop M (.project e k) (.project e' k)
    | reqCall : Sim M e e' -> fs.length = fs'.length ->
        (∀ j, j < fs.length -> Sim M (fs.getD j default) (fs'.getD j default)) ->
        SimTop M (.reqCall e b us fs) (.reqCall e' b us fs')
    | witImpl : SimTop M (.witness (.impl i) P) (.derive i [] [])
    | projRes : es.length = es'.length -> (∀ j, j < es.length -> Sim M (es.getD j default) (es'.getD j default)) ->
        SimTop M (.projRes i σ es a gs) (.projRes i σ es' a gs)
    | compose : es.length = es'.length -> (∀ j, j < es.length -> Sim M (es.getD j default) (es'.getD j default)) ->
        SimTop M (.compose es l r) (.compose es' l r)
    | coerce : Sim M e e' -> es.length = es'.length ->
        (∀ j, j < es.length -> Sim M (es.getD j default) (es'.getD j default)) ->
        SimTop M (.coerce e es c) (.coerce e' es' c')
end

/-- Lists related pointwise by `Sim`. -/
def SimList (M : Module) (es es' : List Ev) : Prop :=
  es.length = es'.length ∧ ∀ j, j < es.length -> Sim M (es.getD j default) (es'.getD j default)

/-- Bases related by `Sim`: one impl, one argument list, operands related pointwise. -/
def SimBase (M : Module) (r r' : ImplId × List Ty × List Ev) : Prop :=
  r.1 = r'.1 ∧ r.2.1 = r'.2.1 ∧ SimList M r.2.2 r'.2.2

theorem sim_refl (M : Module) (e : Ev) : Sim M e e := .top .refl

theorem simList_refl (M : Module) (es : List Ev) : SimList M es es := ⟨rfl, fun _ _ => sim_refl M _⟩

theorem simBase_refl (M : Module) (r : ImplId × List Ty × List Ev) : SimBase M r r := ⟨rfl, rfl, simList_refl M _⟩

/-- A reading of a list's member by position is within the list: the default term reads to nothing. -/
private theorem lt_of_root_getD (M : Module) {ds : List Ev} {m : Nat} {r} (h : Root M (ds.getD m default) r) :
    m < ds.length := by
  by_cases hm : m < ds.length
  · exact hm
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hm)] at h
    cases h

theorem SimList.append {M : Module} {ds ds' fs fs' : List Ev} (h₁ : SimList M ds ds') (h₂ : SimList M fs fs') :
    SimList M (ds ++ fs) (ds' ++ fs') := by
  refine ⟨by simp [h₁.1, h₂.1], fun j hj => ?_⟩
  rw [List.length_append] at hj
  by_cases hlt : j < ds.length
  · rw [List.getD_eq_getElem?_getD, List.getElem?_append_left hlt, ← List.getD_eq_getElem?_getD,
      List.getD_eq_getElem?_getD (l := ds' ++ fs'), List.getElem?_append_left (h₁.1 ▸ hlt), ← List.getD_eq_getElem?_getD]
    exact h₁.2 j hlt
  · have hge := Nat.le_of_not_lt hlt
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right hge, ← List.getD_eq_getElem?_getD,
      List.getD_eq_getElem?_getD (l := ds' ++ fs'), List.getElem?_append_right (h₁.1 ▸ hge), ← List.getD_eq_getElem?_getD,
      h₁.1]
    have := h₁.1
    exact h₂.2 (j - ds'.length) (by omega)

mutual
  /-- Substituting related operand lists for the hypotheses of one term gives related terms. -/
  theorem sim_substHyp {M : Module} {ds ds' : List Ev} (h : SimList M ds ds') :
      ∀ (t : Ev), Sim M (t.substHyp (hypsOf ds)) (t.substHyp (hypsOf ds'))
    | .hyp k => by
      simp only [Ev.substHyp, hypsOf]
      by_cases hk : k < ds.length
      · exact h.2 k hk
      · have hk' : ¬ k < ds'.length := h.1 ▸ hk
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hk),
          List.getD_eq_getElem?_getD (l := ds'), List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hk')]
        exact sim_refl M _
    | .allege c => sim_refl M _
    | .derive i σ es =>
      let hl := sim_substHypList h es
      .top (.derive hl.1 hl.2)
    | .project e k => .top (.project (sim_substHyp h e))
    | .reqCall e b us fs =>
      let hl := sim_substHypList h fs
      .top (.reqCall (sim_substHyp h e) hl.1 hl.2)
    | .witness c P => sim_refl M _
    | .projRes i σ es a gs =>
      let hl := sim_substHypList h es
      .top (.projRes hl.1 hl.2)
    | .refl a => sim_refl M _
    | .compose es l r =>
      let hl := sim_substHypList h es
      .top (.compose hl.1 hl.2)
    | .coerce e es c =>
      let hl := sim_substHypList h es
      .top (.coerce (sim_substHyp h e) hl.1 hl.2)
  theorem sim_substHypList {M : Module} {ds ds' : List Ev} (h : SimList M ds ds') :
      ∀ (ts : List Ev), SimList M (substHypList (hypsOf ds) ts) (substHypList (hypsOf ds') ts)
    | [] => simList_refl M _
    | t :: ts => by
      have ih := sim_substHypList h ts
      refine ⟨by simp [substHypList, ih.1], fun j hj => ?_⟩
      cases j with
      | zero => simpa [substHypList] using sim_substHyp h t
      | succ j => simpa [substHypList] using ih.2 j (by simpa [substHypList] using hj)
end

/-- A term's reading lifts through the head steps of `Sim`: whatever the congruent core reads to, the
term reads to as well. -/
theorem sim_lift {M : Module} {e' : Ev} {P : ImplId × List Ty × List Ev -> Prop}
    (hbase : ∀ e₀, SimTop M e₀ e' -> ∃ r, Root M e₀ r ∧ P r) :
    ∀ {e : Ev}, Sim M e e' -> ∃ r, Root M e r ∧ P r
  | _, .top h => hbase _ h
  | _, .wit hp hs =>
    let ⟨r, hr, hP⟩ := sim_lift hbase hs
    ⟨r, .witProof hp hr, hP⟩
  | _, .projWhere hroot hI hT hk hs =>
    let ⟨r, hr, hP⟩ := sim_lift hbase hs
    ⟨r, .projWhere hroot hI hT hk hr, hP⟩

/-- Forwards: a term's reading survives inlining, to a related base. -/
theorem sim_root_fwd (M : Module) {e : Ev} {r : ImplId × List Ty × List Ev} (hroot : Root M e r) :
    ∀ {e' : Ev}, Sim M e e' -> ∃ r', Root M e' r' ∧ SimBase M r r' := by
  induction hroot with
  | derive =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .derive, simBase_refl M _⟩
      | derive hl hs => exact ⟨_, .derive, rfl, rfl, hl, hs⟩
  | witImpl =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .witImpl, simBase_refl M _⟩
      | witImpl => exact ⟨_, .derive, simBase_refl M _⟩
  | witProof hp hb ih =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .witProof hp hb, simBase_refl M _⟩
    | wit hp' hs =>
      rw [hp] at hp'
      cases hp'
      exact ih hs
  | coerce hb ih =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .coerce hb, simBase_refl M _⟩
      | coerce hs _ _ =>
        obtain ⟨r', hr', hbase⟩ := ih hs
        exact ⟨r', .coerce hr', hbase⟩
  | projReq hb hI hT hk hret ih₁ ih₂ =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .projReq hb hI hT hk hret, simBase_refl M _⟩
      | project hs =>
        obtain ⟨⟨i', σ', ds'⟩, hr', hi, hσ, hl⟩ := ih₁ hs
        simp only at hi hσ hl
        subst hi; subst hσ
        obtain ⟨r', hret', hbase⟩ := ih₂ (sim_substHyp hl _)
        exact ⟨r', .projReq hr' hI hT hk hret', hbase⟩
    | projWhere hb' hI' hT' hk' hs =>
      cases root_unique M hb hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact absurd hk (Nat.not_lt.mpr hk')
  | projWhere hb hI hT hk hop ih₁ ih₂ =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .projWhere hb hI hT hk hop, simBase_refl M _⟩
      | project hs =>
        obtain ⟨⟨i', σ', ds'⟩, hr', hi, hσ, hl⟩ := ih₁ hs
        simp only at hi hσ hl
        subst hi; subst hσ
        obtain ⟨r', hop', hbase⟩ := ih₂ (hl.2 _ (lt_of_root_getD M hop))
        exact ⟨r', .projWhere hr' hI hT hk hop', hbase⟩
    | projWhere hb' hI' hT' hk' hs =>
      cases root_unique M hb hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact ih₂ hs
  | reqCall hb hI hT hB hret ih₁ ih₂ =>
    intro e' hs
    cases hs with
    | top ht =>
      cases ht with
      | refl => exact ⟨_, .reqCall hb hI hT hB hret, simBase_refl M _⟩
      | reqCall hs hl₁ hl₂ =>
        obtain ⟨⟨i', σ', ds'⟩, hr', hi, hσ, hl⟩ := ih₁ hs
        simp only at hi hσ hl
        subst hi; subst hσ
        obtain ⟨r', hret', hbase⟩ := ih₂ (sim_substHyp (hl.append ⟨hl₁, hl₂⟩) _)
        exact ⟨r', .reqCall hr' hI hT hB hret', hbase⟩

/-- Backwards: a reading of the inlined term is a reading of the term, to a related base. -/
theorem sim_root_bwd (M : Module) {e' : Ev} {r' : ImplId × List Ty × List Ev} (hroot : Root M e' r') :
    ∀ {e : Ev}, Sim M e e' -> ∃ r, Root M e r ∧ SimBase M r r' := by
  induction hroot with
  | derive =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .derive, simBase_refl M _⟩
    | derive hl hs => exact ⟨_, .derive, rfl, rfl, hl, hs⟩
    | witImpl => exact ⟨_, .witImpl, simBase_refl M _⟩
  | witImpl =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .witImpl, simBase_refl M _⟩
  | witProof hp hb ih =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .witProof hp hb, simBase_refl M _⟩
  | coerce hb ih =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .coerce hb, simBase_refl M _⟩
    | coerce hs _ _ =>
      obtain ⟨r, hr, hbase⟩ := ih hs
      exact ⟨r, .coerce hr, hbase⟩
  | projReq hb hI hT hk hret ih₁ ih₂ =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .projReq hb hI hT hk hret, simBase_refl M _⟩
    | project hs =>
      obtain ⟨⟨i', σ', ds'⟩, hr, hi, hσ, hl⟩ := ih₁ hs
      simp only at hi hσ hl
      subst hi; subst hσ
      obtain ⟨r, hret', hbase⟩ := ih₂ (sim_substHyp hl _)
      exact ⟨r, .projReq hr hI hT hk hret', hbase⟩
  | projWhere hb hI hT hk hop ih₁ ih₂ =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .projWhere hb hI hT hk hop, simBase_refl M _⟩
    | project hs =>
      obtain ⟨⟨i', σ', ds'⟩, hr, hi, hσ, hl⟩ := ih₁ hs
      simp only at hi hσ hl
      subst hi; subst hσ
      obtain ⟨r, hop', hbase⟩ := ih₂ (hl.2 _ (hl.1 ▸ lt_of_root_getD M hop))
      exact ⟨r, .projWhere hr hI hT hk hop', hbase⟩
  | reqCall hb hI hT hB hret ih₁ ih₂ =>
    intro e hs
    refine sim_lift (fun e₀ ht => ?_) hs
    cases ht with
    | refl => exact ⟨_, .reqCall hb hI hT hB hret, simBase_refl M _⟩
    | reqCall hs hl₁ hl₂ =>
      obtain ⟨⟨i', σ', ds'⟩, hr, hi, hσ, hl⟩ := ih₁ hs
      simp only at hi hσ hl
      subst hi; subst hσ
      obtain ⟨r, hret', hbase⟩ := ih₂ (sim_substHyp (hl.append ⟨hl₁, hl₂⟩) _)
      exact ⟨r, .reqCall hr hI hT hB hret', hbase⟩


/-- Every proof a well-typed term cites is in the table. -/
private theorem cites_lt_of_hasType (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim}
    {e : Ev} {c : Claim} (ht : HasType L M Ok Γ e c) : ∀ q ∈ e.cites, q < M.proofs.length := by
  induction ht with
  | hyp => simp [Ev.cites]
  | derive _ _ _ _ _ _ ih =>
    intro q hq
    simp only [Ev.cites] at hq
    obtain ⟨e', he', hq'⟩ := mem_citesList_inv hq
    obtain ⟨j, hj, hget⟩ := List.getElem_of_mem he'
    exact ih j hj q (by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget]; exact hq')
  | projReq _ _ _ ih => simpa [Ev.cites] using ih
  | projWhere _ _ _ _ _ ih₁ _ => simpa [Ev.cites] using ih₁
  | reqCall _ _ _ _ _ _ _ ih ihs =>
    intro q hq
    simp only [Ev.cites, List.mem_append] at hq
    rcases hq with hq | hq
    · exact ih q hq
    · obtain ⟨e', he', hq'⟩ := mem_citesList_inv hq
      obtain ⟨j, hj, hget⟩ := List.getElem_of_mem he'
      exact ihs j hj q (by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget]; exact hq')
  | witProof hp => intro q hq; simp [Ev.cites] at hq; subst hq; exact hp
  | witImpl => simp [Ev.cites]
  | projRes _ _ _ _ _ _ _ _ ih =>
    intro q hq
    simp only [Ev.cites] at hq
    obtain ⟨e', he', hq'⟩ := mem_citesList_inv hq
    obtain ⟨j, hj, hget⟩ := List.getElem_of_mem he'
    exact ih j hj q (by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget]; exact hq')
  | refl => simp [Ev.cites]
  | compose _ _ _ _ ih =>
    intro q hq
    simp only [Ev.cites] at hq
    obtain ⟨e', he', hq'⟩ := mem_citesList_inv hq
    obtain ⟨j, hj, hget⟩ := List.getElem_of_mem he'
    exact ih j hj q (by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget]; exact hq')
  | coerce _ _ _ _ _ _ ih ihs =>
    intro q hq
    simp only [Ev.cites, List.mem_append] at hq
    rcases hq with hq | hq
    · exact ih q hq
    · obtain ⟨e', he', hq'⟩ := mem_citesList_inv hq
      obtain ⟨j, hj, hget⟩ := List.getElem_of_mem he'
      exact ihs j hj q (by rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, hget]; exact hq')

/-! ## What `inlineAux` does at each constructor -/

private theorem inlineListAux_length (M : Module) (n : Nat) : ∀ (es : List Ev), (inlineListAux M n es).length = es.length
  | [] => by simp [inlineListAux]
  | e :: es => by simp [inlineListAux, inlineListAux_length M n es]

private theorem inlineListAux_getD (M : Module) (n : Nat) : ∀ (es : List Ev) (j : Nat), j < es.length ->
    (inlineListAux M n es).getD j default = inlineAux M n (es.getD j default)
  | [], j, hj => by simp at hj
  | e :: es, 0, _ => by simp [inlineListAux]
  | e :: es, j+1, hj => by simp [inlineListAux]; exact inlineListAux_getD M n es j (by simpa using hj)


/-! ## What `inlineAux` does at a projection -/

/-- A projection is kept over its inlined operand when the operand reveals no derive, or a derive of
an impl whose trait has more plain requirements than the index. -/
private theorem inline_project_keep (M : Module) (n : Nat) (e : Ev) (k : Nat)
    (h : ∀ i σ ds, (inlineAux M (n+1) e).returned = some (.inl (i, σ, ds)) ->
      ∀ I T, M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T -> k < T.reqs.length) :
    inlineAux M (n+1) (.project e k) = .project (inlineAux M (n+1) e) k := by
  simp only [inlineAux]
  cases hret : (inlineAux M (n+1) e).returned with
  | none => rfl
  | some s =>
    cases s with
    | inr q => rfl
    | inl d =>
      obtain ⟨i, σ, ds⟩ := d
      cases hI : M.sig.impls[i]? with
      | none => simp [hI]
      | some I =>
        cases hT : M.sig.traits[I.trait]? with
        | none => simp [hI, hT]
        | some T => simp [hI, hT, Nat.not_le.mpr (h i σ ds hret I T hI hT)]

/-- A projection past the plain requirements of the derive its inlined operand reveals is that
derive's operand at the position (D2). -/
private theorem inline_project_where (M : Module) (n : Nat) (e : Ev) (k : Nat) {i : ImplId} {σ : List Ty}
    {ds : List Ev} {I : ImplDecl} {T : TraitDecl}
    (hret : (inlineAux M (n+1) e).returned = some (.inl (i, σ, ds))) (hI : M.sig.impls[i]? = some I)
    (hT : M.sig.traits[I.trait]? = some T) (hk : T.reqs.length ≤ k) :
    inlineAux M (n+1) (.project e k) = ds.getD (k - T.reqs.length) default := by
  simp only [inlineAux]
  rw [hret]
  simp [hI, hT, hk]

/-! ## What the inlining of a proof body returns -/

/-- A body returning a derive returns, inlined, the derive with its operands inlined at the same fuel. -/
private theorem returned_inline_inl (M : Module) : ∀ {e : Ev} {i : ImplId} {σ : List Ty} {ds : List Ev} (n : Nat),
    e.returned = some (.inl (i, σ, ds)) ->
    (inlineAux M (n+1) e).returned = some (.inl (i, σ, inlineListAux M (n+1) ds))
  | .derive i' σ' es, i, σ, ds, n, h => by
    simp [Ev.returned] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp [inlineAux, Ev.returned]
  | .witness (.impl i') P, i, σ, ds, n, h => by
    simp [Ev.returned] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp [inlineAux, inlineListAux, Ev.returned]
  | .coerce e es c, i, σ, ds, n, h => by
    simp only [inlineAux, Ev.returned]
    exact returned_inline_inl M n (by simpa [Ev.returned] using h)
  | .witness (.proof _) _, _, _, _, _, h | .hyp _, _, _, _, _, h | .allege _, _, _, _, _, h
  | .project _ _, _, _, _, _, h | .reqCall _ _ _ _, _, _, _, _, h | .projRes _ _ _ _ _, _, _, _, _, h
  | .refl _, _, _, _, _, h | .compose _ _ _, _, _, _, _, h => by simp [Ev.returned] at h

/-- A body returning a witness of another proof returns, inlined, what that witness returns inlined
(the witness's written predicate is not read). -/
private theorem returned_inline_inr (M : Module) : ∀ {e : Ev} {q : ProofId} (n : Nat) (P' : Pred),
    e.returned = some (.inr q) ->
    (inlineAux M (n+1) e).returned = (inlineAux M (n+1) (.witness (.proof q) P')).returned
  | .witness (.proof q') P, q, n, P', h => by
    simp [Ev.returned] at h; subst h
    cases hq : M.proofs[q']? with
    | none => simp [inlineAux, hq, Ev.returned]
    | some pr => obtain ⟨Q, body⟩ := pr; simp [inlineAux, hq]
  | .coerce e es c, q, n, P', h => by
    conv => lhs; simp only [inlineAux, Ev.returned]
    exact returned_inline_inr M n P' (by simpa [Ev.returned] using h)
  | .witness (.impl _) _, _, _, _, h | .derive _ _ _, _, _, _, h | .hyp _, _, _, _, h | .allege _, _, _, _, h
  | .project _ _, _, _, _, h | .reqCall _ _ _ _, _, _, _, h | .projRes _ _ _ _ _, _, _, _, h
  | .refl _, _, _, _, h | .compose _ _ _, _, _, _, h => by simp [Ev.returned] at h

/-- Under H2, the inlining of a proof's witness at a fuel above the proof returns the derive the
proof's chain of casts ends at, its operands inlined at a fuel above every proof they cite. -/
private theorem returned_inline_witness (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop}
    (hP : ProofsWF L M Ok) : ∀ (f : Nat) {p : ProofId} {i : ImplId} {σ : List Ty} {ds : List Ev},
    M.proofDeriveAux f p = some (i, σ, ds) -> ∀ (n : Nat) (P : Pred), p + 1 < n ->
    ∃ m, 0 < m ∧ m < n ∧ (∀ q ∈ citesList ds, q + 1 < m) ∧
      (inlineAux M n (.witness (.proof p) P)).returned = some (.inl (i, σ, inlineListAux M m ds))
  | 0, p, i, σ, ds, h, _, _, _ => by simp [Module.proofDeriveAux] at h
  | f+1, p, i, σ, ds, h, n, P, hpn => by
    simp only [Module.proofDeriveAux] at h
    split at h
    · simp at h
    · rename_i Q body hp'
      obtain ⟨hp, hget⟩ := List.getElem?_eq_some_iff.mp hp'
      have hbody : body = M.proofs[p].2 := (congrArg Prod.snd hget).symm
      obtain ⟨_, _, _, hcites⟩ := hP p hp
      match n, hpn with
      | n''+2, hpn =>
        have hinl : inlineAux M (n''+2) (.witness (.proof p) P) = inlineAux M (n''+1) body := by
          simp [inlineAux, hp']
        rw [hinl]
        split at h
        · -- the body's own derive: its operands are inlined at the body's fuel
          rename_i d' hret
          cases h
          refine ⟨n''+1, Nat.succ_pos _, Nat.lt_succ_self _, fun q hq => ?_, returned_inline_inl M n'' hret⟩
          have := hcites q (hbody ▸ cites_of_returned_inl hret q hq)
          exact Nat.lt_of_le_of_lt (Nat.succ_le_of_lt this) (Nat.lt_of_succ_lt_succ hpn)
        · -- a cast: the chain continues at the cited proof, one fuel down
          rename_i q hret
          have hq : q < p := hcites q (hbody ▸ cites_of_returned_inr hret)
          obtain ⟨m, hm0, hmn, hbound, hm⟩ := returned_inline_witness L M hP f h (n''+1) P
            (Nat.lt_of_le_of_lt (Nat.succ_le_of_lt hq) (Nat.lt_of_succ_lt_succ hpn))
          exact ⟨m, hm0, Nat.lt_succ_of_lt hmn, hbound, (returned_inline_inr M n'' P hret).trans hm⟩
        · simp at h

/-! ## Inlining is a `Sim` step at every fuel -/

/-- Related operand lists are related at every position: at a position past the end both read the
default term. -/
private theorem SimList.getD {M : Module} {ds ds' : List Ev} (h : SimList M ds ds') (j : Nat) :
    Sim M (ds.getD j default) (ds'.getD j default) := by
  by_cases hj : j < ds.length
  · exact h.2 j hj
  · have hj' : ¬ j < ds'.length := h.1 ▸ hj
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hj),
      List.getD_eq_getElem?_getD (l := ds'), List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hj')]
    exact sim_refl M _

mutual
  /-- Inlining at any fuel relates a term to its result: every step `inlineAux` takes is a `Sim` step.
  At a projection past the plain requirements of the derive the inlined operand reveals, the operand
  itself reads to a related derive (`sim_root_bwd`), and the D2 step of `Sim` reads its operand. -/
  theorem sim_inline (M : Module) : ∀ (n : Nat) (e : Ev), Sim M e (inlineAux M n e)
    | 0, e => by rw [inlineAux.eq_1]; exact sim_refl M e
    | _+1, .hyp k => by simp only [inlineAux]; exact sim_refl M _
    | _+1, .allege c => by simp only [inlineAux]; exact sim_refl M _
    | n+1, .derive i σ es => by
      simp only [inlineAux]
      exact .top (.derive (sim_inlineList M (n+1) es).1 (sim_inlineList M (n+1) es).2)
    | n+1, .project e k => by
      have hs := sim_inline M (n+1) e
      simp only [inlineAux]
      cases hret : (inlineAux M (n+1) e).returned with
      | none => exact .top (.project hs)
      | some s =>
        cases s with
        | inr q => exact .top (.project hs)
        | inl d =>
          obtain ⟨i, σ, ds'⟩ := d
          cases hI : M.sig.impls[i]? with
          | none => simp only [hI]; exact .top (.project hs)
          | some I =>
            cases hT : M.sig.traits[I.trait]? with
            | none => simp only [hI, hT]; exact .top (.project hs)
            | some T =>
              simp only [hI, hT]
              by_cases hk : T.reqs.length ≤ k
              · rw [ite_eq_left hk]
                obtain ⟨⟨i₀, σ₀, ds⟩, hr, hi, hσ, hl⟩ := sim_root_bwd M (root_of_returned_inl M hret) hs
                simp only at hi hσ hl
                subst hi; subst hσ
                exact .projWhere hr hI hT hk (hl.getD _)
              · rw [ite_eq_right hk]; exact .top (.project hs)
    | n+1, .reqCall e b us fs => by
      simp only [inlineAux]
      exact .top (.reqCall (sim_inline M (n+1) e) (sim_inlineList M (n+1) fs).1 (sim_inlineList M (n+1) fs).2)
    | n+1, .witness (.proof p) P => by
      simp only [inlineAux]
      cases hp : M.proofs[p]? with
      | none => exact sim_refl M _
      | some pr => obtain ⟨Q, body⟩ := pr; exact .wit hp (sim_inline M n body)
    | _+1, .witness (.impl i) P => by simp only [inlineAux]; exact .top .witImpl
    | n+1, .projRes i σ es a gs => by
      simp only [inlineAux]
      exact .top (.projRes (sim_inlineList M (n+1) es).1 (sim_inlineList M (n+1) es).2)
    | _+1, .refl a => by simp only [inlineAux]; exact sim_refl M _
    | n+1, .compose es l r => by
      simp only [inlineAux]
      exact .top (.compose (sim_inlineList M (n+1) es).1 (sim_inlineList M (n+1) es).2)
    | n+1, .coerce e es c => by
      simp only [inlineAux]
      exact .top (.coerce (sim_inline M (n+1) e) (sim_inlineList M (n+1) es).1 (sim_inlineList M (n+1) es).2)
  theorem sim_inlineList (M : Module) : ∀ (n : Nat) (es : List Ev), SimList M es (inlineListAux M n es)
    | n, [] => by simp only [inlineListAux]; exact simList_refl M _
    | n, e :: es => by
      have ih := sim_inlineList M n es
      refine ⟨by simp [inlineListAux, ih.1], fun j hj => ?_⟩
      cases j with
      | zero => simpa [inlineListAux] using sim_inline M n e
      | succ j => simpa [inlineListAux] using ih.2 j (by simpa using hj)
end

/-! ## Inlining preserves the type up to the citation -/

/-- A claim with no citation is well-formed. -/
private theorem claimWF_none (P : Pred) : Claim.WF ⟨P, none⟩ := by cases P <;> simp [Claim.WF]

/-- What the inlining at fuel `n` of a term whose claim cites a proof returns: the cited proof's derive,
its operands inlined at a fuel `m` below `n` and above every proof they cite. This is what a projection
past the plain requirements reads off the inlined operand. -/
private def InlinedCite (M : Module) (n : Nat) (e : Ev) (c : Claim) : Prop :=
  ∀ p i σ ds, c.cite = some (.proof p) -> M.proofDerive p = some (i, σ, ds) ->
    ∃ m, 0 < m ∧ m < n ∧ (∀ q ∈ citesList ds, q + 1 < m) ∧
      (inlineAux M n e).returned = some (.inl (i, σ, inlineListAux M m ds))

/-- A claim with no proof citation makes `InlinedCite` vacuous. -/
private theorem inlinedCite_of_not_proof {M : Module} {n : Nat} {e : Ev} {c : Claim}
    (h : ∀ p, c.cite ≠ some (.proof p)) : InlinedCite M n e c :=
  fun p _ _ _ hc _ => absurd hc (h p)

/-- Inlining at a fuel above every cited proof preserves the type up to the citation, for closed
allege-free terms, and returns what `InlinedCite` says. Strong induction on the fuel, which drops by
one at a witness and to the operands' fuel at a projection past the plain requirements; H2 keeps the
cited proofs below it. -/
private theorem inline_hasType (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hP : ProofsWF L M Ok)
    (n : Nat) : ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> Γ = [] ->
      0 < n -> (∀ q ∈ e.cites, q + 1 < n) ->
      HasType L M Ok [] (inlineAux M n e) c.strip ∧ InlinedCite M n e c := by
  induction n using Nat.strongRecOn with
  | _ n ihn =>
  intro Γ e c ht hΓ hn hc
  cases n with
  | zero => exact absurd hn (Nat.lt_irrefl 0)
  | succ n =>
  induction ht with
  | hyp hk => subst hΓ; exact absurd hk (Nat.not_lt_zero _)
  | @derive i σ es _ hi hσ hok hes πs _ ih =>
    simp only [inlineAux, Claim.strip]
    simp only [Ev.cites] at hc
    refine ⟨.derive hi hσ hok (by rw [inlineListAux_length]; exact hes) (fun _ => none) (fun j hj => ?_),
      inlinedCite_of_not_proof (fun p h => nomatch h)⟩
    rw [inlineListAux_length] at hj
    rw [inlineListAux_getD _ _ _ _ hj]
    exact (ih j hj hΓ (fun q hq => hc q (mem_citesList (getD_mem _ _ hj) hq))).1
  | @projReq _ e₁ t as π k hte ht hk ih =>
    simp only [Ev.cites] at hc
    obtain ⟨ih', _⟩ := ih hΓ hc
    simp only [Claim.strip] at ih' ⊢
    -- the inlined operand reveals a derive of the claim's trait, if any, and `k` is below its
    -- requirement count, so the projection is kept
    rw [inline_project_keep M n e₁ k (fun i σ ds hret I T hI hT => by
      obtain ⟨hi, htrait⟩ := trait_of_returned_inl L M ih' hret
      rw [List.getElem?_eq_getElem hi] at hI; cases hI
      rw [htrait, List.getElem?_eq_getElem ht] at hT; cases hT
      exact hk)]
    exact ⟨.projReq ih' ht hk, inlinedCite_of_not_proof (fun p h => nomatch h)⟩
  | @projWhere _ e₁ t as p i σ ds j c₁ hte ht hder hj hop ih₁ _ =>
    simp only [Ev.cites] at hc
    obtain ⟨ih', hcite⟩ := ih₁ hΓ hc
    simp only [Claim.strip] at ih'
    -- the inlined operand reveals the cited proof's derive, its operands inlined at a fuel `m`
    obtain ⟨m, hm0, hmn, hbound, hret⟩ := hcite p i σ ds rfl hder
    obtain ⟨hi, htrait⟩ := trait_of_returned_inl L M ih' hret
    have heq : inlineAux M (n+1) (.project e₁ (M.sig.traits[t].reqs.length + j)) = inlineAux M m (ds.getD j default) := by
      rw [inline_project_where M n e₁ _ hret (List.getElem?_eq_getElem hi)
        (by rw [htrait]; exact List.getElem?_eq_getElem ht) (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
      exact inlineListAux_getD M m ds j hj
    -- the operand is typed closed, so its inlining at `m` is typed by the induction on the fuel
    obtain ⟨hty, hcite'⟩ := ihn m hmn hop rfl hm0
      (fun q hq => hbound q (mem_citesList (getD_mem _ _ hj) hq))
    refine ⟨heq ▸ hty, fun p' i' σ' ds' hc' hder' => ?_⟩
    obtain ⟨m', hm0', hmm', hbound', hret'⟩ := hcite' p' i' σ' ds' hc' hder'
    exact ⟨m', hm0', Nat.lt_trans hmm' hmn, hbound', heq ▸ hret'⟩
  | @reqCall _ e₁ t as π b us fs hte ht hb hus hfs πs _ ih ihs =>
    simp only [inlineAux, Claim.strip]
    simp only [Ev.cites, List.mem_append] at hc
    obtain ⟨ih', _⟩ := ih hΓ (fun q hq => hc q (Or.inl hq))
    simp only [Claim.strip] at ih'
    refine ⟨.reqCall ih' ht hb hus (by rw [inlineListAux_length]; exact hfs) (fun _ => none) (fun j hj => ?_),
      inlinedCite_of_not_proof (fun p h => nomatch h)⟩
    rw [inlineListAux_length] at hj
    rw [inlineListAux_getD _ _ _ _ hj]
    exact (ihs j hj hΓ (fun q hq => hc q (Or.inr (mem_citesList (getD_mem _ _ hj) hq)))).1
  | @witProof p P _ hp hpred =>
    have hpn : p + 1 < n + 1 := hc p (by simp [Ev.cites])
    obtain ⟨⟨π, hty⟩, _, _, hcites⟩ := hP p hp
    refine ⟨?_, fun p' i σ ds hc' hder => ?_⟩
    · simp only [inlineAux, List.getElem?_eq_getElem hp, Claim.strip]
      have hpn' : p < n := Nat.lt_of_succ_lt_succ hpn
      have := (ihn n (Nat.lt_succ_self n) hty rfl (Nat.lt_of_le_of_lt (Nat.zero_le p) hpn')
        (fun q hq => Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (hcites q hq)) hpn')).1
      simpa [Claim.strip, hpred] using this
    · simp only [Option.some.injEq, Cite.proof.injEq] at hc'
      subst hc'
      exact returned_inline_witness L M hP _ hder (n+1) P hpn
  | @witImpl i _ hi hu hok =>
    simp only [inlineAux, Claim.strip]
    exact ⟨.derive hi (by simp [hu.1]) hok (by simp [hu.2]) (fun _ => none) (fun j hj => by simp at hj),
      inlinedCite_of_not_proof (fun p h => nomatch h)⟩
  | @projRes i σ es _ a labels B gs hi hσ hok hes πs _ hb hg ih =>
    simp only [inlineAux, Claim.strip]
    simp only [Ev.cites] at hc
    refine ⟨.projRes hi hσ hok (by rw [inlineListAux_length]; exact hes) (fun _ => none) (fun j hj => ?_) hb hg,
      inlinedCite_of_not_proof (fun p h => nomatch h)⟩
    rw [inlineListAux_length] at hj
    rw [inlineListAux_getD _ _ _ _ hj]
    exact (ih j hj hΓ (fun q hq => hc q (mem_citesList (getD_mem _ _ hj) hq))).1
  | refl =>
    simp only [inlineAux, Claim.strip]
    exact ⟨.refl, inlinedCite_of_not_proof (fun p h => nomatch h)⟩
  | @compose es E _ l r hne hE _ hent ih =>
    simp only [inlineAux, Claim.strip]
    simp only [Ev.cites] at hc
    refine ⟨.compose (by rw [← List.length_pos_iff, inlineListAux_length]; exact List.length_pos_iff.mpr hne)
      (by rw [inlineListAux_length]; exact hE) (fun j hj => ?_) hent,
      inlinedCite_of_not_proof (fun p h => nomatch h)⟩
    rw [inlineListAux_length] at hj
    rw [inlineListAux_getD _ _ _ _ hj]
    exact (ih j hj hΓ (fun q hq => hc q (mem_citesList (getD_mem _ _ hj) hq))).1
  | @coerce _ e₁ P π E es P' π' hte hE _ hπ hwf hpe ih ihs =>
    simp only [Ev.cites, List.mem_append] at hc
    obtain ⟨ih', hcite⟩ := ih hΓ (fun q hq => hc q (Or.inl hq))
    simp only [Claim.strip] at ih'
    refine ⟨?_, fun p i σ ds hc' hder => ?_⟩
    · simp only [inlineAux, Claim.strip]
      refine .coerce ih' (by rw [inlineListAux_length]; exact hE) (fun j hj => ?_) (Or.inl rfl) (claimWF_none _) hpe
      rw [inlineListAux_length] at hj
      rw [inlineListAux_getD _ _ _ _ hj]
      exact (ihs j hj hΓ (fun q hq => hc q (Or.inr (mem_citesList (getD_mem _ _ hj) hq)))).1
    · -- the coerce keeps its input's citation, and returns what its input returns
      simp only at hc'
      rcases hπ with h | h
      · rw [h] at hc'; cases hc'
      · subst h
        obtain ⟨m, hm0, hmn, hbound, hret⟩ := hcite p i σ ds hc' hder
        exact ⟨m, hm0, hmn, hbound, by simp only [inlineAux, Ev.returned]; exact hret⟩

/-! ## The theorems -/

/-- The root half of `witness_is_let`, as SPEC.md 6.4 states it: inlining the proofs a term cites keeps
its root. No condition on the term is needed: `Sim` relates every term to its inlining at every fuel,
and both root lemmas are proved for `Sim`. -/
theorem witness_is_let_root (M : Module) (e : Ev) :
    ∀ i σ, root M (e.inlineProofs M) i σ ↔ root M e i σ := by
  intro i σ
  have hs := sim_inline M (M.proofs.length + 1) e
  constructor
  · rintro ⟨ds', h⟩
    obtain ⟨⟨i₀, σ₀, ds₀⟩, hr, hi, hσ, _⟩ := sim_root_bwd M h hs
    simp only at hi hσ; subst hi; subst hσ
    exact ⟨ds₀, hr⟩
  · rintro ⟨ds, h⟩
    obtain ⟨⟨i₀, σ₀, ds₀⟩, hr, hi, hσ, _⟩ := sim_root_fwd M h hs
    simp only at hi hσ; subst hi; subst hσ
    exact ⟨ds₀, hr⟩

set_option linter.unusedVariables false in
/-- `trait.proof` is the `let` of the calculus (card 6): a closed allege-free term with every cited
proof's body written in place of its witness has the same claim up to the citation and the same root.
Stated closed, as SPEC.md 6.4 states it: under hypotheses, a proven hypothesis `^k` keeps its citation
and the claim is not stripped (Findings.lean). `hS` and `he` are the specification's hypotheses; the
proof needs neither (a `trait.allege` is its own inlining and keeps its claim). -/
theorem witness_is_let (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {e : Ev} {c : Claim}
    (hS : M.sig.DeclWF) (hP : ProofsWF L M Ok) (he : e.allegeFree) (ht : HasType L M Ok [] e c) :
    HasType L M Ok [] (e.inlineProofs M) c.strip ∧
    ∀ i σ, root M (e.inlineProofs M) i σ ↔ root M e i σ :=
  ⟨(inline_hasType L M hP (M.proofs.length + 1) ht rfl (Nat.succ_pos _)
    (fun q hq => Nat.succ_lt_succ (cites_lt_of_hasType L M ht q hq))).1, witness_is_let_root M e⟩

set_option linter.unusedVariables false in
/-- D2: a projection past the plain requirements is the cited derive's operand in its place: the
operand has the projection's claim under any hypotheses, and, where the projected term reads to the
cited derive, the two read to the same base.

Weakened from the skeleton's statement by H2 (`hP`): the projection index is read against the
requirement list of the trait the cited proof's derive implements, which is the written trait's only
when the proof's body is typed at the application it proves (Findings.lean has the refutation). -/
theorem project_where_is_inversion (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop}
    {Γ : List Claim} {e : Ev} {t : TraitId} {as : List Ty} {p : ProofId} {i : ImplId} {σ : List Ty}
    {ds : List Ev} {j : Nat} {c : Claim} (hP : ProofsWF L M Ok)
    (ht : HasType L M Ok Γ e ⟨.app t as, some (.proof p)⟩) (htr : t < M.sig.traits.length)
    (hder : M.proofDerive p = some (i, σ, ds)) (hj : j < ds.length)
    (hc : HasType L M Ok [] (ds.getD j default) c) :
    HasType L M Ok Γ (ds.getD j default) c ∧
    (Root M e (i, σ, ds) ->
      ∀ r, Root M (.project e (M.sig.traits[t].reqs.length + j)) r ↔ Root M (ds.getD j default) r) := by
  refine ⟨hasType_weaken L M Γ hc, fun hroot r => ?_⟩
  obtain ⟨_, hb⟩ := proofCite_rooted L M hP ht rfl hroot
  obtain ⟨hp, as', hpred⟩ := hb _ _ rfl
  obtain ⟨hi, htrait⟩ := trait_of_proofDeriveAux L M hP _ hder hp hpred
  constructor
  · intro h
    cases h with
    | projReq hb' hI hT hk _ =>
      cases root_unique M hroot hb'
      rw [List.getElem?_eq_getElem hi] at hI; cases hI
      rw [htrait, List.getElem?_eq_getElem htr] at hT; cases hT
      exact absurd hk (by omega)
    | projWhere hb' hI hT hk hop =>
      cases root_unique M hroot hb'
      rw [List.getElem?_eq_getElem hi] at hI; cases hI
      rw [htrait, List.getElem?_eq_getElem htr] at hT; cases hT
      rw [Nat.add_sub_cancel_left] at hop
      exact hop
  · intro h
    refine .projWhere hroot (List.getElem?_eq_getElem hi) ?_ (Nat.le_add_right _ _) ?_
    · rw [htrait]; exact List.getElem?_eq_getElem htr
    · rw [Nat.add_sub_cancel_left]; exact h

end Trait
