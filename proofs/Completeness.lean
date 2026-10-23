import Calculus
/-!
# Completeness: theorem 4 and the cut, 4b (SPEC.md 6.2)

4a: every valid judgment has an allege-free term, built by `elabDeriv` (the frontend's `emit` and the
stage's proof writer read as one function of the derivation). 4b: every closed valid predicate has a
closed allege-free term, so `trait.allege` is a cut.

Exceptions SPEC.md records as findings and this calculus does not see: (CONG) across a
placeholder-declining constructor has no coerce (F9: every `con` is a node to `Entails` here), and the
cross-group (UNIF) has no term (F6: (UNIF) is not a rule of `Deriv`).

How the proofs go. `elabL` writes no `trait.allege` and no `trait.witness` of a proof, so its term is
allege-free and cites nothing; theorem 4a therefore needs no new proof and `M' = M`. The typing is one
induction on the derivation (`elabL_spec`): each rule of SPEC.md 4.1 meets the typing rule of the
operation `elabL` writes for it. The equality rules (SYM)(TRANS)(CONG)(CONG-P)(NORM) and (CONV) become
one `trait.coerce` over the equality leaves of the subderivation; `seeds` lists the equalities those
leaves state, and the coerce's closure premise is proved from them inside `Entails` itself: (SEED) is
a leaf, (EQUIV) is `refl`/`symm`/`trans`, (CONG) is `con`/`proj`, (NORM) is `law`. The stage's form
of the cut (`cut_proof`) writes the term as one new `trait.proof` at the end of the table; `hasType_append`
says the table's existing typings survive the write, and the new proof cites nothing, so the citation
order stays well founded.
-/

namespace Trait

/-! ## Lists built by position (`tabulate`) -/

/-- Reading a tabulated list at a position below its length gives the function's value there. -/
private theorem tabulate_getD {α : Type _} [Inhabited α] :
    ∀ (n : Nat) (f : (j : Nat) → j < n → α) (j : Nat) (hj : j < n),
      (tabulate n f).getD j default = f j hj
  | 0, _, j, hj => absurd hj (Nat.not_lt_zero j)
  | n+1, f, j, hj => by
    simp only [tabulate]
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hj) with h | h
    · rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by rw [tabulate_length]; exact h),
        ← List.getD_eq_getElem?_getD]
      exact tabulate_getD n _ j h
    · subst h
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_append_right (by rw [tabulate_length]; exact Nat.le_refl _),
        tabulate_length, Nat.sub_self]
      rfl

/-- Every value of the function is a member of the tabulated list. -/
private theorem tabulate_mem {α : Type _} :
    ∀ (n : Nat) (f : (j : Nat) → j < n → α) (j : Nat) (hj : j < n), f j hj ∈ tabulate n f
  | 0, _, j, hj => absurd hj (Nat.not_lt_zero j)
  | n+1, f, j, hj => by
    simp only [tabulate, List.mem_append, List.mem_singleton]
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hj) with h | h
    · exact Or.inl (tabulate_mem n _ j h)
    · subst h; exact Or.inr rfl

/-- Every member of a tabulated list is a value of the function. -/
private theorem mem_tabulate {α : Type _} :
    ∀ (n : Nat) (f : (j : Nat) → j < n → α) (x : α), x ∈ tabulate n f → ∃ j hj, x = f j hj
  | 0, _, x, hx => by simp [tabulate] at hx
  | n+1, f, x, hx => by
    simp only [tabulate, List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · obtain ⟨j, hj, rfl⟩ := mem_tabulate n _ x hx
      exact ⟨j, Nat.lt_succ_of_lt hj, rfl⟩
    · exact ⟨n, Nat.lt_succ_self n, hx⟩

/-- Each tabulated list is part of the flattening of the tabulation. -/
private theorem subset_flatten_tabulate {α : Type _} (n : Nat) (f : (j : Nat) → j < n → List α)
    (j : Nat) (hj : j < n) : f j hj ⊆ (tabulate n f).flatten :=
  fun _ hx => List.mem_flatten.mpr ⟨f j hj, tabulate_mem n f j hj, hx⟩

/-! ## `hasAllegeList` and `citesList` over the lists `elabL` builds -/

private theorem hasAllegeList_eq_false_iff :
    ∀ (l : List Ev), hasAllegeList l = false ↔ ∀ e ∈ l, e.hasAllege = false
  | [] => by simp [hasAllegeList]
  | e :: es => by
    rw [hasAllegeList, Bool.or_eq_false_iff, hasAllegeList_eq_false_iff es]
    constructor
    · rintro ⟨h1, h2⟩ x hx
      rcases List.mem_cons.mp hx with rfl | hx
      · exact h1
      · exact h2 x hx
    · intro h
      exact ⟨h e List.mem_cons_self, fun x hx => h x (List.mem_cons_of_mem _ hx)⟩

private theorem citesList_eq_nil_iff :
    ∀ (l : List Ev), citesList l = [] ↔ ∀ e ∈ l, e.cites = []
  | [] => by simp [citesList]
  | e :: es => by
    rw [citesList, List.append_eq_nil_iff, citesList_eq_nil_iff es]
    constructor
    · rintro ⟨h1, h2⟩ x hx
      rcases List.mem_cons.mp hx with rfl | hx
      · exact h1
      · exact h2 x hx
    · intro h
      exact ⟨h e List.mem_cons_self, fun x hx => h x (List.mem_cons_of_mem _ hx)⟩

private theorem hasAllegeList_tabulate {n : Nat} {f : (j : Nat) → j < n → Ev}
    (h : ∀ j hj, (f j hj).hasAllege = false) : hasAllegeList (tabulate n f) = false := by
  rw [hasAllegeList_eq_false_iff]
  intro x hx
  obtain ⟨j, hj, rfl⟩ := mem_tabulate n f x hx
  exact h j hj

private theorem citesList_tabulate {n : Nat} {f : (j : Nat) → j < n → Ev}
    (h : ∀ j hj, (f j hj).cites = []) : citesList (tabulate n f) = [] := by
  rw [citesList_eq_nil_iff]
  intro x hx
  obtain ⟨j, hj, rfl⟩ := mem_tabulate n f x hx
  exact h j hj

private theorem hasAllegeList_flatten_tabulate {n : Nat} {f : (j : Nat) → j < n → List Ev}
    (h : ∀ j hj, hasAllegeList (f j hj) = false) : hasAllegeList (tabulate n f).flatten = false := by
  rw [hasAllegeList_eq_false_iff]
  intro x hx
  obtain ⟨l, hl, hx⟩ := List.mem_flatten.mp hx
  obtain ⟨j, hj, rfl⟩ := mem_tabulate n f l hl
  exact (hasAllegeList_eq_false_iff _).mp (h j hj) x hx

private theorem citesList_flatten_tabulate {n : Nat} {f : (j : Nat) → j < n → List Ev}
    (h : ∀ j hj, citesList (f j hj) = []) : citesList (tabulate n f).flatten = [] := by
  rw [citesList_eq_nil_iff]
  intro x hx
  obtain ⟨l, hl, hx⟩ := List.mem_flatten.mp hx
  obtain ⟨j, hj, rfl⟩ := mem_tabulate n f l hl
  exact (citesList_eq_nil_iff _).mp (h j hj) x hx

private theorem hasAllegeList_append {l₁ l₂ : List Ev} (h₁ : hasAllegeList l₁ = false)
    (h₂ : hasAllegeList l₂ = false) : hasAllegeList (l₁ ++ l₂) = false := by
  rw [hasAllegeList_eq_false_iff] at *
  intro x hx
  rcases List.mem_append.mp hx with hx | hx
  · exact h₁ x hx
  · exact h₂ x hx

private theorem citesList_append {l₁ l₂ : List Ev} (h₁ : citesList l₁ = [])
    (h₂ : citesList l₂ = []) : citesList (l₁ ++ l₂) = [] := by
  rw [citesList_eq_nil_iff] at *
  intro x hx
  rcases List.mem_append.mp hx with hx | hx
  · exact h₁ x hx
  · exact h₂ x hx

/-! ## The closure is monotone in its seeds -/

/-- A closure fact from some seeds is a closure fact from more seeds. -/
private theorem Entails.mono {L : ConLaw} {E E' : List (Ty × Ty)} (hE : E ⊆ E') {a b : Ty}
    (h : Entails L E a b) : Entails L E' a b := by
  induction h with
  | ax h => exact .ax (hE h)
  | refl => exact .refl
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | con hl _ ih => exact .con hl ih
  | proj hl hg _ _ ih₁ ih₂ => exact .proj hl hg ih₁ ih₂
  | law h => exact .law h

/-! ## Typed equality leaves -/

/-- The `j`-th leaf is typed, with no citation, at the equality the `j`-th seed states: the form in
which `trait.coerce` reads its equality operands (card 5). -/
private inductive Leaves (L : ConLaw) (M : Module) (Γ : List Claim) :
    List Ev → List (Ty × Ty) → Prop where
  | nil : Leaves L M Γ [] []
  | cons : HasType L M AnyImpl Γ e ⟨.eq l r, none⟩ → Leaves L M Γ es E →
           Leaves L M Γ (e :: es) ((l, r) :: E)

private theorem Leaves.length {L : ConLaw} {M : Module} {Γ : List Claim} {es : List Ev}
    {E : List (Ty × Ty)} (h : Leaves L M Γ es E) : E.length = es.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

/-- The leaves in the positional form COERCE's premise takes. -/
private theorem Leaves.getD {L : ConLaw} {M : Module} {Γ : List Claim} {es : List Ev}
    {E : List (Ty × Ty)} (h : Leaves L M Γ es E) : ∀ j, j < es.length →
      HasType L M AnyImpl Γ (es.getD j default)
        ⟨.eq (E.getD j default).1 (E.getD j default).2, none⟩ := by
  induction h with
  | nil => intro j hj; exact absurd hj (Nat.not_lt_zero j)
  | cons h _ ih =>
    intro j hj
    cases j with
    | zero => simpa using h
    | succ j => simpa using ih j (Nat.lt_of_succ_lt_succ hj)

private theorem Leaves.append {L : ConLaw} {M : Module} {Γ : List Claim} {es₁ es₂ : List Ev}
    {E₁ E₂ : List (Ty × Ty)} (h₁ : Leaves L M Γ es₁ E₁) (h₂ : Leaves L M Γ es₂ E₂) :
    Leaves L M Γ (es₁ ++ es₂) (E₁ ++ E₂) := by
  induction h₁ with
  | nil => simpa using h₂
  | cons h _ ih => exact .cons h ih

private theorem Leaves.flatten {L : ConLaw} {M : Module} {Γ : List Claim} :
    ∀ (n : Nat) (ls : (j : Nat) → j < n → List Ev) (Es : (j : Nat) → j < n → List (Ty × Ty)),
      (∀ j hj, Leaves L M Γ (ls j hj) (Es j hj)) →
      Leaves L M Γ (tabulate n ls).flatten (tabulate n Es).flatten
  | 0, _, _, _ => by simp only [tabulate, List.flatten_nil]; exact .nil
  | n+1, ls, Es, h => by
    simp only [tabulate, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil]
    exact (Leaves.flatten n _ _ (fun j hj => h j _)).append (h n _)

/-! ## The seeds of an equality derivation -/

/-- The one seed a leaf states when its predicate is an equality; an application states none. -/
private def seedsOf : Pred → List (Ty × Ty)
  | .eq a b => [(a, b)]
  | .app _ _ => []

/-- The equalities the leaves of `elabL` state, in the leaves' order: a leaf's own predicate; the
union through (SYM)(TRANS)(CONG)(CONG-P); nothing for (REFL) and (NORM), which the closure carries. -/
private def seeds (L : ConLaw) (S : Sig) : Deriv L S Ok Γ P → List (Ty × Ty)
  | .hyp .. => seedsOf P
  | .impl .. => []
  | .req .. => seedsOf P
  | .reqA .. => seedsOf P
  | .bind .. => seedsOf P
  | .refl => []
  | .symm d => seeds L S d
  | .trans d₁ d₂ => seeds L S d₁ ++ seeds L S d₂
  | .cong _ ds => (tabulate _ (fun j hj => seeds L S (ds j hj))).flatten
  | .congP _ _ ds es =>
      (tabulate _ (fun j hj => seeds L S (ds j hj))).flatten ++
      (tabulate _ (fun j hj => seeds L S (es j hj))).flatten
  | .norm _ => []
  | .conv .. => []

/-- A leaf typed at an equality is one typed leaf over the one seed it states. -/
private theorem leaf_seeds {L : ConLaw} {M : Module} {Γ : List Claim} {e : Ev} {P : Pred}
    (h : HasType L M AnyImpl Γ e ⟨P, none⟩) (a b : Ty) (hab : P = .eq a b) :
    Leaves L M Γ [e] (seedsOf P) ∧ Entails L (seedsOf P) a b := by
  subst hab
  refine ⟨.cons h .nil, .ax ?_⟩
  simp [seedsOf]

/-- An unproven claim is well formed whatever its predicate. -/
private theorem claimWF_none : ∀ P : Pred, Claim.WF ⟨P, none⟩
  | .app _ _ => trivial
  | .eq _ _ => trivial

/-! ## `elabL` builds a well-typed, allege-free, citation-free term -/

/-- `elabL` is `emit` and the proof writer as one function (SPEC.md 6.2): the term it builds for a
derivation contains no `trait.allege` and cites no proof, and is typed at the derived predicate with
no citation under the hypotheses read as unproven claims. When the predicate is an equality, the
leaves it lists are typed at `seeds` and the seeds entail the equality in the closure, which is
exactly the premise of the one coerce the equality rules become. -/
private theorem elabL_spec (L : ConLaw) (M : Module) {Γ : List Pred} {P : Pred}
    (d : Deriv L M.sig AnyImpl Γ P) :
    (elabL L M.sig d).1.hasAllege = false ∧ hasAllegeList (elabL L M.sig d).2 = false ∧
    (elabL L M.sig d).1.cites = [] ∧ citesList (elabL L M.sig d).2 = [] ∧
    HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim))) (elabL L M.sig d).1 ⟨P, none⟩ ∧
    (∀ a b, P = .eq a b →
      Leaves L M (Γ.map (fun Q => (⟨Q, none⟩ : Claim))) (elabL L M.sig d).2 (seeds L M.sig d) ∧
      Entails L (seeds L M.sig d) a b) := by
  induction d with
  | hyp k hk =>
    -- (HYP) is the block argument `^k`, typed at the `k`-th hypothesis.
    simp only [elabL, seeds]
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim))) (.hyp k) ⟨Γ[k], none⟩ := by
      have hk' : k < (Γ.map (fun Q => (⟨Q, none⟩ : Claim))).length := by simpa using hk
      have := HasType.hyp (L := L) (M := M) (Ok := AnyImpl) hk'
      simpa using this
    exact ⟨rfl, rfl, rfl, rfl, h5, fun a b hab => leaf_seeds h5 a b hab⟩
  | impl i hi σ hσ _ ds ih =>
    -- (IMPL) is a derive whose operands are the where entries' terms, by position.
    simp only [elabL, seeds]
    have h1 : hasAllegeList (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) = false :=
      hasAllegeList_tabulate (fun j hj => (ih j hj).1)
    have h3 : citesList (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) = [] :=
      citesList_tabulate (fun j hj => (ih j hj).2.2.1)
    have h5 := HasType.derive (L := L) (M := M) (Ok := AnyImpl)
      (Γ := Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
      (es := tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) hi hσ trivial (tabulate_length _ _)
      (fun _ => none) (by
        intro j hj
        rw [tabulate_length] at hj
        rw [tabulate_getD _ _ j hj]
        exact (ih j hj).2.2.2.2.1)
    refine ⟨by simpa [Ev.hasAllege] using h1, by simpa [hasAllegeList, Ev.hasAllege] using h1,
      by simpa [Ev.cites] using h3, by simpa [citesList, Ev.cites] using h3, h5, ?_⟩
    intro a b hab
    simp [ImplDecl.header] at hab
  | @req t as k d ht hk ih =>
    -- (REQ) is a projection of the application's term at the requirement's position.
    simp only [elabL, seeds]
    obtain ⟨h1, _, h3, _, h5, _⟩ := ih
    have h5' := HasType.projReq h5 ht hk
    exact ⟨by simpa [Ev.hasAllege] using h1, by simpa [hasAllegeList, Ev.hasAllege] using h1,
      by simpa [Ev.cites] using h3, by simpa [citesList, Ev.cites] using h3, h5',
      fun a b hab => leaf_seeds h5' a b hab⟩
  | @reqA t as b d ht hb us hus prem ih ihp =>
    -- (REQ-A) is the binder call, its premises the terms of the binder's premises by position.
    simp only [elabL, seeds]
    obtain ⟨h1, _, h3, _, h5, _⟩ := ih
    have hp1 : hasAllegeList (tabulate _ (fun j hj => (elabL L M.sig (prem j hj)).1)) = false :=
      hasAllegeList_tabulate (fun j hj => (ihp j hj).1)
    have hp3 : citesList (tabulate _ (fun j hj => (elabL L M.sig (prem j hj)).1)) = [] :=
      citesList_tabulate (fun j hj => (ihp j hj).2.2.1)
    have h5' := HasType.reqCall (fs := tabulate _ (fun j hj => (elabL L M.sig (prem j hj)).1))
      h5 ht hb hus (tabulate_length _ _) (fun _ => none) (by
        intro j hj
        rw [tabulate_length] at hj
        rw [tabulate_getD _ _ j hj]
        exact (ihp j hj).2.2.2.2.1)
    exact ⟨by simp [Ev.hasAllege, h1, hp1], by simp [hasAllegeList, Ev.hasAllege, h1, hp1],
      by simp [Ev.cites, h3, hp3], by simp [citesList, Ev.cites, h3, hp3], h5',
      fun a b hab => leaf_seeds h5' a b hab⟩
  | @bind a labels B gs i hi σ hσ _ ds hb hg ih =>
    -- (BIND) is the `proj_resolve` witness with the where entries' terms as its premises.
    simp only [elabL, seeds]
    have h1 : hasAllegeList (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) = false :=
      hasAllegeList_tabulate (fun j hj => (ih j hj).1)
    have h3 : citesList (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) = [] :=
      citesList_tabulate (fun j hj => (ih j hj).2.2.1)
    have h5 := HasType.projRes (L := L) (M := M) (Ok := AnyImpl)
      (Γ := Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
      (es := tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).1)) hi hσ trivial (tabulate_length _ _)
      (fun _ => none) (by
        intro j hj
        rw [tabulate_length] at hj
        rw [tabulate_getD _ _ j hj]
        exact (ih j hj).2.2.2.2.1) hb hg
    exact ⟨by simpa [Ev.hasAllege] using h1, by simpa [hasAllegeList, Ev.hasAllege] using h1,
      by simpa [Ev.cites] using h3, by simpa [citesList, Ev.cites] using h3, h5,
      fun a' b' hab => leaf_seeds h5 a' b' hab⟩
  | @refl a =>
    -- (REFL) is the `refl` witness, with no leaves and no seeds.
    simp only [elabL, seeds]
    refine ⟨rfl, rfl, rfl, rfl, HasType.refl, ?_⟩
    intro a' b' hab
    cases hab
    exact ⟨.nil, .refl⟩
  | @symm a b d ih =>
    -- (SYM) is a coerce of `refl b` to `b = a` over the leaves of `a = b`; `symm` in the closure.
    simp only [elabL, seeds]
    obtain ⟨_, h2, _, h4, _, h6⟩ := ih
    obtain ⟨hL, hE⟩ := h6 a b rfl
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (.refl b) (elabL L M.sig d).2 ⟨.eq b a, none⟩) ⟨.eq b a, none⟩ :=
      HasType.coerce HasType.refl hL.length hL.getD (Or.inl rfl) (claimWF_none _)
        (PredEntails.eq .refl (.symm hE))
    exact ⟨by simpa [Ev.hasAllege] using h2, h2, by simpa [Ev.cites] using h4, h4, h5,
      fun a' b' hab => by cases hab; exact ⟨hL, .symm hE⟩⟩
  | @trans a b c d₁ d₂ ih₁ ih₂ =>
    -- (TRANS) is a coerce of `refl a` to `a = c` over both leaf lists; `trans` in the closure.
    simp only [elabL, seeds]
    obtain ⟨_, h2₁, _, h4₁, _, h6₁⟩ := ih₁
    obtain ⟨_, h2₂, _, h4₂, _, h6₂⟩ := ih₂
    obtain ⟨hL₁, hE₁⟩ := h6₁ a b rfl
    obtain ⟨hL₂, hE₂⟩ := h6₂ b c rfl
    have hL := hL₁.append hL₂
    have hE : Entails L (seeds L M.sig d₁ ++ seeds L M.sig d₂) a c :=
      .trans (hE₁.mono (List.subset_append_left _ _)) (hE₂.mono (List.subset_append_right _ _))
    have h2 := hasAllegeList_append h2₁ h2₂
    have h4 := citesList_append h4₁ h4₂
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (.refl a) ((elabL L M.sig d₁).2 ++ (elabL L M.sig d₂).2) ⟨.eq a c, none⟩)
        ⟨.eq a c, none⟩ :=
      HasType.coerce HasType.refl hL.length hL.getD (Or.inl rfl) (claimWF_none _)
        (PredEntails.eq .refl hE)
    exact ⟨by simpa [Ev.hasAllege] using h2, h2, by simpa [Ev.cites] using h4, h4, h5,
      fun a' b' hab => by cases hab; exact ⟨hL, hE⟩⟩
  | @cong c as bs hl ds ih =>
    -- (CONG) is a coerce of `refl K(a)` to `K(a) = K(b)` over every child's leaves; `con` in the closure.
    simp only [elabL, seeds]
    have hL := Leaves.flatten _ (fun j hj => (elabL L M.sig (ds j hj)).2)
      (fun j hj => seeds L M.sig (ds j hj)) (fun j hj => ((ih j hj).2.2.2.2.2 _ _ rfl).1)
    have hE : Entails L (tabulate _ (fun j hj => seeds L M.sig (ds j hj))).flatten
        (.con c as) (.con c bs) :=
      .con hl (fun j hj => ((ih j hj).2.2.2.2.2 _ _ rfl).2.mono (subset_flatten_tabulate _ _ j hj))
    have h2 := hasAllegeList_flatten_tabulate (fun j hj => (ih j hj).2.1)
    have h4 := citesList_flatten_tabulate (fun j hj => (ih j hj).2.2.2.1)
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (.refl (.con c as)) (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).2)).flatten
          ⟨.eq (.con c as) (.con c bs), none⟩) ⟨.eq (.con c as) (.con c bs), none⟩ :=
      HasType.coerce HasType.refl hL.length hL.getD (Or.inl rfl) (claimWF_none _)
        (PredEntails.eq .refl hE)
    exact ⟨by simpa [Ev.hasAllege] using h2, h2, by simpa [Ev.cites] using h4, h4, h5,
      fun a' b' hab => by cases hab; exact ⟨hL, hE⟩⟩
  | @congP t as a gs bs hs hl hg ds es ih₁ ih₂ =>
    -- (CONG-P) is a coerce of `refl` over the leaves of both argument lists; `proj` in the closure.
    simp only [elabL, seeds]
    have hL₁ := Leaves.flatten _ (fun j hj => (elabL L M.sig (ds j hj)).2)
      (fun j hj => seeds L M.sig (ds j hj)) (fun j hj => ((ih₁ j hj).2.2.2.2.2 _ _ rfl).1)
    have hL₂ := Leaves.flatten _ (fun j hj => (elabL L M.sig (es j hj)).2)
      (fun j hj => seeds L M.sig (es j hj)) (fun j hj => ((ih₂ j hj).2.2.2.2.2 _ _ rfl).1)
    have hL := hL₁.append hL₂
    have hE : Entails L ((tabulate _ (fun j hj => seeds L M.sig (ds j hj))).flatten ++
        (tabulate _ (fun j hj => seeds L M.sig (es j hj))).flatten)
        (.proj t as a gs) (.proj t bs a hs) :=
      .proj hl hg
        (fun j hj => ((ih₁ j hj).2.2.2.2.2 _ _ rfl).2.mono
          (fun _ hx => List.mem_append_left _ (subset_flatten_tabulate _ _ j hj hx)))
        (fun j hj => ((ih₂ j hj).2.2.2.2.2 _ _ rfl).2.mono
          (fun _ hx => List.mem_append_right _ (subset_flatten_tabulate _ _ j hj hx)))
    have h2 := hasAllegeList_append
      (hasAllegeList_flatten_tabulate (fun j hj => (ih₁ j hj).2.1))
      (hasAllegeList_flatten_tabulate (fun j hj => (ih₂ j hj).2.1))
    have h4 := citesList_append
      (citesList_flatten_tabulate (fun j hj => (ih₁ j hj).2.2.2.1))
      (citesList_flatten_tabulate (fun j hj => (ih₂ j hj).2.2.2.1))
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (.refl (.proj t as a gs))
          ((tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).2)).flatten ++
           (tabulate _ (fun j hj => (elabL L M.sig (es j hj)).2)).flatten)
          ⟨.eq (.proj t as a gs) (.proj t bs a hs), none⟩)
        ⟨.eq (.proj t as a gs) (.proj t bs a hs), none⟩ :=
      HasType.coerce HasType.refl hL.length hL.getD (Or.inl rfl) (claimWF_none _)
        (PredEntails.eq .refl hE)
    exact ⟨by simpa [Ev.hasAllege] using h2, h2, by simpa [Ev.cites] using h4, h4, h5,
      fun a' b' hab => by cases hab; exact ⟨hL, hE⟩⟩
  | @norm a b h =>
    -- (NORM) is a coerce of `refl a` to `a = b` over no leaves; `law` in the closure.
    simp only [elabL, seeds]
    have h5 : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (.refl a) [] ⟨.eq a b, none⟩) ⟨.eq a b, none⟩ :=
      HasType.coerce (E := []) HasType.refl rfl (fun j hj => absurd hj (Nat.not_lt_zero j))
        (Or.inl rfl) (claimWF_none _) (PredEntails.eq .refl (.law h))
    exact ⟨rfl, rfl, rfl, rfl, h5, fun a' b' hab => by cases hab; exact ⟨.nil, .law h⟩⟩
  | @conv t as bs d hl ds ih ihs =>
    -- (CONV) is a coerce of the application's term over the leaves of the argument equalities.
    simp only [elabL, seeds]
    obtain ⟨h1, _, h3, _, h5, _⟩ := ih
    have hL := Leaves.flatten _ (fun j hj => (elabL L M.sig (ds j hj)).2)
      (fun j hj => seeds L M.sig (ds j hj)) (fun j hj => ((ihs j hj).2.2.2.2.2 _ _ rfl).1)
    have hPE : PredEntails L (tabulate _ (fun j hj => seeds L M.sig (ds j hj))).flatten
        (.app t as) (.app t bs) :=
      .app hl (fun j hj => ((ihs j hj).2.2.2.2.2 _ _ rfl).2.mono (subset_flatten_tabulate _ _ j hj))
    have h2 := hasAllegeList_flatten_tabulate (fun j hj => (ihs j hj).2.1)
    have h4 := citesList_flatten_tabulate (fun j hj => (ihs j hj).2.2.2.1)
    have h5' : HasType L M AnyImpl (Γ.map (fun Q => (⟨Q, none⟩ : Claim)))
        (.coerce (elabL L M.sig d).1 (tabulate _ (fun j hj => (elabL L M.sig (ds j hj)).2)).flatten
          ⟨.app t bs, none⟩) ⟨.app t bs, none⟩ :=
      HasType.coerce h5 hL.length hL.getD (Or.inl rfl) (claimWF_none _) hPE
    exact ⟨by simp [Ev.hasAllege, h1, h2], by simp [hasAllegeList, Ev.hasAllege, h1, h2],
      by simp [Ev.cites, h3, h4], by simp [citesList, Ev.cites, h3, h4], h5',
      fun a' b' hab => by cases hab⟩

/-! ## Writing a proof keeps every typing -/

/-- `proofDeriveAux` reads the same derive after proofs are appended to the table, with any fuel at
least the one it had: every proof the chain read is still at its index. -/
private theorem proofDeriveAux_append (M : Module) (extra : List (Pred × Ev)) :
    ∀ (f f' : Nat) (p : ProofId) (r : ImplId × List Ty × List Ev), f ≤ f' →
      M.proofDeriveAux f p = some r →
      Module.proofDeriveAux ⟨M.sig, M.proofs ++ extra⟩ f' p = some r
  | 0, _, _, _, _, h => by simp [Module.proofDeriveAux] at h
  | f+1, 0, _, _, hf, _ => absurd hf (Nat.not_succ_le_zero f)
  | f+1, f'+1, p, r, hf, h => by
    cases hp : M.proofs[p]? with
    | none => simp [Module.proofDeriveAux, hp] at h
    | some pr =>
      obtain ⟨Q, body⟩ := pr
      have hp' : (M.proofs ++ extra)[p]? = some (Q, body) := by
        rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp hp).1]; exact hp
      simp only [Module.proofDeriveAux, hp] at h
      simp only [Module.proofDeriveAux, hp']
      cases hr : body.returned with
      | none => simp [hr] at h
      | some s =>
        cases s with
        | inl d => simp only [hr] at h ⊢; exact h
        | inr q =>
          simp only [hr] at h ⊢
          exact proofDeriveAux_append M extra f f' q r (Nat.le_of_succ_le_succ hf) h

/-- `Pi(p)`'s derive is read alike after proofs are appended. -/
private theorem proofDerive_append (M : Module) (extra : List (Pred × Ev)) {p : ProofId}
    {r : ImplId × List Ty × List Ev} (h : M.proofDerive p = some r) :
    Module.proofDerive ⟨M.sig, M.proofs ++ extra⟩ p = some r :=
  proofDeriveAux_append M extra _ _ p r (by simp only [List.length_append]; omega) h

/-- Appending proofs to the table keeps every typing: the only rules that read the table are WIT-APP
and PROJECT-WHERE, and each reads a proof at an index the longer table answers alike. -/
private theorem hasType_append (L : ConLaw) (M : Module) (extra : List (Pred × Ev))
    {Ok : ImplId -> Subst -> Prop} {Γ : List Claim} {e : Ev} {c : Claim}
    (ht : HasType L M Ok Γ e c) : HasType L ⟨M.sig, M.proofs ++ extra⟩ Ok Γ e c := by
  induction ht with
  | hyp hk => exact .hyp hk
  | derive hi hσ hok hlen πs _ ih =>
    exact .derive (M := ⟨M.sig, M.proofs ++ extra⟩) hi hσ hok hlen πs ih
  | projReq _ ht hk ih => exact .projReq ih ht hk
  | projWhere _ ht hd hj _ ih₁ ih₂ => exact .projWhere ih₁ ht (proofDerive_append M extra hd) hj ih₂
  | reqCall _ ht hb hus hlen πs _ ih ihs => exact .reqCall ih ht hb hus hlen πs ihs
  | witProof hp hP =>
    have hp' : _ < (M.proofs ++ extra).length := Nat.lt_of_lt_of_le hp (by simp)
    exact .witProof hp' (by rw [List.getElem_append_left hp]; exact hP)
  | witImpl hi hu hok => exact .witImpl (M := ⟨M.sig, M.proofs ++ extra⟩) hi hu hok
  | projRes hi hσ hok hlen πs _ hb hg ih =>
    exact .projRes (M := ⟨M.sig, M.proofs ++ extra⟩) hi hσ hok hlen πs ih hb hg
  | refl => exact .refl
  | compose hne hlen _ hent ih => exact .compose hne hlen ih hent
  | coerce _ hlen _ hπ hwf hpe ih₁ ih₂ => exact .coerce ih₁ hlen ih₂ hπ hwf hpe

/-! ## The theorems -/

/-- Theorem 4a, COMPLETENESS: `S; pred(G) |- P => exists M' >= M, exists e allege-free. M'; G |- e : P`.
`elabDeriv` writes no proof, so `M' = M` witnesses the existential. -/
theorem completeness (L : ConLaw) (M : Module) {Γ : List Pred} {P : Pred} (hS : M.sig.DeclWF)
    (hP : ProofsWF L M AnyImpl) (d : Deriv L M.sig AnyImpl Γ P) :
    ∃ M' : Module, M'.sig = M.sig ∧ ProofsWF L M' AnyImpl ∧ (elabDeriv L M d).allegeFree ∧
      HasType L M' AnyImpl (Γ.map (⟨·, none⟩)) (elabDeriv L M d) ⟨P, none⟩ := by
  obtain ⟨h1, _, _, _, h5, _⟩ := elabL_spec L M d
  exact ⟨M, rfl, hP, h1, h5⟩

/-- Theorem 4b, the cut: a closed valid predicate has a closed allege-free term. -/
theorem cut_admissible (L : ConLaw) (M : Module) {P : Pred} (hS : M.sig.DeclWF) (hP : ProofsWF L M AnyImpl)
    (h : Valid L M.sig [] P) :
    ∃ (M' : Module) (e : Ev), M'.sig = M.sig ∧ e.allegeFree ∧ HasType L M' AnyImpl [] e ⟨P, none⟩ := by
  obtain ⟨d⟩ := h
  obtain ⟨M', hsig, _, haf, hty⟩ := completeness L M hS hP d
  exact ⟨M', elabDeriv L M d, hsig, haf, hty⟩

/-- Theorem 4b in the stage's form: a closed valid ground application is proved by a `trait.proof`
written to the table, cited by a `trait.witness` (D/ImplResolution.cpp:466-498, 994-1005). The proof
table stays well founded because the written body cites no proof. -/
theorem cut_proof (L : ConLaw) (M : Module) {t : TraitId} {as : List Ty} (hS : M.sig.DeclWF)
    (hP : ProofsWF L M AnyImpl) (hg : groundList as = true) (h : Valid L M.sig [] (.app t as)) (Γ : List Claim) :
    ∃ (M' : Module) (p : ProofId), M'.sig = M.sig ∧ ProofsWF L M' AnyImpl ∧
      HasType L M' AnyImpl Γ (.witness (.proof p) (.app t as)) ⟨.app t as, some (.proof p)⟩ := by
  obtain ⟨d⟩ := h
  obtain ⟨h1, _, h3, _, h5, _⟩ := elabL_spec L M d
  -- The new proof is the derivation's term, appended at index `M.proofs.length`.
  refine ⟨⟨M.sig, M.proofs ++ [(Pred.app t as, (elabL L M.sig d).1)]⟩, M.proofs.length, rfl, ?_, ?_⟩
  · intro q hq
    have hq' : q < M.proofs.length + 1 := by simpa using hq
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hq') with hlt | heq
    · -- An existing proof: its typing moves to the longer table; the rest is unchanged.
      obtain ⟨⟨π, hty⟩, hgr, haf, hc⟩ := hP q hlt
      simp only [List.getElem_append_left hlt]
      exact ⟨⟨π, hasType_append L M _ hty⟩, hgr, haf, hc⟩
    · -- The new proof: closed, ground, allege-free, and it cites nothing.
      subst heq
      have hget : (M.proofs ++ [(.app t as, (elabL L M.sig d).1)])[M.proofs.length]'(by simp) =
          (.app t as, (elabL L M.sig d).1) := List.getElem_concat_length rfl _
      simp only [hget]
      refine ⟨⟨none, hasType_append L M _ h5⟩, hg, h1, ?_⟩
      rw [h3]
      intro q hq
      cases hq
  · have hp : M.proofs.length < (M.proofs ++ [(Pred.app t as, (elabL L M.sig d).1)]).length := by simp
    exact HasType.witProof hp (congrArg Prod.fst (List.getElem_concat_length rfl hp))

end Trait
