import Defs
/-!
# Logic: the trait logic `S; G |- P`, the ground congruence closure and the denotation

SPEC.md section 4 (the rules), 5.2 (the closure), 4.3 (the primitive fragment and H3), 6.3 (denotation,
`ServedBy`, `Coherent`) and 6.6 (theorem 7's statement), transcribed from the section 7 skeleton.

Decisions taken by the scaffold:
* (REQ-A) is added as `Deriv.reqA`, by the pattern of `Deriv.impl`: the binder's premises are given by
  position, and the binder's labels take the GAT arguments through `Binder.callSubst`.
* The skeleton's `full : Bool` flag and the separate `Env` inductive are replaced by one inductive
  `Deriv` and two predicates on derivations, `Deriv.InEnv` (membership in `Env(G)`, the closure of the
  hypotheses under (REQ) and (REQ-A)) and `Deriv.Primitive` (the fragment `|-p`). The reason is (REQ-A):
  its premises are arbitrary derivations while its application operand must be an `Env(G)` member, so
  `Env` and the derivations are mutually dependent; a predicate on one inductive states that without a
  mutual inductive. `Valid` is inhabitation of `Deriv`; `ValidP` is inhabitation by a primitive one.
* The hypothesis rule carries the position `k` of the hypothesis so that `elab` (Calculus.lean) can
  write `^k`; the skeleton's `P ∈ Γ` is a proposition and cannot be read for the index.
* Shared lemmas that more than one theorem file needs are stated here: weakening, the closure lemma
  (6.1) and its predicate form, and `erasure_core`, which is theorem 7's statement.
  Theorem 5 (Coherence.lean) uses theorem 7 in its COERCE case and the theorem files do not import each
  other, so the statement lives here; Erasure.lean's `erasure` is its instance.

Decisions taken by the integration step, on the provers' requests:
* The denotation `[[.]]` is the computable `denote` with a fuel, and the notions built on it (`Dn`,
  `DnP`, `DnEquiv`, `DenL`, `ServedByL`, `SelSoundL`, `SelCompleteL`, `Sig.CoherentL`) are stated at
  the limit of the fuel: some fuel reaches the answer, and then every larger fuel does (`denote_mono`).
  This is the fixed point SPEC.md 6.3 describes; the stage's bound of 128 is a stage fact (F16). At one
  fixed fuel the denotation is not a congruence (Erasure.lean's `fixed_fuel_erasure_fails`), so the
  skeleton's fuel-indexed `Den`, `ServedBy`, `SelSound` and `Coherent` were transcription artifacts.
* `SelCompleteL` is the clause H4 lacks: the oracle answers every served ground application, as
  `resolveImplFor` does (D/ImplResolution.cpp:172-401) or refuses the program. Without it an oracle
  that declines a served application satisfies `SelSoundL` and refutes theorem 7
  (Erasure.lean's `erasure_needs_sel_complete`).
* `erasure_core` takes a primitive derivation whose (BIND)s `[[.]]` resolves (`ValidGround`) and a
  constructor law that forgets no argument (`ConLaw.KeepsArgs`), and concludes two-sidedly
  (`DnEquiv`). With (REQ) admitted the statement is false: H3 discharges a requirement in the full
  signature, through impls whose headers need not denote (Erasure.lean's `erasure_needs_primitive`).
  `Sig.DeclWF` and `Sig.ReqDuty` are not needed by the primitive proof.
-/
namespace Trait

/-! ## The ground congruence closure (SPEC.md 5.2; `entailedByGroundCongruence`, D/EqualityVerification.cpp:98-276) -/

/-- Ground equational consequence of the seeds `E`: (SEED), (EQUIV), (CONG) on constructors and
projections, and (NORM) through the constructor law. No decomposition, no instantiation, no binding
read. -/
inductive Entails (L : ConLaw) (E : List (Ty × Ty)) : Ty -> Ty -> Prop where
  | ax    : (a, b) ∈ E -> Entails L E a b
  | refl  : Entails L E a a
  | symm  : Entails L E a b -> Entails L E b a
  | trans : Entails L E a b -> Entails L E b c -> Entails L E a c
  | con   : as.length = bs.length ->
            (∀ i, i < as.length -> Entails L E (as.getD i default) (bs.getD i default)) ->
            Entails L E (.con c as) (.con c bs)
  | proj  : as.length = bs.length -> gs.length = hs.length ->
            (∀ i, i < as.length -> Entails L E (as.getD i default) (bs.getD i default)) ->
            (∀ i, i < gs.length -> Entails L E (gs.getD i default) (hs.getD i default)) ->
            Entails L E (.proj t as a gs) (.proj t bs a hs)
  | law   : L.law a b -> Entails L E a b

/-- The closure on predicates, modulo citation (D/EqualityVerification.cpp:30-43): an application
claim is a node over its arguments, an equality claim a node over its two endpoints. -/
inductive PredEntails (L : ConLaw) (E : List (Ty × Ty)) : Pred -> Pred -> Prop where
  | app : as.length = bs.length ->
          (∀ i, i < as.length -> Entails L E (as.getD i default) (bs.getD i default)) ->
          PredEntails L E (.app t as) (.app t bs)
  | eq  : Entails L E a c -> Entails L E b d -> PredEntails L E (.eq a b) (.eq c d)

/-! ## Derivations (SPEC.md 4.1) -/

/-- Derivations `S; G |- P`, Type-valued so that `elab` recurses on them. `Ok i σ` restricts the impl
citations a derivation may make (H4: `DenL` in theorems 5 and 7, `AnyImpl` elsewhere). Rules in the order
of SPEC.md 4.1: (HYP) `hyp`, (IMPL) `impl`, (REQ) `req`, (REQ-A) `reqA`, (BIND) `bind`, (REFL) `refl`,
(SYM) `symm`, (TRANS) `trans`, (CONG) `cong`, (CONG-P) `congP`, (NORM) `norm`, (CONV) `conv`. Premises
given by position are functions of a position below a bound, read with `getD`. -/
inductive Deriv (L : ConLaw) (S : Sig) (Ok : ImplId -> Subst -> Prop) (Γ : List Pred) : Pred -> Type where
  | hyp   : (k : Nat) -> (hk : k < Γ.length) -> Deriv L S Ok Γ Γ[k]
  | impl  : (i : ImplId) -> (hi : i < S.impls.length) -> (σ : List Ty) -> σ.length = S.impls[i].arity ->
            Ok i (Subst.ofList σ) ->
            (∀ j, j < S.impls[i].wheres.length ->
              Deriv L S Ok Γ ((S.impls[i].wheres.getD j default).subst (Subst.ofList σ))) ->
            Deriv L S Ok Γ (S.impls[i].header (Subst.ofList σ))
  | req   : Deriv L S Ok Γ (.app t as) -> (ht : t < S.traits.length) -> (hk : k < S.traits[t].reqs.length) ->
            Deriv L S Ok Γ ((S.traits[t].reqs[k]).subst (Subst.ofList as))
  | reqA  : Deriv L S Ok Γ (.app t as) -> (ht : t < S.traits.length) -> (hb : b < S.traits[t].binders.length) ->
            (us : List Ty) -> us.length = (S.traits[t].binders[b]).labels.length ->
            (∀ j, j < (S.traits[t].binders[b]).premises.length ->
              Deriv L S Ok Γ (((S.traits[t].binders[b]).premises.getD j default).subst
                ((S.traits[t].binders[b]).callSubst as us))) ->
            Deriv L S Ok Γ ((S.traits[t].binders[b]).concl.subst ((S.traits[t].binders[b]).callSubst as us))
  | bind  : (i : ImplId) -> (hi : i < S.impls.length) -> (σ : List Ty) -> σ.length = S.impls[i].arity ->
            Ok i (Subst.ofList σ) ->
            (∀ j, j < S.impls[i].wheres.length ->
              Deriv L S Ok Γ ((S.impls[i].wheres.getD j default).subst (Subst.ofList σ))) ->
            S.impls[i].bindings[a]? = some (some (labels, B)) -> gs.length = labels.length ->
            Deriv L S Ok Γ (.eq (.proj S.impls[i].trait (substList (Subst.ofList σ) S.impls[i].head) a gs)
                                (B.subst ((Subst.ofList σ).extend labels gs)))
  | refl  : Deriv L S Ok Γ (.eq a a)
  | symm  : Deriv L S Ok Γ (.eq a b) -> Deriv L S Ok Γ (.eq b a)
  | trans : Deriv L S Ok Γ (.eq a b) -> Deriv L S Ok Γ (.eq b c) -> Deriv L S Ok Γ (.eq a c)
  | cong  : as.length = bs.length ->
            (∀ i, i < as.length -> Deriv L S Ok Γ (.eq (as.getD i default) (bs.getD i default))) ->
            Deriv L S Ok Γ (.eq (.con c as) (.con c bs))
  | congP : as.length = bs.length -> gs.length = hs.length ->
            (∀ i, i < as.length -> Deriv L S Ok Γ (.eq (as.getD i default) (bs.getD i default))) ->
            (∀ i, i < gs.length -> Deriv L S Ok Γ (.eq (gs.getD i default) (hs.getD i default))) ->
            Deriv L S Ok Γ (.eq (.proj t as a gs) (.proj t bs a hs))
  | norm  : L.law a b -> Deriv L S Ok Γ (.eq a b)
  | conv  : Deriv L S Ok Γ (.app t as) -> as.length = bs.length ->
            (∀ i, i < as.length -> Deriv L S Ok Γ (.eq (as.getD i default) (bs.getD i default))) ->
            Deriv L S Ok Γ (.app t bs)

mutual
  /-- The derivation is an `Env(G)` member (chalk's `FromEnv`; SPEC.md 4.3): a hypothesis, or a plain or
  binder requirement read off an `Env(G)` member with primitive premises. -/
  def Deriv.InEnv : Deriv L S Ok Γ P -> Prop
    | .hyp _ _ => True
    | .req d _ _ => d.InEnv
    | .reqA d _ _ _ _ prem => d.InEnv ∧ ∀ j hj, (prem j hj).Primitive
    | _ => False
  /-- The primitive fragment `|-p` (SPEC.md 4.3): (REQ) and (REQ-A) read requirements off `Env(G)`
  members only. -/
  def Deriv.Primitive : Deriv L S Ok Γ P -> Prop
    | .hyp _ _ => True
    | .impl _ _ _ _ _ ds => ∀ j hj, (ds j hj).Primitive
    | .req d _ _ => d.InEnv
    | .reqA d _ _ _ _ prem => d.InEnv ∧ ∀ j hj, (prem j hj).Primitive
    | .bind _ _ _ _ _ ds _ _ => ∀ j hj, (ds j hj).Primitive
    | .refl => True
    | .symm d => d.Primitive
    | .trans d₁ d₂ => d₁.Primitive ∧ d₂.Primitive
    | .cong _ ds => ∀ j hj, (ds j hj).Primitive
    | .congP _ _ ds es => (∀ j hj, (ds j hj).Primitive) ∧ (∀ j hj, (es j hj).Primitive)
    | .norm _ => True
    | .conv d _ ds => d.Primitive ∧ ∀ j hj, (ds j hj).Primitive
end

/-- No restriction on impl citations. -/
abbrev AnyImpl : ImplId -> Subst -> Prop := fun _ _ => True

/-- `S; G |- P` in the full logic. -/
abbrev Valid (L : ConLaw) (S : Sig) (Γ : List Pred) (P : Pred) : Prop := Nonempty (Deriv L S AnyImpl Γ P)

/-- `S; G |-p P`: derivable in the primitive fragment. -/
def ValidP (L : ConLaw) (S : Sig) (Γ : List Pred) (P : Pred) : Prop :=
  ∃ d : Deriv L S AnyImpl Γ P, d.Primitive

/-- H3 (ReqDuty, SPEC.md 4.3; F13): every impl's plain requirements are derivable in the primitive
fragment from its where entries, and every binder's conclusion from the where entries and the binder's
premises, the binder's variables fresh (`ImplDecl.binderSubst`). -/
def Sig.ReqDuty (L : ConLaw) (S : Sig) : Prop :=
  (∀ i (hi : i < S.impls.length) (ht : S.impls[i].trait < S.traits.length)
      k (hk : k < (S.traits[S.impls[i].trait]).reqs.length),
    ValidP L S S.impls[i].wheres
      (((S.traits[S.impls[i].trait]).reqs[k]).subst (Subst.ofList S.impls[i].head))) ∧
  (∀ i (hi : i < S.impls.length) (ht : S.impls[i].trait < S.traits.length)
      b (hb : b < (S.traits[S.impls[i].trait]).binders.length),
    ValidP L S (S.impls[i].binderHyps ((S.traits[S.impls[i].trait]).binders[b]))
      (((S.traits[S.impls[i].trait]).binders[b]).concl.subst
        (S.impls[i].binderSubst ((S.traits[S.impls[i].trait]).binders[b]))))

/-! ## Lemmas of the logic -/

theorem Deriv.primitive_of_inEnv (d : Deriv L S Ok Γ P) (h : d.InEnv) : d.Primitive := by
  cases d <;> simp_all [Deriv.InEnv, Deriv.Primitive]

/-- A derivation under a restriction on impl citations is one under none. -/
def Deriv.relax : Deriv L S Ok Γ P -> Deriv L S AnyImpl Γ P
  | .hyp k hk => .hyp k hk
  | .impl i hi σ hσ _ ds => .impl i hi σ hσ trivial (fun j hj => (ds j hj).relax)
  | .req d ht hk => .req d.relax ht hk
  | .reqA d ht hb us hus prem => .reqA d.relax ht hb us hus (fun j hj => (prem j hj).relax)
  | .bind i hi σ hσ _ ds hb hg => .bind i hi σ hσ trivial (fun j hj => (ds j hj).relax) hb hg
  | .refl => .refl
  | .symm d => .symm d.relax
  | .trans d₁ d₂ => .trans d₁.relax d₂.relax
  | .cong h ds => .cong h (fun j hj => (ds j hj).relax)
  | .congP h₁ h₂ ds es => .congP h₁ h₂ (fun j hj => (ds j hj).relax) (fun j hj => (es j hj).relax)
  | .norm h => .norm h
  | .conv d h ds => .conv d.relax h (fun j hj => (ds j hj).relax)

theorem valid_relax (h : Nonempty (Deriv L S Ok Γ P)) : Valid L S Γ P := h.elim fun d => ⟨d.relax⟩

/-- A derivation under no hypotheses is one under any: no rule reads `Γ` but (HYP), which has no
instance at `[]`. -/
def Deriv.weakenNil : Deriv L S Ok [] P -> Deriv L S Ok Γ P
  | .hyp k hk => absurd hk (Nat.not_lt_zero k)
  | .impl i hi σ hσ ok ds => .impl i hi σ hσ ok (fun j hj => (ds j hj).weakenNil)
  | .req d ht hk => .req d.weakenNil ht hk
  | .reqA d ht hb us hus prem => .reqA d.weakenNil ht hb us hus (fun j hj => (prem j hj).weakenNil)
  | .bind i hi σ hσ ok ds hb hg => .bind i hi σ hσ ok (fun j hj => (ds j hj).weakenNil) hb hg
  | .refl => .refl
  | .symm d => .symm d.weakenNil
  | .trans d₁ d₂ => .trans d₁.weakenNil d₂.weakenNil
  | .cong h ds => .cong h (fun j hj => (ds j hj).weakenNil)
  | .congP h₁ h₂ ds es => .congP h₁ h₂ (fun j hj => (ds j hj).weakenNil) (fun j hj => (es j hj).weakenNil)
  | .norm h => .norm h
  | .conv d h ds => .conv d.weakenNil h (fun j hj => (ds j hj).weakenNil)

/-- Weakening: a derivation under no hypotheses is one under any. -/
theorem valid_weaken (L : ConLaw) (S : Sig) (Ok : ImplId -> Subst -> Prop) (Γ : List Pred) (P : Pred)
    (h : Nonempty (Deriv L S Ok [] P)) : Nonempty (Deriv L S Ok Γ P) := h.elim fun d => ⟨d.weakenNil⟩

/-- The closure lemma (SPEC.md 6.1): valid seeds and a closure fact give a valid equality. (SEED) is the
premises, (EQUIV) is (REFL)(SYM)(TRANS), (CONG) is (CONG)/(CONG-P), (NORM) is H5. -/
theorem entails_valid (L : ConLaw) (S : Sig) (Ok : ImplId -> Subst -> Prop) (Γ : List Pred)
    (E : List (Ty × Ty)) (hE : ∀ l r, (l, r) ∈ E -> Nonempty (Deriv L S Ok Γ (.eq l r)))
    (h : Entails L E a b) : Nonempty (Deriv L S Ok Γ (.eq a b)) := by
  induction h with
  | ax h => exact hE _ _ h
  | refl => exact ⟨.refl⟩
  | symm _ ih => exact ih.elim fun d => ⟨.symm d⟩
  | trans _ _ ih₁ ih₂ => exact ih₁.elim fun d₁ => ih₂.elim fun d₂ => ⟨.trans d₁ d₂⟩
  | con hlen _ ih => exact ⟨.cong hlen fun i hi => Classical.choice (ih i hi)⟩
  | proj hlen hlen' _ _ ih₁ ih₂ =>
    exact ⟨.congP hlen hlen' (fun i hi => Classical.choice (ih₁ i hi))
      (fun i hi => Classical.choice (ih₂ i hi))⟩
  | law h => exact ⟨.norm h⟩

/-- The closure lemma on predicates: an application moves by (CONV), an equality by (SYM)(TRANS). -/
theorem predEntails_valid (L : ConLaw) (S : Sig) (Ok : ImplId -> Subst -> Prop) (Γ : List Pred)
    (E : List (Ty × Ty)) (hE : ∀ l r, (l, r) ∈ E -> Nonempty (Deriv L S Ok Γ (.eq l r)))
    (h : PredEntails L E P Q) (hP : Nonempty (Deriv L S Ok Γ P)) : Nonempty (Deriv L S Ok Γ Q) := by
  cases h with
  | app hlen hall =>
    exact hP.elim fun d =>
      ⟨.conv d hlen fun i hi => Classical.choice (entails_valid L S Ok Γ E hE (hall i hi))⟩
  | eq hac hbd =>
    exact hP.elim fun d => (entails_valid L S Ok Γ E hE hac).elim fun dac =>
      (entails_valid L S Ok Γ E hE hbd).elim fun dbd => ⟨.trans (.symm dac) (.trans d dbd)⟩

-- (REQ-ADMISSIBLE) as SPEC.md 4.3 states it is false: `Sig.ReqDuty` lets two impls' duties read each other's
-- application with no base. ReqAdmissible.lean states it (`ReqAdmissible`) and refutes it (`req_admissible_fails`).

/-- Under no hypotheses nothing is an `Env(G)` member: (REQ) and (REQ-A) have no primitive instance at
`G = []`. -/
theorem Deriv.not_inEnv_nil {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} :
    ∀ {P : Pred} (d : Deriv L S Ok [] P), ¬ d.InEnv := by
  intro P d
  induction d
  case hyp k hk => exact absurd hk (by simp)
  all_goals simp_all [Deriv.InEnv]

/-! ## Denotation through the selection oracle (SPEC.md 6.3; `resolveImplFor`, D/ImplResolution.cpp:172-401) -/

/-- The stage's selection, as an oracle: for a ground application, the impl and arguments it serves it
with. Its soundness, `SelSoundL`, and its completeness, `SelCompleteL`, are hypotheses. -/
abbrev Sel := Pred -> Option (ImplId × List Ty)

/-- `[[t]]`: each ground projection resolved through the impl selection serves its application with,
to a fixed point, the constructor laws applied; partial where no impl serves, no binding is declared,
or the fuel (the stage's bound of 128, D/TraitTypes.cpp:117-129) is spent. A variable denotes itself; a
projection with a variable in it stays a projection over its denoted children. -/
def denote (sel : Sel) (L : ConLaw) (S : Sig) : Nat -> Ty -> Option Ty
  | 0, _ => none
  | _+1, .var i => some (.var i)
  | fuel+1, .con c as => do
      let bs ← as.mapM (denote sel L S fuel)
      let r := L.rebuild c bs
      if r = .con c bs then pure r else denote sel L S fuel r
  | fuel+1, .proj t as a gs => do
      let as' ← as.mapM (denote sel L S fuel)
      let gs' ← gs.mapM (denote sel L S fuel)
      if groundList as' && groundList gs' then
        match sel (.app t as') with
        | none => none
        | some (i, σ) =>
          match S.impls[i]? with
          | none => none
          | some I =>
            match I.bindings[a]? with
            | some (some (labels, B)) => denote sel L S fuel (B.subst ((Subst.ofList σ).extend labels gs'))
            | _ => none
      else pure (.proj t as' a gs')

/-- `[[P]]`: a predicate with its types denoted. -/
def denotePred (sel : Sel) (L : ConLaw) (S : Sig) (fuel : Nat) : Pred -> Option Pred
  | .app t as => (as.mapM (denote sel L S fuel)).map (.app t)
  | .eq a b => do pure (.eq (← denote sel L S fuel a) (← denote sel L S fuel b))

/-! ## Lists related elementwise -/

/-- Two lists of one length whose elements are related in order. -/
inductive All2 (R : α -> β -> Prop) : List α -> List β -> Prop where
  | nil  : All2 R [] []
  | cons : R a b -> All2 R as bs -> All2 R (a :: as) (b :: bs)

theorem All2.length_eq {R : α -> β -> Prop} {as : List α} {bs : List β} (h : All2 R as bs) :
    as.length = bs.length := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]

theorem All2.getElem? {R : α -> β -> Prop} {as : List α} {bs : List β} (h : All2 R as bs) (i : Nat) :
    (∀ a b, as[i]? = some a -> bs[i]? = some b -> R a b) ∧ (as[i]? = none ↔ bs[i]? = none) := by
  induction h generalizing i with
  | nil => simp
  | cons hab _ ih =>
    cases i with
    | zero =>
      refine ⟨fun a b ha hb => ?_, by simp⟩
      simp at ha hb; subst ha; subst hb; exact hab
    | succ i => simpa using ih i

theorem All2.of_getD {R : α -> β -> Prop} [Inhabited α] [Inhabited β] :
    ∀ {as : List α} {bs : List β}, as.length = bs.length ->
      (∀ i, i < as.length -> R (as.getD i default) (bs.getD i default)) -> All2 R as bs
  | [], [], _, _ => .nil
  | a :: as, b :: bs, hl, h => by
    refine .cons (h 0 (by simp)) (All2.of_getD (by simpa using hl) ?_)
    intro i hi
    simpa using h (i+1) (by simpa using hi)

theorem All2.imp {R Q : α -> β -> Prop} (hRQ : ∀ a b, R a b -> Q a b) {as : List α} {bs : List β}
    (h : All2 R as bs) : All2 Q as bs := by
  induction h with
  | nil => exact .nil
  | cons hab _ ih => exact .cons (hRQ _ _ hab) ih

/-- When every related pair's right element satisfies `Q` with itself, the right list is `Q`-related
to itself. -/
theorem All2.right_self {R : α -> β -> Prop} {Q : β -> β -> Prop} (hRQ : ∀ a b, R a b -> Q b b)
    {as : List α} {bs : List β} (h : All2 R as bs) : All2 Q bs bs := by
  induction h with
  | nil => exact .nil
  | cons hab _ ih => exact .cons (hRQ _ _ hab) ih

theorem All2.flip {R : α -> β -> Prop} {as : List α} {bs : List β} (h : All2 R as bs) :
    All2 (fun b a => R a b) bs as := by
  induction h with
  | nil => exact .nil
  | cons hab _ ih => exact .cons hab ih

/-- `mapM` into `Option` succeeds exactly when it succeeds elementwise. -/
theorem mapM_eq_some_iff {g : α -> Option β} :
    ∀ {as : List α} {bs : List β}, as.mapM g = some bs ↔ All2 (fun a b => g a = some b) as bs
  | [], bs => by
    simp only [List.mapM_nil, pure]
    constructor
    · intro h; cases h; exact .nil
    · intro h; cases h; rfl
  | a :: as, bs => by
    simp only [List.mapM_cons, bind, Option.bind_eq_some_iff, pure]
    constructor
    · rintro ⟨b, hb, bs', hbs', hpure⟩
      cases hpure
      exact .cons hb (mapM_eq_some_iff.mp hbs')
    · intro h
      cases h with
      | cons hb hbs => exact ⟨_, hb, _, mapM_eq_some_iff.mpr hbs, rfl⟩

theorem mapM_mono {g g' : α -> Option β} (h : ∀ a b, g a = some b -> g' a = some b) {as : List α}
    {bs : List β} (hm : as.mapM g = some bs) : as.mapM g' = some bs :=
  mapM_eq_some_iff.mpr ((mapM_eq_some_iff.mp hm).imp h)


/-! ## Groundness, as the oracle reads it -/

theorem groundList_eq_all : ∀ as : List Ty, groundList as = as.all Ty.ground
  | [] => rfl
  | a :: as => by simp [groundList, groundList_eq_all as]

/-! ## The denotation is monotone in the fuel and idempotent -/

theorem denote_zero (sel : Sel) (L : ConLaw) (S : Sig) (t : Ty) : denote sel L S 0 t = none := by
  cases t <;> rfl

/-- The unfolding of `denote` at a constructor. -/
theorem denote_con (sel : Sel) (L : ConLaw) (S : Sig) (f : Nat) (c : ConId) (as : List Ty) :
    denote sel L S (f+1) (.con c as) = (as.mapM (denote sel L S f)).bind fun bs =>
      if L.rebuild c bs = .con c bs then some (L.rebuild c bs) else denote sel L S f (L.rebuild c bs) := rfl

/-- The unfolding of `denote` at a projection. -/
theorem denote_proj (sel : Sel) (L : ConLaw) (S : Sig) (f : Nat) (t : TraitId) (as : List Ty) (a : AssocId)
    (gs : List Ty) :
    denote sel L S (f+1) (.proj t as a gs) = (as.mapM (denote sel L S f)).bind fun as' =>
      (gs.mapM (denote sel L S f)).bind fun gs' =>
        if groundList as' && groundList gs' then
          match sel (.app t as') with
          | none => none
          | some (i, σ) =>
            match S.impls[i]? with
            | none => none
            | some I =>
              match I.bindings[a]? with
              | some (some (labels, B)) => denote sel L S f (B.subst ((Subst.ofList σ).extend labels gs'))
              | _ => none
        else some (.proj t as' a gs') := rfl

theorem denote_var (sel : Sel) (L : ConLaw) (S : Sig) (f : Nat) (i : Var) :
    denote sel L S (f+1) (.var i) = some (.var i) := rfl

/-- More fuel never changes an answer. -/
theorem denote_mono (sel : Sel) (L : ConLaw) (S : Sig) :
    ∀ (f : Nat) (t d : Ty), denote sel L S f t = some d -> denote sel L S (f+1) t = some d
  | 0, t, d, h => by rw [denote_zero] at h; cases h
  | f+1, .var i, d, h => by rw [denote_var] at h ⊢; exact h
  | f+1, .con c as, d, h => by
    rw [denote_con] at h ⊢
    obtain ⟨bs, hbs, hd⟩ := Option.bind_eq_some_iff.mp h
    refine Option.bind_eq_some_iff.mpr ⟨bs, mapM_mono (denote_mono sel L S f) hbs, ?_⟩
    split at hd <;> rename_i hr
    · rw [ite_eq_left hr]; exact hd
    · rw [ite_eq_right hr]; exact denote_mono sel L S f _ d hd
  | f+1, .proj t as a gs, d, h => by
    rw [denote_proj] at h ⊢
    obtain ⟨as', has, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨gs', hgs, h⟩ := Option.bind_eq_some_iff.mp h
    refine Option.bind_eq_some_iff.mpr ⟨as', mapM_mono (denote_mono sel L S f) has, ?_⟩
    refine Option.bind_eq_some_iff.mpr ⟨gs', mapM_mono (denote_mono sel L S f) hgs, ?_⟩
    split at h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · simp_all [denote_mono sel L S f]
          · cases h
    · simp_all

theorem denote_mono_le (sel : Sel) (L : ConLaw) (S : Sig) {f g : Nat} (hfg : f ≤ g) {t d : Ty}
    (h : denote sel L S f t = some d) : denote sel L S g t = some d := by
  induction hfg with
  | refl => exact h
  | step _ ih => exact denote_mono sel L S _ t d ih

/-- Answers of a pointwise idempotent function denote to themselves. -/
theorem mapM_self_of_pointwise {g : α -> Option α} (hg : ∀ a b, g a = some b -> g b = some b)
    {as bs : List α} (h : as.mapM g = some bs) : bs.mapM g = some bs := by
  rw [mapM_eq_some_iff] at h ⊢
  induction h with
  | nil => exact .nil
  | cons hab _ ih => exact .cons (hg _ _ hab) ih

/-- An answer denotes to itself. -/
theorem denote_idem (sel : Sel) (L : ConLaw) (S : Sig) :
    ∀ (f : Nat) (t d : Ty), denote sel L S f t = some d -> denote sel L S f d = some d
  | 0, t, d, h => by rw [denote_zero] at h; cases h
  | f+1, .var i, d, h => by rw [denote_var] at h; cases h; rfl
  | f+1, .con c as, d, h => by
    rw [denote_con] at h
    obtain ⟨bs, hbs, hd⟩ := Option.bind_eq_some_iff.mp h
    split at hd
    · rename_i hr
      rw [hr] at hd; cases hd
      rw [denote_con]
      refine Option.bind_eq_some_iff.mpr ⟨bs, mapM_self_of_pointwise (denote_idem sel L S f) hbs, ?_⟩
      simp [hr]
    · exact denote_mono sel L S f d d (denote_idem sel L S f _ d hd)
  | f+1, .proj t as a gs, d, h => by
    rw [denote_proj] at h
    obtain ⟨as', has, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨gs', hgs, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · split at h
      · cases h
      · split at h
        · cases h
        · split at h
          · exact denote_mono sel L S f d d (denote_idem sel L S f _ d h)
          · cases h
    · cases h
      rw [denote_proj]
      refine Option.bind_eq_some_iff.mpr ⟨as', mapM_self_of_pointwise (denote_idem sel L S f) has, ?_⟩
      refine Option.bind_eq_some_iff.mpr ⟨gs', mapM_self_of_pointwise (denote_idem sel L S f) hgs, ?_⟩
      simp_all

/-! ## The denotation at the limit of the fuel -/

/-- `[[t]] = d`: some fuel reaches `d`, and then (`denote_mono`) every larger fuel does. This is the
fixed point SPEC.md 6.3 describes; the stage's bound of 128 is a stage fact (F16). -/
def Dn (sel : Sel) (L : ConLaw) (S : Sig) (t d : Ty) : Prop := ∃ f, denote sel L S f t = some d

/-- `[[P]] = Q` at the limit. -/
def DnP (sel : Sel) (L : ConLaw) (S : Sig) (P Q : Pred) : Prop := ∃ f, denotePred sel L S f P = some Q

/-- Two types denote alike: to one type, or neither to any. -/
def DnEquiv (sel : Sel) (L : ConLaw) (S : Sig) (a b : Ty) : Prop := ∀ d, Dn sel L S a d ↔ Dn sel L S b d

section Limit
variable {sel : Sel} {L : ConLaw} {S : Sig}

theorem Dn_fun {t d d' : Ty} (h : Dn sel L S t d) (h' : Dn sel L S t d') : d = d' := by
  obtain ⟨f, hf⟩ := h
  obtain ⟨g, hg⟩ := h'
  have h1 := denote_mono_le sel L S (Nat.le_max_left f g) hf
  have h2 := denote_mono_le sel L S (Nat.le_max_right f g) hg
  rw [h1] at h2
  cases h2
  rfl

theorem Dn_idem {t d : Ty} (h : Dn sel L S t d) : Dn sel L S d d :=
  let ⟨f, hf⟩ := h; ⟨f, denote_idem sel L S f t d hf⟩

theorem Dn_var_iff {i : Var} {d : Ty} : Dn sel L S (.var i) d ↔ d = .var i := by
  constructor
  · rintro ⟨f, hf⟩
    cases f with
    | zero => rw [denote_zero] at hf; cases hf
    | succ f => rw [denote_var] at hf; cases hf; rfl
  · rintro rfl; exact ⟨1, rfl⟩

theorem DnEquiv.refl (a : Ty) : DnEquiv sel L S a a := fun _ => Iff.rfl
theorem DnEquiv.symm {a b : Ty} (h : DnEquiv sel L S a b) : DnEquiv sel L S b a := fun d => (h d).symm
theorem DnEquiv.trans {a b c : Ty} (h₁ : DnEquiv sel L S a b) (h₂ : DnEquiv sel L S b c) :
    DnEquiv sel L S a c := fun d => (h₁ d).trans (h₂ d)

/-- A type denotes alike with its answer. -/
theorem DnEquiv_of_Dn {a b : Ty} (h : Dn sel L S a b) : DnEquiv sel L S a b := by
  intro d
  constructor
  · intro hd; rw [← Dn_fun h hd]; exact Dn_idem h
  · intro hd; rw [← Dn_fun (Dn_idem h) hd]; exact h

/-- Elementwise answers at some fuel each share one fuel. -/
theorem All2_Dn_mapM {as bs : List Ty} (h : All2 (Dn sel L S) as bs) :
    ∃ f, as.mapM (denote sel L S f) = some bs := by
  induction h with
  | nil => exact ⟨0, rfl⟩
  | cons hab _ ih =>
    obtain ⟨f, hf⟩ := hab
    obtain ⟨g, hg⟩ := ih
    refine ⟨max f g, ?_⟩
    rw [mapM_eq_some_iff] at hg ⊢
    exact .cons (denote_mono_le sel L S (Nat.le_max_left f g) hf)
      (hg.imp fun a b hab => denote_mono_le sel L S (Nat.le_max_right f g) hab)

theorem All2_Dn_of_mapM {f : Nat} {as bs : List Ty} (h : as.mapM (denote sel L S f) = some bs) :
    All2 (Dn sel L S) as bs :=
  (mapM_eq_some_iff.mp h).imp fun _ _ hab => ⟨f, hab⟩

/-- `[[K(as)]]` at the limit: the children denote, then the constructor law decides. -/
theorem Dn_con_iff {c : ConId} {as : List Ty} {d : Ty} :
    Dn sel L S (.con c as) d ↔ ∃ bs, All2 (Dn sel L S) as bs ∧
      ((L.rebuild c bs = .con c bs ∧ d = .con c bs) ∨
       (L.rebuild c bs ≠ .con c bs ∧ Dn sel L S (L.rebuild c bs) d)) := by
  constructor
  · rintro ⟨f, hf⟩
    cases f with
    | zero => rw [denote_zero] at hf; cases hf
    | succ f =>
      rw [denote_con] at hf
      obtain ⟨bs, hbs, hd⟩ := Option.bind_eq_some_iff.mp hf
      refine ⟨bs, All2_Dn_of_mapM hbs, ?_⟩
      split at hd <;> rename_i hr
      · left; cases hd; exact ⟨hr, hr⟩
      · right; exact ⟨hr, f, hd⟩
  · rintro ⟨bs, hbs, hd⟩
    obtain ⟨f, hf⟩ := All2_Dn_mapM hbs
    rcases hd with ⟨hr, rfl⟩ | ⟨hr, g, hg⟩
    · refine ⟨f+1, ?_⟩
      rw [denote_con]
      exact Option.bind_eq_some_iff.mpr ⟨bs, hf, by simp [hr]⟩
    · refine ⟨max f g + 1, ?_⟩
      rw [denote_con]
      refine Option.bind_eq_some_iff.mpr ⟨bs, mapM_mono (fun a b => denote_mono_le sel L S (Nat.le_max_left f g)) hf, ?_⟩
      rw [ite_eq_right hr]
      exact denote_mono_le sel L S (Nat.le_max_right f g) hg

/-- `[[T[as]::N<gs>]]` at the limit: the children denote; a ground application is resolved through the
impl the oracle serves it with, any other stands. -/
theorem Dn_proj_iff {t : TraitId} {as : List Ty} {a : AssocId} {gs : List Ty} {d : Ty} :
    Dn sel L S (.proj t as a gs) d ↔ ∃ as' gs', All2 (Dn sel L S) as as' ∧ All2 (Dn sel L S) gs gs' ∧
      (((groundList as' && groundList gs') = true ∧
        ∃ i σ I labels B, sel (.app t as') = some (i, σ) ∧ S.impls[i]? = some I ∧
          I.bindings[a]? = some (some (labels, B)) ∧
          Dn sel L S (B.subst ((Subst.ofList σ).extend labels gs')) d) ∨
       ((groundList as' && groundList gs') = false ∧ d = .proj t as' a gs')) := by
  constructor
  · rintro ⟨f, hf⟩
    cases f with
    | zero => rw [denote_zero] at hf; cases hf
    | succ f =>
      rw [denote_proj] at hf
      obtain ⟨as', has, hf⟩ := Option.bind_eq_some_iff.mp hf
      obtain ⟨gs', hgs, hf⟩ := Option.bind_eq_some_iff.mp hf
      refine ⟨as', gs', All2_Dn_of_mapM has, All2_Dn_of_mapM hgs, ?_⟩
      split at hf <;> rename_i hg
      · left
        refine ⟨hg, ?_⟩
        split at hf
        · cases hf
        · rename_i i σ hsel
          split at hf
          · cases hf
          · rename_i I hI
            split at hf
            · rename_i labels B hb
              exact ⟨i, σ, I, labels, B, hsel, hI, hb, f, hf⟩
            · cases hf
      · right
        cases hf
        exact ⟨Bool.eq_false_iff.mpr hg, rfl⟩
  · rintro ⟨as', gs', has, hgs, h⟩
    obtain ⟨f₁, hf₁⟩ := All2_Dn_mapM has
    obtain ⟨f₂, hf₂⟩ := All2_Dn_mapM hgs
    rcases h with ⟨hg, i, σ, I, labels, B, hsel, hI, hb, f₃, hf₃⟩ | ⟨hg, rfl⟩
    · refine ⟨max (max f₁ f₂) f₃ + 1, ?_⟩
      rw [denote_proj]
      refine Option.bind_eq_some_iff.mpr ⟨as', mapM_mono (fun a b => denote_mono_le sel L S
        (Nat.le_trans (Nat.le_max_left f₁ f₂) (Nat.le_max_left _ f₃))) hf₁, ?_⟩
      refine Option.bind_eq_some_iff.mpr ⟨gs', mapM_mono (fun a b => denote_mono_le sel L S
        (Nat.le_trans (Nat.le_max_right f₁ f₂) (Nat.le_max_left _ f₃))) hf₂, ?_⟩
      simp only [hg, hsel, hI, hb, ite_true]
      exact denote_mono_le sel L S (Nat.le_max_right _ f₃) hf₃
    · refine ⟨max f₁ f₂ + 1, ?_⟩
      rw [denote_proj]
      refine Option.bind_eq_some_iff.mpr ⟨as', mapM_mono (fun a b => denote_mono_le sel L S (Nat.le_max_left f₁ f₂)) hf₁, ?_⟩
      refine Option.bind_eq_some_iff.mpr ⟨gs', mapM_mono (fun a b => denote_mono_le sel L S (Nat.le_max_right f₁ f₂)) hf₂, ?_⟩
      simp [hg]

/-- Lists that denote alike elementwise share their answers. -/
theorem All2_Dn_congr {as bs : List Ty} (h : All2 (DnEquiv sel L S) as bs) :
    ∀ cs, All2 (Dn sel L S) as cs ↔ All2 (Dn sel L S) bs cs := by
  induction h with
  | nil => intro cs; exact Iff.rfl
  | cons hab _ ih =>
    intro cs
    constructor
    · intro h; cases h with | cons hac hcs => exact .cons ((hab _).mp hac) ((ih _).mp hcs)
    · intro h; cases h with | cons hbc hcs => exact .cons ((hab _).mpr hbc) ((ih _).mpr hcs)

/-- `[[.]]` is a congruence at a constructor. -/
theorem DnEquiv_con {c : ConId} {as bs : List Ty} (h : All2 (DnEquiv sel L S) as bs) :
    DnEquiv sel L S (.con c as) (.con c bs) := by
  intro d
  rw [Dn_con_iff, Dn_con_iff]
  constructor
  · rintro ⟨cs, hcs, hd⟩; exact ⟨cs, (All2_Dn_congr h cs).mp hcs, hd⟩
  · rintro ⟨cs, hcs, hd⟩; exact ⟨cs, (All2_Dn_congr h cs).mpr hcs, hd⟩

/-- `[[.]]` is a congruence at a projection. -/
theorem DnEquiv_proj {t : TraitId} {as bs : List Ty} {a : AssocId} {gs hs : List Ty}
    (h₁ : All2 (DnEquiv sel L S) as bs) (h₂ : All2 (DnEquiv sel L S) gs hs) :
    DnEquiv sel L S (.proj t as a gs) (.proj t bs a hs) := by
  intro d
  rw [Dn_proj_iff, Dn_proj_iff]
  constructor
  · rintro ⟨as', gs', has, hgs, hd⟩
    exact ⟨as', gs', (All2_Dn_congr h₁ as').mp has, (All2_Dn_congr h₂ gs').mp hgs, hd⟩
  · rintro ⟨as', gs', has, hgs, hd⟩
    exact ⟨as', gs', (All2_Dn_congr h₁ as').mpr has, (All2_Dn_congr h₂ gs').mpr hgs, hd⟩

/-! ## Substitution respects the denotation -/

/-- Two optional values related when both present. -/
def OptRel (R : α -> α -> Prop) : Option α -> Option α -> Prop
  | some x, some y => R x y
  | none, none => True
  | _, _ => False

/-- Two substitutions whose answers denote alike at every label. -/
def EquivSub (sel : Sel) (L : ConLaw) (S : Sig) (τ τ' : Subst) : Prop :=
  ∀ v, OptRel (DnEquiv sel L S) (τ v) (τ' v)

mutual
  /-- Substituting types that denote alike gives types that denote alike. -/
  theorem Ty.subst_dnEquiv {τ τ' : Subst} (h : EquivSub sel L S τ τ') :
      ∀ t : Ty, DnEquiv sel L S (t.subst τ) (t.subst τ')
    | .var v => by
      have hv := h v
      simp only [Ty.subst]
      revert hv
      cases τ v <;> cases τ' v <;> simp [OptRel]
      exact DnEquiv.refl _
    | .con c as => by simp only [Ty.subst]; exact DnEquiv_con (substList_dnEquiv h as)
    | .proj t as a gs => by
      simp only [Ty.subst]; exact DnEquiv_proj (substList_dnEquiv h as) (substList_dnEquiv h gs)
  theorem substList_dnEquiv {τ τ' : Subst} (h : EquivSub sel L S τ τ') :
      ∀ as : List Ty, All2 (DnEquiv sel L S) (substList τ as) (substList τ' as)
    | [] => .nil
    | a :: as => .cons (Ty.subst_dnEquiv h a) (substList_dnEquiv h as)
end

theorem EquivSub_ofList {σ σ' : List Ty} (h : All2 (DnEquiv sel L S) σ σ') :
    EquivSub sel L S (Subst.ofList σ) (Subst.ofList σ') := by
  intro v
  simp only [Subst.ofList]
  have h1 := (h.getElem? v).1
  have h2 := (h.getElem? v).2
  cases hσ : σ[v]? with
  | none => have := h2.mp hσ; simp [this, OptRel]
  | some x =>
    cases hσ' : σ'[v]? with
    | none => exact absurd (h2.mpr hσ') (by simp [hσ])
    | some y => exact h1 x y hσ hσ'

/-- The first label equal to `v` is found at one position in both zips. -/
theorem find_zip_rel {R : Ty -> Ty -> Prop} {gs gs' : List Ty} (h : All2 R gs gs') :
    ∀ (labels : List Var) (v : Var),
      OptRel (fun g g' => R g g') (((labels.zip gs).find? (·.1 == v)).map (·.2))
        (((labels.zip gs').find? (·.1 == v)).map (·.2)) := by
  induction h with
  | nil => intro labels v; cases labels <;> simp [OptRel]
  | cons hab _ ih =>
    intro labels v
    cases labels with
    | nil => simp [OptRel]
    | cons l labels =>
      simp only [List.zip_cons_cons, List.find?_cons]
      cases (l == v) with
      | true => simpa [OptRel] using hab
      | false => exact ih labels v

theorem EquivSub_extend {τ τ' : Subst} (h : EquivSub sel L S τ τ') (labels : List Var) {gs gs' : List Ty}
    (hg : All2 (DnEquiv sel L S) gs gs') :
    EquivSub sel L S (τ.extend labels gs) (τ'.extend labels gs') := by
  intro v
  have hz := find_zip_rel hg labels v
  simp only [Subst.extend]
  revert hz
  cases (labels.zip gs).find? (·.1 == v) <;> cases (labels.zip gs').find? (·.1 == v) <;> simp [OptRel]
  exact h v

/-! ## The constructor laws respect the denotation -/

/-- The variables `0 .. n-1` in order. -/
def vars (n : Nat) : List Ty := (List.range n).map Ty.var

theorem substList_ofList_vars : ∀ as : List Ty, substList (Subst.ofList as) (vars as.length) = as := by
  intro as
  have key : ∀ (as : List Ty) (k : Nat) (pre : List Ty), pre.length = k ->
      substList (Subst.ofList (pre ++ as)) ((List.range' k as.length).map Ty.var) = as := by
    intro as
    induction as with
    | nil => intros; rfl
    | cons a as ih =>
      intro k pre hk
      simp only [List.length_cons, List.range'_succ, List.map_cons, substList, Ty.subst, Subst.ofList]
      rw [List.getElem?_append_right (by omega), hk, Nat.sub_self]
      simp only [List.getElem?_cons_zero, Option.getD_some]
      have := ih (k+1) (pre ++ [a]) (by simp [hk])
      simpa using this
  simpa [vars, List.range_eq_range'] using key as 0 [] rfl

/-- A law's answer is the law at the variables, substituted (H5's `substOk`). -/
theorem ConLaw.rebuild_eq_subst (L : ConLaw) (c : ConId) (as : List Ty) :
    L.rebuild c as = (L.rebuild c (vars as.length)).subst (Subst.ofList as) := by
  rw [L.substOk, substList_ofList_vars]

/-- A constructor law forgets no argument: when the rebuilt type denotes, each argument it was built
from denotes. True of the free algebra (`freeAlgebra_keepsArgs`) and of a law whose answer keeps its
arguments as subterms, such as coop's `Uniform` collapse (CoopTypes.cpp:43-59). -/
def ConLaw.KeepsArgs (L : ConLaw) (sel : Sel) (S : Sig) : Prop :=
  ∀ c as d, Dn sel L S (L.rebuild c as) d -> ∀ x ∈ as, ∃ d', Dn sel L S x d'

theorem All2_Dn_of_each {as : List Ty} (h : ∀ x ∈ as, ∃ d, Dn sel L S x d) :
    ∃ bs, All2 (Dn sel L S) as bs := by
  induction as with
  | nil => exact ⟨[], .nil⟩
  | cons a as ih =>
    obtain ⟨b, hb⟩ := h a (by simp)
    obtain ⟨bs, hbs⟩ := ih (fun x hx => h x (by simp [hx]))
    exact ⟨b :: bs, .cons hb hbs⟩

/-- (NORM) respects the denotation: `K(as)` and its rebuilt form denote alike. -/
theorem DnEquiv_law (hL : L.KeepsArgs sel S) {a b : Ty} (h : L.law a b) : DnEquiv sel L S a b := by
  obtain ⟨c, as, rfl, rfl⟩ := h
  intro d
  rw [Dn_con_iff]
  -- the law at `as` and at the answers `bs` of `as` denote alike
  have law_congr : ∀ bs, All2 (Dn sel L S) as bs -> DnEquiv sel L S (L.rebuild c as) (L.rebuild c bs) := by
    intro bs hbs
    rw [L.rebuild_eq_subst c as, L.rebuild_eq_subst c bs, hbs.length_eq]
    exact Ty.subst_dnEquiv (EquivSub_ofList (hbs.imp fun _ _ h => DnEquiv_of_Dn h)) _
  -- the answers are fixed points, so `K(bs)` denotes to itself
  have fixed : ∀ bs, All2 (Dn sel L S) as bs -> L.rebuild c bs = .con c bs -> Dn sel L S (.con c bs) (.con c bs) :=
    fun bs hbs hr => Dn_con_iff.mpr ⟨bs, hbs.right_self (fun _ _ h => Dn_idem h), .inl ⟨hr, rfl⟩⟩
  constructor
  · rintro ⟨bs, hbs, hd⟩
    rw [law_congr bs hbs d]
    rcases hd with ⟨hr, rfl⟩ | ⟨_, hd⟩
    · rw [hr]; exact fixed bs hbs hr
    · exact hd
  · intro hd
    obtain ⟨bs, hbs⟩ := All2_Dn_of_each (hL c as d hd)
    rw [law_congr bs hbs d] at hd
    refine ⟨bs, hbs, ?_⟩
    by_cases hr : L.rebuild c bs = .con c bs
    · left
      rw [hr] at hd
      exact ⟨hr, Dn_fun hd (fixed bs hbs hr)⟩
    · right; exact ⟨hr, hd⟩

theorem freeAlgebra_keepsArgs (sel : Sel) (S : Sig) : freeAlgebra.KeepsArgs sel S := by
  intro c as d hd x hx
  obtain ⟨bs, hbs, _⟩ := Dn_con_iff.mp hd
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hx
  cases hb : bs[i]? with
  | none => exact absurd ((hbs.getElem? i).2.mpr hb) (by simp [hi])
  | some b => exact ⟨b, (hbs.getElem? i).1 _ _ (List.getElem?_eq_getElem hi) hb⟩

end Limit

/-! ## The oracle at the limit (SPEC.md 6.3, `[[.]]` at the limit) -/

/-- H4 at the limit: an impl citation whose header denotes. -/
def DenL (sel : Sel) (L : ConLaw) (S : Sig) : ImplId -> Subst -> Prop :=
  fun i σ => ∃ hi : i < S.impls.length, ∃ Q, DnP sel L S (S.impls[i].header σ) Q

/-- `ServedBy` at the limit: the impl's header at `σ` and `P` denote one predicate, and the where
entries at `σ` are valid closed in the denotable fragment. -/
def ServedByL (sel : Sel) (L : ConLaw) (S : Sig) (P : Pred) (i : ImplId) (σ : List Ty) : Prop :=
  ∃ hi : i < S.impls.length, σ.length = S.impls[i].arity ∧ (∃ Q, DnP sel L S P Q) ∧
    (∀ Q, DnP sel L S (S.impls[i].header (Subst.ofList σ)) Q ↔ DnP sel L S P Q) ∧
    (∀ W ∈ S.impls[i].wheres, Nonempty (Deriv L S (DenL sel L S) [] (W.subst (Subst.ofList σ))))

/-- The oracle answers only with a server. -/
def SelSoundL (sel : Sel) (L : ConLaw) (S : Sig) : Prop :=
  ∀ P i σ, sel P = some (i, σ) -> ServedByL sel L S P i σ

/-- The clause H4 lacks: the oracle answers every served ground application. `resolveImplFor`
(D/ImplResolution.cpp:172-401) does, or refuses the program. -/
def SelCompleteL (sel : Sel) (L : ConLaw) (S : Sig) : Prop :=
  ∀ t as, as.all Ty.ground = true -> (∃ i σ, ServedByL sel L S (.app t as) i σ) -> (sel (.app t as)).isSome

/-- `Coherent(S)` at the limit: any two servers of a denotable ground application are one impl at
arguments that denote alike. -/
def Sig.CoherentL (S : Sig) (sel : Sel) (L : ConLaw) : Prop :=
  ∀ t as, as.all Ty.ground = true -> (∃ Q, DnP sel L S (.app t as) Q) ->
    ∀ i σ j σ', ServedByL sel L S (.app t as) i σ -> ServedByL sel L S (.app t as) j σ' ->
      i = j ∧ All2 (DnEquiv sel L S) σ σ'

section Oracle
variable {sel : Sel} {L : ConLaw} {S : Sig}

theorem DnP_app_iff {t : TraitId} {as : List Ty} {Q : Pred} :
    DnP sel L S (.app t as) Q ↔ ∃ as', All2 (Dn sel L S) as as' ∧ Q = .app t as' := by
  constructor
  · rintro ⟨f, hf⟩
    obtain ⟨as', has, rfl⟩ := Option.map_eq_some_iff.mp hf
    exact ⟨as', All2_Dn_of_mapM has, rfl⟩
  · rintro ⟨as', has, rfl⟩
    obtain ⟨f, hf⟩ := All2_Dn_mapM has
    exact ⟨f, Option.map_eq_some_iff.mpr ⟨as', hf, rfl⟩⟩

theorem All2_Dn_fun {as bs cs : List Ty} (h₁ : All2 (Dn sel L S) as bs) (h₂ : All2 (Dn sel L S) as cs) :
    bs = cs := by
  induction h₁ generalizing cs with
  | nil => cases h₂; rfl
  | cons hab _ ih => cases h₂ with | cons hac hcs => rw [Dn_fun hab hac, ih hcs]

/-- A premise given at every position is given at every member. -/
theorem forall_mem_of_getD {P : Pred -> Prop} {l : List Pred} (h : ∀ j, j < l.length -> P (l.getD j default)) :
    ∀ W ∈ l, P W := by
  intro W hW
  obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp hW
  have := h j hj
  rwa [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj] at this

/-- (BIND) respects the denotation: under a sound, complete oracle and a coherent signature, a
projection at an application that resolves denotes alike with the cited impl's binding, because the
oracle's server and the citation are one impl at arguments that denote alike (`Coherent`), and the
binding is read at those arguments. -/
theorem DnEquiv_bind (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S) (hC : S.CoherentL sel L)
    {i : ImplId} (hi : i < S.impls.length) {σ : List Ty} (hσ : σ.length = S.impls[i].arity)
    (hW : ∀ j, j < S.impls[i].wheres.length ->
      Nonempty (Deriv L S (DenL sel L S) [] ((S.impls[i].wheres.getD j default).subst (Subst.ofList σ))))
    {a : AssocId} {labels : List Var} {B : Ty} {gs : List Ty}
    (hb : S.impls[i].bindings[a]? = some (some (labels, B)))
    {as' gs' : List Ty} (has : All2 (Dn sel L S) (substList (Subst.ofList σ) S.impls[i].head) as')
    (hgas : groundList as' = true) (hgs : All2 (Dn sel L S) gs gs') (hggs : groundList gs' = true) :
    DnEquiv sel L S (.proj S.impls[i].trait (substList (Subst.ofList σ) S.impls[i].head) a gs)
      (B.subst ((Subst.ofList σ).extend labels gs)) := by
  -- the cited impl serves the resolved application
  have hself : DnP sel L S (.app S.impls[i].trait as') (.app S.impls[i].trait as') :=
    DnP_app_iff.mpr ⟨as', has.right_self (fun _ _ h => Dn_idem h), rfl⟩
  have hserve : ServedByL sel L S (.app S.impls[i].trait as') i σ := by
    refine ⟨hi, hσ, ⟨_, hself⟩, ?_, forall_mem_of_getD hW⟩
    intro Q
    simp only [ImplDecl.header, DnP_app_iff]
    constructor
    · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr (has.imp fun _ _ h => DnEquiv_of_Dn h) cs).mp hcs, rfl⟩
    · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr (has.imp fun _ _ h => DnEquiv_of_Dn h) cs).mpr hcs, rfl⟩
  -- the oracle answers with a server, which `Coherent` identifies with the citation
  have hgall : as'.all Ty.ground = true := by rwa [← groundList_eq_all]
  obtain ⟨⟨i', σ'⟩, hsel⟩ := Option.isSome_iff_exists.mp (hSelC _ _ hgall ⟨i, σ, hserve⟩)
  have hserve' := hSel _ _ _ hsel
  obtain ⟨rfl, hσσ'⟩ := hC _ _ hgall ⟨_, hself⟩ _ _ _ _ hserve hserve'
  have hI : S.impls[i]? = some S.impls[i] := List.getElem?_eq_getElem hi
  -- the binding at the two argument lists denotes alike
  have hBB : DnEquiv sel L S (B.subst ((Subst.ofList σ).extend labels gs))
      (B.subst ((Subst.ofList σ').extend labels gs')) :=
    Ty.subst_dnEquiv (EquivSub_extend (EquivSub_ofList hσσ') labels (hgs.imp fun _ _ h => DnEquiv_of_Dn h)) B
  have hground : (groundList as' && groundList gs') = true := by simp [hgas, hggs]
  intro d
  rw [Dn_proj_iff]
  constructor
  · rintro ⟨as'', gs'', has'', hgs'', h⟩
    obtain rfl := All2_Dn_fun has has''
    obtain rfl := All2_Dn_fun hgs hgs''
    rcases h with ⟨_, i₂, σ₂, I₂, labels₂, B₂, hsel₂, hI₂, hb₂, hd⟩ | ⟨hng, _⟩
    · rw [hsel] at hsel₂
      cases hsel₂
      rw [hI] at hI₂
      cases hI₂
      rw [hb] at hb₂
      cases hb₂
      exact (hBB d).mpr hd
    · rw [hground] at hng; cases hng
  · intro hd
    exact ⟨as', gs', has, hgs, .inl ⟨hground, i, σ', S.impls[i], labels, B, hsel, hI, hb, (hBB d).mp hd⟩⟩

end Oracle

/-! ## The derivations theorem 7 speaks of -/

/-- Every (BIND) node applies its impl at an application and at GAT arguments that denote to ground
types: the projections `[[.]]` resolves. The nodes of an equality derivation are walked; the where
derivations under a (BIND) are applications and are not. -/
def Deriv.Resolvable (sel : Sel) (L : ConLaw) (S : Sig) : Deriv L S Ok Γ P -> Prop
  | .bind (gs := gs) i _ σ _ _ _ _ _ =>
    (∃ as', All2 (Dn sel L S) (substList (Subst.ofList σ) S.impls[i].head) as' ∧ groundList as' = true) ∧
    (∃ gs', All2 (Dn sel L S) gs gs' ∧ groundList gs' = true)
  | .symm d => d.Resolvable sel L S
  | .trans d₁ d₂ => d₁.Resolvable sel L S ∧ d₂.Resolvable sel L S
  | .cong _ ds => ∀ j hj, (ds j hj).Resolvable sel L S
  | .congP _ _ ds es => (∀ j hj, (ds j hj).Resolvable sel L S) ∧ (∀ j hj, (es j hj).Resolvable sel L S)
  | _ => True

/-- `S; [] |-p P` in the denotable fragment, every (BIND) at an application `[[.]]` resolves: the
closed judgments the stage reaches after monomorphization, as theorem 7 reads them. -/
def ValidGround (sel : Sel) (L : ConLaw) (S : Sig) (P : Pred) : Prop :=
  ∃ d : Deriv L S (DenL sel L S) [] P, d.Primitive ∧ d.Resolvable sel L S

/-! ## Theorem 7's statement (SPEC.md 6.6) -/

section Theorems
variable {sel : Sel} {L : ConLaw} {S : Sig}

/-- Theorem 7 in its two-sided form: the endpoints of a closed valid equality denote alike. Induction
on the derivation: (REFL)(SYM)(TRANS) are equality's; (CONG)(CONG-P) by `DnEquiv_con`, `DnEquiv_proj`;
(BIND) by `DnEquiv_bind`; (NORM) by `DnEquiv_law`; (HYP) has no instance at `[]`; (REQ) and (REQ-A)
have no primitive instance at `[]`; the application rules conclude no equality. -/
theorem erasure_equiv (hL : L.KeepsArgs sel S) (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S)
    (hC : S.CoherentL sel L) {P : Pred} (d : Deriv L S (DenL sel L S) [] P) (hp : d.Primitive)
    (hr : d.Resolvable sel L S) : ∀ a b, P = .eq a b -> DnEquiv sel L S a b := by
  induction d with
  | hyp k hk => exact absurd hk (Nat.not_lt_zero k)
  | impl i hi σ hσ ok ds ih => intro a b h; simp [ImplDecl.header] at h
  | req d ht hk ih =>
    simp only [Deriv.Primitive] at hp
    exact absurd hp (Deriv.not_inEnv_nil d)
  | reqA d ht hb us hus prem ih₁ ih₂ =>
    simp only [Deriv.Primitive] at hp
    exact absurd hp.1 (Deriv.not_inEnv_nil d)
  | bind i hi σ hσ ok ds hb hg ih =>
    simp only [Deriv.Resolvable] at hr
    obtain ⟨⟨as', has, hgas⟩, ⟨gs', hgs, hggs⟩⟩ := hr
    intro a b h; cases h
    exact DnEquiv_bind hSel hSelC hC hi hσ (fun j hj => ⟨ds j hj⟩) hb has hgas hgs hggs
  | refl => intro a b h; cases h; exact DnEquiv.refl _
  | symm d ih =>
    simp only [Deriv.Primitive, Deriv.Resolvable] at hp hr
    intro a b h; cases h
    exact (ih hp hr _ _ rfl).symm
  | trans d₁ d₂ ih₁ ih₂ =>
    simp only [Deriv.Primitive, Deriv.Resolvable] at hp hr
    intro a b h; cases h
    exact (ih₁ hp.1 hr.1 _ _ rfl).trans (ih₂ hp.2 hr.2 _ _ rfl)
  | cong hl ds ih =>
    simp only [Deriv.Primitive, Deriv.Resolvable] at hp hr
    intro a b h; cases h
    exact DnEquiv_con (All2.of_getD hl fun j hj => ih j hj (hp j hj) (hr j hj) _ _ rfl)
  | congP hl₁ hl₂ ds es ih₁ ih₂ =>
    simp only [Deriv.Primitive, Deriv.Resolvable] at hp hr
    intro a b h; cases h
    exact DnEquiv_proj (All2.of_getD hl₁ fun j hj => ih₁ j hj (hp.1 j hj) (hr.1 j hj) _ _ rfl)
      (All2.of_getD hl₂ fun j hj => ih₂ j hj (hp.2 j hj) (hr.2 j hj) _ _ rfl)
  | norm hlaw => intro a b h; cases h; exact DnEquiv_law hL hlaw
  | conv d hl ds ih₁ ih₂ => intro a b h; cases h

/-- Theorem 7's statement (SPEC.md 6.6), placed here because theorem 5's COERCE case uses it: a closed
valid equality in the denotable fragment relates two written forms of one type. Stated at the limit of
the fuel, for a primitive derivation whose (BIND)s `[[.]]` resolves (`ValidGround`), under an oracle
that is sound and complete, a coherent signature and a constructor law that forgets no argument; the
conclusion is two-sided. -/
theorem erasure_core (sel : Sel) (L : ConLaw) (S : Sig) {a b : Ty} (hL : L.KeepsArgs sel S)
    (hSel : SelSoundL sel L S) (hSelC : SelCompleteL sel L S) (hC : S.CoherentL sel L)
    (h : ValidGround sel L S (.eq a b)) : DnEquiv sel L S a b :=
  let ⟨d, hp, hr⟩ := h
  erasure_equiv hL hSel hSelC hC d hp hr a b rfl

end Theorems

end Trait
