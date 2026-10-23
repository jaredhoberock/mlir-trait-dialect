/-!
# Defs: the signature and the syntax of evidence terms

Transcribed from SPEC.md section 7 (the Lean skeleton) and section 3 (the signature). Standard library only.
Anchors into the dialect's sources are written as SPEC.md writes them: `D/` is `mlir-trait-dialect/cpp/`,
`S/` is `screech-fable/`.

Decisions taken by the scaffold where the skeleton left a `sorry` in a definition or elided a piece:
* `Ty.ground`, `Sig.DeclWF`, the constructor law and the well-formedness checkers are real definitions.
* The constructor law (H5) is carried as the normalizing function itself, `ConLaw.rebuild`, with its
  substitution stability as the one axiom; (NORM)'s relation `ConLaw.law` is defined from it. A relation
  alone cannot be applied by the denotation of section 6.3, which "applies the constructor laws".
* An impl carries the evidence its `trait.return` supplies, one term per plain requirement (`returns`) and
  one per binder of its trait (`binderReturns`), because the root of a projection (SPEC.md 5.4) and the
  reduction (beta-req) read that term. The skeleton's `ImplDecl` had no field for it. Their typing is
  the term form of H3, stated in Calculus.lean; the logic's form, `Sig.ReqDuty`, is in Logic.lean.
* The evidence term syntax `Ev` is declared here rather than in Calculus.lean because `Module` and
  `ImplDecl` store terms. The seventh constructor, `reqCall`, is the binder call (REQ-CALL) of SPEC.md 5.1.
-/

namespace Trait

abbrev TraitId := Nat   -- index into Sig.traits
abbrev ImplId  := Nat   -- index into Sig.impls
abbrev ProofId := Nat   -- index into Module.proofs
abbrev AssocId := Nat   -- index into TraitDecl.assocs
abbrev Var     := Nat   -- the label of !trait.poly<i>
abbrev ConId   := Nat   -- any non-trait type constructor

/-! ## Types (SPEC.md 3; D/TraitTypes.td:13-57, 196-231) -/

/-- A type: a variable, a constructor over children, or a projection `T[as]::N<gs>`. There is no claim
constructor: hypothesis H1 (F10) is built into the grammar. -/
inductive Ty where
  | var  : Var -> Ty
  | con  : ConId -> List Ty -> Ty
  | proj : TraitId -> List Ty -> AssocId -> List Ty -> Ty
  deriving Repr

instance : Inhabited Ty := ⟨.con 0 []⟩

mutual
  /-- Equality of types is decidable. Written by hand: the derive handler does not cover a nested inductive. -/
  def Ty.decEq : (a b : Ty) → Decidable (a = b)
    | .var i, .var j =>
      if h : i = j then isTrue (h ▸ rfl) else isFalse (fun e => by cases e; exact h rfl)
    | .con c as, .con d bs =>
      if h : c = d then
        match decEqTyList as bs with
        | isTrue e  => isTrue (by subst h; subst e; rfl)
        | isFalse e => isFalse (fun x => by cases x; exact e rfl)
      else isFalse (fun e => by cases e; exact h rfl)
    | .proj t as a gs, .proj u bs b hs =>
      if h : t = u ∧ a = b then
        match decEqTyList as bs, decEqTyList gs hs with
        | isTrue e₁, isTrue e₂ =>
          isTrue (by obtain ⟨h₁, h₂⟩ := h; subst h₁; subst h₂; subst e₁; subst e₂; rfl)
        | isFalse e₁, _ => isFalse (fun x => by cases x; exact e₁ rfl)
        | _, isFalse e₂ => isFalse (fun x => by cases x; exact e₂ rfl)
      else isFalse (fun e => by cases e; exact h ⟨rfl, rfl⟩)
    | .var _, .con _ _ => isFalse (by intro h; cases h)
    | .var _, .proj _ _ _ _ => isFalse (by intro h; cases h)
    | .con _ _, .var _ => isFalse (by intro h; cases h)
    | .con _ _, .proj _ _ _ _ => isFalse (by intro h; cases h)
    | .proj _ _ _ _, .var _ => isFalse (by intro h; cases h)
    | .proj _ _ _ _, .con _ _ => isFalse (by intro h; cases h)
  def decEqTyList : (as bs : List Ty) → Decidable (as = bs)
    | [], [] => isTrue rfl
    | [], _ :: _ => isFalse (by intro h; cases h)
    | _ :: _, [] => isFalse (by intro h; cases h)
    | a :: as, b :: bs =>
      match Ty.decEq a b, decEqTyList as bs with
      | isTrue e₁, isTrue e₂ => isTrue (by subst e₁; subst e₂; rfl)
      | isFalse e₁, _ => isFalse (fun x => by cases x; exact e₁ rfl)
      | _, isFalse e₂ => isFalse (fun x => by cases x; exact e₂ rfl)
end

instance : DecidableEq Ty := Ty.decEq

/-! ## Substitution (D/TraitTypes.hpp:824-831; #trait.impl_args, D/TraitAttributes.td:132-174) -/

/-- A substitution by label. `none` leaves the variable in place. -/
abbrev Subst := Var -> Option Ty

/-- The positional substitution a citation's arguments make: label `i` becomes the `i`-th argument when
`i < |σ|` and stays otherwise. -/
def Subst.ofList (σ : List Ty) : Subst := fun i => σ[i]?

/-- A binding's declared GAT labels extend a substitution (D/TraitOps.cpp:1908-1938): the first label
equal to `i` wins, else the underlying substitution answers. -/
def Subst.extend (σ : Subst) (labels : List Var) (gs : List Ty) : Subst :=
  fun i => match (labels.zip gs).find? (·.1 == i) with
    | some (_, g) => some g
    | none => σ i

mutual
  /-- One substitution pass: a substituted term is never revisited. -/
  def Ty.subst (σ : Subst) : Ty -> Ty
    | .var i          => (σ i).getD (.var i)
    | .con c as       => .con c (substList σ as)
    | .proj t as a gs => .proj t (substList σ as) a (substList σ gs)
  def substList (σ : Subst) : List Ty -> List Ty
    | []      => []
    | a :: as => a.subst σ :: substList σ as
end

mutual
  /-- A type is ground when it has no variable (D/TraitTypes.cpp:351-359). -/
  def Ty.ground : Ty -> Bool
    | .var _          => false
    | .con _ as       => groundList as
    | .proj _ as _ gs => groundList as && groundList gs
  def groundList : List Ty -> Bool
    | []      => true
    | a :: as => a.ground && groundList as
end

mutual
  /-- Every variable of the type satisfies `p`. -/
  def Ty.allVars (p : Var -> Bool) : Ty -> Bool
    | .var i          => p i
    | .con _ as       => allVarsList p as
    | .proj _ as _ gs => allVarsList p as && allVarsList p gs
  def allVarsList (p : Var -> Bool) : List Ty -> Bool
    | []      => true
    | a :: as => a.allVars p && allVarsList p as
end

/-- Every variable of the type is a label below `n`. -/
def Ty.varsBelow (n : Nat) (t : Ty) : Bool := t.allVars (· < n)

mutual
  /-- The variable `v` occurs in the type. -/
  def Ty.mentions (v : Var) : Ty -> Bool
    | .var i          => i == v
    | .con _ as       => mentionsList v as
    | .proj _ as _ gs => mentionsList v as || mentionsList v gs
  def mentionsList (v : Var) : List Ty -> Bool
    | []      => false
    | a :: as => a.mentions v || mentionsList v as
end

/-! ## Predicates and claims (SPEC.md 3; D/TraitAttributes.cpp:126-156; D/TraitTypes.cpp:224-272) -/

/-- A predicate: a trait application or an equality. An equality is ordered as written and read
direction-blind. -/
inductive Pred where
  | app : TraitId -> List Ty -> Pred
  | eq  : Ty -> Ty -> Pred
  deriving Repr, DecidableEq

instance : Inhabited Pred := ⟨.app 0 []⟩

def Pred.subst (σ : Subst) : Pred -> Pred
  | .app t as => .app t (substList σ as)
  | .eq a b   => .eq (a.subst σ) (b.subst σ)

def Pred.ground : Pred -> Bool
  | .app _ as => groundList as
  | .eq a b   => a.ground && b.ground

def Pred.allVars (p : Var -> Bool) : Pred -> Bool
  | .app _ as => allVarsList p as
  | .eq a b   => a.allVars p && b.allVars p

def Pred.varsBelow (n : Nat) (P : Pred) : Bool := P.allVars (· < n)

def Pred.mentions (v : Var) : Pred -> Bool
  | .app _ as => mentionsList v as
  | .eq a b   => a.mentions v || b.mentions v

/-- A citation: a `trait.proof` or an unconditional `trait.impl` (D/TraitOps.cpp:2118-2131). -/
inductive Cite where
  | proof : ProofId -> Cite
  | impl  : ImplId -> Cite
  deriving DecidableEq, Repr

/-- A claim: a predicate with an optional citation. -/
structure Claim where
  pred : Pred
  cite : Option Cite := none
  deriving Repr, DecidableEq

/-- An equality never carries a citation (D/TraitTypes.cpp:224-235; T/invalid_equality_claim_proof.mlir). -/
def Claim.WF : Claim -> Prop
  | ⟨.eq _ _, some _⟩ => False
  | _ => True

/-- `|C|`: the claim with its citation removed (`stripClaimProofs`, D/EqualityVerification.cpp:278-290). -/
def Claim.strip (c : Claim) : Claim := { c with cite := none }

def Claim.subst (σ : Subst) (c : Claim) : Claim := { c with pred := c.pred.subst σ }

/-! ## Evidence terms (SPEC.md 5.1) -/

/-- An evidence term: one constructor per operation, the hypothesis `^k`, and the binder call. Terms
are finite trees (F12). -/
inductive Ev where
  /-- The block argument `^k`, the `k`-th hypothesis in scope. -/
  | hyp     : Nat -> Ev
  /-- `trait.allege C` (card 1): the cut. -/
  | allege  : Claim -> Ev
  /-- `trait.derive hdr_I[σ] from @I[σ] given(es)` (card 2). -/
  | derive  : ImplId -> List Ty -> List Ev -> Ev
  /-- `trait.project e[k]` (card 3): a plain requirement below `|R_T|`, a where entry at or past it. -/
  | project : Ev -> Nat -> Ev
  /-- `trait.method.call e @T[a]::@requirement_b(fs)` at the GAT arguments `us` (REQ-CALL; F4). -/
  | reqCall : Ev -> Nat -> List Ty -> List Ev -> Ev
  /-- `trait.witness @c for P` (card 4, WIT-APP). -/
  | witness : Cite -> Pred -> Ev
  /-- `trait.witness proj_resolve hdr_I[σ]::N<gs> resolves .. by @I[σ] given(es)` (card 4, WIT-PROJ). -/
  | projRes : ImplId -> List Ty -> List Ev -> AssocId -> List Ty -> Ev
  /-- `trait.witness refl : (t = t)` (card 4, WIT-REFL). -/
  | refl    : Ty -> Ev
  /-- `trait.witness compose(es) : (l = r)` (card 4, WIT-COMPOSE; deletion D1). -/
  | compose : List Ev -> Ty -> Ty -> Ev
  /-- `trait.coerce e : tin to C via (es)` on a claim (card 5). The value arm is `Entails` alone. -/
  | coerce  : Ev -> List Ev -> Claim -> Ev
  deriving Inhabited, Repr

/-! ## Declarations and the signature (SPEC.md 3) -/

/-- A binder `Pi X.(P => Q)`: a bodiless `trait.method @requirement_i` whose claim parameters are the
premises and whose result is the conclusion (D/TraitOps.td:186-190). `labels` are the binder's own
variables `X`, declared, distinct from the trait's positions. -/
structure Binder where
  labels   : List Var
  premises : List Pred
  concl    : Pred

/-- A trait: `arity` parameters labelled by position with `Self` first (D/TraitOps.cpp:401-420), the
plain requirements `R_T` (the result list, D/TraitOps.td:161-164), the binders `B_T`, and per associated
type its declared parameter labels (D/TraitOps.cpp:433-451). -/
structure TraitDecl where
  arity   : Nat
  reqs    : List Pred
  binders : List Binder
  assocs  : List (List Var)

/-- An impl: `arity` parameters with dense labels (D/TraitOps.cpp:890-894), the header `T[head]`, the
where entries (block arguments after the first, D/TraitOps.cpp:877-889), one binding per associated type
with the GAT labels it declares (V/v6), and the evidence its `trait.return` supplies: one term per plain
requirement, typed under the where entries as hypotheses `^0 .. ^(k-1)`, and one per binder of the
trait, typed under the where entries followed by the binder's premises. -/
structure ImplDecl where
  arity         : Nat
  trait         : TraitId
  head          : List Ty
  wheres        : List Pred
  bindings      : List (Option (List Var × Ty))
  returns       : List Ev
  binderReturns : List Ev

/-- The signature `S`: the traits and impls of a module. -/
structure Sig where
  traits : List TraitDecl
  impls  : List ImplDecl

/-- `hdr_I[σ]`. -/
def ImplDecl.header (I : ImplDecl) (σ : Subst) : Pred := .app I.trait (substList σ I.head)

/-- Unconditional iff no parameters and no where entries (D/TraitOps.cpp:1107-1113). -/
def ImplDecl.unconditional (I : ImplDecl) : Prop := I.arity = 0 ∧ I.wheres = []

/-- The substitution under which an impl's binder evidence is typed: the trait's positions take the
header, and the binder's own labels take fresh variables above the impl's parameters, so the binder's
variables and the impl's never coincide. -/
def ImplDecl.binderSubst (I : ImplDecl) (B : Binder) : Subst :=
  (Subst.ofList I.head).extend B.labels (B.labels.map (fun l => Ty.var (I.arity + l)))

/-- The hypotheses an impl's binder evidence is typed under: the where entries, then the binder's
premises at `binderSubst`. -/
def ImplDecl.binderHyps (I : ImplDecl) (B : Binder) : List Pred :=
  I.wheres ++ B.premises.map (Pred.subst (I.binderSubst B))

/-- The substitution a binder call at receiver arguments `as` and GAT arguments `us` makes. -/
def Binder.callSubst (B : Binder) (as us : List Ty) : Subst :=
  (Subst.ofList as).extend B.labels us

/-- The substitution that instantiates an impl's binder evidence at a call: the impl's parameters take
the receiver's impl arguments `σ`, and the fresh variables `binderSubst` gave the binder's labels take
the GAT arguments `us`. -/
def ImplDecl.binderCallSubst (I : ImplDecl) (B : Binder) (σ us : List Ty) : Subst :=
  (Subst.ofList σ).extend (B.labels.map (fun l => I.arity + l)) us

/-- A module `M = (S, Pi)`: the signature and the proof table. A proof is the application it proves
and the term its body returns (D/TraitOps.td:1572-1702). -/
structure Module where
  sig    : Sig
  proofs : List (Pred × Ev)

/-! ## Well-formedness of declarations (SPEC.md 3, "Well-formedness enforced") -/

mutual
  /-- A type names existing traits at their arity, existing associated types at their declared GAT
  arity (D/TraitAttributes.cpp:126-156; D/TraitOps.cpp:826-869). -/
  def Ty.wf (S : Sig) : Ty -> Bool
    | .var _          => true
    | .con _ as       => wfList S as
    | .proj t as a gs =>
      match S.traits[t]? with
      | none   => false
      | some T =>
        as.length == T.arity &&
        (match T.assocs[a]? with
          | none   => false
          | some X => gs.length == X.length) &&
        wfList S as && wfList S gs
  def wfList (S : Sig) : List Ty -> Bool
    | []      => true
    | a :: as => a.wf S && wfList S as
end

/-- A predicate names an existing trait at its arity; its endpoints are well-formed types. -/
def Pred.wf (S : Sig) : Pred -> Bool
  | .app t as =>
    match S.traits[t]? with
    | none   => false
    | some T => as.length == T.arity && wfList S as
  | .eq a b => a.wf S && b.wf S

/-- `Sig.DeclWF`: what `TraitOp::verify` (D/TraitOps.cpp:397-501) and `ImplOp::verify` /
`verifySymbolUses` (D/TraitOps.cpp:871-898, 1004-1089, 826-869) enforce on declarations, as far as the
calculus reads them. Each field is one check; the field `impl_params_constrained` states the consequence
of rustc's E0207 rule (every impl parameter is mentioned by the header or a where entry) rather than the
fixed-point rule itself. Nothing here speaks of an impl's returns: that is H3, `Sig.ReqDuty` in
Logic.lean and `Module.ReturnsWF` in Calculus.lean. -/
structure Sig.DeclWF (S : Sig) : Prop where
  /-- Every plain requirement is well-formed over the trait's parameters and applies the trait itself
  only under a projection (D/TraitOps.cpp:471-485; T/cyclic_traits.mlir). -/
  trait_reqs : ∀ (t : TraitId) (T : TraitDecl), S.traits[t]? = some T -> ∀ r ∈ T.reqs,
    r.wf S = true ∧ r.varsBelow T.arity = true ∧ ∀ as, r ≠ .app t as
  /-- Every binder's labels are its own, distinct and above the trait's positions; its premises and
  conclusion are well-formed over the trait's parameters and those labels. -/
  trait_binders : ∀ T ∈ S.traits, ∀ B ∈ T.binders,
    B.labels.Nodup ∧ (∀ l ∈ B.labels, T.arity ≤ l) ∧
    (∀ P ∈ B.premises, P.wf S = true ∧ P.allVars (fun v => decide (v < T.arity) || B.labels.contains v) = true) ∧
    B.concl.wf S = true ∧ B.concl.allVars (fun v => decide (v < T.arity) || B.labels.contains v) = true
  /-- Every associated type's declared parameters are distinct and above the trait's positions
  (D/TraitOps.cpp:433-451). -/
  trait_assocs : ∀ T ∈ S.traits, ∀ X ∈ T.assocs, X.Nodup ∧ ∀ l ∈ X, T.arity ≤ l
  /-- Every impl names an existing trait, states its header at the trait's arity, binds each associated
  type by one entry, and returns one term per plain requirement and per binder. -/
  impl_trait : ∀ I ∈ S.impls, ∃ T : TraitDecl, S.traits[I.trait]? = some T ∧
    I.head.length = T.arity ∧ I.bindings.length = T.assocs.length ∧
    I.returns.length = T.reqs.length ∧ I.binderReturns.length = T.binders.length
  /-- Every impl's header, where entries and bindings are well-formed and mention only the impl's own
  labels (dense below `arity`) and, in a binding, the GAT labels the binding declares, one per declared
  parameter of the associated type (D/TraitOps.cpp:826-869; V/v6). -/
  impl_wf : ∀ I ∈ S.impls,
    (∀ h ∈ I.head, h.wf S = true ∧ h.varsBelow I.arity = true) ∧
    (∀ W ∈ I.wheres, W.wf S = true ∧ W.varsBelow I.arity = true) ∧
    (∀ (T : TraitDecl), S.traits[I.trait]? = some T -> ∀ (a : AssocId) (labels : List Var) (B : Ty), I.bindings[a]? = some (some (labels, B)) ->
      ∃ X : List Var, T.assocs[a]? = some X ∧ labels.length = X.length ∧ labels.Nodup ∧ (∀ l ∈ labels, I.arity ≤ l) ∧
        B.wf S = true ∧ B.allVars (fun v => decide (v < I.arity) || labels.contains v) = true)
  /-- Every impl parameter is mentioned by the header or by a where entry: the consequence of the
  fixed-point rule of D/TraitOps.cpp:1004-1089 (rustc E0207). -/
  impl_params_constrained : ∀ I ∈ S.impls, ∀ v, v < I.arity ->
    mentionsList v I.head = true ∨ ∃ W ∈ I.wheres, W.mentions v = true

/-! ## The constructor laws (H5; D/EqualityVerification.cpp:176-248; F14) -/

/-- H5: a constructor normalizes as a context-free deterministic function of its arguments, stable
under substitution. `rebuild c as` is what the constructor's own `get()` returns for the children `as`;
it need not be `con c as` ("not a free re-application"). The one axiom is stability under substitution,
which `subst_types` needs. -/
structure ConLaw where
  rebuild : ConId -> List Ty -> Ty
  substOk : ∀ (σ : Subst) c as, (rebuild c as).subst σ = rebuild c (substList σ as)

/-- (NORM): `K(a)` rebuilt through its constructor. Only type constructors normalize (:197-205). -/
def ConLaw.law (L : ConLaw) (a b : Ty) : Prop := ∃ c as, a = Ty.con c as ∧ b = L.rebuild c as

theorem ConLaw.law_onCon (L : ConLaw) {a b : Ty} (h : L.law a b) : ∃ c as, a = Ty.con c as :=
  let ⟨c, as, ha, _⟩ := h; ⟨c, as, ha⟩

theorem ConLaw.law_subst (L : ConLaw) (σ : Subst) {a b : Ty} (h : L.law a b) :
    L.law (a.subst σ) (b.subst σ) := by
  obtain ⟨c, as, ha, hb⟩ := h
  subst ha; subst hb
  exact ⟨c, substList σ as, by simp [Ty.subst], L.substOk σ c as⟩

/-- The free algebra: no constructor normalizes. -/
def freeAlgebra : ConLaw := ⟨fun c as => .con c as, fun σ c as => by simp [Ty.subst]⟩

theorem freeAlgebra_law_iff (a b : Ty) : freeAlgebra.law a b ↔ ∃ c as, a = Ty.con c as ∧ b = a := by
  constructor
  · rintro ⟨c, as, ha, hb⟩; exact ⟨c, as, ha, by simp [freeAlgebra, ha, hb]⟩
  · rintro ⟨c, as, ha, hb⟩; exact ⟨c, as, ha, by simp [freeAlgebra, ha, hb]⟩

/-! ## A list helper for premises indexed by position -/

/-- `[f 0, .., f (n-1)]`, for premises given as a function of a position below a bound. -/
def tabulate : (n : Nat) -> ((j : Nat) -> j < n -> α) -> List α
  | 0, _ => []
  | n+1, f => tabulate n (fun j hj => f j (Nat.lt_succ_of_lt hj)) ++ [f n (Nat.lt_succ_self n)]

theorem tabulate_length (n : Nat) (f : (j : Nat) -> j < n -> α) : (tabulate n f).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [tabulate, ih]

end Trait
