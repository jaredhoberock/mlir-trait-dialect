import Logic
/-!
# Calculus: evidence terms and the typing judgment `M; G |- e : C`

SPEC.md section 5 (the calculus), 5.4 (the root of a term), 5.5 (what the reductions read), 6.2 (`elab`)
and 6.5 (the instantiation lemmas), transcribed from the section 7 skeleton.

Decisions taken by the scaffold:
* REQ-CALL is `HasType.reqCall`, by the pattern of `derive`: the receiver's claim names the trait, the
  binder is chosen by index, the GAT arguments `us` are read off the call's written signature, and the
  premises are given by position modulo citation. Its result is unproven.
* The root of a term (SPEC.md 5.4) is the relation `Root M e (I, σ, ds)`, reading to the impl, its
  arguments and the operands supplied for its where entries, instead of a function with a depth bound:
  a reading with no base (`readEvidence`, D/TraitOps.cpp:1696-1767; X/x3) simply has no root. `root`
  is its projection to `(I, σ)`.
* `ProofsWF` (H2) takes the impl restriction `Ok` so that theorems 5 and 7, which restrict citations to
  denotable ones, can use a proof table typed under the same restriction. It also requires the proved
  application to be ground, as H2 states.
* `Module.ReturnsWF` is the term form of H3: the evidence an impl returns is typed under its where
  entries as hypotheses at the requirement it discharges, and is allege-free (F13).
* `elab` returns the term together with the equality leaves its equational part cites, because the
  equality rules (SYM)(TRANS)(CONG)(CONG-P)(NORM) become one `trait.coerce` of a `refl` over those leaves.
* Shared lemmas: `hasType_valid` (theorem 3 for any `Ok`; Soundness.lean's theorem is its instance at
  `AnyImpl`, Coherence.lean's DERIVE case needs it at `DenL`), weakening of the typing, the determinism
  of `Root`, and the two instantiation lemmas of 6.5.

Decisions taken by the integration step, on the provers' requests:
* `HasType.witImpl` carries the premise `Ok i (Subst.ofList [])`, by the pattern of `derive` and
  `projRes`: the citation of an unconditional impl is an impl citation at no arguments, and H4's
  restriction on impl citations covers it exactly as it covers a derive's. Without the premise theorem
  3 has no proof at a general `Ok` (a witness of an impl would have no derivation under `Ok`).
* `inlineAux` reads a projection past the plain requirements off the derive its inlined operand
  returns, as `Root.projWhere` reads it, not off the operand's own text: (PROJECT-WHERE) reads the
  citation off the operand's claim, and a projection whose operand is itself such a projection carries
  the citation without writing it (D/TraitOps.cpp:3159-3161 reads the operand's claim type).
* The instantiation lemmas of 6.5 are stated so that they apply to the return `Root.projReq` reads.
  `subst_evidence` replaces each hypothesis by evidence of its predicate under the hypothesis's own
  citation where it has one and under any well-formed citation where it has none (`Claim.Replaces`),
  and concludes the term's claim likewise: PROJECT-WHERE reads its operand's citation and the operands
  of compose and coerce must be uncited, so "under any citation" alone is not provable. `subst_types`
  keeps `hOk`, trivial at `AnyImpl` and false at `DenL` (a standing projection over a variable denotes;
  its instance at an unserved ground argument does not); the restriction on the instantiated term's
  citations is re-established by `hasType_strengthen` from H4 read off the term (`Ev.CitesOk`) and H2
  under the restriction.
* `subst_types` needs two clauses of H2 that the skeleton's `ProofsWF` leaves out, a proof proving an
  application at its trait's arity and its derive standing at ground arguments. They are
  `Module.ProofsGround`, beside `ProofsWF` rather than in it, because theorem 4b's written proof
  (Completeness.lean's `cut_proof`) is ground only when the derivation it elaborates is. The hypotheses
  in scope must be claims the claim type verifier admits (`ClaimsWF`); the typing preserves both
  checks (`hasType_wf`, `hasType_atArity`).
* `RootWF`, `Module.ReturnsInstWF` and `Module.ReturnsCitesOk` are here so that
  `returnsInstWF_of_returnsWF` derives the instantiated form of H3 Coherence.lean reads from
  `ReturnsWF`, H2 and H4 at the instance. H4 at the instance is a hypothesis in its own right, not a
  consequence of H3 and the base (`Module.ReturnsCitesOk`'s comment has the program).
* `Root.reqCall` is unchanged. The dialect's `inlineMethodAt` (D/TraitOps.cpp:1314-1356) maps a binder
  call's operands onto the method body's parameters with no bridge and bridges only the result with a
  coerce citing nothing, so `Root.reqCall` reads what the stage reads. A premise that the receiver is
  written as the header at `σ` belongs to the theorem that needs it (Coherence.lean's `Reads`), not to
  the reading; `hasType_unique` matches such a premise with the typing.
-/

namespace Trait

/-! ## Functions on terms -/

mutual
  /-- The term contains a `trait.allege`. -/
  def Ev.hasAllege : Ev -> Bool
    | .hyp _ => false
    | .allege _ => true
    | .derive _ _ es => hasAllegeList es
    | .project e _ => e.hasAllege
    | .reqCall e _ _ fs => e.hasAllege || hasAllegeList fs
    | .witness _ _ => false
    | .projRes _ _ es _ _ => hasAllegeList es
    | .refl _ => false
    | .compose es _ _ => hasAllegeList es
    | .coerce e es _ => e.hasAllege || hasAllegeList es
  def hasAllegeList : List Ev -> Bool
    | [] => false
    | e :: es => e.hasAllege || hasAllegeList es
end

/-- Allege-free: no `trait.allege` anywhere in the term. -/
def Ev.allegeFree (e : Ev) : Prop := e.hasAllege = false

mutual
  /-- The proofs the term cites, through `trait.witness`. -/
  def Ev.cites : Ev -> List ProofId
    | .hyp _ => []
    | .allege _ => []
    | .derive _ _ es => citesList es
    | .project e _ => e.cites
    | .reqCall e _ _ fs => e.cites ++ citesList fs
    | .witness (.proof p) _ => [p]
    | .witness (.impl _) _ => []
    | .projRes _ _ es _ _ => citesList es
    | .refl _ => []
    | .compose es _ _ => citesList es
    | .coerce e es _ => e.cites ++ citesList es
  def citesList : List Ev -> List ProofId
    | [] => []
    | e :: es => e.cites ++ citesList es
end

mutual
  /-- Some subterm, the term itself included, satisfies `f`. -/
  def Ev.any (f : Ev -> Bool) : Ev -> Bool
    | .hyp k => f (.hyp k)
    | .allege c => f (.allege c)
    | .derive i σ es => f (.derive i σ es) || anyList f es
    | .project e k => f (.project e k) || e.any f
    | .reqCall e b us fs => f (.reqCall e b us fs) || e.any f || anyList f fs
    | .witness c P => f (.witness c P)
    | .projRes i σ es a gs => f (.projRes i σ es a gs) || anyList f es
    | .refl a => f (.refl a)
    | .compose es l r => f (.compose es l r) || anyList f es
    | .coerce e es c => f (.coerce e es c) || e.any f || anyList f es
  def anyList (f : Ev -> Bool) : List Ev -> Bool
    | [] => false
    | e :: es => e.any f || anyList f es
end

/-- The term uses an operation: some subterm satisfies `f`. -/
def Ev.uses (e : Ev) (f : Ev -> Bool) : Prop := e.any f = true

mutual
  /-- Type substitution into every type position of a term: the clone rule, every claim predicate and
  every `#trait.impl_args` receiving the variable bindings alone (D/Specialization.cpp:106-128). -/
  def Ev.substTy (σ : Subst) : Ev -> Ev
    | .hyp k => .hyp k
    | .allege c => .allege (c.subst σ)
    | .derive i τ es => .derive i (substList σ τ) (substTyList σ es)
    | .project e k => .project (e.substTy σ) k
    | .reqCall e b us fs => .reqCall (e.substTy σ) b (substList σ us) (substTyList σ fs)
    | .witness c P => .witness c (P.subst σ)
    | .projRes i τ es a gs => .projRes i (substList σ τ) (substTyList σ es) a (substList σ gs)
    | .refl a => .refl (a.subst σ)
    | .compose es l r => .compose (substTyList σ es) (l.subst σ) (r.subst σ)
    | .coerce e es c => .coerce (e.substTy σ) (substTyList σ es) (c.subst σ)
  def substTyList (σ : Subst) : List Ev -> List Ev
    | [] => []
    | e :: es => e.substTy σ :: substTyList σ es
end

mutual
  /-- Hypothesis substitution: `^k` becomes `fs k`; an instance's evidence parameters take the caller's
  proofs (D/Passes.cpp:1026-1087). -/
  def Ev.substHyp (fs : Nat -> Ev) : Ev -> Ev
    | .hyp k => fs k
    | .allege c => .allege c
    | .derive i σ es => .derive i σ (substHypList fs es)
    | .project e k => .project (e.substHyp fs) k
    | .reqCall e b us gs => .reqCall (e.substHyp fs) b us (substHypList fs gs)
    | .witness c P => .witness c P
    | .projRes i σ es a gs => .projRes i σ (substHypList fs es) a gs
    | .refl a => .refl a
    | .compose es l r => .compose (substHypList fs es) l r
    | .coerce e es c => .coerce (e.substHyp fs) (substHypList fs es) c
  def substHypList (fs : Nat -> Ev) : List Ev -> List Ev
    | [] => []
    | e :: es => e.substHyp fs :: substHypList fs es
end

/-- The operands a derive supplies for the where entries, as a hypothesis substitution. -/
def hypsOf (ds : List Ev) : Nat -> Ev := fun j => ds.getD j default

/-! ## The proof table (card 6; `ProofOp::getDerive`, D/TraitOps.cpp:2021-2062) -/

/-- What a proof body returns, read through its bridging coerces: a derive (a root), an unconditional
impl, or a witness of another proof (a cast). -/
def Ev.returned : Ev -> Option (Sum (ImplId × List Ty × List Ev) ProofId)
  | .derive i σ es => some (.inl (i, σ, es))
  | .witness (.impl i) _ => some (.inl (i, [], []))
  | .witness (.proof q) _ => some (.inr q)
  | .coerce e _ _ => e.returned
  | _ => none

/-- The derive a proof's chain of casts ends at, within `fuel` proofs. -/
def Module.proofDeriveAux (M : Module) : Nat -> ProofId -> Option (ImplId × List Ty × List Ev)
  | 0, _ => none
  | fuel+1, p =>
    match M.proofs[p]? with
    | none => none
    | some (_, body) =>
      match body.returned with
      | some (.inl d) => some d
      | some (.inr q) => M.proofDeriveAux fuel q
      | none => none

/-- `Pi(p)`'s derive: the impl, its arguments and its operands, read off the proof's own derive or, for
a cast, its root's (D/TraitOps.cpp:2021-2062). A chain longer than the proof table has a cycle and reads
as nothing. -/
def Module.proofDerive (M : Module) (p : ProofId) : Option (ImplId × List Ty × List Ev) :=
  M.proofDeriveAux (M.proofs.length + 1) p

/-! ## Typing (SPEC.md 5.1, one rule per card) -/

/-- `M; G |- e : C`. `G` is an index because PROJECT-WHERE types the cited proof's operand at `[]`. The
results of `derive`, `projReq`, `reqCall`, `projRes`, `refl` and `compose` are unproven (F1). There is
no rule for `allege`: it is the cut (theorem 4b). The value arm of `coerce` is `Entails L E a b` alone
(theorem 7). Premises matched modulo citation take their citations as a function `πs` of the position,
which is how an inductive proposition states "some citation" without nesting an existential. -/
inductive HasType (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) : List Claim -> Ev -> Claim -> Prop where
  /-- (HYP): the block argument `^k`. -/
  | hyp     : (hk : k < Γ.length) -> HasType L M Ok Γ (.hyp k) Γ[k]
  /-- (DERIVE), card 2 (D/TraitOps.cpp:2413-2431). -/
  | derive  : (hi : i < M.sig.impls.length) -> σ.length = M.sig.impls[i].arity -> Ok i (Subst.ofList σ) ->
              es.length = M.sig.impls[i].wheres.length -> (πs : Nat -> Option Cite) ->
              (∀ j, j < es.length -> HasType L M Ok Γ (es.getD j default)
                  ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), πs j⟩) ->
              HasType L M Ok Γ (.derive i σ es) ⟨M.sig.impls[i].header (Subst.ofList σ), none⟩
  /-- (PROJECT-REQ), card 3 (D/TraitTypes.cpp:448-455). -/
  | projReq : HasType L M Ok Γ e ⟨.app t as, π⟩ -> (ht : t < M.sig.traits.length) ->
              (hk : k < M.sig.traits[t].reqs.length) ->
              HasType L M Ok Γ (.project e k) ⟨(M.sig.traits[t].reqs[k]).subst (Subst.ofList as), none⟩
  /-- (PROJECT-WHERE), card 3, deletion D2 (D/TraitTypes.cpp:457-464). -/
  | projWhere : HasType L M Ok Γ e ⟨.app t as, some (.proof p)⟩ -> (ht : t < M.sig.traits.length) ->
              M.proofDerive p = some (i, σ, ds) -> (hj : j < ds.length) ->
              HasType L M Ok [] (ds.getD j default) c ->
              HasType L M Ok Γ (.project e (M.sig.traits[t].reqs.length + j)) c
  /-- (REQ-CALL), SPEC.md 5.1, F4 (D/TraitOps.cpp:2774-2795, 2857-2887). -/
  | reqCall : HasType L M Ok Γ e ⟨.app t as, π⟩ -> (ht : t < M.sig.traits.length) ->
              (hb : b < M.sig.traits[t].binders.length) ->
              us.length = (M.sig.traits[t].binders[b]).labels.length ->
              fs.length = (M.sig.traits[t].binders[b]).premises.length -> (πs : Nat -> Option Cite) ->
              (∀ j, j < fs.length -> HasType L M Ok Γ (fs.getD j default)
                  ⟨((M.sig.traits[t].binders[b]).premises.getD j default).subst
                      ((M.sig.traits[t].binders[b]).callSubst as us), πs j⟩) ->
              HasType L M Ok Γ (.reqCall e b us fs)
                ⟨(M.sig.traits[t].binders[b]).concl.subst ((M.sig.traits[t].binders[b]).callSubst as us), none⟩
  /-- (WIT-APP) of a proof, card 4 (D/TraitTypes.cpp:362-382). -/
  | witProof : (hp : p < M.proofs.length) -> M.proofs[p].1 = P ->
              HasType L M Ok Γ (.witness (.proof p) P) ⟨P, some (.proof p)⟩
  /-- (WIT-APP) of an unconditional impl, card 4 (D/TraitOps.cpp:1107-1113): an impl citation at no
  arguments, under the same restriction `Ok` as a derive's. -/
  | witImpl : (hi : i < M.sig.impls.length) -> M.sig.impls[i].unconditional -> Ok i (Subst.ofList []) ->
              HasType L M Ok Γ (.witness (.impl i) (M.sig.impls[i].header (Subst.ofList [])))
                ⟨M.sig.impls[i].header (Subst.ofList []), some (.impl i)⟩
  /-- (WIT-PROJ), card 4 (D/TraitOps.cpp:2433-2460). -/
  | projRes : (hi : i < M.sig.impls.length) -> σ.length = M.sig.impls[i].arity -> Ok i (Subst.ofList σ) ->
              es.length = M.sig.impls[i].wheres.length -> (πs : Nat -> Option Cite) ->
              (∀ j, j < es.length -> HasType L M Ok Γ (es.getD j default)
                  ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), πs j⟩) ->
              M.sig.impls[i].bindings[a]? = some (some (labels, B)) -> gs.length = labels.length ->
              HasType L M Ok Γ (.projRes i σ es a gs)
                ⟨.eq (.proj M.sig.impls[i].trait (substList (Subst.ofList σ) M.sig.impls[i].head) a gs)
                     (B.subst ((Subst.ofList σ).extend labels gs)), none⟩
  /-- (WIT-REFL), card 4 (D/TraitOps.cpp:2319-2326). -/
  | refl    : HasType L M Ok Γ (.refl a) ⟨.eq a a, none⟩
  /-- (WIT-COMPOSE), card 4, deletion D1 (D/TraitOps.cpp:2337-2373). -/
  | compose : es ≠ [] -> E.length = es.length ->
              (∀ j, j < es.length -> HasType L M Ok Γ (es.getD j default)
                  ⟨.eq (E.getD j default).1 (E.getD j default).2, none⟩) ->
              Entails L E l r -> HasType L M Ok Γ (.compose es l r) ⟨.eq l r, none⟩
  /-- (COERCE) on a claim, card 5 (D/TraitOps.cpp:2607-2662): no citation minted or exchanged. -/
  | coerce  : HasType L M Ok Γ e ⟨P, π⟩ -> E.length = es.length ->
              (∀ j, j < es.length -> HasType L M Ok Γ (es.getD j default)
                  ⟨.eq (E.getD j default).1 (E.getD j default).2, none⟩) ->
              (π' = none ∨ π' = π) -> Claim.WF ⟨P', π'⟩ -> PredEntails L E P P' ->
              HasType L M Ok Γ (.coerce e es ⟨P', π'⟩) ⟨P', π'⟩

/-! ## The standing hypotheses on a module (SPEC.md 6) -/

/-- H2: every proof is typed under `G = []` at the ground application it proves, is allege-free, and
cites only earlier proofs, so that "p cites q" is well founded (F2). -/
def ProofsWF (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) : Prop :=
  ∀ p (hp : p < M.proofs.length),
    (∃ π : Option Cite, HasType L M Ok [] M.proofs[p].2 ⟨M.proofs[p].1, π⟩) ∧ (M.proofs[p].1).ground = true ∧
    (M.proofs[p].2).allegeFree ∧ (∀ q ∈ (M.proofs[p].2).cites, q < p)

/-- G-WF: a proven hypothesis names a proof or an unconditional impl of its own predicate
(`verifyCitation`, D/TraitTypes.cpp:362-382; checked on module-level `func.func` signatures only,
D/Passes.cpp:278-291; F1). -/
def GammaWF (M : Module) (Γ : List Claim) : Prop :=
  ∀ c ∈ Γ, (∀ p, c.cite = some (.proof p) -> ∃ hp : p < M.proofs.length, M.proofs[p].1 = c.pred) ∧
           (∀ i, c.cite = some (.impl i) -> ∃ hi : i < M.sig.impls.length,
              M.sig.impls[i].unconditional ∧ M.sig.impls[i].header (Subst.ofList []) = c.pred)

/-- The term form of H3 (F13): the evidence an impl returns for its `k`-th plain requirement is typed
under the where entries as unproven hypotheses `^0 .. ^(k-1)` at `rho_k[h]`, and the evidence its
method for binder `b` returns is typed under the where entries followed by the binder's premises at the
binder's conclusion, the binder's variables fresh; every such term is allege-free
(`verifyRequirementEvidence`, D/TraitOps.cpp:774-810; S/src/codegen/mlir/generator.rs:928-962). -/
def Module.ReturnsWF (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) : Prop :=
  ∀ i (hi : i < M.sig.impls.length) (ht : M.sig.impls[i].trait < M.sig.traits.length),
    M.sig.impls[i].returns.length = (M.sig.traits[M.sig.impls[i].trait]).reqs.length ∧
    M.sig.impls[i].binderReturns.length = (M.sig.traits[M.sig.impls[i].trait]).binders.length ∧
    (∀ k (hk : k < (M.sig.traits[M.sig.impls[i].trait]).reqs.length),
      (M.sig.impls[i].returns.getD k default).allegeFree ∧
      ∃ π : Option Cite, HasType L M Ok (M.sig.impls[i].wheres.map (⟨·, none⟩)) (M.sig.impls[i].returns.getD k default)
        ⟨((M.sig.traits[M.sig.impls[i].trait]).reqs[k]).subst (Subst.ofList M.sig.impls[i].head), π⟩) ∧
    (∀ b (hb : b < (M.sig.traits[M.sig.impls[i].trait]).binders.length),
      (M.sig.impls[i].binderReturns.getD b default).allegeFree ∧
      ∃ π : Option Cite, HasType L M Ok
        ((M.sig.impls[i].binderHyps ((M.sig.traits[M.sig.impls[i].trait]).binders[b])).map (⟨·, none⟩))
        (M.sig.impls[i].binderReturns.getD b default)
        ⟨((M.sig.traits[M.sig.impls[i].trait]).binders[b]).concl.subst
            (M.sig.impls[i].binderSubst ((M.sig.traits[M.sig.impls[i].trait]).binders[b])), π⟩)

/-! ## The root of a term (SPEC.md 5.4; `evidenceOf`, `getRootOf`, `readEvidence`) -/

/-- `Root M e (I, σ, ds)`: the term reads to the impl `I` at arguments `σ` with operands `ds` for the
where entries. A hypothesis is open, an allegation has no root until the stage replaces it, an
equality term carries none. A projection of a plain requirement reads the impl's return at `σ` with the
operands for its hypotheses; a projection past the plain requirements is the operand itself (D2); a
binder call reads the impl's method body at `σ` and the GAT arguments with the operands followed by the
premises for its hypotheses (beta-A). -/
inductive Root (M : Module) : Ev -> ImplId × List Ty × List Ev -> Prop where
  | derive   : Root M (.derive i σ es) (i, σ, es)
  | witImpl  : Root M (.witness (.impl i) P) (i, [], [])
  | witProof : M.proofs[p]? = some (Q, body) -> Root M body r -> Root M (.witness (.proof p) P) r
  | coerce   : Root M e r -> Root M (.coerce e es c) r
  | projReq  : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
               k < T.reqs.length ->
               Root M (((I.returns.getD k default).substTy (Subst.ofList σ)).substHyp (hypsOf ds)) r ->
               Root M (.project e k) r
  | projWhere : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
               T.reqs.length ≤ k -> Root M (ds.getD (k - T.reqs.length) default) r ->
               Root M (.project e k) r
  | reqCall  : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
               T.binders[b]? = some B ->
               Root M (((I.binderReturns.getD b default).substTy (I.binderCallSubst B σ us)).substHyp
                         (hypsOf (ds ++ fs))) r ->
               Root M (.reqCall e b us fs) r

/-- `root(e) = (I, σ)`: the impl and arguments the term reads to. -/
def root (M : Module) (e : Ev) (i : ImplId) (σ : List Ty) : Prop := ∃ ds, Root M e (i, σ, ds)

/-! ## Inlining proofs: the `let` of the calculus (card 6; SPEC.md 6.4) -/

mutual
  /-- Within `fuel` proof bodies: a witness of a proof becomes the proof's body, a witness of an
  unconditional impl becomes its derive with no operands (F8), a projection past the plain requirements
  of a term that now reveals a derive becomes that derive's operand (D2), and a coerce drops the
  citation it carried. The fuel drops at a witness only: the operands of a revealed derive are
  already inlined. -/
  def inlineAux (M : Module) : Nat -> Ev -> Ev
    | 0, e => e
    | _+1, .hyp k => .hyp k
    | _+1, .allege c => .allege c
    | n+1, .derive i σ es => .derive i σ (inlineListAux M (n+1) es)
    | n+1, .project e k =>
      -- D2 through the inlined operand: a projection past the plain requirements of the derive the
      -- operand now reveals is that derive's operand at the position, read as `Root.projWhere` reads it.
      let e' := inlineAux M (n+1) e
      match e'.returned with
      | some (.inl (i, _, ds)) =>
        match M.sig.impls[i]? with
        | some I =>
          match M.sig.traits[I.trait]? with
          | some T => if T.reqs.length ≤ k then ds.getD (k - T.reqs.length) default else .project e' k
          | none => .project e' k
        | none => .project e' k
      | _ => .project e' k
    | n+1, .reqCall e b us fs => .reqCall (inlineAux M (n+1) e) b us (inlineListAux M (n+1) fs)
    | n+1, .witness (.proof p) P =>
      match M.proofs[p]? with
      | some (_, body) => inlineAux M n body
      | none => .witness (.proof p) P
    | _+1, .witness (.impl i) _ => .derive i [] []
    | n+1, .projRes i σ es a gs => .projRes i σ (inlineListAux M (n+1) es) a gs
    | _+1, .refl a => .refl a
    | n+1, .compose es l r => .compose (inlineListAux M (n+1) es) l r
    | n+1, .coerce e es c => .coerce (inlineAux M (n+1) e) (inlineListAux M (n+1) es) c.strip
  def inlineListAux (M : Module) : Nat -> List Ev -> List Ev
    | _, [] => []
    | n, e :: es => inlineAux M n e :: inlineListAux M n es
end

/-- The term with every proof it cites written in place of its witness. Under H2 a citation chain is
shorter than the proof table, so the fuel is never spent. -/
def Ev.inlineProofs (M : Module) (e : Ev) : Ev := inlineAux M (M.proofs.length + 1) e

/-! ## Predicates on terms used by the independence lemmas (theorem 6) -/

def Ev.isDerive  : Ev -> Bool | .derive .. => true | _ => false
def Ev.isProject : Ev -> Bool | .project .. => true | _ => false
def Ev.isProjRes : Ev -> Bool | .projRes .. => true | _ => false
def Ev.isRefl    : Ev -> Bool | .refl .. => true | _ => false
def Ev.isCoerce  : Ev -> Bool | .coerce .. => true | _ => false
def Ev.isReqCall : Ev -> Bool | .reqCall .. => true | _ => false

/-! ## `elab`: the frontend's `emit` and the stage's proof writer as one function of the derivation (SPEC.md 6.2) -/

/-- The term for a derivation together with the equality leaves its equational part cites. (HYP) is
`^k`; (IMPL) a derive; (REQ) a project; (REQ-A) a binder call; (BIND) a `proj_resolve` witness; (REFL)
a `refl` witness; (SYM)(TRANS)(CONG)(CONG-P)(NORM) one coerce of a `refl` over the leaves; (CONV) a
coerce over the leaves of the argument equalities. A leaf is a hypothesis, a plain or binder requirement
read as an equality, or a binding; (NORM) needs none, the closure carries it. -/
def elabL (L : ConLaw) (S : Sig) : Deriv L S Ok Γ P -> Ev × List Ev
  | .hyp k _ => (.hyp k, [.hyp k])
  | .impl i _ σ _ _ ds =>
      let e := Ev.derive i σ (tabulate _ (fun j hj => (elabL L S (ds j hj)).1)); (e, [e])
  | .req (k := k) d _ _ => let e := Ev.project (elabL L S d).1 k; (e, [e])
  | .reqA (b := b) d _ _ us _ prem =>
      let e := Ev.reqCall (elabL L S d).1 b us (tabulate _ (fun j hj => (elabL L S (prem j hj)).1)); (e, [e])
  | .bind (a := a) (gs := gs) i _ σ _ _ ds _ _ =>
      let e := Ev.projRes i σ (tabulate _ (fun j hj => (elabL L S (ds j hj)).1)) a gs; (e, [e])
  | .refl (a := a) => (.refl a, [])
  | .symm (a := a) (b := b) d =>
      let r := elabL L S d; (.coerce (.refl b) r.2 ⟨.eq b a, none⟩, r.2)
  | .trans (a := a) (c := c) d₁ d₂ =>
      let E := (elabL L S d₁).2 ++ (elabL L S d₂).2; (.coerce (.refl a) E ⟨.eq a c, none⟩, E)
  | .cong (c := c) (as := as) (bs := bs) _ ds =>
      let E := (tabulate _ (fun j hj => (elabL L S (ds j hj)).2)).flatten
      (.coerce (.refl (.con c as)) E ⟨.eq (.con c as) (.con c bs), none⟩, E)
  | .congP (t := t) (as := as) (bs := bs) (a := a) (gs := gs) (hs := hs) _ _ ds es =>
      let E := (tabulate _ (fun j hj => (elabL L S (ds j hj)).2)).flatten ++
               (tabulate _ (fun j hj => (elabL L S (es j hj)).2)).flatten
      (.coerce (.refl (.proj t as a gs)) E ⟨.eq (.proj t as a gs) (.proj t bs a hs), none⟩, E)
  | .norm (a := a) (b := b) _ => (.coerce (.refl a) [] ⟨.eq a b, none⟩, [])
  | .conv (t := t) (bs := bs) d _ ds =>
      let E := (tabulate _ (fun j hj => (elabL L S (ds j hj)).2)).flatten
      let e := Ev.coerce (elabL L S d).1 E ⟨.app t bs, none⟩; (e, [e])

/-- The term `elab` writes for a derivation (`elab` is a reserved word in Lean, hence the name). -/
def elabDeriv (L : ConLaw) (M : Module) (d : Deriv L M.sig Ok Γ P) : Ev := (elabL L M.sig d).1

/-! ## Operands read by position -/

/-- An operand read by position is a member of the operand list. -/
theorem getD_mem_of_lt {α : Type} [Inhabited α] (l : List α) {j : Nat} (hj : j < l.length) :
    l.getD j default ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
  exact List.getElem_mem hj

/-- Reading by position below the length is indexing. -/
theorem getD_eq_getElem_of_lt {α : Type} [Inhabited α] (l : List α) {j : Nat} (hj : j < l.length) :
    l.getD j default = l[j] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
  rfl

/-- An allege-free operand list has allege-free members. -/
theorem hasAllegeList_eq_false {es : List Ev} (h : hasAllegeList es = false) :
    ∀ e ∈ es, e.hasAllege = false := by
  induction es with
  | nil => intro e he; nomatch he
  | cons e es ih =>
    simp only [hasAllegeList, Bool.or_eq_false_iff] at h
    intro e' he'
    rcases List.mem_cons.mp he' with rfl | he'
    · exact h.1
    · exact ih h.2 e' he'

/-- A member's citations are among the list's. -/
theorem mem_citesList {es : List Ev} {e : Ev} (he : e ∈ es) {q : ProofId} (hq : q ∈ e.cites) :
    q ∈ citesList es := by
  induction es with
  | nil => nomatch he
  | cons e' es ih =>
    simp only [citesList, List.mem_append]
    rcases List.mem_cons.mp he with rfl | he
    · exact Or.inl hq
    · exact Or.inr (ih he)

/-! ## Reading a proof's derive (card 6; `ProofOp::getDerive`) -/

/-- A body whose chain of casts ends at another proof cites it. -/
theorem returned_inr_cites : ∀ (e : Ev) (q : ProofId), e.returned = some (.inr q) → q ∈ e.cites
  | .witness (.proof p) _, q, h => by
    simp only [Ev.returned, Option.some.injEq, Sum.inr.injEq] at h
    subst h
    simp [Ev.cites]
  | .coerce e _ _, q, h => by
    simp only [Ev.returned] at h
    simp only [Ev.cites, List.mem_append]
    exact Or.inl (returned_inr_cites e q h)
  | .hyp _, _, h | .allege _, _, h | .derive _ _ _, _, h | .project _ _, _, h | .reqCall _ _ _ _, _, h
  | .witness (.impl _) _, _, h | .projRes _ _ _ _ _, _, h | .refl _, _, h | .compose _ _ _, _, h => by
    simp [Ev.returned] at h

/-- The operands of the derive a body's chain of casts ends at are subterms of the body: their
citations are among the body's, and they are allege-free when the body is. -/
theorem returned_inl_sub : ∀ (e : Ev) {i : ImplId} {σ : List Ty} {ds : List Ev},
    e.returned = some (.inl (i, σ, ds)) → ∀ d ∈ ds,
      (∀ q ∈ d.cites, q ∈ e.cites) ∧ (e.hasAllege = false → d.hasAllege = false)
  | .derive _ _ es, _, _, ds, h, d, hd => by
    simp only [Ev.returned, Option.some.injEq, Sum.inl.injEq, Prod.mk.injEq] at h
    obtain ⟨_, _, rfl⟩ := h
    refine ⟨fun q hq => ?_, fun hf => ?_⟩
    · simp only [Ev.cites]; exact mem_citesList hd hq
    · simp only [Ev.hasAllege] at hf; exact hasAllegeList_eq_false hf d hd
  | .witness (.impl _) _, _, _, ds, h, d, hd => by
    simp only [Ev.returned, Option.some.injEq, Sum.inl.injEq, Prod.mk.injEq] at h
    obtain ⟨_, _, rfl⟩ := h
    nomatch hd
  | .coerce e es _, i, σ, ds, h, d, hd => by
    simp only [Ev.returned] at h
    obtain ⟨h₁, h₂⟩ := returned_inl_sub e h d hd
    refine ⟨fun q hq => ?_, fun hf => ?_⟩
    · simp only [Ev.cites, List.mem_append]; exact Or.inl (h₁ q hq)
    · simp only [Ev.hasAllege, Bool.or_eq_false_iff] at hf; exact h₂ hf.1
  | .hyp _, _, _, _, h, _, _ | .allege _, _, _, _, h, _, _ | .project _ _, _, _, _, h, _, _
  | .reqCall _ _ _ _, _, _, _, h, _, _ | .witness (.proof _) _, _, _, _, h, _, _
  | .projRes _ _ _ _ _, _, _, _, h, _, _ | .refl _, _, _, _, h, _, _ | .compose _ _ _, _, _, _, h, _, _ => by
    simp [Ev.returned] at h

/-- Under H2, the derive `proofDeriveAux` reads for `p` is returned by the body of a proof at or
below `p`: each cast step cites a proof, and a cited proof is earlier. -/
theorem proofDeriveAux_spec {L : ConLaw} {M : Module} {Ok : ImplId -> Subst -> Prop}
    (hP : ProofsWF L M Ok) :
    ∀ (fuel p : Nat) {i : ImplId} {σ : List Ty} {ds : List Ev},
      M.proofDeriveAux fuel p = some (i, σ, ds) →
      ∃ q, q ≤ p ∧ ∃ hq : q < M.proofs.length, (M.proofs[q].2).returned = some (.inl (i, σ, ds)) := by
  intro fuel
  induction fuel with
  | zero => intro p i σ ds h; simp [Module.proofDeriveAux] at h
  | succ fuel ih =>
    intro p i σ ds h
    simp only [Module.proofDeriveAux] at h
    cases hpr : M.proofs[p]? with
    | none => rw [hpr] at h; simp at h
    | some pr =>
      obtain ⟨Q, body⟩ := pr
      rw [hpr] at h
      simp only at h
      obtain ⟨hp, hbody⟩ := List.getElem?_eq_some_iff.mp hpr
      cases hret : body.returned with
      | none => rw [hret] at h; simp at h
      | some s =>
        rw [hret] at h
        cases s with
        | inl d =>
          simp only [Option.some.injEq] at h
          subst h
          exact ⟨p, Nat.le_refl p, hp, by rw [hbody]; exact hret⟩
        | inr q =>
          simp only at h
          obtain ⟨q', hq'q, hq', hret'⟩ := ih q h
          have hq_cites : q ∈ (M.proofs[p].2).cites := by
            rw [hbody]; exact returned_inr_cites body q hret
          have hqp : q < p := (hP p hp).2.2.2 q hq_cites
          exact ⟨q', Nat.le_of_lt (Nat.lt_of_le_of_lt hq'q hqp), hq', hret'⟩

/-- `Pi(p)`'s derive, under H2: its operands are returned by a proof at or below `p`. -/
theorem proofDerive_spec {L : ConLaw} {M : Module} {Ok : ImplId -> Subst -> Prop}
    (hP : ProofsWF L M Ok) {p : ProofId} {i : ImplId} {σ : List Ty} {ds : List Ev}
    (h : M.proofDerive p = some (i, σ, ds)) :
    ∃ q, q ≤ p ∧ ∃ hq : q < M.proofs.length, (M.proofs[q].2).returned = some (.inl (i, σ, ds)) :=
  proofDeriveAux_spec hP _ p h

/-- The seeds of a compose or coerce, read off its operands by position, are valid when each operand
is. -/
theorem seeds_valid {L : ConLaw} {S : Sig} {Ok : ImplId -> Subst -> Prop} {Γ : List Pred}
    {E : List (Ty × Ty)} {es : List Ev} (hlen : E.length = es.length)
    (h : ∀ j, j < es.length →
      Nonempty (Deriv L S Ok Γ (.eq (E.getD j default).1 (E.getD j default).2))) :
    ∀ l r, (l, r) ∈ E → Nonempty (Deriv L S Ok Γ (.eq l r)) := by
  intro l r hlr
  obtain ⟨j, hj, hEj⟩ := List.mem_iff_getElem.mp hlr
  have := h j (hlen ▸ hj)
  rwa [getD_eq_getElem_of_lt E hj, hEj] at this

/-! ## Theorem 3 for any impl restriction (SPEC.md 6.1) -/

/-- The proof citations the hypotheses carry are below `n`. -/
def ClaimsBelow (n : Nat) (Γ : List Claim) : Prop :=
  ∀ c ∈ Γ, ∀ p, c.cite = some (.proof p) → p < n

/-- Theorem 3 with a bound: an allege-free term whose witnesses all cite proofs below `n`, under
hypotheses whose citations are below `n`, has a derivation citing the same impls, and its claim's
citation (if a proof) is below `n`. WIT-APP of a proof is not a sub-derivation of the term's typing (the
body is typed on its own in the proof table), so the proof runs an outer strong induction on `n`,
which H2's clause "p cites q only for q < p" makes descend at every `witProof`, and an inner
induction on the typing derivation. The bound on the claim's citation is what PROJECT-WHERE needs:
its operand is read off the cited proof's derive, so its citations are below that proof. -/
theorem hasType_deriv (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hP : ProofsWF L M Ok)
    (n : Nat) :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c →
      e.allegeFree → (∀ q ∈ e.cites, q < n) → ClaimsBelow n Γ →
      Nonempty (Deriv L M.sig Ok (Γ.map Claim.pred) c.pred) ∧
        (∀ p, c.cite = some (.proof p) → p < n) := by
  induction n using Nat.strongRecOn with
  | _ n ihn =>
  intro Γ e c ht
  induction ht with
  | hyp hk =>
    intro _ _ hΓ
    refine ⟨⟨?_⟩, fun p hp => hΓ _ (List.getElem_mem hk) p hp⟩
    have hk' := Nat.lt_of_lt_of_eq hk (List.length_map Claim.pred).symm
    have d := Deriv.hyp (L := L) (S := M.sig) (Ok := Ok) _ hk'
    rw [List.getElem_map] at d
    exact d
  | derive hi hσ ok hes πs hprem ih =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    simp only [Ev.cites] at hc
    refine ⟨⟨Deriv.impl _ hi _ hσ ok fun j hj => ?_⟩, fun p hp => nomatch hp⟩
    have hj' := Nat.lt_of_lt_of_eq hj hes.symm
    have hmem := getD_mem_of_lt _ hj'
    exact Classical.choice (ih j hj' (hasAllegeList_eq_false he _ hmem)
      (fun q hq => hc q (mem_citesList hmem hq)) hΓ).1
  | projReq h ht hk ih =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    simp only [Ev.cites] at hc
    obtain ⟨⟨d⟩, _⟩ := ih he hc hΓ
    exact ⟨⟨.req d ht hk⟩, fun p hp => nomatch hp⟩
  | projWhere h ht hpd hj hc ih_e ih_c =>
    intro he hcites hΓ
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    simp only [Ev.cites] at hcites
    obtain ⟨_, hpn⟩ := ih_e he hcites hΓ
    have hp : _ < n := hpn _ rfl
    -- the operand is returned by a proof at or below the cited one, so H2 bounds its citations
    -- and makes it allege-free
    obtain ⟨q, hqp, hq, hret⟩ := proofDerive_spec hP hpd
    obtain ⟨_, _, hfree, hcq⟩ := hP q hq
    obtain ⟨hsub, hfr⟩ := returned_inl_sub _ hret _ (getD_mem_of_lt _ hj)
    obtain ⟨d, hcd⟩ := ih_c (hfr hfree)
      (fun q' hq' => Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le (hcq q' (hsub q' hq')) hqp) (Nat.le_of_lt hp))
      (fun c hc => nomatch hc)
    exact ⟨d.elim fun d => ⟨d.weakenNil⟩, hcd⟩
  | reqCall h ht hb hus hfs πs hprem ih_e ih_f =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.cites, List.mem_append] at hc
    obtain ⟨⟨d⟩, _⟩ := ih_e he.1 (fun q hq => hc q (Or.inl hq)) hΓ
    refine ⟨⟨.reqA d ht hb _ hus fun j hj => ?_⟩, fun p hp => nomatch hp⟩
    have hj' := Nat.lt_of_lt_of_eq hj hfs.symm
    have hmem := getD_mem_of_lt _ hj'
    exact Classical.choice (ih_f j hj' (hasAllegeList_eq_false he.2 _ hmem)
      (fun q hq => hc q (Or.inr (mem_citesList hmem hq))) hΓ).1
  | witProof hp hPeq =>
    intro _ hc _
    subst hPeq
    simp only [Ev.cites, List.mem_singleton] at hc
    have hpn := hc _ rfl
    obtain ⟨⟨π, hbody⟩, _, hfree, hcites⟩ := hP _ hp
    have hb := (ihn _ hpn hbody hfree hcites (fun c hc => nomatch hc)).1
    refine ⟨hb.elim fun d => ⟨d.weakenNil⟩, fun p' hp' => ?_⟩
    simp only [Option.some.injEq, Cite.proof.injEq] at hp'
    subst hp'
    exact hpn
  | witImpl hi hunc ok =>
    intro _ _ _
    refine ⟨⟨Deriv.impl _ hi [] ?_ ok fun j hj => ?_⟩, fun p hp => nomatch hp⟩
    · exact hunc.1.symm
    · rw [hunc.2] at hj; exact absurd hj (Nat.not_lt_zero j)
  | projRes hi hσ ok hes πs hprem hb hg ih =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    simp only [Ev.cites] at hc
    refine ⟨⟨Deriv.bind _ hi _ hσ ok (fun j hj => ?_) hb hg⟩, fun p hp => nomatch hp⟩
    have hj' := Nat.lt_of_lt_of_eq hj hes.symm
    have hmem := getD_mem_of_lt _ hj'
    exact Classical.choice (ih j hj' (hasAllegeList_eq_false he _ hmem)
      (fun q hq => hc q (mem_citesList hmem hq)) hΓ).1
  | refl => intro _ _ _; exact ⟨⟨.refl⟩, fun p hp => nomatch hp⟩
  | compose hne hlen hprem hent ih =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    simp only [Ev.cites] at hc
    refine ⟨entails_valid L M.sig Ok _ _ (seeds_valid hlen fun j hj => ?_) hent, fun p hp => nomatch hp⟩
    have hmem := getD_mem_of_lt _ hj
    exact (ih j hj (hasAllegeList_eq_false he _ hmem) (fun q hq => hc q (mem_citesList hmem hq)) hΓ).1
  | coerce h hlen hprem hπ hwf hpe ih_e ih_s =>
    intro he hc hΓ
    simp only [Ev.allegeFree, Ev.hasAllege, Bool.or_eq_false_iff] at he
    simp only [Ev.cites, List.mem_append] at hc
    obtain ⟨hd, hcite⟩ := ih_e he.1 (fun q hq => hc q (Or.inl hq)) hΓ
    refine ⟨predEntails_valid L M.sig Ok _ _ (seeds_valid hlen fun j hj => ?_) hpe hd, ?_⟩
    · have hmem := getD_mem_of_lt _ hj
      exact (ih_s j hj (hasAllegeList_eq_false he.2 _ hmem)
        (fun q hq => hc q (Or.inr (mem_citesList hmem hq))) hΓ).1
    · intro p hp
      rcases hπ with rfl | rfl
      · nomatch hp
      · exact hcite p hp

/-- Every finite list of proof indices is bounded. -/
theorem exists_bound (l : List Nat) : ∃ n, ∀ q ∈ l, q < n := by
  induction l with
  | nil => exact ⟨0, fun q h => nomatch h⟩
  | cons q l ih =>
    obtain ⟨n, hn⟩ := ih
    refine ⟨max (q + 1) n, fun q' hq' => ?_⟩
    rcases List.mem_cons.mp hq' with rfl | hq'
    · exact Nat.lt_of_lt_of_le (Nat.lt_succ_self q') (Nat.le_max_left _ _)
    · exact Nat.lt_of_lt_of_le (hn q' hq') (Nat.le_max_right _ _)

/-- The proof citations of a finite context are bounded. -/
theorem exists_claimsBelow (Γ : List Claim) : ∃ n, ClaimsBelow n Γ := by
  obtain ⟨n, hn⟩ := exists_bound
    (Γ.filterMap fun c => match c.cite with | some (.proof p) => some p | _ => none)
  refine ⟨n, fun c hc p hp => hn p (List.mem_filterMap.mpr ⟨c, hc, ?_⟩)⟩
  simp [hp]

set_option linter.unusedVariables false in
/-- Theorem 3 for any impl restriction `Ok` (SPEC.md 6.1): an allege-free term's claim is derivable
under the predicates of its hypotheses, citing the same impls. Soundness.lean's `soundness` is this
at `AnyImpl`; Coherence.lean needs it at `DenL`. `hS` and `hΓ` are the specification's hypotheses;
the proof needs neither: soundness speaks of the predicate of a claim, never of its citation
(PROJECT-WHERE concludes the operand's own closed type, and a proven hypothesis is used by (HYP)
alone). G-WF matters where the root of a term is read (theorem 5). -/
theorem hasType_valid (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) {Γ : List Claim} {e : Ev}
    {c : Claim} (hS : M.sig.DeclWF) (hP : ProofsWF L M Ok) (hΓ : GammaWF M Γ) (he : e.allegeFree)
    (ht : HasType L M Ok Γ e c) :
    Nonempty (Deriv L M.sig Ok (Γ.map Claim.pred) c.pred) := by
  obtain ⟨n₁, h₁⟩ := exists_bound e.cites
  obtain ⟨n₂, h₂⟩ := exists_claimsBelow Γ
  exact (hasType_deriv L M hP (max n₁ n₂) ht he
    (fun q hq => Nat.lt_of_lt_of_le (h₁ q hq) (Nat.le_max_left _ _))
    (fun c hc p hp => Nat.lt_of_lt_of_le (h₂ c hc p hp) (Nat.le_max_right _ _))).1

/-! ## Substitution: composition, agreement on a type's variables, ground types -/

theorem substList_length (σ : Subst) : ∀ as : List Ty, (substList σ as).length = as.length
  | [] => rfl
  | _ :: as => by simp [substList, substList_length σ as]

theorem substList_getElem? (σ : Subst) :
    ∀ (as : List Ty) (v : Nat), (substList σ as)[v]? = (as[v]?).map (Ty.subst σ)
  | [], v => by simp [substList]
  | a :: as, 0 => by simp [substList]
  | a :: as, v+1 => by simp [substList, substList_getElem? σ as v]

/-- One pass of `τ` and then one pass of `σ`, as one substitution: `τ`'s answer with `σ` applied to it,
or `σ`'s own answer where `τ` has none. -/
def Subst.comp (σ τ : Subst) : Subst := fun v =>
  match τ v with
  | some u => some (u.subst σ)
  | none => σ v

mutual
  /-- Two passes are one pass of the composition. -/
  theorem Ty.subst_subst (σ τ : Subst) : ∀ t : Ty, (t.subst τ).subst σ = t.subst (Subst.comp σ τ)
    | .var v => by
      simp only [Ty.subst]
      cases h : τ v with
      | none => simp [Subst.comp, h, Ty.subst]
      | some u => simp [Subst.comp, h]
    | .con c as => by simp only [Ty.subst, substList_subst σ τ as]
    | .proj t as a gs => by simp only [Ty.subst, substList_subst σ τ as, substList_subst σ τ gs]
  theorem substList_subst (σ τ : Subst) :
      ∀ as : List Ty, substList σ (substList τ as) = substList (Subst.comp σ τ) as
    | [] => rfl
    | a :: as => by simp only [substList, Ty.subst_subst σ τ a, substList_subst σ τ as]
end

theorem Pred.subst_subst (σ τ : Subst) (P : Pred) : (P.subst τ).subst σ = P.subst (Subst.comp σ τ) := by
  cases P with
  | app t as => simp only [Pred.subst, substList_subst]
  | eq a b => simp only [Pred.subst, Ty.subst_subst]

mutual
  /-- Substitutions that agree on the variables of a type substitute alike into it. -/
  theorem Ty.subst_congr (σ τ : Subst) :
      ∀ t : Ty, (∀ v, t.mentions v = true -> σ v = τ v) -> t.subst σ = t.subst τ
    | .var v, h => by simp only [Ty.subst]; rw [h v (by simp [Ty.mentions])]
    | .con c as, h => by
      simp only [Ty.subst]
      rw [substList_congr σ τ as (fun v hv => h v (by simpa [Ty.mentions] using hv))]
    | .proj t as a gs, h => by
      simp only [Ty.subst]
      rw [substList_congr σ τ as (fun v hv => h v (by simp [Ty.mentions, hv])),
        substList_congr σ τ gs (fun v hv => h v (by simp [Ty.mentions, hv]))]
  theorem substList_congr (σ τ : Subst) :
      ∀ as : List Ty, (∀ v, mentionsList v as = true -> σ v = τ v) -> substList σ as = substList τ as
    | [], _ => rfl
    | a :: as, h => by
      simp only [substList]
      rw [Ty.subst_congr σ τ a (fun v hv => h v (by simp [mentionsList, hv])),
        substList_congr σ τ as (fun v hv => h v (by simp [mentionsList, hv]))]
end

theorem Pred.subst_congr (σ τ : Subst) (P : Pred) (h : ∀ v, P.mentions v = true -> σ v = τ v) :
    P.subst σ = P.subst τ := by
  cases P with
  | app t as => simp only [Pred.subst]; rw [substList_congr σ τ as (fun v hv => h v (by simp [Pred.mentions, hv]))]
  | eq a b =>
    simp only [Pred.subst]
    rw [Ty.subst_congr σ τ a (fun v hv => h v (by simp [Pred.mentions, hv])),
      Ty.subst_congr σ τ b (fun v hv => h v (by simp [Pred.mentions, hv]))]

mutual
  /-- A variable a type mentions satisfies what all its variables satisfy. -/
  theorem Ty.allVars_mentions (p : Var -> Bool) :
      ∀ t : Ty, t.allVars p = true -> ∀ v, t.mentions v = true -> p v = true
    | .var i, h, v, hv => by
      simp only [Ty.allVars] at h
      simp only [Ty.mentions, beq_iff_eq] at hv
      subst hv; exact h
    | .con c as, h, v, hv =>
      allVarsList_mentions p as (by simpa [Ty.allVars] using h) v (by simpa [Ty.mentions] using hv)
    | .proj t as a gs, h, v, hv => by
      simp only [Ty.allVars, Bool.and_eq_true] at h
      simp only [Ty.mentions, Bool.or_eq_true] at hv
      rcases hv with hv | hv
      · exact allVarsList_mentions p as h.1 v hv
      · exact allVarsList_mentions p gs h.2 v hv
  theorem allVarsList_mentions (p : Var -> Bool) :
      ∀ as : List Ty, allVarsList p as = true -> ∀ v, mentionsList v as = true -> p v = true
    | [], _, v, hv => by simp [mentionsList] at hv
    | a :: as, h, v, hv => by
      simp only [allVarsList, Bool.and_eq_true] at h
      simp only [mentionsList, Bool.or_eq_true] at hv
      rcases hv with hv | hv
      · exact Ty.allVars_mentions p a h.1 v hv
      · exact allVarsList_mentions p as h.2 v hv
end

theorem Pred.allVars_mentions (p : Var -> Bool) (P : Pred) (h : P.allVars p = true) :
    ∀ v, P.mentions v = true -> p v = true := by
  intro v hv
  cases P with
  | app t as => exact allVarsList_mentions p as h v hv
  | eq a b =>
    simp only [Pred.allVars, Bool.and_eq_true] at h
    simp only [Pred.mentions, Bool.or_eq_true] at hv
    rcases hv with hv | hv
    · exact Ty.allVars_mentions p a h.1 v hv
    · exact Ty.allVars_mentions p b h.2 v hv

theorem allVarsList_of_forall (p : Var -> Bool) :
    ∀ as : List Ty, (∀ a ∈ as, a.allVars p = true) -> allVarsList p as = true
  | [], _ => rfl
  | a :: as, h => by
    simp only [allVarsList, Bool.and_eq_true]
    exact ⟨h a (by simp), allVarsList_of_forall p as (fun a' ha' => h a' (by simp [ha']))⟩

theorem groundList_of_forall : ∀ as : List Ty, (∀ a ∈ as, a.ground = true) -> groundList as = true
  | [], _ => rfl
  | a :: as, h => by
    simp only [groundList, Bool.and_eq_true]
    exact ⟨h a (by simp), groundList_of_forall as (fun a' ha' => h a' (by simp [ha']))⟩

theorem groundList_getElem {as : List Ty} (h : groundList as = true) {i : Nat} (hi : i < as.length) :
    as[i].ground = true := by
  rw [groundList_eq_all, List.all_eq_true] at h
  exact h _ (List.getElem_mem hi)

mutual
  /-- A type with no variable below zero has none. -/
  theorem Ty.ground_of_varsBelow_zero : ∀ t : Ty, t.varsBelow 0 = true -> t.ground = true
    | .var i, h => by simp [Ty.varsBelow, Ty.allVars] at h
    | .con c as, h => groundList_of_varsBelow_zero as (by simpa [Ty.varsBelow, Ty.allVars] using h)
    | .proj t as a gs, h => by
      simp only [Ty.varsBelow, Ty.allVars, Bool.and_eq_true] at h
      simp only [Ty.ground, Bool.and_eq_true]
      exact ⟨groundList_of_varsBelow_zero as h.1, groundList_of_varsBelow_zero gs h.2⟩
  theorem groundList_of_varsBelow_zero :
      ∀ as : List Ty, allVarsList (fun v => decide (v < 0)) as = true -> groundList as = true
    | [], _ => rfl
    | a :: as, h => by
      simp only [allVarsList, Bool.and_eq_true] at h
      simp only [groundList, Bool.and_eq_true]
      exact ⟨Ty.ground_of_varsBelow_zero a h.1, groundList_of_varsBelow_zero as h.2⟩
end

mutual
  /-- A ground type is fixed by every substitution. -/
  theorem Ty.subst_of_ground (σ : Subst) : ∀ t : Ty, t.ground = true -> t.subst σ = t
    | .var _, h => by simp [Ty.ground] at h
    | .con c as, h => by simp only [Ty.subst]; rw [substList_of_ground σ as (by simpa [Ty.ground] using h)]
    | .proj t as a gs, h => by
      simp only [Ty.ground, Bool.and_eq_true] at h
      simp only [Ty.subst]; rw [substList_of_ground σ as h.1, substList_of_ground σ gs h.2]
  theorem substList_of_ground (σ : Subst) : ∀ as : List Ty, groundList as = true -> substList σ as = as
    | [], _ => rfl
    | a :: as, h => by
      simp only [groundList, Bool.and_eq_true] at h
      simp only [substList]; rw [Ty.subst_of_ground σ a h.1, substList_of_ground σ as h.2]
end

theorem Pred.subst_of_ground (σ : Subst) (P : Pred) (h : P.ground = true) : P.subst σ = P := by
  cases P with
  | app t as => simp only [Pred.subst]; rw [substList_of_ground σ as (by simpa [Pred.ground] using h)]
  | eq a b =>
    simp only [Pred.ground, Bool.and_eq_true] at h
    simp only [Pred.subst]; rw [Ty.subst_of_ground σ a h.1, Ty.subst_of_ground σ b h.2]

mutual
  /-- Substituting ground types for every variable of a type gives a ground type. -/
  theorem Ty.ground_subst (σ : Subst) :
      ∀ t : Ty, (∀ v, t.mentions v = true -> ∃ d, σ v = some d ∧ d.ground = true) -> (t.subst σ).ground = true
    | .var v, h => by
      obtain ⟨d, hd, hg⟩ := h v (by simp [Ty.mentions])
      simp [Ty.subst, hd, hg]
    | .con c as, h => by
      simp only [Ty.subst, Ty.ground]
      exact groundList_subst σ as (fun v hv => h v (by simpa [Ty.mentions] using hv))
    | .proj t as a gs, h => by
      simp only [Ty.subst, Ty.ground, Bool.and_eq_true]
      exact ⟨groundList_subst σ as (fun v hv => h v (by simp [Ty.mentions, hv])),
        groundList_subst σ gs (fun v hv => h v (by simp [Ty.mentions, hv]))⟩
  theorem groundList_subst (σ : Subst) :
      ∀ as : List Ty, (∀ v, mentionsList v as = true -> ∃ d, σ v = some d ∧ d.ground = true) ->
        groundList (substList σ as) = true
    | [], _ => rfl
    | a :: as, h => by
      simp only [substList, groundList, Bool.and_eq_true]
      exact ⟨Ty.ground_subst σ a (fun v hv => h v (by simp [mentionsList, hv])),
        groundList_subst σ as (fun v hv => h v (by simp [mentionsList, hv]))⟩
end

theorem Pred.ground_subst (σ : Subst) (P : Pred)
    (h : ∀ v, P.mentions v = true -> ∃ d, σ v = some d ∧ d.ground = true) : (P.subst σ).ground = true := by
  cases P with
  | app t as => exact groundList_subst σ as (fun v hv => h v (by simp [Pred.mentions, hv]))
  | eq a b =>
    simp only [Pred.subst, Pred.ground, Bool.and_eq_true]
    exact ⟨Ty.ground_subst σ a (fun v hv => h v (by simp [Pred.mentions, hv])),
      Ty.ground_subst σ b (fun v hv => h v (by simp [Pred.mentions, hv]))⟩

/-! ## Substitution composed with a positional substitution, and with a binding's or a binder's -/

/-- Below `|τ|`, substituting `τ` and then `σ` is substituting `τ[σ]`. -/
theorem comp_ofList (σ : Subst) (τ : List Ty) (v : Var) (hv : v < τ.length) :
    Subst.comp σ (Subst.ofList τ) v = Subst.ofList (substList σ τ) v := by
  simp only [Subst.comp, Subst.ofList, substList_getElem?, List.getElem?_eq_getElem hv, Option.map_some]

/-- Substituting the header's arguments for the trait's positions and then the impl's arguments for its
own labels is substituting the instantiated header, for a type whose variables are the trait's
positions: `t[h][σ] = t[h[σ]]`. -/
theorem Ty.subst_ofList_subst (σ : Subst) (head : List Ty) (t : Ty) (h : t.varsBelow head.length = true) :
    (t.subst (Subst.ofList head)).subst σ = t.subst (Subst.ofList (substList σ head)) := by
  rw [Ty.subst_subst]
  exact Ty.subst_congr _ _ t (fun v hv =>
    comp_ofList σ head v (by simpa using Ty.allVars_mentions _ t h v hv))

theorem substList_ofList_subst (σ : Subst) (head : List Ty) (as : List Ty)
    (h : allVarsList (fun v => decide (v < head.length)) as = true) :
    substList σ (substList (Subst.ofList head) as) = substList (Subst.ofList (substList σ head)) as := by
  rw [substList_subst]
  exact substList_congr _ _ as (fun v hv =>
    comp_ofList σ head v (by simpa using allVarsList_mentions _ as h v hv))

theorem Pred.subst_ofList_subst (σ : Subst) (head : List Ty) (P : Pred)
    (h : P.varsBelow head.length = true) :
    (P.subst (Subst.ofList head)).subst σ = P.subst (Subst.ofList (substList σ head)) := by
  rw [Pred.subst_subst]
  exact Pred.subst_congr _ _ P (fun v hv =>
    comp_ofList σ head v (by simpa using Pred.allVars_mentions _ P h v hv))

/-- The first label equal to `v` is found at one position whether the arguments are substituted or not. -/
theorem find_zip_substList (σ : Subst) : ∀ (labels : List Var) (us : List Ty) (v : Var),
    (labels.zip (substList σ us)).find? (·.1 == v) =
      ((labels.zip us).find? (·.1 == v)).map (fun p => (p.1, p.2.subst σ))
  | [], us, v => by cases us <;> simp
  | l :: labels, [], v => by simp [substList]
  | l :: labels, u :: us, v => by
    simp only [substList, List.zip_cons_cons, List.find?_cons]
    cases (l == v) with
    | true => simp
    | false => simpa using find_zip_substList σ labels us v

/-- Two extensions agree at a label when their underlying substitutions agree there whenever no
extended label is found. -/
theorem extend_agree (τ τ' : Subst) (labels : List Var) (us : List Ty) (v : Var)
    (h : (labels.zip us).find? (·.1 == v) = none -> τ v = τ' v) :
    τ.extend labels us v = τ'.extend labels us v := by
  simp only [Subst.extend]
  cases hf : (labels.zip us).find? (·.1 == v) with
  | none => exact h hf
  | some p => rfl

/-- Substituting an extension and then `σ` is substituting the extension at the substituted arguments. -/
theorem comp_extend (σ τ : Subst) (labels : List Var) (us : List Ty) (v : Var) :
    Subst.comp σ (τ.extend labels us) v = (Subst.comp σ τ).extend labels (substList σ us) v := by
  simp only [Subst.comp, Subst.extend, find_zip_substList]
  cases hf : (labels.zip us).find? (·.1 == v) with
  | none => rfl
  | some p => obtain ⟨l, g⟩ := p; rfl

/-- When the arguments are as many as the labels, a label not found is not among the labels. -/
theorem find_zip_none_of_lengths : ∀ (labels : List Var) (us : List Ty) (v : Var),
    us.length = labels.length -> (labels.zip us).find? (·.1 == v) = none -> labels.contains v = false
  | [], _, v, _, _ => rfl
  | l :: labels, [], v, h, _ => by simp at h
  | l :: labels, u :: us, v, h, hf => by
    simp only [List.zip_cons_cons, List.find?_cons] at hf
    simp only [List.length_cons, Nat.succ.injEq] at h
    cases hlv : (l == v) with
    | true => simp [hlv] at hf
    | false =>
      simp only [hlv] at hf
      rw [List.contains_cons, Bool.or_eq_false_iff]
      refine ⟨?_, find_zip_none_of_lengths labels us v h hf⟩
      cases hvl : (v == l) with
      | false => rfl
      | true => simp only [beq_iff_eq] at hvl; subst hvl; simp at hlv

/-- Instantiating a type over the positions `0 .. |as|-1` and the labels `labels` at `as, us`, then
substituting `σ`, is instantiating it at `as[σ], us[σ]`: the clone rule commutes with a binding's
or a binder's substitution. -/
theorem Ty.subst_extend_subst (σ : Subst) (as us : List Ty) (labels : List Var)
    (hlen : us.length = labels.length) (t : Ty)
    (h : t.allVars (fun v => decide (v < as.length) || labels.contains v) = true) :
    (t.subst ((Subst.ofList as).extend labels us)).subst σ =
      t.subst ((Subst.ofList (substList σ as)).extend labels (substList σ us)) := by
  rw [Ty.subst_subst]
  apply Ty.subst_congr
  intro v hv
  rw [comp_extend]
  apply extend_agree
  intro hnone
  rw [find_zip_substList, Option.map_eq_none_iff] at hnone
  have hall := Ty.allVars_mentions _ t h v hv
  rw [find_zip_none_of_lengths labels us v hlen hnone] at hall
  simp only [Bool.or_false, decide_eq_true_eq] at hall
  exact comp_ofList σ as v hall

theorem Pred.subst_extend_subst (σ : Subst) (as us : List Ty) (labels : List Var)
    (hlen : us.length = labels.length) (P : Pred)
    (h : P.allVars (fun v => decide (v < as.length) || labels.contains v) = true) :
    (P.subst ((Subst.ofList as).extend labels us)).subst σ =
      P.subst ((Subst.ofList (substList σ as)).extend labels (substList σ us)) := by
  rw [Pred.subst_subst]
  apply Pred.subst_congr
  intro v hv
  rw [comp_extend]
  apply extend_agree
  intro hnone
  rw [find_zip_substList, Option.map_eq_none_iff] at hnone
  have hall := Pred.allVars_mentions _ P h v hv
  rw [find_zip_none_of_lengths labels us v hlen hnone] at hall
  simp only [Bool.or_false, decide_eq_true_eq] at hall
  exact comp_ofList σ as v hall

/-! ## Substitution into the closure and into terms -/

/-- The ground congruence closure is stable under substitution: every rule is, and (NORM) by H5. -/
theorem entails_subst (L : ConLaw) (σ : Subst) {E : List (Ty × Ty)} {a b : Ty} (h : Entails L E a b) :
    Entails L (E.map fun p => (p.1.subst σ, p.2.subst σ)) (a.subst σ) (b.subst σ) := by
  induction h with
  | ax hab => exact .ax (List.mem_map.mpr ⟨_, hab, rfl⟩)
  | refl => exact .refl
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | con hl _ ih =>
    simp only [Ty.subst]
    refine .con (by simp [substList_length, hl]) (fun i hi => ?_)
    rw [substList_length] at hi
    rw [substList_getD σ _ hi, substList_getD σ _ (hl ▸ hi)]
    exact ih i hi
  | proj hl₁ hl₂ _ _ ih₁ ih₂ =>
    simp only [Ty.subst]
    refine .proj (by simp [substList_length, hl₁]) (by simp [substList_length, hl₂]) (fun i hi => ?_) (fun i hi => ?_)
    · rw [substList_length] at hi
      rw [substList_getD σ _ hi, substList_getD σ _ (hl₁ ▸ hi)]
      exact ih₁ i hi
    · rw [substList_length] at hi
      rw [substList_getD σ _ hi, substList_getD σ _ (hl₂ ▸ hi)]
      exact ih₂ i hi
  | law hlaw => exact .law (L.law_subst σ hlaw)
where
  substList_getD (σ : Subst) (as : List Ty) {i : Nat} (hi : i < as.length) :
      (substList σ as).getD i default = (as.getD i default).subst σ := by
    rw [getD_eq_getElem_of_lt _ (by rw [substList_length]; exact hi), getD_eq_getElem_of_lt _ hi]
    have := substList_getElem? σ as i
    rw [List.getElem?_eq_getElem (by rw [substList_length]; exact hi), List.getElem?_eq_getElem hi] at this
    simpa using this

theorem predEntails_subst (L : ConLaw) (σ : Subst) {E : List (Ty × Ty)} {P Q : Pred} (h : PredEntails L E P Q) :
    PredEntails L (E.map fun p => (p.1.subst σ, p.2.subst σ)) (P.subst σ) (Q.subst σ) := by
  cases h with
  | app hl hall =>
    simp only [Pred.subst]
    refine .app (by simp [substList_length, hl]) (fun i hi => ?_)
    rw [substList_length] at hi
    rw [entails_subst.substList_getD σ _ hi, entails_subst.substList_getD σ _ (hl ▸ hi)]
    exact entails_subst L σ (hall i hi)
  | eq hac hbd => exact .eq (entails_subst L σ hac) (entails_subst L σ hbd)

theorem substTyList_length (σ : Subst) : ∀ es : List Ev, (substTyList σ es).length = es.length
  | [] => rfl
  | _ :: es => by simp [substTyList, substTyList_length σ es]

theorem substTyList_getD (σ : Subst) : ∀ (es : List Ev) {j : Nat}, j < es.length ->
    (substTyList σ es).getD j default = (es.getD j default).substTy σ
  | [], j, hj => absurd hj (Nat.not_lt_zero j)
  | e :: es, 0, _ => by simp [substTyList]
  | e :: es, j+1, hj => by
    simp only [substTyList, List.getD_cons_succ]
    exact substTyList_getD σ es (by simpa using hj)

theorem substHypList_length (fs : Nat -> Ev) : ∀ es : List Ev, (substHypList fs es).length = es.length
  | [] => rfl
  | _ :: es => by simp [substHypList, substHypList_length fs es]

theorem substHypList_getD (fs : Nat -> Ev) : ∀ (es : List Ev) {j : Nat}, j < es.length ->
    (substHypList fs es).getD j default = (es.getD j default).substHyp fs
  | [], j, hj => absurd hj (Nat.not_lt_zero j)
  | e :: es, 0, _ => by simp [substHypList]
  | e :: es, j+1, hj => by
    simp only [substHypList, List.getD_cons_succ]
    exact substHypList_getD fs es (by simpa using hj)

theorem getD_map_of_lt {α β : Type} [Inhabited α] [Inhabited β] (f : α -> β) (l : List α) {j : Nat}
    (hj : j < l.length) : (l.map f).getD j default = f (l.getD j default) := by
  rw [getD_eq_getElem_of_lt _ (by simpa using hj), getD_eq_getElem_of_lt _ hj, List.getElem_map]

mutual
  /-- Type substitution writes no allegation and erases none. -/
  theorem hasAllege_substTy (σ : Subst) : ∀ e : Ev, (e.substTy σ).hasAllege = e.hasAllege
    | .hyp _ => rfl
    | .allege _ => rfl
    | .derive _ _ es => by simp only [Ev.substTy, Ev.hasAllege, hasAllegeList_substTy σ es]
    | .project e _ => by simp only [Ev.substTy, Ev.hasAllege, hasAllege_substTy σ e]
    | .reqCall e _ _ fs => by simp only [Ev.substTy, Ev.hasAllege, hasAllege_substTy σ e, hasAllegeList_substTy σ fs]
    | .witness _ _ => rfl
    | .projRes _ _ es _ _ => by simp only [Ev.substTy, Ev.hasAllege, hasAllegeList_substTy σ es]
    | .refl _ => rfl
    | .compose es _ _ => by simp only [Ev.substTy, Ev.hasAllege, hasAllegeList_substTy σ es]
    | .coerce e es _ => by simp only [Ev.substTy, Ev.hasAllege, hasAllege_substTy σ e, hasAllegeList_substTy σ es]
  theorem hasAllegeList_substTy (σ : Subst) : ∀ es : List Ev, hasAllegeList (substTyList σ es) = hasAllegeList es
    | [] => rfl
    | e :: es => by simp only [substTyList, hasAllegeList, hasAllege_substTy σ e, hasAllegeList_substTy σ es]
end

mutual
  /-- Hypothesis substitution with allege-free replacements keeps a term allege-free. -/
  theorem hasAllege_substHyp (fs : Nat -> Ev) (hfs : ∀ k, (fs k).hasAllege = false) :
      ∀ e : Ev, e.hasAllege = false -> (e.substHyp fs).hasAllege = false
    | .hyp k, _ => hfs k
    | .allege _, h => h
    | .derive _ _ es, h => hasAllegeList_substHyp fs hfs es h
    | .project e _, h => hasAllege_substHyp fs hfs e h
    | .reqCall e _ _ gs, h => by
      simp only [Ev.hasAllege, Ev.substHyp, Bool.or_eq_false_iff] at h ⊢
      exact ⟨hasAllege_substHyp fs hfs e h.1, hasAllegeList_substHyp fs hfs gs h.2⟩
    | .witness _ _, h => h
    | .projRes _ _ es _ _, h => hasAllegeList_substHyp fs hfs es h
    | .refl _, h => h
    | .compose es _ _, h => hasAllegeList_substHyp fs hfs es h
    | .coerce e es _, h => by
      simp only [Ev.hasAllege, Ev.substHyp, Bool.or_eq_false_iff] at h ⊢
      exact ⟨hasAllege_substHyp fs hfs e h.1, hasAllegeList_substHyp fs hfs es h.2⟩
  theorem hasAllegeList_substHyp (fs : Nat -> Ev) (hfs : ∀ k, (fs k).hasAllege = false) :
      ∀ es : List Ev, hasAllegeList es = false -> hasAllegeList (substHypList fs es) = false
    | [], h => h
    | e :: es, h => by
      simp only [hasAllegeList, substHypList, Bool.or_eq_false_iff] at h ⊢
      exact ⟨hasAllege_substHyp fs hfs e h.1, hasAllegeList_substHyp fs hfs es h.2⟩
end

theorem substTyList_ne_nil (σ : Subst) {es : List Ev} (h : es ≠ []) : substTyList σ es ≠ [] := by
  cases es with
  | nil => exact absurd rfl h
  | cons e es => simp [substTyList]

theorem substHypList_ne_nil (fs : Nat -> Ev) {es : List Ev} (h : es ≠ []) : substHypList fs es ≠ [] := by
  cases es with
  | nil => exact absurd rfl h
  | cons e es => simp [substHypList]

/-! ## Claims as the claim type verifier admits them -/

/-- An application names its trait at the trait's arity (D/TraitAttributes.cpp:126-156): the part of a
predicate's well-formedness the typing preserves. An equality is unconstrained. -/
def Pred.AtArity (S : Sig) : Pred -> Prop
  | .app t as => ∃ ht : t < S.traits.length, as.length = S.traits[t].arity
  | .eq _ _ => True

theorem Pred.atArity_of_wf {S : Sig} {P : Pred} (h : P.wf S = true) : P.AtArity S := by
  cases P with
  | app t as =>
    simp only [Pred.wf] at h
    split at h
    · simp at h
    · rename_i T hT
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨ht, hT'⟩ := List.getElem?_eq_some_iff.mp hT
      exact ⟨ht, by rw [hT']; exact h.1⟩
  | eq _ _ => trivial

theorem Pred.atArity_subst {S : Sig} (σ : Subst) {P : Pred} (h : P.AtArity S) : (P.subst σ).AtArity S := by
  cases P with
  | app t as =>
    obtain ⟨ht, hlen⟩ := h
    exact ⟨ht, by rw [substList_length]; exact hlen⟩
  | eq _ _ => trivial

/-- Every hypothesis in scope is a claim the claim type verifier admits: well formed (no citation on an
equality, D/TraitTypes.cpp:224-235) and at its trait's arity. -/
def ClaimsWF (S : Sig) (Γ : List Claim) : Prop := ∀ c ∈ Γ, c.WF ∧ c.pred.AtArity S

theorem claimsWF_nil (S : Sig) : ClaimsWF S [] := fun _ hc => nomatch hc

theorem Claim.WF_none (P : Pred) : Claim.WF ⟨P, none⟩ := by cases P <;> trivial

theorem Claim.WF_app (t : TraitId) (as : List Ty) (π : Option Cite) : Claim.WF ⟨.app t as, π⟩ := by
  cases π <;> trivial

theorem Claim.WF_eq_cite {a b : Ty} {π : Option Cite} (h : Claim.WF ⟨.eq a b, π⟩) : π = none := by
  cases π with
  | none => rfl
  | some _ => exact h.elim

theorem Claim.WF_subst (σ : Subst) {c : Claim} (h : c.WF) : (c.subst σ).WF := by
  obtain ⟨P, π⟩ := c
  cases P with
  | app t as => exact Claim.WF_app _ _ _
  | eq a b => have := Claim.WF_eq_cite h; subst this; exact Claim.WF_none _

/-- The two clauses of H2 that `ProofsWF` states only for the proved predicate's groundness: a proof
proves a trait application at the trait's arity (an equality is never a proof's claim,
D/TraitTypes.cpp:224-235), and the derive its citation reads to is at ground arguments, as the stage
writes it at a ground resolution (D/ImplResolution.cpp:816-873). Stated beside `ProofsWF` rather than
in it because theorem 4b's written proof (Completeness.lean, `cut_proof`) is ground only when the
derivation it elaborates is. -/
def Module.ProofsGround (M : Module) : Prop :=
  ∀ p (hp : p < M.proofs.length),
    (∃ t as, M.proofs[p].1 = .app t as ∧ (Pred.app t as).AtArity M.sig) ∧
    ∀ i σ ds, M.proofDerive p = some (i, σ, ds) -> groundList σ = true

/-- A proof whose derive is read is in the table. -/
theorem proofDerive_lt (M : Module) {p : ProofId} {d : ImplId × List Ty × List Ev}
    (h : M.proofDerive p = some d) : p < M.proofs.length := by
  simp only [Module.proofDerive, Module.proofDeriveAux] at h
  cases hp : M.proofs[p]? with
  | none => rw [hp] at h; simp at h
  | some pr => exact (List.getElem?_eq_some_iff.mp hp).1

/-! ## A term has one claim, and the restriction on citations does not enter it -/

/-- Every rule computes its conclusion from the term and the declarations it names, so a term has one
claim under any two restrictions on impl citations. -/
theorem hasType_unique (L : ConLaw) (M : Module) {Ok Ok' : ImplId -> Subst -> Prop} :
    ∀ {Γ : List Claim} {e : Ev} {c c' : Claim}, HasType L M Ok Γ e c -> HasType L M Ok' Γ e c' -> c = c' := by
  intro Γ e c c' h₁
  induction h₁ generalizing c' with
  | hyp hk => intro h₂; cases h₂; rfl
  | derive hi hσ ok hes πs hprem ih => intro h₂; cases h₂; rfl
  | projReq h ht hk ih =>
    intro h₂
    cases h₂ with
    | projReq h' ht' hk' => cases ih h'; rfl
    | projWhere h' ht' hder hj hop => cases ih h'; omega
  | @projWhere Γ e t as p i σ ds j c h ht hder hj hop ih₁ ih₂ =>
    intro h₂
    generalize hk₀ : M.sig.traits[t].reqs.length + j = k₀ at h₂
    cases h₂ with
    | projReq h' ht' hk' => cases ih₁ h'; omega
    | projWhere h' ht' hder' hj' hop' =>
      cases ih₁ h'
      rw [hder] at hder'; cases hder'
      cases Nat.add_left_cancel hk₀
      exact ih₂ hop'
  | reqCall h ht hb hus hfs πs hprem ih ihs =>
    intro h₂
    cases h₂ with
    | reqCall h' => cases ih h'; rfl
  | witProof hp hPeq => intro h₂; cases h₂; rfl
  | witImpl hi hunc ok => intro h₂; cases h₂; rfl
  | projRes hi hσ ok hes πs hprem hb hg ih =>
    intro h₂
    cases h₂ with
    | projRes _ _ _ _ _ _ hb' _ => rw [hb] at hb'; cases hb'; rfl
  | refl => intro h₂; cases h₂; rfl
  | compose hne hlen hprem hent ih => intro h₂; cases h₂; rfl
  | coerce h hlen hprem hπ hwf hpe ih ihs => intro h₂; cases h₂; rfl

/-- A body whose chain of casts ends at a derive types that derive's operands at the impl's where
entries. -/
theorem returned_inl_typed (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim} {i : ImplId} {σ : List Ty} {ds : List Ev},
      HasType L M Ok Γ e c -> e.returned = some (.inl (i, σ, ds)) ->
      ∃ hi : i < M.sig.impls.length, σ.length = M.sig.impls[i].arity ∧
        ds.length = M.sig.impls[i].wheres.length ∧
        ∃ πs : Nat -> Option Cite, ∀ j, j < ds.length ->
          HasType L M Ok Γ (ds.getD j default)
            ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), πs j⟩ := by
  intro Γ e c i σ ds ht
  induction ht with
  | derive hi hσ ok hes πs hprem _ =>
    intro hr
    simp only [Ev.returned, Option.some.injEq, Sum.inl.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl, rfl⟩ := hr
    exact ⟨hi, hσ, hes, πs, hprem⟩
  | witImpl hi hunc ok =>
    intro hr
    simp only [Ev.returned, Option.some.injEq, Sum.inl.injEq, Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl, rfl⟩ := hr
    exact ⟨hi, hunc.1.symm, by simp [hunc.2], fun _ => none, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
  | coerce h _ _ _ _ _ ih _ => intro hr; exact ih (by simpa [Ev.returned] using hr)
  | hyp | projReq | projWhere | reqCall | witProof | projRes | refl | compose =>
    intro hr; simp [Ev.returned] at hr

/-- A restriction on impl citations may be loosened. -/
theorem hasType_mono (L : ConLaw) (M : Module) {Ok Ok' : ImplId -> Subst -> Prop}
    (hOk : ∀ i σ, Ok i σ -> Ok' i σ) :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> HasType L M Ok' Γ e c := by
  intro Γ e c ht
  induction ht with
  | hyp hk => exact .hyp hk
  | derive hi hσ ok hes πs _ ih => exact .derive hi hσ (hOk _ _ ok) hes πs ih
  | projReq _ ht hk ih => exact .projReq ih ht hk
  | projWhere _ ht hder hj _ ih₁ ih₂ => exact .projWhere ih₁ ht hder hj ih₂
  | reqCall _ ht hb hus hfs πs _ ih ihs => exact .reqCall ih ht hb hus hfs πs ihs
  | witProof hp hpred => exact .witProof hp hpred
  | witImpl hi hu ok => exact .witImpl hi hu (hOk _ _ ok)
  | projRes hi hσ ok hes πs _ hb hg ih => exact .projRes hi hσ (hOk _ _ ok) hes πs ih hb hg
  | refl => exact .refl
  | compose hne hE _ hent ih => exact .compose hne hE ih hent
  | coerce _ hE _ hπ hwf hpe ih ihs => exact .coerce ih hE ihs hπ hwf hpe

theorem hasType_relax (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {Γ : List Claim} {e : Ev}
    {c : Claim} (ht : HasType L M Ok Γ e c) : HasType L M AnyImpl Γ e c :=
  hasType_mono L M (fun _ _ _ => trivial) ht

theorem proofsWF_relax (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hP : ProofsWF L M Ok) :
    ProofsWF L M AnyImpl := by
  intro p hp
  obtain ⟨⟨π, hty⟩, hgr, hfree, hcites⟩ := hP p hp
  exact ⟨⟨π, hasType_relax L M hty⟩, hgr, hfree, hcites⟩

theorem returnsWF_relax (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hR : M.ReturnsWF L Ok) :
    M.ReturnsWF L AnyImpl := by
  intro i hi ht
  obtain ⟨h₁, h₂, h₃, h₄⟩ := hR i hi ht
  refine ⟨h₁, h₂, fun k hk => ?_, fun b hb => ?_⟩
  · obtain ⟨hfree, π, hty⟩ := h₃ k hk
    exact ⟨hfree, π, hasType_relax L M hty⟩
  · obtain ⟨hfree, π, hty⟩ := h₄ b hb
    exact ⟨hfree, π, hasType_relax L M hty⟩

/-! ## What the typing preserves of the claim type verifier's checks -/

/-- A well-typed term's claim is well formed when the hypotheses' are: no rule mints a citation on an
equality, and a proof's claim is an application (H2). -/
theorem hasType_wf (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hG : M.ProofsGround) :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> (∀ c ∈ Γ, c.WF) -> c.WF := by
  intro Γ e c ht
  induction ht with
  | hyp hk => intro hΓ; exact hΓ _ (List.getElem_mem hk)
  | derive => intro _; exact Claim.WF_none _
  | projReq => intro _; exact Claim.WF_none _
  | projWhere _ _ _ _ _ _ ih => intro _; exact ih (fun c hc => nomatch hc)
  | reqCall => intro _; exact Claim.WF_none _
  | @witProof p P Γ hp hPeq =>
    intro _
    obtain ⟨⟨t, as, hpa, _⟩, _⟩ := hG _ hp
    subst hPeq
    rw [hpa]
    exact Claim.WF_app _ _ _
  | witImpl => intro _; exact Claim.WF_app _ _ _
  | projRes => intro _; exact Claim.WF_none _
  | refl => intro _; exact Claim.WF_none _
  | compose => intro _; exact Claim.WF_none _
  | coerce _ _ _ _ hwf _ => intro _; exact hwf

/-- A well-typed term's application claim is at its trait's arity when the hypotheses' are: the
declarations' predicates are (`DeclWF`), a proof's is (H2), and substitution and the closure keep
the argument count. -/
theorem hasType_atArity (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hS : M.sig.DeclWF)
    (hG : M.ProofsGround) :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> ClaimsWF M.sig Γ -> c.pred.AtArity M.sig := by
  intro Γ e c ht
  induction ht with
  | hyp hk => intro hΓ; exact (hΓ _ (List.getElem_mem hk)).2
  | derive hi =>
    intro _
    obtain ⟨T, hT, hlen, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
    obtain ⟨ht, hT'⟩ := List.getElem?_eq_some_iff.mp hT
    simp only [ImplDecl.header, Pred.AtArity, substList_length]
    exact ⟨ht, by rw [hT']; exact hlen⟩
  | projReq _ ht hk =>
    intro _
    exact Pred.atArity_subst _
      (Pred.atArity_of_wf (hS.trait_reqs _ _ (List.getElem?_eq_getElem ht) _ (List.getElem_mem hk)).1)
  | projWhere _ _ _ _ _ _ ih => intro _; exact ih (claimsWF_nil _)
  | reqCall _ ht hb =>
    intro _
    have hB := hS.trait_binders _ (List.getElem_mem ht) _ (List.getElem_mem hb)
    exact Pred.atArity_subst _ (Pred.atArity_of_wf hB.2.2.2.1)
  | @witProof p P Γ hp hPeq =>
    intro _
    obtain ⟨⟨t, as, hpa, hat⟩, _⟩ := hG _ hp
    subst hPeq
    simp only [hpa]
    exact hat
  | witImpl hi =>
    intro _
    obtain ⟨T, hT, hlen, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
    obtain ⟨ht, hT'⟩ := List.getElem?_eq_some_iff.mp hT
    simp only [ImplDecl.header, Pred.AtArity, substList_length]
    exact ⟨ht, by rw [hT']; exact hlen⟩
  | projRes => intro _; trivial
  | refl => intro _; trivial
  | compose => intro _; trivial
  | coerce _ _ _ _ _ hpe ih =>
    intro hΓ
    cases hpe with
    | app hlen _ =>
      obtain ⟨ht, ha⟩ := ih hΓ
      exact ⟨ht, by rw [← hlen]; exact ha⟩
    | eq _ _ => trivial

/-- Under H2 and `DeclWF`, the claim of a where-entry operand read off a proof's derive (D2) is ground:
the derive is at ground arguments and the where entry's variables are the impl's labels. -/
theorem projWhere_operand_ground (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (hS : M.sig.DeclWF)
    (hP : ProofsWF L M Ok) (hG : M.ProofsGround) {p : ProofId} {i : ImplId} {σ : List Ty} {ds : List Ev}
    (hder : M.proofDerive p = some (i, σ, ds)) {j : Nat} (hj : j < ds.length) {c : Claim}
    (hop : HasType L M Ok [] (ds.getD j default) c) : c.pred.ground = true := by
  obtain ⟨q, _, hq, hret⟩ := proofDerive_spec hP hder
  obtain ⟨⟨π₁, hbody⟩, _, _, _⟩ := hP q hq
  obtain ⟨hi, hσ, hds, πs, hops⟩ := returned_inl_typed L M hbody hret
  cases hasType_unique L M hop (hops j hj)
  have hg := (hG p (proofDerive_lt M hder)).2 _ _ _ hder
  obtain ⟨_, hwh, _⟩ := hS.impl_wf _ (List.getElem_mem hi)
  have hW := (hwh _ (getD_mem_of_lt _ (hds ▸ hj))).2
  apply Pred.ground_subst
  intro v hv
  have hv' : v < σ.length := by
    rw [hσ]; simpa using Pred.allVars_mentions _ _ hW v hv
  exact ⟨σ[v], by simp [Subst.ofList, List.getElem?_eq_getElem hv'], groundList_getElem hg hv'⟩

/-! ## The instantiation lemmas (SPEC.md 6.5) -/

/-- (subst-types), SPEC.md 6.5: the clone rule. Every claim predicate and every `#trait.impl_args`
receives the variable bindings alone (D/Specialization.cpp:106-128); the proof table is not cloned,
so a where-entry projection off a proof keeps its claim, which is ground (H2). `hOk` is trivial at
`AnyImpl`; at `DenL` it is false (a standing projection over a variable denotes, its instance at an
unserved ground argument does not), so the restriction on the instantiated term's citations is
re-established separately (`hasType_strengthen`). -/
theorem subst_types (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (σ : Subst) (hS : M.sig.DeclWF)
    (hP : ProofsWF L M Ok) (hG : M.ProofsGround)
    (hOk : ∀ i τ, Ok i (Subst.ofList τ) -> Ok i (Subst.ofList (substList σ τ))) :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> ClaimsWF M.sig Γ ->
      HasType L M Ok (Γ.map (Claim.subst σ)) (e.substTy σ) (c.subst σ) := by
  intro Γ e c ht
  induction ht with
  | @hyp k Γ hk =>
    intro _
    have hk' : k < (Γ.map (Claim.subst σ)).length := by simpa using hk
    have := HasType.hyp (L := L) (M := M) (Ok := Ok) hk'
    rwa [List.getElem_map] at this
  | @derive i τ Γ es hi hτ ok hes πs hprem ih =>
    intro hΓ
    obtain ⟨hhead, hwh, _⟩ := hS.impl_wf _ (List.getElem_mem hi)
    have hhv : allVarsList (fun v => decide (v < τ.length)) M.sig.impls[i].head = true := by
      rw [hτ]; exact allVarsList_of_forall _ _ (fun h hh => (hhead h hh).2)
    simp only [Ev.substTy, Claim.subst, ImplDecl.header, Pred.subst]
    rw [substList_ofList_subst σ τ _ hhv]
    refine HasType.derive hi (by rw [substList_length]; exact hτ) (hOk _ _ ok)
      (by rw [substTyList_length]; exact hes) πs (fun j hj => ?_)
    rw [substTyList_length] at hj
    rw [substTyList_getD σ es hj]
    have := ih j hj hΓ
    simp only [Claim.subst] at this
    have hW := (hwh _ (getD_mem_of_lt _ (hes ▸ hj))).2
    rwa [Pred.subst_ofList_subst σ τ _ (by rw [hτ]; exact hW)] at this
  | @projReq Γ e t as π k h ht hk ih =>
    intro hΓ
    obtain ⟨_, hlen⟩ := hasType_atArity L M hS hG h hΓ
    have ih' := ih hΓ
    simp only [Claim.subst, Pred.subst] at ih'
    simp only [Ev.substTy, Claim.subst]
    have := HasType.projReq ih' ht hk
    rwa [← Pred.subst_ofList_subst σ as _ (by
      rw [hlen]; exact (hS.trait_reqs _ _ (List.getElem?_eq_getElem ht) _ (List.getElem_mem hk)).2.1)] at this
  | @projWhere Γ e t as p i τ ds j c h ht hder hj hop ih _ =>
    intro hΓ
    have ih' := ih hΓ
    simp only [Claim.subst, Pred.subst] at ih'
    simp only [Ev.substTy]
    have hres := HasType.projWhere ih' ht hder hj hop
    have hcg : c.subst σ = c := by
      obtain ⟨P, π⟩ := c
      simp only [Claim.subst]
      rw [Pred.subst_of_ground σ _ (projWhere_operand_ground L M hS hP hG hder hj hop)]
    rw [hcg]
    exact hres
  | @reqCall Γ e t as π b us gs h ht hb hus hgs πs hprem ih ihs =>
    intro hΓ
    obtain ⟨_, hlen⟩ := hasType_atArity L M hS hG h hΓ
    have hB := hS.trait_binders _ (List.getElem_mem ht) _ (List.getElem_mem hb)
    have ih' := ih hΓ
    simp only [Claim.subst, Pred.subst] at ih'
    simp only [Ev.substTy, Claim.subst, Binder.callSubst]
    rw [Pred.subst_extend_subst σ as us _ hus _ (by rw [hlen]; exact hB.2.2.2.2)]
    refine HasType.reqCall ih' ht hb (by rw [substList_length]; exact hus)
      (by rw [substTyList_length]; exact hgs) πs (fun j hj => ?_)
    rw [substTyList_length] at hj
    rw [substTyList_getD σ gs hj]
    have := ihs j hj hΓ
    simp only [Claim.subst, Binder.callSubst] at this
    rwa [Pred.subst_extend_subst σ as us _ hus _
      (by rw [hlen]; exact (hB.2.2.1 _ (getD_mem_of_lt _ (hgs ▸ hj))).2)] at this
  | @witProof p P Γ hp hPeq =>
    intro _
    obtain ⟨_, hgr, _, _⟩ := hP _ hp
    rw [hPeq] at hgr
    simp only [Ev.substTy, Claim.subst, Pred.subst_of_ground σ _ hgr]
    exact HasType.witProof hp hPeq
  | @witImpl i Γ hi hunc ok =>
    intro _
    have hg : groundList M.sig.impls[i].head = true := by
      obtain ⟨hhead, _, _⟩ := hS.impl_wf _ (List.getElem_mem hi)
      exact groundList_of_forall _ (fun h hh => Ty.ground_of_varsBelow_zero h (hunc.1 ▸ (hhead h hh).2))
    have := HasType.witImpl (L := L) (M := M) (Ok := Ok) (Γ := Γ.map (Claim.subst σ)) hi hunc ok
    simpa [Ev.substTy, Claim.subst, ImplDecl.header, Pred.subst, substList_of_ground σ _ hg,
      substList_of_ground (Subst.ofList []) _ hg] using this
  | @projRes i τ Γ a labels B es gs hi hτ ok hes πs hprem hb hg ih =>
    intro hΓ
    obtain ⟨hhead, hwh, hbind⟩ := hS.impl_wf _ (List.getElem_mem hi)
    obtain ⟨T, hT, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
    obtain ⟨X, hX, hlab, _, _, _, hBv⟩ := hbind T hT _ _ _ hb
    have hhv : allVarsList (fun v => decide (v < τ.length)) M.sig.impls[i].head = true := by
      rw [hτ]; exact allVarsList_of_forall _ _ (fun h hh => (hhead h hh).2)
    simp only [Ev.substTy, Claim.subst, Pred.subst, Ty.subst]
    rw [substList_ofList_subst σ τ _ hhv, Ty.subst_extend_subst σ τ gs labels hg _ (by rw [hτ]; exact hBv)]
    refine HasType.projRes hi (by rw [substList_length]; exact hτ) (hOk _ _ ok)
      (by rw [substTyList_length]; exact hes) πs (fun j hj => ?_) hb (by rw [substList_length, hg])
    rw [substTyList_length] at hj
    rw [substTyList_getD σ es hj]
    have := ih j hj hΓ
    simp only [Claim.subst] at this
    have hW := (hwh _ (getD_mem_of_lt _ (hes ▸ hj))).2
    rwa [Pred.subst_ofList_subst σ τ _ (by rw [hτ]; exact hW)] at this
  | refl => intro _; simp only [Ev.substTy, Claim.subst, Pred.subst]; exact HasType.refl
  | @compose es Γ E l r hne hlen hprem hent ih =>
    intro hΓ
    simp only [Ev.substTy, Claim.subst, Pred.subst]
    refine HasType.compose (E := E.map fun p => (p.1.subst σ, p.2.subst σ)) (substTyList_ne_nil σ hne)
      (by simp [substTyList_length, hlen]) (fun j hj => ?_) (entails_subst L σ hent)
    rw [substTyList_length] at hj
    rw [substTyList_getD σ es hj, getD_map_of_lt _ _ (hlen ▸ hj)]
    have := ih j hj hΓ
    simpa [Claim.subst, Pred.subst] using this
  | @coerce Γ e P π π' P' E es h hlen hprem hπ hwf hpe ih ihs =>
    intro hΓ
    have ih' := ih hΓ
    simp only [Claim.subst] at ih'
    simp only [Ev.substTy, Claim.subst]
    refine HasType.coerce (E := E.map fun p => (p.1.subst σ, p.2.subst σ)) ih'
      (by simp [substTyList_length, hlen]) (fun j hj => ?_) hπ (Claim.WF_subst σ hwf) (predEntails_subst L σ hpe)
    rw [substTyList_length] at hj
    rw [substTyList_getD σ es hj, getD_map_of_lt _ _ (hlen ▸ hj)]
    have := ihs j hj hΓ
    simpa [Claim.subst, Pred.subst] using this

/-- `c'` may replace `c`: the predicates agree, and the citation is `c`'s, or `c` carries none and `c'` is
well formed (so an equality stays uncited). A derive's operand replaces a where hypothesis stated
without a citation under any citation of its own. -/
def Claim.Replaces (c' c : Claim) : Prop :=
  c'.pred = c.pred ∧ (c'.cite = c.cite ∨ (c.cite = none ∧ c'.WF))

theorem Claim.replaces_refl (c : Claim) : c.Replaces c := ⟨rfl, Or.inl rfl⟩

/-- A function of the position giving each operand's citation, from the citations given one by one. -/
theorem exists_cites {n : Nat} {Q : Nat -> Option Cite -> Prop} (h : ∀ j, j < n -> ∃ π, Q j π) :
    ∃ πs : Nat -> Option Cite, ∀ j, j < n -> Q j (πs j) :=
  ⟨fun j => if hj : j < n then Classical.choose (h j hj) else none,
   fun j hj => by simp only [hj, ↓reduceDIte]; exact Classical.choose_spec (h j hj)⟩

/-- (subst-evidence), SPEC.md 6.5: an instance's evidence parameters take the caller's proofs
(D/Passes.cpp:1026-1087; D/Specialization.cpp:329-357). A hypothesis stated with a citation is replaced
by evidence under that citation (PROJECT-WHERE reads it); one stated without is replaced by any
well-formed evidence of its predicate. The term's claim is kept likewise. -/
theorem subst_evidence (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} (fs : Nat -> Ev) :
    ∀ {Γ Δ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c ->
      (∀ k (hk : k < Γ.length), ∃ c', HasType L M Ok Δ (fs k) c' ∧ c'.Replaces Γ[k]) ->
      ∃ c', HasType L M Ok Δ (e.substHyp fs) c' ∧ c'.Replaces c := by
  intro Γ Δ e c ht
  induction ht generalizing Δ with
  | @hyp k Γ hk => intro hd; exact hd _ hk
  | @derive i σ Γ es hi hσ ok hes πs hprem ih =>
    intro hd
    have hops : ∀ j, j < es.length -> ∃ π, HasType L M Ok Δ ((substHypList fs es).getD j default)
        ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), π⟩ := by
      intro j hj
      obtain ⟨⟨P', π'⟩, hc', hp, _⟩ := ih j hj hd
      simp only at hp; subst hp
      rw [substHypList_getD fs es hj]
      exact ⟨π', hc'⟩
    obtain ⟨πs', hπs'⟩ := exists_cites hops
    exact ⟨_, .derive hi hσ ok (by rw [substHypList_length]; exact hes) πs'
      (fun j hj => hπs' j (by rwa [substHypList_length] at hj)), Claim.replaces_refl _⟩
  | @projReq Γ e t as π k h ht hk ih =>
    intro hd
    obtain ⟨⟨P', π'⟩, hc', hp, _⟩ := ih hd
    simp only at hp; subst hp
    exact ⟨_, .projReq hc' ht hk, Claim.replaces_refl _⟩
  | @projWhere Γ e t as p i σ ds j c h ht hder hj hop ih _ =>
    intro hd
    obtain ⟨⟨P', π'⟩, hc', hp, hc⟩ := ih hd
    simp only at hp hc; subst hp
    have hcite : π' = some (.proof p) := by
      rcases hc with hc | ⟨hc, _⟩
      · exact hc
      · cases hc
    subst hcite
    exact ⟨_, .projWhere hc' ht hder hj hop, Claim.replaces_refl _⟩
  | @reqCall Γ e t as π b us gs h ht hb hus hgs πs hprem ih ihs =>
    intro hd
    obtain ⟨⟨P', π'⟩, hc', hp, _⟩ := ih hd
    simp only at hp; subst hp
    have hops : ∀ j, j < gs.length -> ∃ π, HasType L M Ok Δ ((substHypList fs gs).getD j default)
        ⟨((M.sig.traits[t].binders[b]).premises.getD j default).subst
          ((M.sig.traits[t].binders[b]).callSubst as us), π⟩ := by
      intro j hj
      obtain ⟨⟨P'', π''⟩, hc'', hp'', _⟩ := ihs j hj hd
      simp only at hp''; subst hp''
      rw [substHypList_getD fs gs hj]
      exact ⟨π'', hc''⟩
    obtain ⟨πs', hπs'⟩ := exists_cites hops
    exact ⟨_, .reqCall hc' ht hb hus (by rw [substHypList_length]; exact hgs) πs'
      (fun j hj => hπs' j (by rwa [substHypList_length] at hj)), Claim.replaces_refl _⟩
  | witProof hp hPeq => intro _; exact ⟨_, .witProof hp hPeq, Claim.replaces_refl _⟩
  | witImpl hi hunc ok => intro _; exact ⟨_, .witImpl hi hunc ok, Claim.replaces_refl _⟩
  | @projRes i σ Γ a labels B es gs hi hσ ok hes πs hprem hb hg ih =>
    intro hd
    have hops : ∀ j, j < es.length -> ∃ π, HasType L M Ok Δ ((substHypList fs es).getD j default)
        ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), π⟩ := by
      intro j hj
      obtain ⟨⟨P', π'⟩, hc', hp, _⟩ := ih j hj hd
      simp only at hp; subst hp
      rw [substHypList_getD fs es hj]
      exact ⟨π', hc'⟩
    obtain ⟨πs', hπs'⟩ := exists_cites hops
    exact ⟨_, .projRes hi hσ ok (by rw [substHypList_length]; exact hes) πs'
      (fun j hj => hπs' j (by rwa [substHypList_length] at hj)) hb hg, Claim.replaces_refl _⟩
  | refl => intro _; exact ⟨_, .refl, Claim.replaces_refl _⟩
  | @compose es Γ E l r hne hlen hprem hent ih =>
    intro hd
    refine ⟨_, .compose (substHypList_ne_nil fs hne) (by rw [substHypList_length]; exact hlen)
      (fun j hj => ?_) hent, Claim.replaces_refl _⟩
    rw [substHypList_length] at hj
    obtain ⟨⟨P', π'⟩, hc', hp, hc⟩ := ih j hj hd
    simp only at hp hc; subst hp
    have hcite : π' = none := by
      rcases hc with hc | ⟨_, hwf⟩
      · exact hc
      · exact Claim.WF_eq_cite hwf
    subst hcite
    rw [substHypList_getD fs es hj]
    exact hc'
  | @coerce Γ e P π π' P' E es h hlen hprem hπ hwf hpe ih ihs =>
    intro hd
    obtain ⟨⟨P₁, ρ⟩, hc', hp, hc⟩ := ih hd
    simp only at hp hc; subst hp
    have hops : ∀ j, j < (substHypList fs es).length -> HasType L M Ok Δ ((substHypList fs es).getD j default)
        ⟨.eq (E.getD j default).1 (E.getD j default).2, none⟩ := by
      intro j hj
      rw [substHypList_length] at hj
      obtain ⟨⟨P₂, ρ₂⟩, hc₂, hp₂, hc₃⟩ := ihs j hj hd
      simp only at hp₂ hc₃; subst hp₂
      have hcite : ρ₂ = none := by
        rcases hc₃ with hc₃ | ⟨_, hwf⟩
        · exact hc₃
        · exact Claim.WF_eq_cite hwf
      subst hcite
      rw [substHypList_getD fs es hj]
      exact hc₂
    rcases hπ with rfl | rfl
    · exact ⟨_, .coerce hc' (by rw [substHypList_length]; exact hlen) hops (Or.inl rfl) (Claim.WF_none _) hpe,
        Claim.replaces_refl _⟩
    · rcases hc with hc | ⟨hc, _⟩
      · subst hc
        exact ⟨_, .coerce hc' (by rw [substHypList_length]; exact hlen) hops (Or.inr rfl) hwf hpe,
          Claim.replaces_refl _⟩
      · subst hc
        exact ⟨_, .coerce hc' (by rw [substHypList_length]; exact hlen) hops (Or.inl rfl) (Claim.WF_none _) hpe,
          Claim.replaces_refl _⟩

/-! ## H4 read off a term: the impl citations it writes -/

mutual
  /-- Every impl citation the term writes, at its arguments, satisfies `Ok`; at `DenL`, its header
  denotes there. The operands a where-entry projection reads off a proof are the proof table's, not
  the term's; H2 on the table covers them. -/
  def Ev.CitesOk (Ok : ImplId -> Subst -> Prop) : Ev -> Prop
    | .hyp _ => True
    | .allege _ => True
    | .derive i σ es => Ok i (Subst.ofList σ) ∧ citesOkList Ok es
    | .project e _ => e.CitesOk Ok
    | .reqCall e _ _ fs => e.CitesOk Ok ∧ citesOkList Ok fs
    | .witness (.impl i) _ => Ok i (Subst.ofList [])
    | .witness (.proof _) _ => True
    | .projRes i σ es _ _ => Ok i (Subst.ofList σ) ∧ citesOkList Ok es
    | .refl _ => True
    | .compose es _ _ => citesOkList Ok es
    | .coerce e es _ => e.CitesOk Ok ∧ citesOkList Ok es
  def citesOkList (Ok : ImplId -> Subst -> Prop) : List Ev -> Prop
    | [] => True
    | e :: es => e.CitesOk Ok ∧ citesOkList Ok es
end

theorem citesOkList_getD {Ok : ImplId -> Subst -> Prop} :
    ∀ {es : List Ev}, citesOkList Ok es -> ∀ {j : Nat}, j < es.length -> (es.getD j default).CitesOk Ok
  | [], _, j, hj => absurd hj (Nat.not_lt_zero j)
  | e :: es, h, 0, _ => by
    simp only [citesOkList] at h
    simp only [List.getD_cons_zero]
    exact h.1
  | e :: es, h, j+1, hj => by
    simp only [citesOkList] at h
    simp only [List.getD_cons_succ]
    exact citesOkList_getD h.2 (by simpa using hj)

theorem citesOkList_of_getD {Ok : ImplId -> Subst -> Prop} :
    ∀ {es : List Ev}, (∀ j, j < es.length -> (es.getD j default).CitesOk Ok) -> citesOkList Ok es
  | [], _ => trivial
  | e :: es, h => by
    simp only [citesOkList]
    exact ⟨by simpa using h 0 (by simp),
      citesOkList_of_getD (fun j hj => by simpa using h (j+1) (by simpa using hj))⟩

/-- Typing under a restriction puts every impl citation the term writes under it. -/
theorem hasType_citesOk (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> e.CitesOk Ok := by
  intro Γ e c ht
  induction ht with
  | hyp => trivial
  | derive _ _ ok _ _ _ ih => exact And.intro ok (citesOkList_of_getD ih)
  | projReq _ _ _ ih => exact ih
  | projWhere _ _ _ _ _ ih _ => exact ih
  | reqCall _ _ _ _ _ _ _ ih ihs => exact And.intro ih (citesOkList_of_getD ihs)
  | witProof => trivial
  | witImpl _ _ ok => exact ok
  | projRes _ _ ok _ _ _ _ _ ih => exact And.intro ok (citesOkList_of_getD ih)
  | refl => trivial
  | compose _ _ _ _ ih => exact citesOkList_of_getD ih
  | coerce _ _ _ _ _ _ ih ihs => exact And.intro ih (citesOkList_of_getD ihs)

/-- A term typed under one restriction is typed under another when every impl citation it writes
satisfies the other and the proof table does (H2 under it): the where-entry operands a projection
reads off a proof are retyped from the table. -/
theorem hasType_strengthen (L : ConLaw) (M : Module) {Ok Ok' : ImplId -> Subst -> Prop}
    (hP : ProofsWF L M Ok') :
    ∀ {Γ : List Claim} {e : Ev} {c : Claim}, HasType L M Ok Γ e c -> e.CitesOk Ok' -> HasType L M Ok' Γ e c := by
  intro Γ e c ht
  induction ht with
  | hyp hk => intro _; exact .hyp hk
  | derive hi hσ _ hes πs _ ih =>
    intro hc
    have hc' : Ok' _ _ ∧ citesOkList Ok' _ := hc
    exact .derive hi hσ hc'.1 hes πs (fun j hj => ih j hj (citesOkList_getD hc'.2 hj))
  | projReq _ ht hk ih => intro hc; exact .projReq (ih hc) ht hk
  | @projWhere Γ e t as p i σ ds j c h ht hder hj hop ih _ =>
    intro hc
    obtain ⟨q, _, hq, hret⟩ := proofDerive_spec hP hder
    obtain ⟨⟨π₁, hbody⟩, _, _, _⟩ := hP q hq
    obtain ⟨_, _, _, πs, hops⟩ := returned_inl_typed L M hbody hret
    cases hasType_unique L M hop (hops j hj)
    exact .projWhere (ih hc) ht hder hj (hops j hj)
  | reqCall _ ht hb hus hfs πs _ ih ihs =>
    intro hc
    have hc' : _ ∧ citesOkList Ok' _ := hc
    exact .reqCall (ih hc'.1) ht hb hus hfs πs (fun j hj => ihs j hj (citesOkList_getD hc'.2 hj))
  | witProof hp hpred => intro _; exact .witProof hp hpred
  | witImpl hi hu _ => intro hc; exact .witImpl hi hu hc
  | projRes hi hσ _ hes πs _ hb hg ih =>
    intro hc
    have hc' : Ok' _ _ ∧ citesOkList Ok' _ := hc
    exact .projRes hi hσ hc'.1 hes πs (fun j hj => ih j hj (citesOkList_getD hc'.2 hj)) hb hg
  | refl => intro _; exact .refl
  | compose hne hE _ hent ih =>
    intro hc; exact .compose hne hE (fun j hj => ih j hj (citesOkList_getD hc hj)) hent
  | coerce _ hE _ hπ hwf hpe ih ihs =>
    intro hc
    have hc' : _ ∧ citesOkList Ok' _ := hc
    exact .coerce (ih hc'.1) hE (fun j hj => ihs j hj (citesOkList_getD hc'.2 hj)) hπ hwf hpe

mutual
  /-- Hypothesis substitution with replacements citing within `Ok` keeps a term's citations within it. -/
  theorem citesOk_substHyp {Ok : ImplId -> Subst -> Prop} (fs : Nat -> Ev) (hfs : ∀ k, (fs k).CitesOk Ok) :
      ∀ e : Ev, e.CitesOk Ok -> (e.substHyp fs).CitesOk Ok
    | .hyp k, _ => hfs k
    | .allege _, h => h
    | .derive _ _ es, h => by
      simp only [Ev.CitesOk, Ev.substHyp] at h ⊢
      exact ⟨h.1, citesOkList_substHyp fs hfs es h.2⟩
    | .project e _, h => citesOk_substHyp fs hfs e h
    | .reqCall e _ _ gs, h => by
      simp only [Ev.CitesOk, Ev.substHyp] at h ⊢
      exact ⟨citesOk_substHyp fs hfs e h.1, citesOkList_substHyp fs hfs gs h.2⟩
    | .witness (.impl _) _, h => h
    | .witness (.proof _) _, h => h
    | .projRes _ _ es _ _, h => by
      simp only [Ev.CitesOk, Ev.substHyp] at h ⊢
      exact ⟨h.1, citesOkList_substHyp fs hfs es h.2⟩
    | .refl _, h => h
    | .compose es _ _, h => citesOkList_substHyp fs hfs es h
    | .coerce e es _, h => by
      simp only [Ev.CitesOk, Ev.substHyp] at h ⊢
      exact ⟨citesOk_substHyp fs hfs e h.1, citesOkList_substHyp fs hfs es h.2⟩
  theorem citesOkList_substHyp {Ok : ImplId -> Subst -> Prop} (fs : Nat -> Ev) (hfs : ∀ k, (fs k).CitesOk Ok) :
      ∀ es : List Ev, citesOkList Ok es -> citesOkList Ok (substHypList fs es)
    | [], h => h
    | e :: es, h => by
      simp only [citesOkList, substHypList] at h ⊢
      exact ⟨citesOk_substHyp fs hfs e h.1, citesOkList_substHyp fs hfs es h.2⟩
end

/-! ## H3 and H4 at an instance: what the reading of a projection asks of an impl's return -/

/-- A base is well formed when its arguments are at the impl's arity and its operands are closed,
allege-free terms for the impl's where entries at those arguments, with any citations. -/
def RootWF (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) (i : ImplId) (σ : List Ty)
    (ds : List Ev) : Prop :=
  ∃ hi : i < M.sig.impls.length, σ.length = M.sig.impls[i].arity ∧ ds.length = M.sig.impls[i].wheres.length ∧
    ∀ j, j < ds.length -> (ds.getD j default).allegeFree ∧
      ∃ π : Option Cite, HasType L M Ok [] (ds.getD j default)
        ⟨(M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ), π⟩

/-- H4 at an instance: every impl the return for a plain requirement cites, at a well-formed base's
arguments, satisfies `Ok`; at `DenL`, its header denotes there. Not a consequence of H3 and the base:
a return `derive Baz[T[a0]::N] from @Baz_all[T[a0]::N]` types generically, and at the base `[i32]` of a
served `Foo[i32]` its citation `Baz_all[T[i32]::N]` denotes only where an impl of `T` serves `i32`,
which nothing about `Foo`'s impl says; the stage refuses the standing projection at exit (F17). -/
def Module.ReturnsCitesOk (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) : Prop :=
  ∀ i (hi : i < M.sig.impls.length) (ht : M.sig.impls[i].trait < M.sig.traits.length)
    k (hk : k < (M.sig.traits[M.sig.impls[i].trait]).reqs.length) (σ : List Ty) (ds : List Ev),
    RootWF L M Ok i σ ds -> ((M.sig.impls[i].returns.getD k default).substTy (Subst.ofList σ)).CitesOk Ok

/-- The term form of H3 after instantiation, as `Root.projReq` reads it: at a well-formed base
`(I, σ, ds)`, the return for the `k`-th plain requirement with `σ` for the impl's labels and `ds` for
its hypotheses is a closed, allege-free term for `rho_k[h][σ]`, cited within `Ok`. Derived from
`Module.ReturnsWF` (H3), H2 and H4 at the instance by `returnsInstWF_of_returnsWF`. -/
def Module.ReturnsInstWF (L : ConLaw) (M : Module) (Ok : ImplId -> Subst -> Prop) : Prop :=
  ∀ i (hi : i < M.sig.impls.length) (ht : M.sig.impls[i].trait < M.sig.traits.length)
    k (hk : k < (M.sig.traits[M.sig.impls[i].trait]).reqs.length) (σ : List Ty) (ds : List Ev),
    RootWF L M Ok i σ ds ->
    (((M.sig.impls[i].returns.getD k default).substTy (Subst.ofList σ)).substHyp (hypsOf ds)).allegeFree ∧
    ∃ π : Option Cite, HasType L M Ok []
      (((M.sig.impls[i].returns.getD k default).substTy (Subst.ofList σ)).substHyp (hypsOf ds))
      ⟨(((M.sig.traits[M.sig.impls[i].trait]).reqs[k]).subst (Subst.ofList M.sig.impls[i].head)).subst
          (Subst.ofList σ), π⟩

/-- H3 instantiates: `Module.ReturnsWF` under no restriction, (subst-types) at the base's arguments,
(subst-evidence) at its operands, and the restriction `Ok` re-established from H4 at the instance and
H2 under `Ok`. -/
theorem returnsInstWF_of_returnsWF (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop}
    (hS : M.sig.DeclWF) (hP : ProofsWF L M Ok) (hG : M.ProofsGround) (hR : M.ReturnsWF L AnyImpl)
    (hH4 : M.ReturnsCitesOk L Ok) : M.ReturnsInstWF L Ok := by
  intro i hi ht k hk σ ds hW
  obtain ⟨_, _, hret, _⟩ := hR i hi ht
  obtain ⟨hfree, π, hty⟩ := hret k hk
  obtain ⟨_, hσ, hds, hops⟩ := hW
  -- the operands, at every position: allege-free, and citing within `Ok`
  have hfs_free : ∀ k', (hypsOf ds k').hasAllege = false := by
    intro k'
    by_cases hk' : k' < ds.length
    · exact (hops k' hk').1
    · show (ds.getD k' default).hasAllege = false
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.le_of_not_lt hk'), Option.getD_none]
      rfl
  have hfs_ok : ∀ k', (hypsOf ds k').CitesOk Ok := by
    intro k'
    by_cases hk' : k' < ds.length
    · obtain ⟨π', hty'⟩ := (hops k' hk').2
      exact hasType_citesOk L M hty'
    · show (ds.getD k' default).CitesOk Ok
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.le_of_not_lt hk'), Option.getD_none]
      trivial
  -- (subst-types) at σ, under no restriction: the where hypotheses are well-formed claims at arity
  have hΓ : ClaimsWF M.sig (M.sig.impls[i].wheres.map (⟨·, none⟩)) := by
    intro c hc
    obtain ⟨W, hWm, rfl⟩ := List.mem_map.mp hc
    exact ⟨Claim.WF_none _, Pred.atArity_of_wf ((hS.impl_wf _ (List.getElem_mem hi)).2.1 W hWm).1⟩
  have h1 := subst_types L M (Subst.ofList σ) hS (proofsWF_relax L M hP) hG (fun _ _ _ => trivial) hty hΓ
  -- (subst-evidence) at ds: each operand replaces the uncited where hypothesis at its position
  have hd : ∀ k' (hk' : k' < ((M.sig.impls[i].wheres.map (⟨·, none⟩)).map (Claim.subst (Subst.ofList σ))).length),
      ∃ c', HasType L M AnyImpl [] (hypsOf ds k') c' ∧
        c'.Replaces ((M.sig.impls[i].wheres.map (⟨·, none⟩)).map (Claim.subst (Subst.ofList σ)))[k'] := by
    intro k' hk'
    simp only [List.length_map] at hk'
    have hk'' : k' < ds.length := by rw [hds]; exact hk'
    obtain ⟨π', hty'⟩ := (hops k' hk'').2
    refine ⟨⟨_, π'⟩, hasType_relax L M hty', ?_⟩
    simp only [List.getElem_map, Claim.subst]
    exact ⟨by rw [getD_eq_getElem_of_lt _ hk'], Or.inr ⟨rfl, hasType_wf L M hG hty' (fun c hc => nomatch hc)⟩⟩
  obtain ⟨⟨P', π'⟩, h2, hp, _⟩ := subst_evidence L M (hypsOf ds) h1 hd
  simp only [Claim.subst] at hp
  subst hp
  refine ⟨hasAllege_substHyp _ hfs_free _ (by rw [hasAllege_substTy]; exact hfree), π', ?_⟩
  exact hasType_strengthen L M hP h2
    (citesOk_substHyp _ hfs_ok _ (hH4 i hi ht k hk σ ds ⟨hi, hσ, hds, hops⟩))

/-! ## Weakening of the typing, and the determinism of the root -/

/-- A closed term types under any hypotheses: no rule reads `Γ` but (HYP), which has no instance at `[]`. -/
theorem hasType_weaken (L : ConLaw) (M : Module) {Ok : ImplId -> Subst -> Prop} {e : Ev} {c : Claim}
    (Γ : List Claim) (ht : HasType L M Ok [] e c) : HasType L M Ok Γ e c := by
  generalize hΓ : ([] : List Claim) = Γ₀ at ht
  induction ht with
  | hyp hk => subst hΓ; exact absurd hk (Nat.not_lt_zero _)
  | derive hi hσ hok hes πs _ ih => exact .derive hi hσ hok hes πs (fun j hj => ih j hj hΓ)
  | projReq _ ht hk ih => exact .projReq (ih hΓ) ht hk
  | projWhere _ ht hder hj hop ih₁ _ => exact .projWhere (ih₁ hΓ) ht hder hj hop
  | reqCall _ ht hb hus hfs πs _ ih ihs => exact .reqCall (ih hΓ) ht hb hus hfs πs (fun j hj => ihs j hj hΓ)
  | witProof hp hpred => exact .witProof hp hpred
  | witImpl hi hu hok => exact .witImpl hi hu hok
  | projRes hi hσ hok hes πs _ hb hg ih => exact .projRes hi hσ hok hes πs (fun j hj => ih j hj hΓ) hb hg
  | refl => exact .refl
  | compose hne hE _ hent ih => exact .compose hne hE (fun j hj => ih j hj hΓ) hent
  | coerce _ hE _ hπ hwf hpe ih ihs => exact .coerce (ih hΓ) hE (fun j hj => ihs j hj hΓ) hπ hwf hpe

/-- A term reads to at most one base: induction on the first reading, inversion on the second; a
projection read as a plain requirement by one and as a where entry by the other contradicts on the
index against the trait's requirement count. -/
theorem root_unique (M : Module) {e : Ev} {r₁ r₂ : ImplId × List Ty × List Ev}
    (h₁ : Root M e r₁) (h₂ : Root M e r₂) : r₁ = r₂ := by
  induction h₁ generalizing r₂ with
  | derive => cases h₂; rfl
  | witImpl => cases h₂; rfl
  | witProof hp hb ih =>
    cases h₂ with
    | witProof hp' hb' => rw [hp] at hp'; cases hp'; exact ih hb'
  | coerce hb ih =>
    cases h₂ with
    | coerce hb' => exact ih hb'
  | projReq hb hI hT hk hret ih₁ ih₂ =>
    cases h₂ with
    | projReq hb' hI' hT' hk' hret' =>
      cases ih₁ hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact ih₂ hret'
    | projWhere hb' hI' hT' hk' _ =>
      cases ih₁ hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact absurd hk (Nat.not_lt.mpr hk')
  | projWhere hb hI hT hk hop ih₁ ih₂ =>
    cases h₂ with
    | projReq hb' hI' hT' hk' _ =>
      cases ih₁ hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact absurd hk' (Nat.not_lt.mpr hk)
    | projWhere hb' hI' hT' hk' hop' =>
      cases ih₁ hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      exact ih₂ hop'
  | reqCall hb hI hT hB hret ih₁ ih₂ =>
    cases h₂ with
    | reqCall hb' hI' hT' hB' hret' =>
      cases ih₁ hb'
      rw [hI] at hI'; cases hI'
      rw [hT] at hT'; cases hT'
      rw [hB] at hB'; cases hB'
      exact ih₂ hret'

end Trait
