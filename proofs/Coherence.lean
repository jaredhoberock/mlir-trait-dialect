import Calculus
/-!
# Coherence: theorem 5 (SPEC.md 6.3)

Under a coherent signature every closed term for an application has, up to denotation of the
arguments, the unique server as its root; so `trait.method.call` through any two such terms resolves
to one method.

Proof plan from SPEC.md 6.3: induction on the typing derivation. DERIVE: the root is `(I, σ)`, the
premises valid by theorem 3 (`hasType_valid` at `DenL`), so `(I, σ)` serves the header. WIT-APP: a
proof's body is closed, induction through H2; an unconditional impl serves its header. COERCE: the
root is preserved and the two applications denote alike by theorem 7 on the cited equalities.
PROJECT-REQ: the root is that of the impl's return at `σ` with the derive's operands (`Root.projReq`);
by H3 in its term form and the instantiation lemmas that return is a closed term for `rho_k[h[σ]]`.
PROJECT-WHERE: inversion. Equality arms carry no root.

Where the proof departs from the plan, and what it takes beyond the skeleton's hypotheses:
* The induction is on the reading of the term, not on its typing: the instantiated return a
  PROJECT-REQ reads (`Root.projReq`) is not a sub-derivation of the term's typing, and its own reading
  may be arbitrarily long. `Reads` below is that reading, with the two facts the stage establishes
  about a term before it dispatches: every projection's reading reaches a base (`readEvidence`,
  D/TraitOps.cpp:1696-1767; X/x3 and X/x10 are the shapes with no base), and every `proj_resolve`
  stands at an application the denotation resolves (the exit walk's refusal of standing projections,
  D/Passes.cpp:1962-1990; F17). The first is what the skeleton's `root(e) = (I, σ)` presupposed; the
  second is `Deriv.Resolvable` of theorem 7, stated on the term. Without it the statement is false:
  `impl<a> T for a { N = i32 }` and `impl<b> V for (b, T[b]::N)` give `derive V[a0, T[a0]::N] from
  @V_all[a0]` coerced to `V[a0, i32]` through `proj_resolve T[a0]::N = i32 by @T_all[a0]`, a closed
  allege-free term whose root `(V_all, [a0])` does not serve `V[a0, i32]`: `[[T[a0]::N]]` is the
  standing projection, `V[a0, T[a0]::N]` and `V[a0, i32]` denote differently.
* The COERCE case needs theorem 7 on the term's equality operands. `erasure_core` takes a primitive
  derivation whose (BIND)s resolve (`ValidGround`); the typing of a closed equality term gives neither
  (a `project` at an equality requirement is (REQ), non-primitive at `G = []`). The induction on
  `Reads` proves the needed statement for terms directly: every closed allege-free equality term the
  reading reaches has endpoints that denote alike (`reads_spec`'s second conclusion).
* The PROJECT-REQ case needs the instantiated return typed closed at the instantiated requirement:
  `Module.ReturnsInstWF`, derived in Calculus.lean (`returnsInstWF_of_returnsWF`) from `ReturnsWF` under
  no restriction (H3), (subst-types) and (subst-evidence) of SPEC.md 6.5, H2's clauses `ProofsGround`,
  and H4 at the instance (`Module.ReturnsCitesOk`: the impls the return cites denote at the base's
  arguments). The last is a hypothesis of its own: the return's citations at the base need not denote
  although the base serves a denotable application (the report's finding on 6.5).
* Binder calls are outside `Reads`. `Root.reqCall` substitutes the call's premises, typed at the
  receiver's written arguments `a`, for the method's hypotheses, typed at the header `h[σ]`
  (`ImplDecl.binderHyps`); the instantiated body is well typed only when `a` and `h[σ]` coincide as
  written, since every typing rule compares predicates by identity. The dialect's (beta-A) inlines
  the body (D/TraitOps.cpp:1477-1506) where (beta-req) bridges with a coerce; whether the stage bridges
  a method call's premises is a finding for the owner, and `Root.reqCall` a definition request.
* `hD` (H3 in the logic) and `hden` are the specification's hypotheses; the proofs need neither: `hR`
  (H3 on the terms) does the work through `ReturnsInstWF`, and a reading that reaches a base denotes
  its own claim, so `hden` is derived rather than assumed.

The statements are at the limit of the fuel (`DnP`, `DenL`, `ServedByL`, `SelSoundL`, `Sig.CoherentL`
in Logic.lean), as `erasure_core` is, and take the two hypotheses `erasure_core` needs beyond the
skeleton's: the oracle answers every served ground application (`SelCompleteL`) and the constructor
law forgets no argument (`ConLaw.KeepsArgs`).
-/

namespace Trait

/-! ## Lemmas on the denotation -/

section Den
variable {sel : Sel} {L : ConLaw} {S : Sig}

/-- Lists that denote to one list denote alike. -/
theorem All2_dnEquiv_of_Dn {as bs cs : List Ty} (h₁ : All2 (Dn sel L S) as cs)
    (h₂ : All2 (Dn sel L S) bs cs) : All2 (DnEquiv sel L S) as bs := by
  induction h₁ generalizing bs with
  | nil => cases h₂; exact .nil
  | cons hac _ ih =>
    cases h₂ with
    | cons hbc hbs => exact .cons ((DnEquiv_of_Dn hac).trans (DnEquiv_of_Dn hbc).symm) (ih hbs)

/-- Two applications whose arguments denote alike denote alike. -/
theorem DnP_app_congr {t : TraitId} {as bs : List Ty} (h : All2 (DnEquiv sel L S) as bs) :
    ∀ Q, DnP sel L S (.app t as) Q ↔ DnP sel L S (.app t bs) Q := by
  intro Q
  rw [DnP_app_iff, DnP_app_iff]
  constructor
  · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr h cs).mp hcs, rfl⟩
  · rintro ⟨cs, hcs, rfl⟩; exact ⟨cs, (All2_Dn_congr h cs).mpr hcs, rfl⟩

/-- The closure lemma under the denotation, from seeds that denote alike: (SEED) is the seeds,
(EQUIV) is equality's, (CONG) is `DnEquiv_con`/`DnEquiv_proj`, (NORM) is `DnEquiv_law`. -/
theorem entails_dnEquiv_of_seeds (hL : L.KeepsArgs sel S) {E : List (Ty × Ty)}
    (hE : ∀ l r, (l, r) ∈ E -> DnEquiv sel L S l r) {a b : Ty} (h : Entails L E a b) :
    DnEquiv sel L S a b := by
  induction h with
  | ax hab => exact hE _ _ hab
  | refl => exact DnEquiv.refl _
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | con hl _ ih => exact DnEquiv_con (All2.of_getD hl ih)
  | proj hl₁ hl₂ _ _ ih₁ ih₂ => exact DnEquiv_proj (All2.of_getD hl₁ ih₁) (All2.of_getD hl₂ ih₂)
  | law hlaw => exact DnEquiv_law hL hlaw

/-- The seeds of a compose or coerce, read off its operands by position, denote alike when each
operand's endpoints do. -/
theorem seeds_dnEquiv {E : List (Ty × Ty)} {es : List Ev} (hlen : E.length = es.length)
    (h : ∀ j, j < es.length -> DnEquiv sel L S (E.getD j default).1 (E.getD j default).2) :
    ∀ l r, (l, r) ∈ E -> DnEquiv sel L S l r := by
  intro l r hlr
  obtain ⟨j, hj, hEj⟩ := List.mem_iff_getElem.mp hlr
  have := h j (hlen ▸ hj)
  rwa [getD_eq_getElem_of_lt E hj, hEj] at this

/-- A server of one application serves every application denoting alike. -/
theorem servedBy_congr {P P' : Pred} {i : ImplId} {σ : List Ty} (h : ServedByL sel L S P i σ)
    (hPP' : ∀ Q, DnP sel L S P Q ↔ DnP sel L S P' Q) : ServedByL sel L S P' i σ := by
  obtain ⟨hi, hσ, ⟨Q, hQ⟩, hiff, hW⟩ := h
  exact ⟨hi, hσ, ⟨Q, (hPP' Q).mp hQ⟩, fun Q => (hiff Q).trans (hPP' Q), hW⟩

/-- A server's trait is the application's, and its header's arguments denote alike with the
application's. -/
theorem servedBy_app {t : TraitId} {as : List Ty} {i : ImplId} {σ : List Ty}
    (h : ServedByL sel L S (.app t as) i σ) :
    ∃ hi : i < S.impls.length, S.impls[i].trait = t ∧
      All2 (DnEquiv sel L S) (substList (Subst.ofList σ) S.impls[i].head) as := by
  obtain ⟨hi, _, ⟨Q, hQ⟩, hiff, _⟩ := h
  obtain ⟨cs, hcs, rfl⟩ := DnP_app_iff.mp hQ
  have h2 := (hiff _).mpr hQ
  simp only [ImplDecl.header] at h2
  obtain ⟨cs', hcs', heq⟩ := DnP_app_iff.mp h2
  cases heq
  exact ⟨hi, rfl, All2_dnEquiv_of_Dn hcs' hcs⟩

end Den

/-- Under `DeclWF`, a plain requirement's variables are the trait's positions, which the impl's
header fills. -/
theorem req_varsBelow_head {S : Sig} (hS : S.DeclWF) {i : ImplId} (hi : i < S.impls.length)
    (ht : S.impls[i].trait < S.traits.length) {k : Nat}
    (hk : k < (S.traits[S.impls[i].trait]).reqs.length) :
    ((S.traits[S.impls[i].trait]).reqs[k]).varsBelow S.impls[i].head.length = true := by
  obtain ⟨T, hT, hlen, _, _, _⟩ := hS.impl_trait _ (List.getElem_mem hi)
  rw [List.getElem?_eq_getElem ht] at hT
  cases hT
  rw [hlen]
  exact (hS.trait_reqs _ _ (List.getElem?_eq_getElem ht) _ (List.getElem_mem hk)).2.1

/-- A substituted predicate that is an application came from an application. -/
theorem Pred.subst_eq_app {τ : Subst} {P : Pred} {t : TraitId} {as : List Ty}
    (h : P.subst τ = .app t as) : ∃ bs, P = .app t bs ∧ substList τ bs = as := by
  cases P with
  | app t' bs => simp only [Pred.subst, Pred.app.injEq] at h; exact ⟨bs, by rw [h.1], h.2⟩
  | eq a b => simp [Pred.subst] at h

/-- A substituted predicate that is an equality came from an equality. -/
theorem Pred.subst_eq_eq {τ : Subst} {P : Pred} {l r : Ty}
    (h : P.subst τ = .eq l r) : ∃ a b, P = .eq a b ∧ a.subst τ = l ∧ b.subst τ = r := by
  cases P with
  | app t' bs => simp [Pred.subst] at h
  | eq a b => simp only [Pred.subst, Pred.eq.injEq] at h; exact ⟨a, b, rfl, h.1, h.2⟩

/-! ## The reading of a term, as the stage establishes it -/

/-- `Reads sel L M e`: the term's reading reaches a base everywhere and every `proj_resolve` in it
stands at an application the denotation resolves. The projection cases name the base the operand
reads to and continue into what the projection reads: the impl's return instantiated at that base
(`Root.projReq`), or the base's operand (`Root.projWhere`). A `proj_resolve` asks that its header's
arguments and its GAT arguments denote to ground lists, which is `Deriv.Resolvable` on the term. The
operands of a derive or a `proj_resolve` need no reading: a server's where entries need only be
valid. There is no case for a binder call (see the header), a hypothesis (none at `G = []`) or an
allegation. -/
inductive Reads (sel : Sel) (L : ConLaw) (M : Module) : Ev -> Prop where
  | derive   : Reads sel L M (.derive i σ es)
  | witImpl  : Reads sel L M (.witness (.impl i) P)
  | witProof : M.proofs[p]? = some (Q, body) -> Reads sel L M body -> Reads sel L M (.witness (.proof p) P)
  | coerce   : Reads sel L M e -> (∀ j, j < es.length -> Reads sel L M (es.getD j default)) ->
               Reads sel L M (.coerce e es c)
  | projReq  : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
               k < T.reqs.length -> Reads sel L M e ->
               Reads sel L M (((I.returns.getD k default).substTy (Subst.ofList σ)).substHyp (hypsOf ds)) ->
               Reads sel L M (.project e k)
  | projWhere : Root M e (i, σ, ds) -> M.sig.impls[i]? = some I -> M.sig.traits[I.trait]? = some T ->
               T.reqs.length ≤ k -> Reads sel L M e -> Reads sel L M (ds.getD (k - T.reqs.length) default) ->
               Reads sel L M (.project e k)
  | refl     : Reads sel L M (.refl a)
  | compose  : (∀ j, j < es.length -> Reads sel L M (es.getD j default)) -> Reads sel L M (.compose es l r)
  | projRes  : M.sig.impls[i]? = some I ->
               (∃ as', All2 (Dn sel L M.sig) (substList (Subst.ofList σ) I.head) as' ∧ groundList as' = true) ->
               (∃ gs', All2 (Dn sel L M.sig) gs gs' ∧ groundList gs' = true) ->
               Reads sel L M (.projRes i σ es a gs)

/-! ## A proof citation on a closed term names the term's own base (`ProofOp::getDerive` against `Root`) -/

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

/-- A proof whose chain of casts ends at a derive reads to that derive. -/
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

/-- A term with a root whose claim cites a proof reads to what the proof's witness reads to, and the
proof proves an application of the claim's trait. A hypothesis has no root; a derive, a projection
at a plain requirement and a binder call conclude unproven (F1), so the citation is carried by a
witness, a coerce or a projection past the plain requirements alone. -/
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

/-! ## The induction on the reading -/

/-- No hypothesis, nothing to check: G-WF at `G = []`. -/
theorem gammaWF_nil (M : Module) : GammaWF M [] := fun _ hc => nomatch hc

/-- What a reading establishes about a closed allege-free term: an application claim has a base that
serves it and is well formed; an equality claim's endpoints denote alike. Induction on `Reads`,
with inversion of the typing at every node. -/
theorem reads_spec (sel : Sel) (L : ConLaw) (M : Module) (hS : M.sig.DeclWF)
    (hP : ProofsWF L M (DenL sel L M.sig)) (hG : M.ProofsGround) (hR : M.ReturnsWF L AnyImpl)
    (hH4 : M.ReturnsCitesOk L (DenL sel L M.sig))
    (hL : L.KeepsArgs sel M.sig) (hSel : SelSoundL sel L M.sig) (hSelC : SelCompleteL sel L M.sig)
    (hC : M.sig.CoherentL sel L) :
    ∀ {e : Ev}, Reads sel L M e -> ∀ {P : Pred} {π : Option Cite}, e.allegeFree ->
      HasType L M (DenL sel L M.sig) [] e ⟨P, π⟩ ->
      (∀ t as, P = .app t as -> ∃ i σ ds, Root M e (i, σ, ds) ∧ ServedByL sel L M.sig (.app t as) i σ ∧
          RootWF L M (DenL sel L M.sig) i σ ds) ∧
      (∀ l r, P = .eq l r -> DnEquiv sel L M.sig l r) := by
  -- H3 at the instance, from H3, H2 and H4 at the instance
  have hRI := returnsInstWF_of_returnsWF L M hS hP hG hR hH4
  intro e hrd
  induction hrd with
  | @derive i σ es =>
    intro P π he ht
    cases ht with
    | derive hi hσ ok hes πs hprem =>
      simp only [Ev.allegeFree, Ev.hasAllege] at he
      -- the where entries at σ are valid: theorem 3 on each operand
      have hW : ∀ j, j < M.sig.impls[i].wheres.length -> Nonempty (Deriv L M.sig (DenL sel L M.sig) []
          ((M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ))) := fun j hj =>
        hasType_valid L M _ hS hP (gammaWF_nil M)
          (hasAllegeList_eq_false he _ (getD_mem_of_lt _ (hes ▸ hj))) (hprem j (hes ▸ hj))
      refine ⟨fun t as hP => ⟨_, _, _, .derive, ?_, hi, hσ, hes,
        fun j hj => ⟨hasAllegeList_eq_false he _ (getD_mem_of_lt _ hj), πs j, hprem j hj⟩⟩,
        fun l r hP => by simp [ImplDecl.header] at hP⟩
      rw [← hP]
      exact ⟨hi, hσ, ok.2, fun Q => Iff.rfl, forall_mem_of_getD hW⟩
  | witImpl =>
    intro P π he ht
    cases ht with
    | witImpl hi hunc ok =>
      refine ⟨fun t as hP => ⟨_, [], [], .witImpl, ?_, hi, hunc.1.symm, by simp [hunc.2],
        fun j hj => absurd hj (Nat.not_lt_zero j)⟩, fun l r hP => by simp [ImplDecl.header] at hP⟩
      rw [← hP]
      exact ⟨hi, hunc.1.symm, ok.2, fun Q => Iff.rfl, fun W hW => by rw [hunc.2] at hW; cases hW⟩
  | witProof hp hb ih =>
    intro P π he ht
    cases ht with
    | witProof hp' hPeq =>
      obtain ⟨_, hpb⟩ := List.getElem?_eq_some_iff.mp hp
      obtain ⟨⟨π', hty⟩, _, hfree, _⟩ := hP _ hp'
      simp only [hpb] at hty hfree hPeq
      subst hPeq
      obtain ⟨ih₁, ih₂⟩ := ih hfree hty
      refine ⟨fun t as hP => ?_, ih₂⟩
      obtain ⟨i, σ, ds, hr, hs, hw⟩ := ih₁ t as hP
      exact ⟨i, σ, ds, .witProof hp hr, hs, hw⟩
  | coerce hre hres ih_e ih_es =>
    intro P π he ht
    cases ht with
    | coerce h hlen hprem hπ hwf hpe =>
      simp only [Ev.allegeFree, Ev.hasAllege, Bool.or_eq_false_iff] at he
      -- the seeds denote alike: the equality operands are read
      have hseeds := seeds_dnEquiv hlen fun j hj =>
        (ih_es j hj (hasAllegeList_eq_false he.2 _ (getD_mem_of_lt _ hj)) (hprem j hj)).2 _ _ rfl
      refine ⟨fun t as hP => ?_, fun l r hP => ?_⟩
      · subst hP
        cases hpe with
        | app hlen' hall =>
          obtain ⟨i, σ, ds, hr, hs, hw⟩ := (ih_e he.1 h).1 _ _ rfl
          have hargs := All2.of_getD hlen' fun j hj => entails_dnEquiv_of_seeds hL hseeds (hall j hj)
          exact ⟨i, σ, ds, .coerce hr, servedBy_congr hs (DnP_app_congr hargs), hw⟩
      · subst hP
        cases hpe with
        | eq hal hbr =>
          have hab := (ih_e he.1 h).2 _ _ rfl
          exact ((entails_dnEquiv_of_seeds hL hseeds hal).symm.trans hab).trans
            (entails_dnEquiv_of_seeds hL hseeds hbr)
  | projReq hroot hI hT hk hre hret ih_e ih_ret =>
    intro P π he ht
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.mp hI
    obtain ⟨htr, rfl⟩ := List.getElem?_eq_some_iff.mp hT
    cases ht with
    | projReq h ht' hk' =>
      -- the operand's base is the reading's, and serves the operand's claim
      obtain ⟨i₀, σ₀, ds₀, hr₀, hs₀, hw₀⟩ := (ih_e he h).1 _ _ rfl
      cases root_unique M hroot hr₀
      obtain ⟨_, htrait, hargs⟩ := servedBy_app hs₀
      subst htrait
      -- the instantiated return is a closed term for the requirement at the header
      obtain ⟨hfree, π₁, hty⟩ := hRI _ hi htr _ hk _ _ hw₀
      rw [Pred.subst_ofList_subst _ _ _ (req_varsBelow_head hS hi htr hk)] at hty
      obtain ⟨ih₁, ih₂⟩ := ih_ret hfree hty
      refine ⟨fun t₁ as₁ hP => ?_, fun l r hP => ?_⟩
      · obtain ⟨bs, hRk, rfl⟩ := Pred.subst_eq_app hP
        rw [hRk] at ih₁
        obtain ⟨i', σ', ds', hr', hs', hw'⟩ := ih₁ _ _ rfl
        refine ⟨i', σ', ds', .projReq hroot hI hT hk hr', ?_, hw'⟩
        exact servedBy_congr hs' (DnP_app_congr (substList_dnEquiv (EquivSub_ofList hargs) bs))
      · obtain ⟨a, b, hRk, rfl, rfl⟩ := Pred.subst_eq_eq hP
        rw [hRk] at ih₂
        have hab := ih₂ _ _ rfl
        exact ((Ty.subst_dnEquiv (EquivSub_ofList hargs) a).symm.trans hab).trans
          (Ty.subst_dnEquiv (EquivSub_ofList hargs) b)
    | projWhere h ht' hder hj hop =>
      -- a projection read as a plain requirement by the reading and as a where entry by the typing
      obtain ⟨i₀, σ₀, ds₀, hr₀, hs₀, hw₀⟩ := (ih_e he h).1 _ _ rfl
      cases root_unique M hroot hr₀
      obtain ⟨_, htrait, _⟩ := servedBy_app hs₀
      subst htrait
      exact absurd hk (by omega)
  | projWhere hroot hI hT hk hre hd ih_e ih_d =>
    intro P π he ht
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.mp hI
    obtain ⟨htr, rfl⟩ := List.getElem?_eq_some_iff.mp hT
    cases ht with
    | projReq h ht' hk' =>
      obtain ⟨i₀, σ₀, ds₀, hr₀, hs₀, hw₀⟩ := (ih_e he h).1 _ _ rfl
      cases root_unique M hroot hr₀
      obtain ⟨_, htrait, _⟩ := servedBy_app hs₀
      subst htrait
      exact absurd hk' (by omega)
    | projWhere h ht' hder hj hop =>
      obtain ⟨i₀, σ₀, ds₀, hr₀, hs₀, hw₀⟩ := (ih_e he h).1 _ _ rfl
      cases root_unique M hroot hr₀
      obtain ⟨_, htrait, _⟩ := servedBy_app hs₀
      subst htrait
      -- the cited proof's derive is the operand's base, so the typing's operand is the reading's
      have hroot' := (proofCite_rooted L M hP h rfl hroot).1 default
      have hder' := root_of_proofDeriveAux M _ (Q := default) hder
      cases root_unique M hroot' hder'
      rw [Nat.add_sub_cancel_left] at hd ih_d
      obtain ⟨_, _, _, hops⟩ := hw₀
      obtain ⟨ih₁, ih₂⟩ := ih_d (hops _ hj).1 hop
      refine ⟨fun t₁ as₁ hP => ?_, ih₂⟩
      obtain ⟨i', σ', ds', hr', hs', hw'⟩ := ih₁ t₁ as₁ hP
      refine ⟨i', σ', ds', .projWhere hroot hI hT hk ?_, hs', hw'⟩
      rw [Nat.add_sub_cancel_left]
      exact hr'
  | refl =>
    intro P π he ht
    cases ht with
    | refl => exact ⟨fun t as hP => (nomatch hP), fun l r hP => (by cases hP; exact DnEquiv.refl _)⟩
  | compose hres ih =>
    intro P π he ht
    cases ht with
    | compose hne hlen hprem hent =>
      simp only [Ev.allegeFree, Ev.hasAllege] at he
      have hseeds := seeds_dnEquiv hlen fun j hj =>
        (ih j hj (hasAllegeList_eq_false he _ (getD_mem_of_lt _ hj)) (hprem j hj)).2 _ _ rfl
      exact ⟨fun t as hP => (nomatch hP),
        fun l' r' hP => (by cases hP; exact entails_dnEquiv_of_seeds hL hseeds hent)⟩
  | @projRes i I σ gs es a hI hres_as hres_gs =>
    intro P π he ht
    simp only [Ev.allegeFree, Ev.hasAllege] at he
    obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.mp hI
    cases ht with
    | projRes hi' hσ ok hes πs hprem hb hg =>
      have hW : ∀ j, j < M.sig.impls[i].wheres.length -> Nonempty (Deriv L M.sig (DenL sel L M.sig) []
          ((M.sig.impls[i].wheres.getD j default).subst (Subst.ofList σ))) := fun j hj =>
        hasType_valid L M _ hS hP (gammaWF_nil M)
          (hasAllegeList_eq_false he _ (getD_mem_of_lt _ (hes ▸ hj))) (hprem j (hes ▸ hj))
      obtain ⟨as', has, hgas⟩ := hres_as
      obtain ⟨gs', hgs, hggs⟩ := hres_gs
      exact ⟨fun t as hP => (nomatch hP),
        fun l r hP => (by cases hP; exact DnEquiv_bind hSel hSelC hC hi hσ hW hb has hgas hgs hggs)⟩

/-! ## The theorems -/

set_option linter.unusedVariables false in
/-- Whenever a closed allege-free term for a denotable application reads to a base, the base serves
the application. Restated with the stage's two facts about the term (`Reads`), H2's clauses
`ProofsGround` and H4 at the instance (`ReturnsCitesOk`); `hD` and `hden` are the specification's
hypotheses and are not needed (see the header). -/
theorem root_serves (sel : Sel) (L : ConLaw) (M : Module) {e : Ev} {t : TraitId} {as : List Ty}
    {π : Option Cite} {i : ImplId} {σ : List Ty} {ds : List Ev}
    (hS : M.sig.DeclWF) (hD : M.sig.ReqDuty L) (hG : M.ProofsGround) (hR : M.ReturnsWF L AnyImpl)
    (hH4 : M.ReturnsCitesOk L (DenL sel L M.sig))
    (hP : ProofsWF L M (DenL sel L M.sig)) (hL : L.KeepsArgs sel M.sig) (hSel : SelSoundL sel L M.sig)
    (hSelC : SelCompleteL sel L M.sig) (hC : M.sig.CoherentL sel L) (hRd : Reads sel L M e)
    (he : e.allegeFree) (ht : HasType L M (DenL sel L M.sig) [] e ⟨.app t as, π⟩)
    (hden : ∃ Q, DnP sel L M.sig (.app t as) Q) (hr : Root M e (i, σ, ds)) :
    ServedByL sel L M.sig (.app t as) i σ := by
  obtain ⟨i', σ', ds', hr', hs, _⟩ := (reads_spec sel L M hS hP hG hR hH4 hL hSel hSelC hC hRd he ht).1 t as rfl
  cases root_unique M hr hr'
  exact hs

set_option linter.unusedVariables false in
/-- Theorem 5, COHERENCE: `Coherent(S), H1-H5. e allege-free, M; [] |- e : T[a] (by c)?, [[T[a]]] defined
=> root(e) = (I, σ) with ServedBy(S, T[a], I, σ)`. Restated with `Reads`, `ProofsGround` and
`ReturnsCitesOk` as `root_serves` is; existence of the root is the first of `Reads`' two facts. -/
theorem coherence (sel : Sel) (L : ConLaw) (M : Module) {e : Ev} {t : TraitId} {as : List Ty}
    {π : Option Cite}
    (hS : M.sig.DeclWF) (hD : M.sig.ReqDuty L) (hG : M.ProofsGround) (hR : M.ReturnsWF L AnyImpl)
    (hH4 : M.ReturnsCitesOk L (DenL sel L M.sig))
    (hP : ProofsWF L M (DenL sel L M.sig)) (hL : L.KeepsArgs sel M.sig) (hSel : SelSoundL sel L M.sig)
    (hSelC : SelCompleteL sel L M.sig) (hC : M.sig.CoherentL sel L) (hRd : Reads sel L M e)
    (he : e.allegeFree) (ht : HasType L M (DenL sel L M.sig) [] e ⟨.app t as, π⟩)
    (hden : ∃ Q, DnP sel L M.sig (.app t as) Q) :
    ∃ i σ, root M e i σ ∧ ServedByL sel L M.sig (.app t as) i σ := by
  obtain ⟨i, σ, ds, hr, hs, _⟩ := (reads_spec sel L M hS hP hG hR hH4 hL hSel hSelC hC hRd he ht).1 t as rfl
  exact ⟨i, σ, ⟨ds, hr⟩, hs⟩

/-- The corollary SPEC.md 6.3 states: two closed terms for one ground application root at one impl, at
arguments that denote alike. -/
theorem coherence_two_terms (sel : Sel) (L : ConLaw) (M : Module) {e₁ e₂ : Ev} {t : TraitId}
    {as : List Ty} {π₁ π₂ : Option Cite}
    (hS : M.sig.DeclWF) (hD : M.sig.ReqDuty L) (hG : M.ProofsGround) (hR : M.ReturnsWF L AnyImpl)
    (hH4 : M.ReturnsCitesOk L (DenL sel L M.sig))
    (hP : ProofsWF L M (DenL sel L M.sig)) (hL : L.KeepsArgs sel M.sig) (hSel : SelSoundL sel L M.sig)
    (hSelC : SelCompleteL sel L M.sig) (hC : M.sig.CoherentL sel L) (hg : as.all Ty.ground = true)
    (hRd₁ : Reads sel L M e₁) (he₁ : e₁.allegeFree) (ht₁ : HasType L M (DenL sel L M.sig) [] e₁ ⟨.app t as, π₁⟩)
    (hRd₂ : Reads sel L M e₂) (he₂ : e₂.allegeFree) (ht₂ : HasType L M (DenL sel L M.sig) [] e₂ ⟨.app t as, π₂⟩)
    (hden : ∃ Q, DnP sel L M.sig (.app t as) Q) :
    ∃ i σ₁ σ₂, root M e₁ i σ₁ ∧ root M e₂ i σ₂ ∧ All2 (DnEquiv sel L M.sig) σ₁ σ₂ := by
  obtain ⟨i₁, σ₁, hr₁, hs₁⟩ := coherence sel L M hS hD hG hR hH4 hP hL hSel hSelC hC hRd₁ he₁ ht₁ hden
  obtain ⟨i₂, σ₂, hr₂, hs₂⟩ := coherence sel L M hS hD hG hR hH4 hP hL hSel hSelC hC hRd₂ he₂ ht₂ hden
  obtain ⟨rfl, hσ⟩ := hC t as hg hden i₁ σ₁ i₂ σ₂ hs₁ hs₂
  exact ⟨i₁, σ₁, σ₂, hr₁, hr₂, hσ⟩

end Trait
