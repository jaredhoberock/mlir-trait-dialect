#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ImplResolution.hpp"
#include "Trait.hpp.inc"

namespace mlir::trait {

class ImplResolver;

/// Converts/lowers a dialect's ops to participate in the trait pipeline.
///
/// ### Pipeline phases
///  1) prove-claims
///     - Resolves trait.allege -> trait.witness by constructing proofs.
///     - Pushes proofs into types (applySubstitutionInPlace).
///     - **No general dialect rewrites run here.**
///     - Invariant after this phase: **there is no `trait.allege` left**.
///
///  2) instantiate-monomorphs
///     - Lowers trait.func.call / trait.method.call when operands are
///       monomorphic and claims are proven.
///     - **This is where dialects contribute convert-to-trait patterns**
///       to specialize/ground polymorphic regions and helper ops
///       (e.g., tuple.foldl body specialization).
///
///  3) monomorphize
///     - Erases trait/impl/proof ops and !trait.claim uses via TypeConverter
///       and op conversions contributed by dialects.
///
/// ### Contracts for dialect implementations
///  - `populateConvertToTraitConversionPatterns`
///      *Runs in phase (2), after all alleges are gone and claims are proven.*
///      - MUST NOT introduce `trait.allege`.
///      - May assume `trait.witness` and proven `!trait.claim` exist.
///      - Patterns should be idempotent and predicate on monomorphic operands,
///        enabling the greedy driver to reach a local fixed point.
///  - `populateEraseClaimsPatterns`
///      *Runs in phase (3).*
///      - Provide type/op conversions to remove all residual `!trait.claim`
///        mentions from types/regions (e.g., recursively convert tuple types,
///        drop claim-typed region args/results, rebuild ops accordingly).
///
/// ### Rationale
/// Keeping `trait.allege` quarantined to the prove-claims pass preserves a
/// simple, linear pipeline and prevents dialect patterns from depending on
/// transient IR that must disappear before monomorphization.
///
struct ConvertToTraitPatternInterface : DialectInterface {
  inline ConvertToTraitPatternInterface(Dialect *dialect)
    : DialectInterface(dialect, TypeID::get<ConvertToTraitPatternInterface>())
  {}

  /// Called during instantiate-monomorphs
  /// Register patterns that *prepare and specialize* your dialect’s IR
  /// for monomorphization (e.g., concretize polymorphic region signatures,
  /// specialize helpers that carry !trait.claim values, etc.).
  /// Constraints:
  ///   - MUST NOT introduce `trait.allege`.
  ///   - May assume proven claims / witnesses exist.
  virtual void populateConvertToTraitConversionPatterns(RewritePatternSet& patterns) const = 0;

  /// Called during monomorphize
  /// Provide a TypeConverter + rewrite patterns that erase all residual
  /// !trait.claim in your dialect’s types/regions and legalize ops after
  /// claims are dropped.
  virtual void populateEraseClaimsPatterns(TypeConverter &typeConverter, RewritePatternSet& patterns) const = 0;

  inline static StringRef getInterfaceName() { return "ConvertToTraitPatternInterface"; }

  inline static ::mlir::TypeID getInterfaceID() {
    return ::mlir::TypeID::get<ConvertToTraitPatternInterface>();
  }
};

class ImplGeneratorSet;

struct GenerateImplsInterface : DialectInterface {
  inline GenerateImplsInterface(Dialect *dialect)
    : DialectInterface(dialect, TypeID::get<GenerateImplsInterface>())
  {}

  virtual void populateImplGenerators(ImplGeneratorSet& generators) const = 0;

  inline static StringRef getInterfaceName() { return "GenerateImplsInterface"; }

  inline static TypeID getInterfaceID() {
    return TypeID::get<GenerateImplsInterface>();
  }
};

} // end mlir::trait
