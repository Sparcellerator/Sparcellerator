//===- DimLvlMap.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DimLvlMap.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::sparse_tensor::ir_detail;

//===----------------------------------------------------------------------===//
// `DimLvlExpr` implementation.
//===----------------------------------------------------------------------===//

SymVar DimLvlExpr::castSymVar() const {
  return SymVar(llvm::cast<AffineSymbolExpr>(expr));
}

std::optional<SymVar> DimLvlExpr::dyn_castSymVar() const {
  if (const auto s = dyn_cast_or_null<AffineSymbolExpr>(expr))
    return SymVar(s);
  return std::nullopt;
}

Var DimLvlExpr::castDimLvlVar() const {
  return Var(getAllowedVarKind(), llvm::cast<AffineDimExpr>(expr));
}

std::optional<Var> DimLvlExpr::dyn_castDimLvlVar() const {
  if (const auto x = dyn_cast_or_null<AffineDimExpr>(expr))
    return Var(getAllowedVarKind(), x);
  return std::nullopt;
}

std::tuple<DimLvlExpr, AffineExprKind, DimLvlExpr>
DimLvlExpr::unpackBinop() const {
  const auto ak = getAffineKind();
  const auto binop = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  const DimLvlExpr lhs(kind, binop ? binop.getLHS() : nullptr);
  const DimLvlExpr rhs(kind, binop ? binop.getRHS() : nullptr);
  return {lhs, ak, rhs};
}

//===----------------------------------------------------------------------===//
// `DimSpec` implementation.
//===----------------------------------------------------------------------===//

DimSpec::DimSpec(DimVar var, DimExpr expr, SparseTensorDimSliceAttr slice)
    : var(var), expr(expr), slice(slice) {}

bool DimSpec::isValid(Ranks const &ranks) const {
  // Nothing in `slice` needs additional validation.
  // We explicitly consider null-expr to be vacuously valid.
  return ranks.isValid(var) && (!expr || ranks.isValid(expr));
}

//===----------------------------------------------------------------------===//
// `LvlSpec` implementation.
//===----------------------------------------------------------------------===//

LvlSpec::LvlSpec(LvlVar var, LvlExpr expr, LevelType type, unsigned eb,
                 unsigned ec)
    : var(var), expr(expr), type(type), ellBlockSize(eb), ellCols(ec) {
  assert(expr);
  assert(isValidLT(type) && !isUndefLT(type));
}

bool LvlSpec::isValid(Ranks const &ranks) const {
  // Nothing in `type` needs additional validation.
  return ranks.isValid(var) && ranks.isValid(expr);
}

//===----------------------------------------------------------------------===//
// `DimLvlMap` implementation.
//===----------------------------------------------------------------------===//

DimLvlMap::DimLvlMap(unsigned symRank, ArrayRef<DimSpec> dimSpecs,
                     ArrayRef<LvlSpec> lvlSpecs)
    : symRank(symRank), dimSpecs(dimSpecs), lvlSpecs(lvlSpecs),
      mustPrintLvlVars(false) {
  // First, check integrity of the variable-binding structure.
  // NOTE: This establishes the invariant that calls to `VarSet::add`
  // below cannot cause OOB errors.
  assert(isWF());

  VarSet usedVars(getRanks());
  for (const auto &dimSpec : dimSpecs)
    if (!dimSpec.canElideExpr())
      usedVars.add(dimSpec.getExpr());
  for (auto &lvlSpec : this->lvlSpecs) {
    // Is this LvlVar used in any overt expression?
    const bool isUsed = usedVars.contains(lvlSpec.getBoundVar());
    // This LvlVar can be elided iff it isn't overtly used.
    lvlSpec.setElideVar(!isUsed);
    // If any LvlVar cannot be elided, then must forward-declare all LvlVars.
    mustPrintLvlVars = mustPrintLvlVars || isUsed;
  }
}

bool DimLvlMap::isWF() const {
  const auto ranks = getRanks();
  unsigned dimNum = 0;
  for (const auto &dimSpec : dimSpecs)
    if (dimSpec.getBoundVar().getNum() != dimNum++ || !dimSpec.isValid(ranks))
      return false;
  assert(dimNum == ranks.getDimRank());
  unsigned lvlNum = 0;
  for (const auto &lvlSpec : lvlSpecs)
    if (lvlSpec.getBoundVar().getNum() != lvlNum++ || !lvlSpec.isValid(ranks))
      return false;
  assert(lvlNum == ranks.getLvlRank());
  return true;
}

AffineMap DimLvlMap::getDimToLvlMap(MLIRContext *context) const {
  SmallVector<AffineExpr> lvlAffines;
  lvlAffines.reserve(getLvlRank());
  
  for (const auto &lvlSpec : lvlSpecs) {
    AffineExpr expr = lvlSpec.getExpr().getAffineExpr();
    
    // Handle BELL block decomposition
    if (lvlSpec.getEllBlockSize() > 0) {
      const unsigned blockSize = lvlSpec.getEllBlockSize();
      lvlAffines.pop_back();
      // First level: block index (floorDiv)
      lvlAffines.push_back(expr.floorDiv(blockSize));
      
      // Second level: intra-block offset (mod)
      lvlAffines.push_back(expr % blockSize);
    } else {
      // Regular level mapping
      lvlAffines.push_back(expr);
    }
  }
  
  return AffineMap::get(getDimRank(), getSymRank(), lvlAffines, context);
}


AffineMap DimLvlMap::getLvlToDimMap(MLIRContext *context) const {
  SmallVector<AffineExpr> dimAffines;
  dimAffines.reserve(getDimRank());

  const Level lvlRank = getLvlRank();
  Level l = 0;
  Level dim = 0;

  while (l < lvlRank) {
    const auto &lvlSpec = lvlSpecs[l];
    if (lvlSpec.getEllBlockSize() > 0) {
      // Handle BELL blocked dimension
      const unsigned blockSize = lvlSpec.getEllBlockSize();
      const unsigned ellCols = lvlSpec.getEllCols();

      // Consume two levels per BELL block
      AffineExpr blockIdx = getAffineDimExpr(l -1 , context);     // Block index level
      AffineExpr offset = getAffineDimExpr(l , context);   // Offset level
      dimAffines.push_back(blockIdx * blockSize + offset);
      l += 1; 
      dim++;
    } else {
      // Regular level mapping
      dimAffines.push_back(getAffineDimExpr(l, context));
      l++;
      dim++;
    }
  }

  return AffineMap::get(/*dimCount=*/getLvlRank(),
                        /*symbolCount=*/getSymRank(), dimAffines, context);
}

//===----------------------------------------------------------------------===//
