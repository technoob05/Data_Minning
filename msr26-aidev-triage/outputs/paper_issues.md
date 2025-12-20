# Paper Consistency Issues Found

## CRITICAL INCONSISTENCIES

### 1. **GHOSTING RATE MISMATCH** (Line 32 - Abstract)
**Current**: "among PRs _rejected after receiving human feedback_, **64.5%** are abandoned"
**Verified**: 11.3% of rejected PRs are ghosted (818/7,270)
**Action**: UPDATE abstract to reflect actual rate

### 2. **Abstract vs Code Discrepancy**
Paper abstract claims 64.5% but fresh code shows 11.3%
This is a MAJOR inconsistency that must be fixed

## MINOR ISSUES

###  3. **Overfull hbox warnings**
- Several lines too wide (cosmetic, not critical)

## VERIFIED CORRECT

✓ Dataset size: 33,596 PRs
✓ Agents: 5  
✓ High-cost rate: Top 20% (24.1% actual)
✓ LOAO: 0.956-0.965
✓ SOTA: 0.958
✓ Instant merge: 32.6%

## ACTION ITEMS

1. Fix abstract ghosting rate from 64.5% to 11.3%
2. Ensure consistency throughout paper
3. Verify all numerical claims one more time
