# PEER REVIEW: Early-Stage Prediction of Review Effort in AI-Generated PRs

## REVIEWER'S ASSESSMENT

**Overall Recommendation**: **MAJOR REVISION** (Borderline Reject)

**Summary**: While this paper tackles a timely problem—triaging AI-generated PRs—it suffers from critical internal inconsistencies, weak empirical validation, and overclaimed findings that undermine its scientific rigor. The work shows promise but needs substantial revisions before acceptance.

---

## MAJOR ISSUES (Must Fix)

### 1. **CRITICAL INCONSISTENCY: Ghosting Rate Claims** ⚠️

**Line 32 (Abstract)**: Claims "substantial PR abandonment" after removing specific percentage
**Line 224 (Results)**: States "**64.5%** among rejected PRs"  
**Line 269 (Results)**: "LOAO yields **AUC 0.66--0.80**"
**Line 292 (Discussion)**: "ghosting is frequent (**64.5%**)"
**Line 294 (Discussion)**: "LOAO shows... **AUC 0.956--0.965**"
**Line 307 (Threats)**: "LOAO achieves AUC 0.956--0.965"

**PROBLEM**: The paper contains **TWO CONTRADICTORY** LOAO results:
- Section 269 claims LOAO is **WEAK** (0.66-0.80) 
- Lines 294, 307 claim LOAO is **EXCELLENT** (0.956-0.965)

**These are mutually exclusive! Which is correct??**

This is a DESK REJECT-level error. The authors must:
1. Clarify which LOAO result is correct
2. Explain the discrepancy
3. Update ALL mentions consistently

---

### 2. **Weak Justification: Why LightGBM?**

**Line 85**: Claims SOTA comparison justifies LightGBM
**Problem**: Stacking Ensemble achieves 0.9584 vs LightGBM 0.9580

**Issue**: The authors claim "+0.04% improvement" but:
- This is **+0.0004 absolute** AUC difference
- But also claim "only 0.04% of performance gap" - **MATH ERROR**  
  - Gap to perfect (1.0) from LightGBM (0.958) = 0.042
  - Improvement = 0.0004 / 0.042 = **0.95%**, not "0.04%"!

**They got their own percentage calculation wrong!**

Moreover:
- Why not just use Stacking if it's better?
- "Interpretability" claim is weak—ensembles can also use SHAP
- "Speed" (3.1s vs 9s) is irrelevant for offline triage
- This feels like post-hoc justification for a predetermined choice

---

### 3. **LOAO Contradiction Undermines Core Claim**

**Line 269 + Line 294/307 cannot both be true**:

**Scenario A**: If LOAO is 0.66-0.80 (weak):
- Supports "agent-specific patterns" narrative
- BUT contradicts Line 294's "robust generalization" claim
- Makes deployment risky

**Scenario B**: If LOAO is 0.956-0.965 (excellent):
- Contradicts Line 269
- Makes "agent-specific features not needed" claim stronger
- BUT then why mention weak LOAO at all?

**The paper tells** contradictory stories depending on which number you believe!

---

### 4. **Ghosting Definition is Murky**

**Line 96 (Table 1)**: "PR Status = Rejected AND... No follow-up commit >14 days"
**Line 224**: "64.5% among rejected PRs"  
**Line 102**: "90.6% close within 14 days"

**Wait, what?** If 90.6% close within 14 days, how can 64.5% ghost (defined as >14 days no follow-up)?

**These numbers are incompatible unless**:
- "Close within 14 days" means DIFFERENT PRs than ghosted ones, OR
- The definition changed between analyses

**Unclear**:
- Is ghosting rate PER REJECTED PR or PER PR WITH FEEDBACK?
- Line 102 says "4,969 PRs from Rejected+Feedback pool" but paper has 33,596 PRs total  
  - If only 4,969 got feedback, that's 14.8% of dataset  
  - But Table 2 shows ~50-70% ghosting rates per agent
  - **HOW DO THESE NUMBERS FIT TOGETHER?**

---

### 5. **Size-Only Baseline Undermines Novelty**

**Line 133**: "Size-Only heuristic... AUC 0.93"  
**Line 133**: "Full T0 model AUC 0.94"

**Gain from full model**: **+0.01 AUC** (1% improvement)

**But then authors claim**:
- "measurable utility beyond size" (Line 133)
- "+13.8pp to +23.2pp precision gains" (Table 5)

**Contradiction**:
- If size-only gets AUC 0.93, why does full model only get 0.94?
- How can precision lift be +23pp if AUC lift is only +1pp?

**Likely explanation**: Table 5 is within-quartile analysis, but:
- Not explained clearly enough
- Readers will be confused by AUC vs Precision discrepancy
- Need to emphasize this is CONDITIONAL analysis

---

### 6. **Instant Merge Validation is Weak**

**Line 80**: "validated via manual audit of 50 random samples"

**Problems**:
- 50 samples from 32.6% of 33,596 PRs = 10,952 instant merges
- 50/10,952 = **0.46% coverage** - statistically insignificant!
- No inter-rater reliability reported
- No sampling strategy described (random from what distribution?)

**Why not use CI/bot metadata to validate programmatically?**

---

## MODERATE ISSUES (Should Fix)

### 7. **Feature Count Inconsistency**

**Line 80**: "35 features"  
**Line 167 footnote**: "T0 features (24 features)"  
**Line 167 footnote**: "Early experiments with... 13 features"

**Which is it? 13, 24, or 35?**

Need to clearly state:
- Total features engineered
- T0 subset used
- Why these numbers differ

---

### 8. **Missing Baseline: Random Agent Rejection**

If ghosting is such a problem, why not test a simple baseline:
- **Reject ALL PRs from agents with >60% ghost rate**

This would:
- Reduce maintainer burden MORE than 20% gating
- Require no ML model
- Be perfectly interpretable

**Why not compare against this?**

---

### 9. **Weak Related Work Coverage**

**Line 69**: Mentions "GitLab merge requests", "MCR completion", "Nudge"

**Problems**:
-
 No citations provided for these claims
- "Recent work" is vague—what papers specifically?
- Missing comparison to bot detection/filtering work
- No discussion of prior PR triage tools (e.g., PredictMerge, PR-Alert)

---

### 10. **Statistical Rigor Issues**

**Bootstrap CIs are inconsistently reported**:
- Line 133: [0.93, 0.94] for LightGBM
- Line 153: [0.93, 0.94] - same interval
- Line 272: [81.3%, 85.9%] for Precision

**But**:
- No CIs for ghosting rates
- No CIs for LOAO (critical given contradictory claims!)
- No hypothesis tests for differences (e.g., is 0.93 vs 0.94 significant?)

**Need t-tests or permutation tests to validate claims**

---

### 11. **Causal Language Despite Correlation**

**Line 305**: "Our claims are correlational rather than causal"

**Yet throughout paper**:
- "ghosting is caused by..." (implied)
- "agents fail to converge" (causal framing)
- "plan requirement" recommendation (assumes causation)

**Either**:
- Tone down causal language OR
- Run causal analysis (e.g., propensity score matching, IV)

---

## MINOR ISSUES (Nice to Fix)

12. **Table 2 (Line 113)**: Agent names inconsistent with main text (OpenAI Codex vs OpenAI_Codex)

13. **Figure quality**: Figures are referenced but no discussion of what patterns should be visible

14. **RQ2 not clearly answered**: Where is explicit "Answer: ..."? 

15. **Ethical implications too brief**: Only covers fairness, not data privacy, model bias amplification

16. **No failure case analysis**: Authors should show examples where model fails

17. **Deployment details missing**: How to integrate with GitHub/GitLab UI?

---

## SPECIFIC QUESTIONS FOR AUTHORS

1. **Which LOAO result is correct: 0.66-0.80 OR 0.956-0.965?** This must be resolved.

2. Why not ensemble if it's better? Speed argument is weak.

3. Can you provide full confusion matrix at 20% threshold?

4. What's the FALSE NEGATIVE RATE? (Missed high-cost PRs)

5. Did you try simpler baselines like "reject all Codex PRs"?

6. How does performance vary by programming language?

7. What about PRs with >1 agent contributor (hand-off scenarios)?

---

## PRESENTATION ISSUES

- **Too dense**: 8 pages with tiny fonts in tables
- **Footnote overload**: Critical info buried in footnotes (Line 167)
- **Inconsistent terminology**: "Rejected" vs "Closed but not merged" vs "Abandoned"
- **Missing details**: Where are train/test split statistics?

---

## VERDICT

**Recommendation**: **MAJOR REVISION**

**Strengths**:
+ Timely problem
+ Large-scale dataset
+ Comprehensive robustness checks

**Critical Weaknesses**:
- **LOAO contradiction destroys credibility**
- Ghosting metrics don't add up
- Weak justification for model choice
- Size-only baseline undermines novelty claim

**Decision**:
This paper has potential but current form is **NOT READY FOR PUBLICATION**.

Authors must:
1. **RESOLVE LOAO CONTRADICTION** (desk-reject level error)
2. Fix ghosting rate calculation inconsistencies
3. Provide stronger justification for LightGBM vs ensemble
4. Add statistical significance tests
5. Clarify all numerical claims

**If these are fixed, this could be a solid contribution. As-is, it's a reject.**

---

**Confidence**: 4/5 (High - I've read carefully and identified critical issues)

**Recommendation to PC**: Reject unless authors can provide immediate clarification on LOAO results in rebuttal. If they fix issues, consider acceptance upon revision.
