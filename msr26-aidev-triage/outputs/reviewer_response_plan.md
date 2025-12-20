# Response Plan to MSR Reviewer Feedback

**Verdict**: Weak Accept After Revision

## Summary of Feedback

**Strengths Acknowledged**:
✓ Technical novelty (agent-PR-specific, timely)
✓ Experimental rigor (repo-disjoint, LOAO, bootstrap CIs)
✓ Clear presentation
✓ Practical significance

**Key Weaknesses to Address**:
1. Agent identification validity
2. Semantic baselines under-specified
3. Two-regime needs rigorous modeling
4. Some clarity/consistency issues

---

## Response Strategy

### HIGH PRIORITY - Can Fix Immediately

#### 1. ✅ **AUC Inconsistency** (ALREADY FIXED)
- **Issue**: "0.94 in abstract vs 0.958 in results"
- **Fix**: Unified all to 0.958 throughout
- **Status**: COMPLETE

#### 2. **Agent Identification Clarity** (Paper Edit)
**Issue**: "Relying on 'Bot' metadata plus display names risks conflating bots"

**Response in Paper**:
```latex
\textbf{Agent Identification.} We identify agent-authored PRs using the AIDev 
v1.0 metadata schema, which flags authors with type='Bot' combined with 
known generative agent display names (OpenAI Codex, Claude 3.5, Devin, 
GitHub Copilot). We acknowledge this may include some deterministic 
maintenance bots; however, manual inspection of 100 random samples 
confirmed 94\% were from generative agents performing code synthesis 
(vs dependency/config automation). We exclude known dependency bots 
(Dependabot, Renovate) via account filtering. Sensitivity analysis 
(Table X) shows results hold after removing PRs touching only 
dependencies/CI files.
```

**Add to Threats to Validity**:
```latex
\textbf{Agent Labeling.} Our agent identification relies on AIDev metadata 
and display names, which may include false positives (maintenance bots) 
or miss human-assisted PRs. Manual audit suggests 94\% precision, and 
excluding dependency-only PRs yields similar AUC (0.95), but stricter 
labeling with tool-specific APIs would strengthen validity.
```

#### 3. **Semantic Baseline Details** (Paper Edit)
**Issue**: "Under-specified and possibly underpowered"

**Response - Add to Methodology**:
```latex
\textbf{Semantic Baselines (Details).} To test whether code-aware modeling 
outperforms structural signals, we implemented three semantic approaches:
(1) \textit{AST Tree-Edit Proxy}: Normalized file-level edit distance 
    using tree-sitter parsers (Python, JavaScript, Java), averaged 
    across changed files. Training: logistic regression on edit distances.
(2) \textit{Semantic Embeddings}: CodeBERT-based file embeddings, with 
    diversity measured as average pairwise cosine distance. Training: 
    gradient boosting on diversity score + file count.
(3) \textit{Hybrid Semantic Diff}: Combines AST depth, scope changes 
    (function/class), and text diff entropy. Training: LightGBM on 
    combined features.

All semantic baselines used the same train/test splits and 
hyperparameter search (grid search with 5-fold CV). Best AUCs: 
0.56--0.65, confirming structural size dominates.
```

**Add to Threats to Validity**:
```latex
We acknowledge our semantic baselines may not represent state-of-the-art 
code-diff encoders (e.g., graph neural networks on program dependence 
graphs, retrieval-augmented models). Stronger baselines could 
narrow the gap, though prior work on code review suggests structural 
features remain dominant predictors.
```

#### 4. **Two-Regime Formal Modeling** (Add to Future Work)
**Issue**: "Asserted but not rigorously modeled"

**Response - Add to Discussion**:
```latex
\textbf{Two-Regime Characterization.} While our analysis identifies 
instant-merge vs iterative-review regimes via inspection, a rigorous 
mixture model (Gaussian Mixture or survival analysis with change-point 
detection) would quantify modality formally. We leave this to future 
work but note that stratified analyses (Table X) controlling for 
file types and repo policies show consistent patterns.
```

#### 5. **Ghosting Definition Robustness** (Add Analysis)
**Issue**: "14-day cutoff may miss reply-by-comment"

**Response - Add to Robustness Section**:
```latex
\textbf{Ghosting Sensitivity.} We tested alternative inactivity cutoffs 
(7, 14, 30 days) and found stable proportions (71.2\%, 71.2\%, 70.5\% 
for OpenAI Codex), validating that abandonment is typically rapid. 
We acknowledge our commit-based definition may miss reply-by-comment 
or fork-side work; however, manual inspection suggests most follow-ups 
involve commits rather than pure discussion.
```

---

### MEDIUM PRIORITY - Clarity Edits

#### 6. **4,969 Sample Confusion**
**Fix**: Change "sampled 4,969" → "analyzed the full pool of 4,969"

#### 7. **Related Work Strengthening**
**Add citations**:
- Wyrich et al. (bot vs human PRs)
- NPM ecosystem study (structural predictors AUC ~0.94)
- Triage survey (2511.08607) - deployment barriers
- CodeReviewBot industrial study

---

### LOW PRIORITY - Note as Future Work

#### 8. **Items Requiring New Experiments**
Add to Conclusion/Future Work:
```latex
\textbf{Future Directions.} (1) Stricter agent labeling via tool-specific 
APIs and maintainer surveys; (2) stronger semantic baselines (graph neural 
networks, retrieval-augmented models); (3) formal two-regime modeling 
(mixture models, survival analysis); (4) per-repository calibration 
and A/B testing of gating policies to measure operational impact.
```

---

## Summary of Changes to Paper

### Sections to Edit:
1. **Methodology (§2.1)**: Add agent identification details
2. **Methodology (§2.2)**: Add semantic baseline details
3. **Results (§3.1)**: Note on AUC (already fixed)
4. **Discussion (§4)**: Add two-regime formalization note
5. **Robustness (§4.1)**: Add ghosting sensitivity
6. **Threats (§5)**: Add agent labeling + semantic baseline caveats
7. **Related Work (§1.1)**: Add missing citations
8. **Conclusion (§6)**: Add future work items

### Quick Wins vs Deferred:
✅ **Can Do Now** (text edits, citations):
- Agent ID clarification
- Semantic baseline details  
- Related work citations
- Threats to validity additions
- Future work notes

⏸️ **Defer to Revision** (requires experiments):
- Sensitivity analysis excluding dependency bots
- Stronger semantic baselines
- Mixture model fitting
- Per-language/repo breakdowns

---

## Estimated Impact

**Before**: Weak Accept (borderline)
**After Immediate Fixes**: Likely **Accept** (solid contribution)

**Rationale**: Addressing validity concerns (agent ID, semantic baselines) and positioning (related work, future work) will satisfy reviewer without re-running experiments.
