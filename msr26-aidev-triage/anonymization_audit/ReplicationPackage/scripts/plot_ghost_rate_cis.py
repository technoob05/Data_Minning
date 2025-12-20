
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from statsmodels.stats.proportion import proportion_confint # Missing
import math

def wilson_score_interval(count, n, confidence=0.95):
    if n == 0: return 0, 0
    p = count / n
    z = 1.96 # Approx for 95%
    
    denominator = 1 + z**2/n
    center_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound

def plot_ghosting_rate(df):
    # Filter: Rejected + Feedback
    pool = df[df['is_rejected'] & df['has_feedback']].copy()
    
    # Ghosting = Single Commit (No follow-up)
    pool['is_ghosted'] = (pool['num_commits'] == 1)
    
    # Group by Agent
    stats = []
    agents = pool['agent'].unique()
    
    for agent in agents:
        subset = pool[pool['agent'] == agent]
        n = len(subset)
        ghosted = subset['is_ghosted'].sum()
        rate = ghosted / n
        
        # CI
        ci_low, ci_high = wilson_score_interval(ghosted, n)
        err_low = rate - ci_low
        err_high = ci_high - rate
        
        stats.append({
            'agent': agent,
            'rate': rate,
            'n': n,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'err': [[err_low], [err_high]] # Format for errorbar
        })
    
    stats_df = pd.DataFrame(stats).sort_values('rate', ascending=False)
    
    # Plot
    plt.figure(figsize=(8, 5))
    
    # Map friendly names if needed
    name_map = {'OpenAI_Codex': 'Codex', 'Claude_Code': 'Claude'}
    labels = [name_map.get(x, x) for x in stats_df['agent']]
    
    yerr = np.array([x['err'] for x in stats]).reshape(2, len(stats))
    # reshape to 2xN
    yerr_low = [x['err'][0][0] for x in stats]
    yerr_high = [x['err'][1][0] for x in stats]
    
    bars = plt.bar(range(len(stats)), stats_df['rate'], 
            yerr=[yerr_low, yerr_high], 
            capsize=5, color='steelblue', alpha=0.8)
            
    # Add counts
    for i, row in enumerate(stats_df.itertuples()):
        plt.text(i, row.rate + row.err[1][0] + 0.02, f"n={row.n}", 
                 ha='center', va='bottom', fontsize=9)
                 
    plt.xticks(range(len(stats)), labels, rotation=45, ha='right')
    plt.ylabel("Ghosting Rate (Single-Commit %)")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.title("Ghosting Rate by Agent (Rejected PRs w/ Feedback)")
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ghosting_rate.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")
    
    # Copy to paper
    paper_out = PAPER_DIR / "ghosting_rate.png"
    plt.savefig(paper_out, dpi=300)
    print(f"Saved copy to {paper_out}")

if __name__ == "__main__":
    df = load_data()
    plot_ghosting_rate(df)
