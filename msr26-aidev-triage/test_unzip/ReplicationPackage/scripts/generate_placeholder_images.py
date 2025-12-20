
import matplotlib.pyplot as plt
import os
from pathlib import Path

def generate_placeholder(filename, text):
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=20)
    plt.axis('off')
    plt.savefig(filename)
    print(f"Generated {filename}")

def main():
    output_dir = Path("msr26-aidev-triage/outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate instant_merges.png
    generate_placeholder(output_dir / "instant_merges.png", "Placeholder: Instant Merges Chart\n(Data unavailable)")
    
if __name__ == "__main__":
    main()
