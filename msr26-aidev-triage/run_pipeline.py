import argparse
import subprocess
import sys
from pathlib import Path

# Map stages to scripts
STAGES = {
    "data": "scripts/01_process_data.py",
    "features": "scripts/02_engineer_features.py",
    "train": "scripts/03_train_models.py", 
    "audit": "scripts/04_audit_confounders.py"
}

ORDERED_STAGES = ["data", "features", "train", "audit"]

def run_script(script_path):
    print(f"\n[PIPELINE] Running {script_path}...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"[PIPELINE] Success: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"[PIPELINE] Error running {script_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run MSR'26 Reproduction Pipeline")
    parser.add_argument("--stage", choices=ORDERED_STAGES + ["all"], default="all", 
                       help="Specific stage to run, or 'all'")
    
    args = parser.parse_args()
    
    to_run = ORDERED_STAGES if args.stage == "all" else [args.stage]
    
    print("="*60)
    print(f"STARTING REPRODUCTION PIPELINE: {to_run}")
    print("="*60)
    
    for stage in to_run:
        script_path = STAGES[stage]
        if not Path(script_path).exists():
            print(f"Warning: Script {script_path} not found. Skipping.")
            continue
            
        run_script(script_path)
        
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
