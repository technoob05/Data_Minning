
import pandas as pd

def inspect_event_order():
    pr_id = 3205734508 # From previous output
    print(f"Inspecting events for PR {pr_id}...")
    try:
        df = pd.read_parquet("msr26-aidev-triage/data/pr_timeline.parquet")
        events = df[df["pr_id"] == pr_id].copy()
        
        # Determine if there is a 'created_at' to sort by?
        # If created_at is mixed (some null), we rely on default sort?
        # Let's print them as they are in the DF, assuming DF order is load order.
        
        print(f"Total events: {len(events)}")
        print(events[["event", "created_at", "actor"]].to_string())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_event_order()
