import pandas as pd
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_PATH = os.path.join(os.getcwd(), "D2D_48_Task_Benchmark.xlsx")

# Precision levels from your suite
PRECISIONS = ["Novice", "10%", "5%", "2%"]

# Pattern combinations (Source_Pattern, Dest_Pattern)
# This fulfills the "one end steady, one end pulse" and SS/PP request
PATTERN_COMBOS = [
    ("Steady", "Steady"),
    ("Pulse", "Pulse"),
    ("Steady", "Pulse"),
    ("Pulse", "Steady")
]

# Speed Ranges from auto-d2s.py
SPEED_RANGES = {
    "V1": "0.02-0.04",
    "V2": "0.04-0.06",
    "V3": "0.06-0.08"
}

# High-Quality Dynamic Configurations (Removed easy Sync-Same-Speed)
# 1. Same Direction, Different Speeds (Async)
# 2. Opposite Directions (Opposite)
DYNA_CONFIGS = [
    {"Label": "Async-Intercept", "Direction": "Same", "V_Src": "V2", "V_Dst": "V1", "Difficulty": "Medium"},
    {"Label": "Async-Overtake", "Direction": "Same", "V_Src": "V1", "V_Dst": "V2", "Difficulty": "Medium"},
    {"Label": "Opposite-Pass", "Direction": "Opposite", "V_Src": "V2", "V_Dst": "V2", "Difficulty": "Hard"}
]

# ==============================================================================
# LOGIC: 4 Precisions * 4 Pattern Combos * 3 Dyna Configs = 48 Tasks
# ==============================================================================
data = []
task_id = 1

for precision in PRECISIONS:
    for src_p, dst_p in PATTERN_COMBOS:
        for config in DYNA_CONFIGS:
            
            row = {
                "Task ID": f"D2D_{task_id:02d}",
                "Suite": "D2D_Industry",
                "Precision": precision,
                "Source_Pattern": src_p,
                "Dest_Pattern": dst_p,
                "Combination_Type": f"{src_p[0]}-{dst_p[0]}", # e.g., S-P
                "Movement_Type": config["Label"],
                "Direction": config["Direction"],
                "Difficulty": config["Difficulty"],
                "V_Source Range (m/s)": SPEED_RANGES[config["V_Src"]],
                "V_Dest Range (m/s)": SPEED_RANGES[config["V_Dst"]],
                "Source_XY_Align": "[0.1, -0.2]", # Ghost alignment
                "Dest_XY_Align": "[0.1, 0.2]",   # Ghost alignment
                "Status": "Planned"
            }
            data.append(row)
            task_id += 1

# ==============================================================================
# EXPORT
# ==============================================================================
df = pd.DataFrame(data)
df.to_excel(OUTPUT_PATH, index=False)

print(f"Successfully generated {len(df)} High-Quality D2D task combinations.")
print(f"Saved to: {OUTPUT_PATH}")