#!/usr/bin/env python3
import os, json, glob

# where your individual results.json live
INPUT_DIR = os.path.expanduser("~/Documents/local_results")
PATTERN   = os.path.join(INPUT_DIR, "*_results.json")

combined = {}

for fn in glob.glob(PATTERN):
    with open(fn, "r") as f:
        data = json.load(f)
    for relpath, info in data.items():
        view = info.get("predicted_view")
        if view:
            # use the path exactly as it appears in your JSON
            combined[relpath] = [view]

# write a single combined JSON
out_path = os.path.join(INPUT_DIR, "combined_views.json")
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)

print(f"✅ Wrote {len(combined)} entries to:\n  {out_path}")
