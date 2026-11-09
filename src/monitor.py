import json
from pathlib import Path
from statistics import mean

path = Path("logs/predictions.jsonl")
if not path.exists():
    raise SystemExit("No log yet. Send some requests to /predict first.")

rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
uncertain = sum(r["uncertain"] for r in rows)

print("requests:", len(rows))
print("avg confidence:", round(mean(r["confidence"] for r in rows), 3))
print(f"uncertain: {uncertain} ({uncertain / len(rows):.0%})")
print("avg brightness:", round(mean(r["brightness"] for r in rows), 1))
print("avg size:", round(mean(r["width"] for r in rows)), "x", round(mean(r["height"] for r in rows)))
print("avg inference ms:", round(mean(r["inference_ms"] for r in rows), 1))