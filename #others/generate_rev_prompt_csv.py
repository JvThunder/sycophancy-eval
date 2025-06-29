import json, csv, pathlib, re
from pathlib import Path

def extract_prompt(json_path: Path) -> str | None:
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('metadata', {}).get('reversal_prompt')
    except Exception:
        return None

rows = []
for folder in sorted(Path('.').glob('mcq_outputs_*')):
    first_json = next(folder.glob('*.json'), None)
    if not first_json:
        continue
    prompt = extract_prompt(first_json)
    rows.append((folder.name, prompt))

csv_path = Path('reversal_prompts_summary.csv')
with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['folder', 'reversal_prompt'])
    writer.writerows(rows)

print(f'Wrote summary CSV to {csv_path.resolve()}') 