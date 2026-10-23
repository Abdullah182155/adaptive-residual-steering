
new_getting_started = '''## 🚀 Getting Started

### Local Setup
```bash
git clone https://github.com/Abdullah182155/adaptive-residual-steering.git
cd adaptive-residual-steering
git checkout development
pip install -r requirements.txt
python -m unittest discover -s tests -v  # unit + hygiene tests
```

### Full Reproduction
```bash
# 1. Train (S0 -> S1 -> S0b -> S2 -> S3)
python scripts/run_ars_pipeline.py --stage all  # runs every stage incl. S5 + bench

# ...or stage by stage:
python scripts/run_ars_pipeline.py --stage S0
python scripts/run_ars_pipeline.py --stage S1
python scripts/run_ars_pipeline.py --stage S0b
python scripts/run_ars_pipeline.py --stage S2
python scripts/run_ars_pipeline.py --stage S3

# 2. Calibrate the deployment contract (selection split n=120)
python scripts/run_calibration.py --contract_path deployment_contract.json

# 3. 5-arm benchmark (GSM8K test n=1319; 0- and 8-shot; add --secondary for 2/4-shot)
python scripts/run_benchmark.py --contract deployment_contract.json --out experiment_record.json

# 4. Report (Markdown + LaTeX)
python scripts/make_report.py --record experiment_record.json --out_dir report
```

Outputs: `deployment_contract.json`, `experiment_record.json`, `report/REPORT.md`, `report/tables/*.tex`.

Generations are cached in `checkpoints/bench_cache`, so interrupted benchmark runs resume without recomputation.

### Reproducible Kaggle Execution
See [notebooks/kaggle_ars_runner.ipynb](notebooks/kaggle_ars_runner.ipynb) for a self-contained execution notebook.

---
'''

lines = open('README.md', 'r', encoding='utf-8').readlines()
gs_idx = None
for i, l in enumerate(lines):
    if '## ' in l and 'Getting Started' in l:
        gs_idx = i
        break

# Find next ## section or end
next_section = len(lines)
for i in range(gs_idx + 1, len(lines)):
    if lines[i].startswith('## ') and 'Getting Started' not in lines[i]:
        next_section = i
        break

# Replace
new_lines = lines[:gs_idx] + [new_getting_started] + lines[next_section:]
open('README.md', 'w', encoding='utf-8').writelines(new_lines)
print(f'Replaced lines {gs_idx+1}-{next_section} with new Getting Started section')
print(f'New total lines: {len(new_lines)}')
