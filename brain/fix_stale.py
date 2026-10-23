lines = open('scripts/run_experiment.py', 'r', encoding='utf-8').readlines()
fixed = 0
for i, l in enumerate(lines):
    if 'set_routing_override' in l and '"all"' in l:
        lines[i] = l.replace('"all"', '"only", model.target_layers')
        fixed += 1
        print(f'fixed line {i+1}: {lines[i].rstrip()}')
open('scripts/run_experiment.py', 'w', encoding='utf-8').writelines(lines)
print(f'done, {fixed} fix(es)')
