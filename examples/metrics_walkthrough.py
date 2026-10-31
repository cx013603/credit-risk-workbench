"""Recompute denominators from saved aggregate counts; standard library only."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / 'evidence/results-summary.json').read_text(encoding='utf-8'))
    model = evidence['validation']['extended71']
    c = model['threshold']
    tp, fp, tn, fn = (c[k] for k in ('TP', 'FP', 'TN', 'FN'))
    assert tp + fp + tn + fn == model['rows']
    assert tp + fn == model['events']
    rates = {
        'recall': tp / (tp + fn),
        'false_positive_rate': fp / (fp + tn),
        'precision': tp / (tp + fp),
        'selected_fraction': (tp + fp) / model['rows'],
    }
    for name, value in rates.items():
        assert abs(value - c[name]) < 1e-12, name
        print(f'{name}: {value:.2%}')
    ks = rates['recall'] - rates['false_positive_rate']
    assert abs(ks - model['ks']) < 1e-12
    print(f'KS at the saved threshold: {ks:.6f}')
    print('Scope: reused validation evidence; no new evaluation or inference.')


if __name__ == '__main__':
    main()
