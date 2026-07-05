# Competition utilities

This directory contains practical, opt-in scripts that use `ovvo-nns` outside the core package API.

## NeuroGolf 2026 starter

`neurogolf_2026_entry.py` generates a Kaggle NeuroGolf-style starter submission by combining two NNS passes:

1. `nns.nns_part` estimates spatial cell importance from the identity residual, so the script can identify where a grid task is active.
2. `nns.nns_dep` ranks candidate local stencil offsets on those important cells, so the script can fit a sparse one-hot convolution for same-size local rules.

The script then exports one ONNX model per task and writes `submission.zip` plus a JSON diagnostics report.

Install the competition extras before running it:

```bash
pip install ovvo-nns onnx numpy
```

Run on Kaggle data:

```bash
python examples/competitions/neurogolf_2026_entry.py \
  --data_dir /kaggle/input/neurogolf-2026 \
  --out submission.zip \
  --report nns_neurogolf_report.json
```

Run the built-in smoke-test demo:

```bash
python examples/competitions/neurogolf_2026_entry.py \
  --make_demo_data demo_neurogolf_data \
  --out demo_submission.zip \
  --report demo_report.json
```

The script is intentionally conservative. It attempts same-size local rules and falls back to a 1x1 identity or zero convolution when a task is not solved exactly on the available examples. The report is the important first artifact: it shows which tasks NNS solved as sparse local rules and which tasks need object, crop, symmetry, connected-component, or mask-algebra primitives.
