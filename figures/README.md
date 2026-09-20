# Figures

- `d4a_phase_alignment`: formal D4a q90 threshold-error comparison across the eight split × variant × K cells. Source: `results/d4a_repair_alignment.csv`.
- `d4b_residual_desynchronization`: report-level grouped D4b summary across residual strength and split. Source: `results/d4b_strength_summary_report.csv`. It is explicitly not a formal raw-data replacement.

Regenerate with:

```powershell
python .\repro\make_figures.py
```
