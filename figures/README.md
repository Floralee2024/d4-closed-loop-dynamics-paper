# Figures

- `d4a_phase_alignment`: formal D4a q90 threshold-error comparison across the eight split × variant × K cells. Source: `results/d4a_repair_alignment.csv`.
- `d4b_residual_desynchronization`: formal grouped D4b summary of residual-associated operational marker separation across residual strength and split. The filename is retained for artifact compatibility. Source: `results/d4b_formal_gpu/d4b_strength_summary.csv`. It is a grouped summary, not a formal raw-data replacement.

Regenerate with:

```powershell
python .\repro\make_figures.py
```
