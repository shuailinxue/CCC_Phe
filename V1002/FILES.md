# Current V1002 repair files

- src/phenoniche/v1002/repair_primary.py: six-factor matching, primary protocol gate, per-view gradient and warm-start audits.
- src/phenoniche/v1002/repair_final.py: authorized staged solution, unchanged bulk/Cox, final-method-only robustness grid.
- src/phenoniche/v1002/repair_figures.py: primary maps, twin diagnostic, repaired-method main figure.
- tests/test_v1002_repair.py: analytic gradient audit, callback equivalence, and explicit background/gate tests.
- outputs/v1002_repair/: audit JSON, final model checkpoints, figures, and regression log.
# v2 controlled simulation

- `src/phenoniche/v1002/final_simulation.py`: fixed two-pair truth, cell-level spatial wrapper, geometry-only topology, ALR bulk truth.
- `src/phenoniche/v1002/final_model.py`: balanced joint matrix factorization and six-factor recovery gate.
- `src/phenoniche/v1002/final_experiment.py`: primary comparison, filtering audit, conditional bulk/Cox and robustness, result serialization.
- `src/phenoniche/v1002/final_figures.py`: four audit figures; failed panels are marked clearly.
- `tests/test_v1002_final_simulation.py`: hard-case truth, balancing, ALR identifiability, and gate tests.
- `outputs/v1002_final_sim/`: v2 primary records, explicit downstream stop status, figures, pytest log.
