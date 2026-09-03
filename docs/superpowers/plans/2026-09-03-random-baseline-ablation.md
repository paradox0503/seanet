# Random Baseline Ablation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline in the current task. Steps use checkbox (`- [ ]`) syntax for tracking. The user has approved the design and requested implementation; do not create another task or commit automatically.

**Goal:** Add a full-pool random baseline with the same number of optimizer-update batches as the existing loss curriculum.

**Architecture:** Put index-only scheduling in `util/batch_schedule.py`; keep PyTorch permutation generation and all model training in `Experiment`. Select the schedule by configuration and use a separate baseline configuration directory for outputs.

**Tech Stack:** Python, existing PyTorch training code, standard-library unittest.

## Global Constraints

- Default `training_schedule` is `loss_curriculum`; the alternative is `random_baseline`.
- Budget is `max(1, total_num_batches // 10)` batches per formal epoch.
- Curriculum boundaries remain 80 epochs, 8 stages, 10 epochs per stage.
- Random baseline skips loss evaluation and uses all original batch indices every epoch.
- Model, loss, optimizer, query construction, validation and checkpoint timing stay unchanged.
- Baseline logs, checkpoints, exported encoder and elapsed-time log must not overwrite curriculum outputs.
- Do not install dependencies or execute real training locally. PyTorch is unavailable.
- Use tests before production changes. Do not commit or push.

## File Map

- Create `util/batch_schedule.py`: pure index budget, candidate pool and selection.
- Modify `util/experiment.py`: conditional initial scoring, shared schedule helpers, configured encoder export.
- Modify `util/conf.py`: schedule default and legal values.
- Modify `conf/example.json`: explicit legacy schedule.
- Create `conf/random_baseline/example.json`: same hyperparameters, random schedule and separate outputs.
- Modify `run.py`: elapsed-time log beside the configured training log.
- Create `tests/test_batch_schedule.py`: real pure scheduling behavior.
- Create `tests/source_loader.py`: load selected real methods without GPU-only imports.
- Create `tests/test_random_baseline.py`: execute real run/configuration methods with lightweight external-operation doubles.
- Modify `README.md`: paired commands, output locations, equal-update-budget definition and fresh-run warning.

## Task 1: Pure Index Scheduling

**Interfaces:**
- `batches_per_epoch(total_num_batches, divisor=10) -> int`
- `candidate_batch_indices(schedule, ordered_indices, epoch, curriculum_epochs=80, stages=8) -> list`
- `select_batch_indices(candidate_indices, batch_count, shuffled_positions) -> list`

- [x] Write tests with literal boundaries and original IDs, including:

```python
self.assertEqual(batches_per_epoch(15630), 1563)
self.assertEqual(candidate_batch_indices('loss_curriculum', list(range(80)), 9), list(range(10)))
self.assertEqual(candidate_batch_indices('loss_curriculum', list(range(80)), 10), list(range(20)))
self.assertEqual(candidate_batch_indices('random_baseline', [7, 2, 99], 0), [7, 2, 99])
self.assertEqual(select_batch_indices([7, 2, 99, 6], 2, [2, 0]), [99, 7])
```

- [x] Run `python -m unittest discover -s tests -p test_batch_schedule.py -v`; confirm the missing scheduling behavior fails.
- [x] Implement budget validation for positive integers and the exact integer floor budget.
- [x] Implement the candidate pool using:

```python
stage = min(epoch // (curriculum_epochs // stages), stages - 1)
pool_end = (len(ordered_indices) * (stage + 1) + stages - 1) // stages
```

Return the full pool for random baseline or epoch >= curriculum_epochs. Reject unknown schedules, negative/noninteger epoch and invalid stage settings.
- [x] Implement selection: validate nonempty unique IDs, positive budget and unique in-range integer positions; return all candidates if the pool fits, otherwise map the first budget positions back to original IDs. Do not mutate inputs.
- [x] Run scheduling tests until green; include tiny pools, remainder batches and invalid inputs.

## Task 2: Runtime Selection and Output Isolation

**Consumes:** Task 1 scheduling functions and existing `Configuration` / `Experiment`.
**Produces:** Configurable runtime behavior plus a separately runnable baseline.

- [x] Add standard-library tests that execute actual extracted `Experiment.run`, configuration methods and `run.main` without importing PyTorch. Double only unavailable GPU/model work; assert selected loader IDs, completed epochs and real output files.
- [x] Baseline run fixture: 20 batches, max_epoch=2, a deterministic reverse permutation; assert both training calls receive IDs [19, 18], scoring is never called, and encoder export uses the configured file. Curriculum fixture: hand-authored scores put IDs [10, 9, 8] first; assert first-epoch selected IDs are [8, 9].
- [x] Configuration tests assert defaults choose curriculum, explicit baseline is accepted, unknown schedule is rejected, all non-output hyperparameters match, and output directories differ. Elapsed-time test runs real `main` against a temporary log directory and checks `0time.log` there.
- [x] Run `python -m unittest discover -s tests -v`; confirm failures identify missing baseline branch/config and fixed output paths.
- [x] Add to configuration defaults and legal values:

```python
'training_schedule': 'loss_curriculum'
# legal values:
'training_schedule': {'loss_curriculum', 'random_baseline'}
```

- [x] In pretrain run, read/validate schedule and compute the shared budget. Keep the entire initial scoring block only inside `if schedule == 'loss_curriculum'`; use `list(range(total_num_batches))` otherwise. Print/log a schedule summary and explicit baseline skip message.
- [x] Obtain candidate indices through the helper each epoch. Generate positions only when pool > budget:

```python
positions = torch.randperm(len(pool))[:batch_size_per_round].tolist()
selected_indices = select_batch_indices(pool, batch_size_per_round, positions)
```

Use an empty position list when the pool fits. Keep curriculum stage logs only in curriculum mode. Leave all code after loader construction unchanged.
- [x] Replace hardcoded encoder output with:

```python
pickle_path = os.path.abspath(self.__conf.getHP('pickle'))
os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
with open(pickle_path, 'wb') as f:
    pickle.dump(self.model._AEBuilder__encoder, f)
```

- [x] Add `training_schedule` to the course example. Copy its configuration values into `conf/random_baseline/example.json`, changing only schedule, name, pickle and result_path. Use `conf/random_baseline/our_pretrain.pkl` and `conf/random_baseline/` for the latter two paths; default output resolution follows the new config parent.
- [x] Set elapsed-time path in `run.py` using `Path(conf.getHP('log_filepath')).with_name('0time.log')`.
- [x] Run the full test suite; verify both schedules use the same actual training method and update budget. No full PyTorch integration claim.

## Task 3: Documentation and Final Verification

- [x] Add commands to README:

```bash
python run.py -C conf/example.json
python run.py -C conf/random_baseline/example.json
```

Explain that runs share cached samples, output folders differ, old checkpoints resume automatically, and equal budget excludes the curriculum's extra no-gradient scoring pass.
- [x] Run `python -m unittest discover -s tests -v` and `python -m compileall -q run.py util model tests`.
- [x] Run `git diff --check`, review the diff against the approved spec, and confirm the body of `__train`, `__validate`, optimizer and loss remain unchanged.
- [x] Review output isolation and tests before delivery; state that real GPU training remains unverified. Leave changes uncommitted.

## Execution Results (2026-09-03)

- Test-first evidence: 15 scheduling tests failed for the missing implementation, then passed. Seven integration/config/output assertions failed against the old behavior, then passed after the branch and path changes.
- Final standard-library suite: 23 tests passed. Control-flow tests execute real selected methods with GPU work replaced; they do not prove full PyTorch integration.
- Compileall and Git whitespace checks passed.
- Differential scheduling check: 50,601 cases matched the legacy curriculum pool rule and preserved equal update budgets; baseline pools were always full.
- AST comparison confirmed the training loss, training/validation methods, optimizer, LR/WD/SRIP adjustments and checkpoint method were unchanged.
- Independent read-only review found no critical or important issues. No dependency installation, real training, commit, push or branch change was performed.
