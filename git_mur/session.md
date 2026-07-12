# Murmuration — Session Summary

**Date:** July 12, 2026
**Test suite:** 565 passed, 0 failures, 24 skipped, 26 deselected, 2 xfailed
**Doc-link tests:** 4/4 passed

---

## Session Overview

Seven major workstreams completed in this session:

| # | Workstream | Key Deliverable |
|---|---|---|
| 1 | Roadmap rework | Reading guide, 3D compliance, modularity "Ships:" lines, phase acceptance |
| 2 | CI/Docker rework | Guard-rails wired to test files, test.yml updated, gymnasium in reqs |
| 3 | Guard marker registration | `pytest.ini` with 9 markers, `pytest -m guard` selects 20 tests |
| 4 | regenerate_golden.py | Standalone CLI to regenerate all 5 golden .npz files |
| 5 | Migration merge | migration-reference.md + roadmap_steps.csv → Appendices G/H, old files deleted |
| 6 | Final roadmap audit | 15 phases, 119 steps, 8 appendices, 0 orphan refs |
| 7 | Session.md rewrite | This comprehensive summary |

---

## 1. Roadmap Rework (roadmap_deepseek.md)

### Changes applied

| Change | Detail |
|---|---|
| **Reading Guide** (top) | 9-entry table: "Where to start based on your goal" — understand the plan, implement P0, find a module, check acceptance, look up a term, see the architecture, migration, schedule, or CI |
| **3D Compliance** (Design Guide) | Explicit constraints: no `(N,2)` arrays in `physics/`, `depth>0` validator, SO(3) invariance tests, 3D boxsize, Θ′ is the only 2D projection (diagnostic only in `analysis/`), all rendering is 3D |
| **"Ships:" lines** (15 phases) | Each phase header declares what independently testable module it delivers |
| **Phase Acceptance** (Design Guide) | Cross-links to Appendix F (229 CI-auditable checkboxes) |

### Architecture verification

| Concern | Result |
|---|---|
| Micro→Macro (L0→L3) | Clean DAG: `core(L0) → physics atoms(L0) → assemblies(L1) → subsystems(L2) → system(L3)` |
| Macro→Micro (7 subsystems) | All 119 steps target exactly one subsystem (A–F2) |
| Modularity | Each phase ships independently testable modules |
| Strictly 3D | All simulation math in 3D `(N,3) float32`, z-up. All viz in 3D (ModernGL) |

---

## 2. CI & Docker Rework

### guard-rails.yml

| Guard | Status |
|---|---|
| guard-rail-golden | Wired to `test_golden.py` (replaced inline fallback) |
| guard-rail-config-drift | Verified — already references `test_config_drift.py` |
| guard-rail-collection-count | Verified — already references `test_collection_count.py` |

### test.yml

- `test-imports` job: Added `test_architecture.py`, `test_docs.py`, `test_golden.py`

### requirements-optional.txt

- Added `gymnasium>=0.29,<2.0` for MARL (P12)

### test.md

- Updated CI fast suite command to include new test files
- Added guard-rail test section (`pytest test/test_architecture.py test/test_docs.py test/test_golden.py -v`)
- Added `'golden'` marker documentation

---

## 3. Guard Marker Registration

### pytest.ini (created)

9 registered markers:
```
slow, gl, gpu, golden, guard, numba, pygame, e2e, integration
```

### Test files marked

| File | Marker |
|---|---|
| `test/test_architecture.py` | `pytestmark = pytest.mark.guard` |
| `test/test_docs.py` | `pytestmark = pytest.mark.guard` |
| `test/test_golden.py` | `pytestmark = [pytest.mark.golden, pytest.mark.guard]` |
| `test/test_config_drift.py` | `pytestmark = pytest.mark.guard` + skip |
| `test/test_collection_count.py` | `pytestmark = pytest.mark.guard` + skip |

### Usage

```bash
pytest -m guard          # 20 selected, 18 passed, 2 xfailed
pytest -m golden         # 11 selected, 9 passed, 2 xfailed
pytest -m "guard and not golden"  # Guard tests excluding golden
```

---

## 4. regenerate_golden.py

Standalone CLI script at `test/regenerate_golden.py`.

### Features

| Feature | Detail |
|---|---|
| Default run | All 5 modes, seed=77, N=15, 30 frames |
| Single mode | `--mode projection` |
| Custom params | `--seed 42 --birds 20 --frames 60` |
| Dry run | `--dry-run` |
| Verification | Shape, dtype, NaN check, bounds check |
| Exit code | 0 = all OK, 1 = any mode failed |
| Future modes | TODO comments for angle (P5) and marl (P12) |

### Golden files (test/data/)

| File | Mode | Non-deterministic? |
|---|---|---|
| `golden_projection.npz` | projection | No |
| `golden_spatial.npz` | spatial | No |
| `golden_field.npz` | field | No |
| `golden_vicsek.npz` | vicsek | Yes (until P0.4) |
| `golden_influencer.npz` | influencer | Yes (until P0.4) |

---

## 5. Migration Merge

### What was done

- `migration-reference.md` → merged into `roadmap_deepseek.md` as **Appendix G** ("Old→New Identifier Mapping")
- `roadmap_steps.csv` → merged into `roadmap_deepseek.md` as **Appendix H** ("Step Index" — 119-step flat table)
- Both source files deleted
- 0 orphan references found across entire codebase (verified by code searcher)

### Appendix G — Old→New Identifier Mapping

Maps the retired `roadmap.md` scheme (D0–D9, S1–S7, T0–T6) to the new `roadmap_deepseek.md` scheme (P0–P14). Detailed mapping tables for:
- Architecture Foundation (D0→P0–P2)
- Test Infrastructure (T0–T6→P0, P2, P8, P13)
- Science Portfolio (S1–S7→P3–P12)

### Appendix H — Step Index

Flat markdown table: Phase | Step | Title | Level | Files | Test Files | Citations

---

## 6. Final Roadmap Audit

### Statistics

| Metric | Value |
|---|---|
| Lines | 3,720 |
| Phase headers | 15 (P0–P14) |
| Steps | 119 |
| Appendices | 8 (A–H) |
| Parts | 2 (Part I: Current State, Part II: Phases) |

### Reference integrity

| Reference | Count | Status |
|---|---|---|
| `arch.md` | 33 | Valid |
| `sci/todo_claude*.md` | 56 | Valid |
| `.github/gantt-schedule.md` | 3 | Valid |
| `migration-reference.md` | 0 | Deleted — content in Appendix G |
| `roadmap_steps.csv` | 0 | Deleted — content in Appendix H |
| Internal heading anchors | 9 (Reading Guide) | All resolve |

### Appendices index

| Appendix | Content |
|---|---|
| A | Sci/ document traceability |
| B | Force term reference |
| C | Phase dependency graph |
| D | Glossary |
| E | Module→Phase reverse index |
| F | Phase boundary checklists (229 items) |
| G | Old→New identifier mapping |
| H | Step index (119 steps) |

---

## 7. Test Suite

### Final state

| Metric | Value |
|---|---|
| Fast tests passed | **565** |
| Skipped | 24 |
| Xfailed | 2 (vicsek, influencer — non-deterministic) |
| Deselected (slow) | 26 |
| Doc-link tests | 4/4 |
| `pytest -m guard` | 20 selected, 18 passed, 2 xfailed |
| `pytest -m golden` | 11 selected, 9 passed, 2 xfailed |
| Marker warnings | 0 |

---

## Files Changed This Session

| File | Change |
|---|---|
| `roadmap_deepseek.md` | Reading Guide, 3D compliance, "Ships:" lines, phase acceptance ref, Appendices G/H merged |
| `session.md` | This comprehensive rewrite |
| `.github/workflows/guard-rails.yml` | Wired test_golden.py into golden guard |
| `.github/workflows/test.yml` | Added 3 test files to test-imports |
| `requirements-optional.txt` | Added gymnasium |
| `test.md` | Updated CI command, added guard-rail section, golden marker |
| `pytest.ini` | **Created** — 9 registered markers |
| `pyproject.toml` | Removed redundant marker section |
| `test/test_architecture.py` | Added `import pytest`, `pytestmark = pytest.mark.guard` |
| `test/test_docs.py` | Added `import pytest`, `pytestmark = pytest.mark.guard` |
| `test/test_golden.py` | Added `pytestmark = [pytest.mark.golden, pytest.mark.guard]` |
| `test/test_config_drift.py` | Added `pytestmark = pytest.mark.guard` |
| `test/test_collection_count.py` | Added `pytestmark = pytest.mark.guard` |
| `test/regenerate_golden.py` | **Created** — stand-alone golden generation script |

### Files deleted

| File | Reason |
|---|---|
| `migration-reference.md` | Merged into Appendix G |
| `roadmap_steps.csv` | Merged into Appendix H |

---

## Quick-Start for Next Session

```bash
# Run all guard-rail tests
pytest -m guard -v

# Run golden trajectory tests
pytest -m golden -v

# Full fast suite
pytest test/ -q -m 'not slow'

# Regenerate golden files after physics changes
python test/regenerate_golden.py

# Doc-link integrity
pytest test/test_docs.py -v

# Architecture enforcement
pytest test/test_architecture.py -v
```

---

*Session completed July 12, 2026. All changes verified with 565 passing tests, 0 failures.*
