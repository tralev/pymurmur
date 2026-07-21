"""Part VII whole-system check — CI workflow integrity.

Every other guard test verifies "does job X's own test/logic pass."
This file verifies the *pipeline* — `.github/workflows/*.yml` as CI would
actually execute it — because a per-job check can't catch bugs that only
exist in how jobs are wired together or in scripts embedded directly in
the YAML (not backed by a pytest file at all).

This is exactly how 4 real bugs were found during a 2026-07-21
whole-system pass (see roadmap_deepseek.md's Part VII entry for that
date): a YAML line-folding artifact that garbled `guard-rails-summary`'s
log output, two dead-code fallback branches with heredocs whose
terminator was indented (bash requires an exact unindented match for
plain `<<`), a silent no-op (`cfg.flock.seed = 42` — not a real field)
inside one of those fallbacks, and a broken AST class-scoping loop inside
the other. All four were only found by extracting each job's *resolved*
run script and actually executing it — including forcing dead `else`
branches to run for real — not by reading the YAML or checking job names
line up in docs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.guard

WORKFLOW_FILES = [
    Path(".github/workflows/guard-rails.yml"),
    Path(".github/workflows/test.yml"),
]

GHA_EXPR = re.compile(r"\$\{\{[^}]*\}\}")

# `if [ -f <marker file> ]; then <primary> else <fallback> fi` — the
# pattern used throughout guard-rails.yml so a job degrades to a manual
# equivalent when its dedicated test file doesn't exist yet.
IF_DASH_F = re.compile(r"if \[ -f ([^\]]+?) \]; then")


def _sanitize(script: str) -> str:
    """Replace GitHub Actions `${{ ... }}` expressions with a dummy
    literal so `bash -n`/`bash` can parse the script — GHA substitutes
    these before the shell ever sees them, so raw `${{ }}` is not valid
    bash on its own."""
    return GHA_EXPR.sub("DUMMY", script)


def _iter_job_steps():
    """Yield (workflow_path, job_name, step_index, step_name, run_script)
    for every step with a `run:` key across both workflow files."""
    for wf in WORKFLOW_FILES:
        doc = yaml.safe_load(wf.read_text())
        for job_name, job in doc["jobs"].items():
            for i, step in enumerate(job.get("steps", [])):
                run = step.get("run")
                if run:
                    yield wf, job_name, i, step.get("name", "?"), run


_JOB_STEPS = list(_iter_job_steps())
_JOB_STEP_IDS = [f"{wf.name}:{job}:{i}" for wf, job, i, _name, _run in _JOB_STEPS]


# ── Every job's run script is valid bash ──────────────────────────


@pytest.mark.parametrize(
    "wf, job_name, step_i, step_name, run", _JOB_STEPS, ids=_JOB_STEP_IDS
)
def test_job_run_script_is_valid_bash(wf, job_name, step_i, step_name, run):
    """Every job step's `run:` script is syntactically valid bash once
    GitHub Actions expressions are stripped out.

    Catches YAML line-folding artifacts (an unescaped physical line
    break silently turning into a space instead of a newline — this is
    exactly what garbled `guard-rails-summary`'s status output) and
    heredocs with a mismatched terminator (bash's plain `<<` requires an
    exact match — an indented `PYEOF` under an unindented `<< 'PYEOF'`
    produces "unexpected end of file", not a clean error).
    """
    sanitized = _sanitize(run)
    proc = subprocess.run(
        ["bash", "-n"], input=sanitized, capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"{wf}::{job_name} step {step_i} ({step_name}) is not valid bash:\n"
        f"{proc.stderr}"
    )


# ── Heredoc terminators actually match their opener ───────────────


def test_heredoc_terminators_are_not_indented():
    """No job script opens a heredoc with plain `<<` (not `<<-`) and
    then indents the terminator line — bash requires an exact,
    unindented match for plain `<<`, so an indented terminator causes
    "unexpected end of file" if the branch containing it ever runs.

    This is the precise, generalized version of the bug found in two
    dead fallback branches (determinism smoke-test, config-drift scan):
    both opened `python3 << 'PYEOF'` and closed with `  PYEOF` (two
    leading spaces) — silently broken until the branch was actually
    forced to execute.
    """
    failures: list[str] = []
    for wf, job_name, step_i, _step_name, run in _JOB_STEPS:
        lines = run.split("\n")
        for j, line in enumerate(lines):
            m = re.search(r"<<(-?)\s*'?(\w+)'?\s*$", line)
            if not m:
                continue
            dash, marker = m.group(1), m.group(2)
            for k in range(j + 1, len(lines)):
                if lines[k].strip() == marker:
                    if not dash and lines[k] != marker:
                        failures.append(
                            f"{wf}::{job_name} step {step_i}: heredoc <<{marker} "
                            f"terminator is indented ({lines[k]!r}) but opened "
                            f"with plain << (requires exact match, not <<-)"
                        )
                    break
            else:
                failures.append(
                    f"{wf}::{job_name} step {step_i}: heredoc <<{marker} has "
                    f"no matching terminator line at all"
                )

    assert not failures, "\n" + "\n".join(failures)


# ── Dead fallback branches actually run, not just parse ───────────


def _force_else_branch(run: str, marker_file: str) -> str:
    """Rewrite `if [ -f <marker_file> ]; then` to reference a path that
    never exists, forcing the script's `else` (fallback) branch."""
    return run.replace(f"if [ -f {marker_file} ]; then", "if [ -f /nonexistent_xyz_probe ]; then")


def _fallback_jobs_with_heredocs():
    """Jobs with an `if [ -f ... ]; then ... else ... fi` fallback that
    also embeds a heredoc — the fallback branches with actual Python
    logic (not just an alternate pytest invocation), and therefore the
    ones where a semantic bug (not just a syntax bug) could hide."""
    out = []
    for wf, job_name, step_i, step_name, run in _JOB_STEPS:
        m = IF_DASH_F.search(run)
        if m and "<<" in run:
            out.append((wf, job_name, step_i, step_name, run, m.group(1)))
    return out


_FALLBACK_HEREDOC_JOBS = _fallback_jobs_with_heredocs()
_FALLBACK_IDS = [f"{wf.name}:{job}:{i}" for wf, job, i, _n, _r, _m in _FALLBACK_HEREDOC_JOBS]


@pytest.mark.parametrize(
    "wf, job_name, step_i, step_name, run, marker_file",
    _FALLBACK_HEREDOC_JOBS,
    ids=_FALLBACK_IDS,
)
def test_dead_fallback_branch_actually_runs(wf, job_name, step_i, step_name, run, marker_file):
    """Force a fallback branch's `else` path (its marker file always
    exists today, so this branch never runs in real CI) and execute the
    resolved script for real — not `bash -n`, an actual run — to prove
    it doesn't *crash*.

    `bash -n` only proves the script *parses*; it caught the heredoc
    terminator bug but would have missed both semantic bugs found in
    the same branches: a silent no-op (`cfg.flock.seed = 42`, not a
    real field) and a broken AST scoping loop (every field misattributed
    to one class). Only real execution catches those.

    Deliberately does NOT assert `returncode == 0` — a fallback whose
    business logic finds a real problem (e.g. the config-drift scan
    legitimately reporting orphans, given its own known heuristic
    limitations) is supposed to `sys.exit(1)`; that is not a crash. The
    crash signal is an uncaught Python traceback or a bash-level fatal
    error, checked below.
    """
    forced = _force_else_branch(run, marker_file)
    assert "/nonexistent_xyz_probe" in forced, (
        f"Marker-file substitution didn't match — test fixture is stale "
        f"for {wf}::{job_name} step {step_i}"
    )
    proc = subprocess.run(
        ["bash", "-c", forced], capture_output=True, text=True, timeout=60
    )
    assert "Traceback" not in proc.stdout and "Traceback" not in proc.stderr, (
        f"{wf}::{job_name} step {step_i} ({step_name}) fallback branch "
        f"raised an uncaught Python exception:\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.returncode in (0, 1), (
        f"{wf}::{job_name} step {step_i} ({step_name}) fallback branch "
        f"exited {proc.returncode} — expected 0 (passed) or 1 (a clean, "
        f"intentional sys.exit(1)), not a bash-level fatal error:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_determinism_fallback_smoke_test_actually_passes():
    """The determinism fallback isn't just crash-free — its own
    assertion (`same seed -> bit-identical positions`) must genuinely
    hold, since unlike config-drift's heuristic-quality caveat, this one
    has an unambiguous correct answer.  Regression guard for the exact
    bug found: `cfg.flock.seed = 42` is a silent no-op (not a real
    `FlockConfig` field), so both engines got fresh, divergent entropy
    and the assertion always failed until fixed to `cfg.seed = 42`.
    """
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    run = doc["jobs"]["guard-rail-golden"]["steps"][6]["run"]
    forced = _force_else_branch(
        run, "test/l4_crosscutting/guards/test_determinism.py"
    )
    assert "/nonexistent_xyz_probe" in forced

    proc = subprocess.run(
        ["bash", "-c", forced], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "Determinism smoke test passed" in proc.stdout


def test_config_drift_fallback_attributes_fields_to_correct_class():
    """The config-drift fallback's field→class attribution is correct —
    regression guard for the exact bug found: a shared mutable
    `dataclass_stack` pushed during a flat (non-block-scoped) `ast.walk`
    meant every field in the file got attributed to whichever dataclass
    was visited last (confirmed: all 180+ fields were mislabeled as
    `CaptureConfig`).  Doesn't require the scan's orphan-detection
    heuristic itself to be accurate (a separate, known limitation) —
    only that a real field lands under its real class."""
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    run = doc["jobs"]["guard-rail-config-drift"]["steps"][3]["run"]
    forced = _force_else_branch(
        run, "test/l4_crosscutting/guards/test_config_drift.py"
    )
    assert "/nonexistent_xyz_probe" in forced

    proc = subprocess.run(
        ["bash", "-c", forced], capture_output=True, text=True, timeout=60
    )
    # A handful of fields with well-known, distinct owning classes —
    # if the scoping bug regresses, these would all collapse onto one
    # (previously: whichever class was defined last in config.py).
    assert "SpatialConfig.separation_weight" in proc.stdout
    assert "FieldConfig.field_noise" in proc.stdout
    assert "VicsekConfig.vicsek_diffusion" in proc.stdout
    assert "DomainConfig.width" in proc.stdout


# ── guard-rails-summary's merge-gate aggregation is correct ───────


def _load_summary_script() -> str:
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    return doc["jobs"]["guard-rails-summary"]["steps"][0]["run"]


def _guard_job_names() -> list[str]:
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    return sorted(
        j for j in doc["jobs"]["guard-rails-summary"]["needs"]
    )


def _render_summary(results: dict[str, str]) -> str:
    script = _load_summary_script()
    for job in _guard_job_names():
        placeholder = "${{ needs.%s.result }}" % job
        script = script.replace(placeholder, results.get(job, "success"))
    assert "${{" not in script, "left-over unrendered GHA expression"
    return script


@pytest.mark.parametrize(
    "scenario, override, expected_exit",
    [
        ("all success", {}, 0),
        ("single failure", {"guard-rail-composers": "failure"}, 1),
        ("single cancelled", {"guard-rail-mypy": "cancelled"}, 1),
        ("first job failed", {"guard-rail-dag": "failure"}, 1),
        (
            "multiple failures",
            {
                "guard-rail-dag": "failure",
                "guard-rail-composers": "failure",
                "guard-rail-evolved": "cancelled",
            },
            1,
        ),
        ("skipped is not a failure", {"guard-rail-evolved": "skipped"}, 0),
    ],
)
def test_guard_rails_summary_aggregation(scenario, override, expected_exit):
    """`guard-rails-summary`'s merge-gate bash logic correctly aggregates
    every combination of job results — not just that job names are
    listed, but that the actual pass/fail arithmetic is right.

    Runs the real, resolved bash script (GitHub Actions `${{ }}`
    expressions substituted with literal result strings, exactly as GHA
    itself would do before invoking the shell) via a real `bash`
    subprocess for each scenario.
    """
    results = {job: "success" for job in _guard_job_names()}
    results.update(override)
    script = _render_summary(results)

    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode == expected_exit, (
        f"scenario '{scenario}': exit={proc.returncode}, expected {expected_exit}\n"
        f"{proc.stdout}"
    )


def test_guard_rails_summary_echo_lines_are_not_concatenated():
    """Every status echo appears on its own output line — regression
    guard for the exact bug found: a missing newline between the Mypy
    and Evolved-artifact echo statements silently merged them into one
    bash command (`echo "A" echo "B"` — a single `echo` call with the
    literal word "echo" as one of its arguments), garbling the log."""
    script = _render_summary({})
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    assert not any(" echo " in line for line in lines), (
        f"An echo'd line contains the literal word 'echo' — statements were "
        f"concatenated instead of newline-separated:\n{proc.stdout}"
    )
    # Every named guard-rail job gets its own status line.
    status_lines = [line for line in lines if line.rstrip().endswith("success")]
    assert len(status_lines) == len(_guard_job_names()), (
        f"Expected one status line per job ({len(_guard_job_names())}), "
        f"got {len(status_lines)}:\n{proc.stdout}"
    )


# ── needs graph is structurally sound ──────────────────────────────


def test_needs_graph_has_no_dangling_references():
    """Every job's `needs:` entries reference jobs that actually exist
    in the workflow — a renamed/removed job would otherwise silently
    break the dependency graph (GitHub Actions itself would reject this
    at workflow-parse time, but failing fast in a unit test is faster
    to diagnose than a red CI run)."""
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    jobs = doc["jobs"]
    for job_name, job in jobs.items():
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        for n in needs:
            assert n in jobs, f"{job_name} needs undefined job '{n}'"


def test_guard_rails_summary_covers_every_other_job():
    """`guard-rails-summary` depends on (and its bash loop checks) every
    other job in the file — a newly-added guard job that's forgotten
    from the summary's `needs` list would silently never block a merge
    on its own failure."""
    doc = yaml.safe_load(Path(".github/workflows/guard-rails.yml").read_text())
    jobs = doc["jobs"]
    other_jobs = set(jobs) - {"guard-rails-summary"}
    summary_needs = set(jobs["guard-rails-summary"]["needs"])
    assert other_jobs == summary_needs, (
        f"Jobs not covered by guard-rails-summary's needs: {other_jobs - summary_needs}\n"
        f"needs entries with no matching job: {summary_needs - other_jobs}"
    )
