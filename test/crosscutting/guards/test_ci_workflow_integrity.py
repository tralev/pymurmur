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

Later the same day, both workflows were rewritten so every job's test/
lint/type-check execution runs inside Docker (`docker build` +
`docker run`/`docker compose`) instead of installing dependencies
directly on the bare runner. That rewrite deleted the `if [ -f ... ];
then <pytest> else <heredoc fallback> fi` scaffolding entirely (every
guarded test file now exists, so the fallback branches were dead code) —
which also retired the fallback-specific tests below (their subjects no
longer exist). `test_no_bare_test_invocation_outside_docker` replaces
them: it asserts the *reason* those fallbacks could be deleted safely
stays true going forward — no job may invoke pytest/ruff/mypy directly
on the runner.
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


# ── Every test/lint job runs through Docker, not the bare runner ──

# Tool invocations that must never appear in a run: script unless that
# script also goes through Docker (docker build/compose/run).
_TEST_TOOL_MARKERS = ("pytest ", "pytest\n", "ruff check", "mypy ")


def _invokes_test_tool_outside_docker(run: str) -> bool:
    invokes_tool = any(marker in run for marker in _TEST_TOOL_MARKERS)
    return invokes_tool and "docker" not in run


def test_no_bare_test_invocation_outside_docker():
    """No job step invokes pytest/ruff/mypy directly on the bare GitHub
    Actions runner — every test/lint/type-check execution must go
    through `docker build`/`docker compose`/`docker run` (per the
    2026-07-21 Docker-first CI rewrite: OpenGL, gymnasium, scipy, ruff,
    mypy, and every plain CPU test all run inside a container)."""
    offenders = [
        f"{wf}::{job_name} step {step_i} ({step_name})"
        for wf, job_name, step_i, step_name, run in _JOB_STEPS
        if _invokes_test_tool_outside_docker(run)
    ]
    assert not offenders, (
        "The following steps invoke a test/lint tool without going "
        "through Docker:\n" + "\n".join(offenders)
    )


def test_bare_test_invocation_is_actually_detected():
    """The detection mechanism itself catches a real bare invocation —
    not just "no job happens to have one today"."""
    assert _invokes_test_tool_outside_docker("pip install pytest\npytest test/ -v\n")
    assert not _invokes_test_tool_outside_docker(
        "docker run --rm pymurmur-test:latest pytest test/ -v\n"
    )


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
