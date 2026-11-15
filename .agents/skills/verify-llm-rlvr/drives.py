"""llm-rlvr drives: the DRIVES table, the control, and the subprocess primitives they share."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
CLI = ROOT / ".venv" / "bin" / "llm-rlvr"
PACKAGE = "llm_rlvr"
INSTALL_HINT = "make install"
SHARED = ["pyproject.toml", "uv.lock", ".agents/skills/verify-llm-rlvr/**", "src/llm_rlvr/_base.py"]
TRIPWIRE_TOOLS = ["modal", "wandb", "huggingface-cli", "hf", "nvidia-smi"]
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class DriveError(Exception):
    pass


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: list[str]
    run: Callable[[Path, dict[str, str]], str]


def run(
    evidence: Path, env: dict[str, str], name: str, argv: list[str], timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        raise DriveError(f"{name}: timed out after {timeout}s") from None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    redacted = {
        k: ("<redacted>" if any(m in k for m in SECRET_MARKERS) else v) for k, v in env.items()
    }
    (evidence / f"{name}.cmd.json").write_text(
        json.dumps({"argv": argv, "env": redacted, "exit": proc.returncode}, indent=2)
    )
    (evidence / f"{name}.stdout").write_text(out)
    (evidence / f"{name}.stderr").write_text(err)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise DriveError(message)


def ok(result: subprocess.CompletedProcess[str], name: str) -> None:
    tail = result.stderr.strip()[-400:]
    expect(result.returncode == 0, f"{name} exited {result.returncode}: {tail}")


def fixtures(evidence: Path, env: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(dir=env["HOME"]))
    ok(
        run(evidence, env, "fixtures", [str(PYTHON), str(HERE / "fixtures.py"), str(root)]),
        "fixtures",
    )
    return root


def drive_score(evidence: Path, env: dict[str, str], expected: float = 0.5) -> str:
    root = fixtures(evidence, env)
    out = evidence / "score-report.json"
    argv = [str(CLI), "score", "--completions", str(root / "completions.json")]
    result = run(
        evidence, env, "score", [*argv, "--db-root", str(root / "database"), "--out", str(out)]
    )
    ok(result, "llm-rlvr score")
    report = json.loads(out.read_text())
    accuracy = report["execution_accuracy"]
    expect(
        abs(accuracy - expected) < 1e-9, f"expected execution accuracy {expected}, got {accuracy}"
    )
    expect(report["n"] == 2 and report["n_correct"] == 1, f"expected 1/2 correct, got {report}")
    expect(
        "execution_accuracy=0.5000" in result.stdout, f"stdout lacks the EX line: {result.stdout!r}"
    )
    return f"scored 2 completions on a SQLite DB: EX {accuracy:.3f}, reasons {report['reasons']}"


def control_score(evidence: Path, env: dict[str, str]) -> str:
    return drive_score(evidence, env, expected=0.75)


def drive_build_traces(evidence: Path, env: dict[str, str]) -> str:
    root = fixtures(evidence, env)
    traces, manifest = evidence / "traces.jsonl", evidence / "traces.manifest.json"
    argv = [str(CLI), "build-traces", "--candidates", str(root / "candidates.json")]
    argv += ["--db-root", str(root / "database"), "--out", str(traces), "--manifest", str(manifest)]
    ok(run(evidence, env, "build-traces", argv), "llm-rlvr build-traces")
    counts = json.loads(manifest.read_text())["counts"]
    expect(
        counts["n_candidates"] == 3 and counts["n_kept"] == 1, f"expected 1 of 3 kept, got {counts}"
    )
    rows = [json.loads(line) for line in traces.read_text().splitlines()]
    kept = rows[0]["completion"][0]["content"] if len(rows) == 1 else None
    expect(kept is not None and kept.startswith("<think>"), f"unexpected trace rows {rows}")
    return "kept 1 of 3 candidates: dropped the wrong SQL and the answer with no <think>"


def drive_public_evidence(evidence: Path, env: dict[str, str]) -> str:
    result = run(evidence, env, "reproduce", [str(CLI), "reproduce", "--mode", "public"])
    ok(result, "llm-rlvr reproduce --mode public")
    for needle in [
        "GRPO  <think>                          0.779  [0.736, 0.823]",
        "ok: README headline table matches results/public/headline_summary.json",
    ]:
        expect(needle in result.stdout, f"expected {needle!r} in the public report")
    return "rendered the Spider and BIRD headline tables and matched the README"


def drive_launch_plan(evidence: Path, env: dict[str, str]) -> str:
    probe = [str(PYTHON), "-W", "ignore", str(HERE / "launch_plan.py")]
    result = run(evidence, env | {"PYTHONPATH": str(ROOT)}, "launch-plan", probe)
    ok(result, "launch plan")
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    (evidence / "launch-plan.json").write_text(json.dumps(plan, indent=2))
    expect(plan["entrypoints_are_modal"], "expected real modal LocalEntrypoints")
    expect(all(plan["gpus"].values()), f"expected every function to declare a GPU: {plan['gpus']}")
    refusals = plan["refusals"]
    expect(refusals, "launch plan reported no launch paths")
    for name, refusal in refusals.items():
        expect(
            "--approved" in refusal and not refusal.startswith(("DISPATCHED", "CRASH")),
            f"{name}: expected a refusal naming --approved, got {refusal!r}",
        )
    invalid = plan["invalid_grpo"]
    expect(
        "ValidationError" in invalid and "--approved" not in invalid,
        f"invalid GRPO request reached the gate: {invalid!r}",
    )
    return f"modal_app resolves the plan for {plan['model']}; {len(refusals)} launch paths refuse; bad request rejected first"


DRIVES = {
    "score": Drive(
        "llm-rlvr score turns a CompletionSet and SQLite DBs into execution accuracy",
        [
            "src/llm_rlvr/eval/**",
            "src/llm_rlvr/generation/**",
            "src/llm_rlvr/harness/**",
            "src/llm_rlvr/tasks/**",
            "src/llm_rlvr/env/**",
            "src/llm_rlvr/cli.py",
            "examples/no_gpu_demo.py",
        ],
        drive_score,
    ),
    "build-traces": Drive(
        "llm-rlvr build-traces keeps only correct, well-formed SFT traces and writes a manifest",
        [
            "src/llm_rlvr/train/traces.py",
            "src/llm_rlvr/generation/**",
            "src/llm_rlvr/harness/**",
            "src/llm_rlvr/tasks/**",
            "src/llm_rlvr/cli.py",
        ],
        drive_build_traces,
    ),
    "public-evidence": Drive(
        "llm-rlvr reproduce --mode public renders the headline and matches the README",
        [
            "src/llm_rlvr/report/**",
            "src/llm_rlvr/eval/aggregate.py",
            "results/public/**",
            "README.md",
        ],
        drive_public_evidence,
    ),
    "launch-plan": Drive(
        "modal_app resolves the plan; every entrypoint and GPU function refuses unapproved",
        ["modal_app.py", "src/llm_rlvr/modal_helpers.py", "src/llm_rlvr/train/**"],
        drive_launch_plan,
    ),
}
SMOKE = "score"
# The control passes only when its drive fails on exactly this planted expectation.
CONTROL_FAILURE = "expected execution accuracy 0.75, got 0.5"
CONTROL = ("score", control_score)
