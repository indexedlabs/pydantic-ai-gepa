"""Audit raw I/O sites, including reads, in the modules a harness can reach.

This is a review tripwire, not a taint analysis: new call sites (including a
second occurrence in an existing function) need a documented disposition.
SafeDir/record calls are recognized explicitly. Everything else is enumerated
below, including legacy non-held-out branches and private files.
"""

import ast
from collections import Counter
from pathlib import Path

MODULES = "harness eval select run runs spend lanes reflector events front pareto notes reflector_recovery".split()
OPERATIONS = {
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
    "open",
    "mkstemp",
    "replace",
    "mkdir",
    "unlink",
    "glob",
    "rglob",
    "iterdir",
    "rmdir",
    "rename",
    "link",
    "symlink_to",
    "touch",
}


class Inventory(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.safe_directories = set()
        self.safe_functions = set()
        self.calls = {}

    def visit_ImportFrom(self, node):
        if node.module == "harness_record":
            self.safe_functions.update(item.asname or item.name for item in node.names)

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node):
        saved = self.safe_directories.copy()
        for item in node.items:
            expression = item.context_expr
            if (
                isinstance(expression, ast.Call)
                and ast.unparse(expression.func)
                in {"SafeDir.open", "harness_record.SafeDir.open"}
                and isinstance(item.optional_vars, ast.Name)
            ):
                self.safe_directories.add(item.optional_vars.id)
        self.generic_visit(node)
        self.safe_directories = saved

    def visit_Call(self, node):
        expression = ast.unparse(node.func)
        operation = (
            node.func.attr if isinstance(node.func, ast.Attribute) else expression
        )
        safe = (
            expression in self.safe_functions
            or expression.startswith(("harness_record.", "SafeDir."))
            or any(expression.startswith(name + ".") for name in self.safe_directories)
        )
        # str.replace is formatting, not filesystem mutation. Candidate.write
        # is an indirect pathname writer and belongs in the inventory too.
        relevant = operation in OPERATIONS and (
            operation != "replace" or expression == "os.replace"
        )
        relevant |= expression == "candidate.write"
        if relevant and not safe:
            self.calls.setdefault(".".join(self.scope), Counter())[expression] += 1
        self.generic_visit(node)


# Each reason describes the branch containing precisely these call counts.
# Moving/removing a guard still requires review; adding raw calls fails this test.
ALLOWLIST = {
    "reflector_recovery.cleanup_continuation": (
        "Deletes private replay evidence outside GEPA_DIR, never public artifacts.",
        {"path.unlink": 1},
    ),
    "harness._atomic_json": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {
            "path.parent.mkdir": 1,
            "tempfile.mkstemp": 1,
            "os.replace": 1,
            "os.unlink": 1,
        },
    ),
    "harness._requests": (
        "Reflector fallback after the held-out descriptor-list/read branch returns.",
        {"path.read_text": 1, "(run_dir(run_id) / 'nominations').glob": 1},
    ),
    "eval._expose_trace_path": (
        "mkdir is in the non-held-out else branch; harness uses SafeDir.",
        {"path.parent.mkdir": 1},
    ),
    "eval._write_trace_file": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "eval.run_eval_once": (
        "Run mkdir is non-held-out; report write is after record.write_text returns False.",
        {
            "run_dir(active_run_id, workspace_root).mkdir": 1,
            "reports_dir.mkdir": 1,
            "report_path.write_text": 1,
        },
    ),
    "select._journal_lane_outcome": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "select._append_journal_once": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "select._emit_merge_opportunities": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"merge_dir.mkdir": 1, "stat_path.write_text": 1},
    ),
    "select._select_lock": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"lock_path.parent.mkdir": 1, "os.open": 1},
    ),
    "run.RunState.save": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {
            "path.parent.mkdir": 1,
            "tempfile.mkstemp": 1,
            "os.replace": 1,
            "os.unlink": 1,
        },
    ),
    "run._write_final_report": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.write_text": 1},
    ),
    "run.start": (
        "Run mkdir is in the non-held-out else branch.",
        {"run_dir(run_id).mkdir": 1},
    ),
    "runs.MinibatchStore.save": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"self._dir.mkdir": 1, "path.write_text": 1},
    ),
    "runs.ParetoLog.append": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"self._path.parent.mkdir": 1, "os.open": 2},
    ),
    "spend._lock": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "spend._reservations": (
        "Reads harness-private validation reservations outside GEPA_DIR.",
        {"private.read_text": 1},
    ),
    "spend._write_reservations": (
        "Private reservations outside GEPA_DIR, or non-held-out fallback after record publication.",
        {"path.parent.mkdir": 1, "temporary.open": 1, "os.replace": 1},
    ),
    "spend.EvalSpendMeter.flush": (
        "Private spend outside GEPA_DIR, or non-held-out fallback after record publication.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "spend.EvalSpendMeter.finish": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.open": 1},
    ),
    "spend.evaluation_spend.register_private_path": (
        "Pointer fallback only without record handling; remaining opens are private validation spend outside GEPA_DIR.",
        {
            "temporary.open": 1,
            "os.replace": 1,
            "validation_spend_path.parent.mkdir": 1,
            "validation_spend_path.open": 1,
        },
    ),
    "lanes._lane_lock": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "os.open": 1},
    ),
    "lanes._atomic_write_json": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {
            "path.parent.mkdir": 1,
            "tempfile.mkstemp": 1,
            "os.replace": 1,
            "os.unlink": 1,
        },
    ),
    "lanes.ensure_worktrees_ignored": (
        "Controller Git common-dir info/exclude, outside GEPA_DIR; independent lane storage is owned by lane_repositories.",
        {"info.mkdir": 1, "exclude.read_text": 1, "exclude.open": 1},
    ),
    "lanes._append_journal": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.open": 1, "os.open": 1},
    ),
    "lanes.write_packet": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.write_text": 1},
    ),
    "lanes._candidate_gate": (
        "Candidate source and prediction.json in the lane checkout, not GEPA_DIR view I/O.",
        {"(worktree / relative).read_text": 1, "prediction_path.read_text": 1},
    ),
    "lanes._write_comparison": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"path.parent.mkdir": 1, "path.write_text": 1},
    ),
    "lanes.lane_continue": (
        "Detached reflector worker log; explicit refuse_heldout_git_mutations before mkdir/open.",
        {"lane_dir.mkdir": 1, "log_path.open": 1},
    ),
    "reflector._file_lock": (
        "Private authoritative lock outside GEPA_DIR or reflector public lock; harness public handles come from view_file.",
        {"path.parent.mkdir": 1, "path.open": 1},
    ),
    "reflector.write_packet": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"tempfile.mkstemp": 1, "os.replace": 1, "os.unlink": 1},
    ),
    "events.advance_cursor": (
        "Reflector-only consumer ack (gepa next/ack); never called by held-out select/status.",
        {
            "path.parent.mkdir": 1,
            "tempfile.mkstemp": 1,
            "os.replace": 1,
            "os.unlink": 1,
        },
    ),
    "events._scan_next_seq": (
        "Only reached from the non-held-out emit branch; harness lists by directory fd.",
        {"events_path.iterdir": 1},
    ),
    "events.emit": (
        "Non-held-out fallback after descriptor-relative exclusive publication returns.",
        {"path.mkdir": 1, "tempfile.mkstemp": 1, "os.link": 1, "os.unlink": 3},
    ),
    "events._claim_reaper_key": (
        "Non-held-out fallback after descriptor-relative sentinel claim returns.",
        {"sentinels.mkdir": 1, "os.open": 1},
    ),
    "events.run_reaper_pass": (
        "mkdir is in the non-held-out else branch; harness uses SafeDir.",
        {"events_path.mkdir": 1},
    ),
    "front._read_front": (
        "Private validation front outside GEPA_DIR.",
        {"path.read_text": 1},
    ),
    "front._write_front": (
        "Private validation front outside GEPA_DIR.",
        {
            "path.parent.mkdir": 1,
            "tempfile.mkstemp": 1,
            "os.replace": 1,
            "os.unlink": 1,
        },
    ),
    "front.write_parent_packet": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"tempfile.mkstemp": 1, "os.replace": 1, "os.unlink": 1},
    ),
    "front.snapshot_components": (
        "Non-held-out fallback after SafeDir/record publication returns or an explicit held-out branch.",
        {"candidate.write": 1},
    ),
    "front.ValidationFront.retire_replay": (
        "Deletes private validation evidence outside GEPA_DIR.",
        {
            "validation_evidence_path(self.dataset, project_root=self.root, run_id=f'{self.run_id}:eval:{eval_id}').unlink": 1
        },
    ),
}


def test_harness_io_calls_have_a_reviewed_disposition():
    directory = Path(__file__).resolve().parents[2] / "src/pydantic_ai_gepa/cli"
    actual = {}
    for name in MODULES:
        visitor = Inventory()
        visitor.visit(ast.parse((directory / f"{name}.py").read_text()))
        actual.update(
            {
                f"{name}.{function}": dict(calls)
                for function, calls in visitor.calls.items()
            }
        )
    # layout has initialization/import plumbing beyond this view-I/O scope;
    # include the run-discovery helper that the harness calls before lookup.
    layout = ast.parse((directory / "layout.py").read_text())
    discovery = next(
        node
        for node in layout.body
        if isinstance(node, ast.FunctionDef) and node.name == "latest_run_id"
    )
    visitor = Inventory()
    visitor.visit(discovery)
    assert not visitor.calls
    assert all(reason for reason, _ in ALLOWLIST.values())
    assert actual == {name: calls for name, (_, calls) in ALLOWLIST.items()}
