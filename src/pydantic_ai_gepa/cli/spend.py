"""Durable aggregate rollout spend shared by every CLI process in a run."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, NoReturn

from pydantic_ai.messages import ModelResponse

import typer

from ..spend import (
    COST_STOP_REASON,
    CostBudgetExceeded,
    SpendMeter,
    PriceFn,
    SpendCategory,
    rollout_spend,
)
from .layout import run_dir, run_state_path

if TYPE_CHECKING:
    from .run import RunState


def validate_cap(value: float | None) -> None:
    try:
        SpendMeter(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_rows(
    path: Path, warnings: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    contents = path.read_text()
    lines = contents.splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if warnings is None or index != len(lines) - 1 or contents.endswith("\n"):
                raise
            warnings["ledger_torn_tail"] = True
    return rows


class MissingValidationSpend(ValueError):
    """A registered validation eval has lost its authoritative spend."""


def _validation_owners(directory: Path) -> dict[str, int]:
    path = directory / "validation-spend-owners.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _rows(
    run_id: str, root: Path | None, warnings: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    directory = run_dir(run_id, root)
    rows = _read_rows(directory / "spend.jsonl", warnings)
    pointer = directory / "validation-spend-path"
    if pointer.exists():
        from .lanes import _pid_alive

        completed = {row["eval_id"] for row in rows if row["kind"] == "validation"}
        owners = _validation_owners(directory)
        private = Path(pointer.read_text())
        if not private.exists():
            if warnings is not None:
                warnings["validation_checkpoint_missing"] = True
            elif set(owners) - completed or not owners:
                raise MissingValidationSpend(
                    "Missing private validation spend checkpoint"
                )
        private_rows = _read_rows(private, warnings)
        live: set[str] = set()
        if warnings is None:
            # Admission retains private high-water costs even after publication.
            highest: dict[str, float] = {}
            for item in private_rows:
                key = item["eval_id"]
                highest[key] = max(highest.get(key, 0.0), item["max_rollout_dollars"])
            for row in rows:
                if row["kind"] == "validation":
                    # If all evals were published before private-file cleanup,
                    # the aggregate cost is a conservative scheduling bound.
                    row["max_rollout_dollars"] = highest.get(
                        row["eval_id"], row["total_dollars"]
                    )
        else:
            live = {
                key
                for key, pid in owners.items()
                if key not in completed and _pid_alive(pid)
            }
            if live:
                warnings["validation_in_progress"] = True
        for row in private_rows:
            if row["eval_id"] in completed:
                continue
            if warnings is not None:
                pid = owners.get(row["eval_id"])
                if pid is None or row["eval_id"] in live:
                    warnings["validation_in_progress"] = True
                    continue
            rows.append(row)
    return rows


def _add_models(target: dict[str, Any], source: dict[str, Any]) -> None:
    for name, usage in source.items():
        combined = target.setdefault(name, {})
        for key, value in usage.items():
            combined[key] = combined.get(key, 0) + value


def _report(rows: list[dict[str, Any]], cap: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "max_token_cost": cap,
        "total_dollars": 0.0,
        "training_dollars": 0.0,
        "validation_dollars": 0.0,
        "by_model": {},
        "unpriced_usage": {},
        "unmetered_rollouts": 0,
        "stopped_by_cost": False,
        "stop_reason": None,
    }
    for row in rows:
        if cap is None:
            result["max_token_cost"] = row.get("max_token_cost")
        result["unmetered_rollouts"] += row.get("unmetered_rollouts", 0)
        dollars = row["total_dollars"]
        result["total_dollars"] += dollars
        side = "validation" if row["kind"] == "validation" else "training"
        result[f"{side}_dollars"] += dollars
        _add_models(result["by_model"], row["by_model"])
        _add_models(result["unpriced_usage"], row["unpriced_usage"])
        if row["stop_reason"]:
            result["stopped_by_cost"] = True
            result["stop_reason"] = row["stop_reason"]
    return result


def spend_report(
    run_id: str, root: Path | None = None, cap: float | None = None
) -> dict[str, Any]:
    with _lock(run_dir(run_id, root) / "spend.lock"):
        warnings: dict[str, Any] = {}
        return dict(_report(_rows(run_id, root, warnings), cap), **warnings)


def _kind_costs(rows: list[dict[str, Any]], kind: str) -> tuple[int, float, float]:
    matching = [row for row in rows if row["kind"] == kind]
    observations = sum(row["rollouts_completed"] for row in matching)
    dollars = sum(row.get("completed_rollout_dollars", 0.0) for row in matching)
    highest = max(
        (row.get("max_rollout_dollars", 0.0) for row in matching), default=0.0
    )
    return observations, dollars / observations if observations else 0.0, highest


def _reservations(run_id: str, root: Path | None) -> dict[str, Any]:
    """Read under spend.lock; dead owners no longer reserve future work."""
    from .lanes import _pid_alive

    path = run_dir(run_id, root) / "spend-reservations.json"
    reservations = json.loads(path.read_text()) if path.exists() else {}
    private = _private_reservations_path(run_id, root)
    if private is not None and private.exists():
        reservations.update(json.loads(private.read_text()))
    return {
        key: value for key, value in reservations.items() if _pid_alive(value["pid"])
    }


def _private_reservations_path(run_id: str, root: Path | None) -> Path | None:
    pointer = run_dir(run_id, root) / "validation-spend-path"
    return (
        Path(pointer.read_text()).with_suffix(".reservations.json")
        if pointer.exists()
        else None
    )


def _write_reservations(path: Path, reservations: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(reservations, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_reservations(
    run_id: str, root: Path | None, reservations: dict[str, Any]
) -> None:
    private = _private_reservations_path(run_id, root)
    # Validation reservation updates can also reveal individual response costs.
    public_rows = {
        key: value
        for key, value in reservations.items()
        if value.get("kind") != "validation"
    }
    _write_reservations(run_dir(run_id, root) / "spend-reservations.json", public_rows)
    if private is not None:
        _write_reservations(
            private,
            {
                key: value
                for key, value in reservations.items()
                if value.get("kind") == "validation"
            },
        )


def _reserved_other(
    reservations: dict[str, Any], rows: list[dict[str, Any]], eval_id: str
) -> float:
    paid: dict[str, float] = {}
    for row in rows:
        key = row["eval_id"]
        paid[key] = paid.get(key, 0.0) + row["total_dollars"]
    return sum(
        max(0.0, value["dollars"] - paid.get(key, 0.0))
        for key, value in reservations.items()
        if key != eval_id
    )


@dataclass
class _RolloutUsage:
    requests: int = 0
    dollars: float = 0.0


class EvalSpendMeter(SpendMeter):
    """Append aggregate deltas after each response, including failed responses.

    Validation deltas stay private until an eval aggregate is published.
    Checkpointing each response preserves received usage after a process dies.
    """

    def __init__(
        self,
        run_id: str,
        root: Path | None,
        eval_id: str,
        kind: str,
        cap: float | None,
        price_fn: PriceFn | None,
        count: int,
        concurrency: int,
        private_path: Path | None = None,
        persist_stop: bool = True,
    ) -> None:
        super().__init__(cap, price_fn)
        self.run_id, self.root, self.eval_id, self.kind = run_id, root, eval_id, kind
        self.run_cap = cap
        self.private_path = private_path
        self.persist_stop = persist_stop
        self.started = self.completed = self.unmetered = 0
        self.count, self.concurrency = count, max(1, concurrency)
        self._condition = asyncio.Condition()
        self._active = 0
        self.completed_dollars = self.highest = 0.0
        self._requests: ContextVar[_RolloutUsage | None] = ContextVar(
            "rollout_requests", default=None
        )
        self._persist_lock = RLock()
        self._saved: dict[str, Any] = {}
        self._finished = False

    def flush(self) -> None:
        with self._persist_lock:
            report = self.report().model_dump()
            current = {
                "total_dollars": report["total_dollars"],
                "by_model": report["by_model"],
                "unpriced_usage": report["unpriced_usage"],
                "rollouts_started": self.started,
                "rollouts_completed": self.completed,
                "unmetered_rollouts": self.unmetered,
                "max_token_cost": self.run_cap,
                "completed_rollout_dollars": self.completed_dollars,
                "max_rollout_dollars": self.highest,
                "stop_reason": report["stop_reason"] if self.persist_stop else None,
            }
            if current == self._saved:
                return
            row = dict(current, eval_id=self.eval_id, kind=self.kind)
            for field in (
                "total_dollars",
                "rollouts_started",
                "rollouts_completed",
                "unmetered_rollouts",
                "completed_rollout_dollars",
            ):
                row[field] -= self._saved.get(field, 0)
            for field in ("by_model", "unpriced_usage"):
                row[field] = {
                    name: {
                        key: value
                        - self._saved.get(field, {}).get(name, {}).get(key, 0)
                        for key, value in usage.items()
                    }
                    for name, usage in current[field].items()
                }
            directory = run_dir(self.run_id, self.root)
            with _lock(directory / "spend.lock"):
                path = self.private_path or directory / "spend.jsonl"
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._saved = current

    def finish(self) -> None:
        if self._finished:
            return
        self.flush()
        if self.private_path is not None:
            # Publish once per validation eval. The private deltas remain the
            # authority until this aggregate exists, including after a crash.
            with _lock(run_dir(self.run_id, self.root) / "spend.lock"):
                row = dict(self._saved, eval_id=self.eval_id, kind=self.kind)
                row.pop("max_rollout_dollars")
                with (run_dir(self.run_id, self.root) / "spend.jsonl").open(
                    "a"
                ) as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        self._finished = True

    def callable_completed(self) -> None:
        """Require metering only after both callable and metric succeeded."""
        usage = self._requests.get()
        if usage is not None and not usage.requests:
            self.unmetered += 1
            if self.run_cap is not None:
                self.stop_reason = (
                    "Evaluate callable reported no spend (no metered response)"
                )
                raise CostBudgetExceeded(self.stop_reason)

    def _check_shared(self, *, after_response: bool = False) -> None:
        if self.run_cap is None:
            return
        with _lock(run_dir(self.run_id, self.root) / "spend.lock"):
            rows = _rows(self.run_id, self.root)
            report = _report(rows, self.run_cap)
            other = _reserved_other(
                _reservations(self.run_id, self.root), rows, self.eval_id
            )
        remaining = self.run_cap - report["total_dollars"] - other
        reason = report["stop_reason"]
        if report["total_dollars"] > self.run_cap or (
            not after_response and remaining <= 0
        ):
            reason = reason or COST_STOP_REASON
        if reason:
            self.stop_reason = reason
            self.flush()
            raise CostBudgetExceeded(reason)
        # Other processes may have settled below their reservation. Refresh the
        # local backstop from current shared headroom before another request.
        if not after_response:
            self.max_token_cost = self.report().total_dollars + remaining

    def check(self) -> None:
        self._check_shared()
        super().check()

    def record(self, category: SpendCategory, response: ModelResponse) -> None:
        with self._persist_lock:
            usage = self._requests.get()
            before = self.report().total_dollars
            if usage is not None:
                usage.requests += 1
            try:
                super().record(category, response)
            finally:
                if usage is not None:
                    usage.dollars += self.report().total_dollars - before
                self.flush()
            self._check_shared(after_response=True)

    def _admit_rollout(self) -> bool:
        """Reserve a start, or wait for this process's existing rollouts to drain."""
        self.check()
        if self.run_cap is None:
            return True
        with _lock(run_dir(self.run_id, self.root) / "spend.lock"):
            rows = _rows(self.run_id, self.root)
            reservations = _reservations(self.run_id, self.root)
            spent = sum(row["total_dollars"] for row in rows)
            remaining = (
                self.run_cap - spent - _reserved_other(reservations, rows, self.eval_id)
            )
            observations, mean, highest = _kind_costs(rows, self.kind)
            limit = (
                self.concurrency
                if observations and remaining >= self.concurrency * highest
                else 1
            )
            if self._active >= limit:
                return False
            # The batch mean reserves future work; a highest-cost floor also
            # covers in-flight slots so concurrent processes cannot each spend
            # the same apparently free headroom using an underestimated mean.
            inflight = (self._active + 1) * highest
            # Near the cap a single rollout may use the last projected budget;
            # its actual cost is guarded by the response backstop.
            if limit == 1 and not self._active:
                inflight = min(inflight, remaining)
            reservation = max(mean * (self.count - self.completed), inflight)
            if remaining <= 0 or reservation > remaining:
                if self._active:
                    return False
                self.stop_reason = COST_STOP_REASON
                raise CostBudgetExceeded()
            reservations[self.eval_id] = {
                "pid": os.getpid(),
                "dollars": self.report().total_dollars + reservation,
                "kind": self.kind,
            }
            _save_reservations(self.run_id, self.root, reservations)
        return True

    @asynccontextmanager
    async def rollout(self) -> AsyncIterator[None]:
        async with self._condition:
            while not self._admit_rollout():
                await self._condition.wait()
            self._active += 1
            self.started += 1
        usage = _RolloutUsage()
        token = self._requests.set(usage)
        self.flush()
        try:
            yield
            self.completed += 1
            self.completed_dollars += usage.dollars
        finally:
            self._requests.reset(token)
            self.highest = max(self.highest, usage.dollars)
            self.flush()
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()


def _finish_cost_stop(
    run_id: str,
    root: Path | None,
    state: RunState | None,
    reason: str,
    cap: float | None,
) -> None:
    from .events import EventDraft, emit, list_events
    from .run import RunState, _public_state, _write_final_report
    from .runs import ParetoLog, utc_now_iso

    path = run_state_path(run_id, root)
    if not path.exists():
        typer.echo(
            json.dumps(
                {"spend": spend_report(run_id, root, cap), "stop_reason": reason}
            )
        )
        return
    state = state or RunState.from_dict(json.loads(path.read_text()))
    state = replace(
        state,
        status="done",
        continuation=None,
        select_phase=None,
        select_context=None,
        iterations=ParetoLog(run_id, root).count_budget_rows()
        + state.gate_consumed_iterations,
        updated_at=utc_now_iso(),
        last_comparison={"reason_code": "cost_budget_exhausted", "stop_reason": reason},
    )
    state.save(root)
    final_path, _ = _write_final_report(state, root=root)
    if not any(event.type == "run_done" for event in list_events(run_id, root)):
        emit(
            run_id,
            "run",
            EventDraft(
                type="run_done",
                lane=None,
                payload={
                    "final_report_path": str(final_path),
                },
            ),
            root=root,
        )
    payload = _public_state(state, outcomes=[], final_report=final_path, root=root)
    payload["spend"] = spend_report(run_id, root, state.max_token_cost)
    typer.echo(json.dumps({"run": payload}))


@contextmanager
def evaluation_spend(
    *,
    run_id: str,
    root: Path | None,
    eval_id: str,
    kind: str,
    count: int,
    cap: float | None,
    price_fn: PriceFn | None,
    concurrency: int = 1,
    state: RunState | None = None,
    validation_spend_path: Path | None = None,
) -> Iterator[EvalSpendMeter]:
    """Reserve an eval under a short lock; release it after actual spend settles."""
    validate_cap(cap)
    from .run import RunState
    from .runs import ParetoLog

    path = run_state_path(run_id, root)
    managed = (
        RunState.from_dict(json.loads(path.read_text())) if path.exists() else None
    )
    if managed and managed.max_token_cost is not None:
        cap = (
            min(cap, managed.max_token_cost)
            if cap is not None
            else managed.max_token_cost
        )
    own_cap = managed is None or cap == managed.max_token_cost
    ad_hoc_cap = cap is not None and (
        managed is None
        or managed.max_token_cost is None
        or cap < managed.max_token_cost
    )
    if kind == "validation" and validation_spend_path is None:
        raise ValueError("Validation spend requires a private checkpoint path")
    if validation_spend_path is not None:
        from .validation import validation_dataset_path

        validation_spend_path = validation_dataset_path(
            str(validation_spend_path),
            project_root=root or Path.cwd(),
            allow_missing=True,
        )
    meter = EvalSpendMeter(
        run_id,
        root,
        eval_id,
        kind,
        cap,
        price_fn,
        count,
        concurrency,
        validation_spend_path,
        persist_stop=own_cap,
    )
    admitted = False

    def refuse(reason: str) -> NoReturn:
        typer.echo(reason, err=True)
        raise typer.Exit(code=2)

    def register_private_path() -> None:
        if validation_spend_path is not None:
            pointer = run_dir(run_id, root) / "validation-spend-path"
            if not pointer.exists():
                temporary = pointer.with_suffix(".tmp")
                with temporary.open("w") as handle:
                    handle.write(str(validation_spend_path))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, pointer)
            # Register ownership for uncapped evals too. No spend or per-case
            # progress belongs in this reflector-readable manifest.
            validation_spend_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not validation_spend_path.exists():
                with validation_spend_path.open("a") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            owners = _validation_owners(run_dir(run_id, root))
            if eval_id not in owners:
                owners[eval_id] = os.getpid()
                _write_reservations(
                    run_dir(run_id, root) / "validation-spend-owners.json", owners
                )

    try:
        with _lock(run_dir(run_id, root) / "spend.lock"):
            try:
                rows = _rows(run_id, root)
            except json.JSONDecodeError:
                refuse("Cannot evaluate with a malformed spend ledger")
            except MissingValidationSpend as exc:
                refuse(str(exc))
            report = _report(rows, cap)
            if ad_hoc_cap and cap is not None and report["total_dollars"] >= cap:
                refuse(
                    "One-off max-token-cost cannot cover run spend and this evaluation"
                )
            pointer = run_dir(run_id, root) / "validation-spend-path"
            if (
                validation_spend_path is not None
                and pointer.exists()
                and pointer.read_text() != str(validation_spend_path)
            ):
                refuse("Validation spend checkpoint path changed")
            if report["stopped_by_cost"]:
                raise CostBudgetExceeded(report["stop_reason"])
            if cap is not None:
                known_evals = {row["eval_id"] for row in rows}
                if report["unmetered_rollouts"] or any(
                    row.extra.get("eval_id") not in known_evals
                    for row in ParetoLog(run_id, root).iter_rows()
                ):
                    refuse("Cannot cap prior evaluations that reported no spend")
                if report["unpriced_usage"]:
                    refuse(
                        "Cannot cap previously unpriced models: "
                        + ", ".join(report["unpriced_usage"])
                    )
                reservations = _reservations(run_id, root)
                remaining = (
                    cap
                    - report["total_dollars"]
                    - _reserved_other(reservations, rows, eval_id)
                )
                _, mean, _ = _kind_costs(rows, kind)
                projected = mean * count
                if remaining <= 0 or projected > remaining:
                    if not own_cap:
                        refuse(
                            "One-off max-token-cost cannot cover run spend and this evaluation"
                        )
                    raise CostBudgetExceeded()
                register_private_path()
                reservations[eval_id] = {
                    "pid": os.getpid(),
                    "dollars": projected,
                    "kind": kind,
                }
                _save_reservations(run_id, root, reservations)
                meter.max_token_cost = remaining
            register_private_path()
            admitted = True
        with rollout_spend(meter):
            yield meter
    except CostBudgetExceeded as exc:
        admitted = True
        if (
            managed is not None
            and managed.max_token_cost is not None
            and (meter.unmetered or meter.report().unpriced_usage)
        ):
            # Missing accounting invalidates the managed cap too, even when
            # this particular eval requested a tighter one-off limit.
            own_cap = meter.persist_stop = True
        with _lock(run_dir(run_id, root) / "spend.lock"):
            register_private_path()
        meter.stop_reason = exc.stop_reason
        meter.finish()
        # Only terminal state/report emission is serialized, never paid work.
        with _lock(run_dir(run_id, root) / "spend-finalize.lock"):
            latest = (
                RunState.from_dict(json.loads(path.read_text()))
                if path.exists()
                else None
            )
            terminal = latest if latest and latest.status == "done" else state or latest
            if own_cap:
                _finish_cost_stop(run_id, root, terminal, exc.stop_reason, cap)
            else:
                typer.echo(
                    json.dumps(
                        {
                            "spend": spend_report(run_id, root, cap),
                            "stop_reason": exc.stop_reason,
                        }
                    )
                )
        raise typer.Exit(code=70) from exc
    finally:
        if admitted:
            meter.finish()
        if admitted and cap is not None:
            with _lock(run_dir(run_id, root) / "spend.lock"):
                reservations = _reservations(run_id, root)
                reservations.pop(eval_id, None)
                _save_reservations(run_id, root, reservations)
