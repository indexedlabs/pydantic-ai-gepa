"""ComponentStore + agent introspection for the external-reflection CLI.

The store is the single source of truth for *values* of optimizable components.
*Identity* (the set of valid slot names) comes from the live agent via
``introspect_agent``. See pydanticaigepa-spec-973 and pydanticaigepa-dec-0ky.

Slot names contain colons (e.g. ``tool:foo:description``) which would split
into subdirectories on disk, so we round-trip them through a filesystem-safe
encoding (``__`` separator) when persisting under ``.gepa/components/`` and
``.gepa/staged/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .layout import components_dir, staged_dir

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


SLOT_SEPARATOR = "__"
SLOT_SUFFIX = ".md"


def slot_to_filename(slot: str) -> str:
    """Encode a slot name into a filesystem-safe basename."""
    if not slot:
        raise ValueError("Slot name must not be empty.")
    encoded = slot.replace(":", SLOT_SEPARATOR).replace("/", SLOT_SEPARATOR)
    return f"{encoded}{SLOT_SUFFIX}"


def filename_to_slot(name: str) -> str:
    """Decode a filesystem basename back to its slot name."""
    if not name.endswith(SLOT_SUFFIX):
        raise ValueError(
            f"Unexpected component filename {name!r}: missing {SLOT_SUFFIX} suffix."
        )
    return name[: -len(SLOT_SUFFIX)].replace(SLOT_SEPARATOR, ":")


class SlotStatus(str, Enum):
    """Lifecycle status of a component slot relative to the live agent."""

    CONFIRMED = "confirmed"
    """A confirmed value file exists in `.gepa/components/`."""

    STAGED = "staged"
    """A stub file exists in `.gepa/staged/` awaiting `gepa components confirm`."""

    INTROSPECTED_ONLY = "introspected_only"
    """The slot exists on the agent but no confirmed or staged file is on disk yet."""

    ORPHAN = "orphan"
    """A confirmed/staged file exists but the slot is no longer present on the agent."""


@dataclass(frozen=True)
class SlotRecord:
    """A single slot's view across the agent and the filesystem."""

    name: str
    status: SlotStatus
    introspected_seed: str | None
    """The seed value derived from the live agent (instruction text, tool docstring, etc.), if any."""

    confirmed_text: str | None
    staged_text: str | None


class ComponentStore:
    """Read/write component values under `.gepa/components/` and `.gepa/staged/`."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._components_dir = components_dir(root)
        self._staged_dir = staged_dir(root)

    @property
    def components_dir(self) -> Path:
        return self._components_dir

    @property
    def staged_dir(self) -> Path:
        return self._staged_dir

    # ---------- filesystem helpers ----------

    def _component_path(self, slot: str) -> Path:
        return self._components_dir / slot_to_filename(slot)

    def _staged_path(self, slot: str) -> Path:
        return self._staged_dir / slot_to_filename(slot)

    def staged_path(self, slot: str) -> Path:
        """Public accessor for the staged path (used by verbs to print the user-facing path)."""
        return self._staged_path(slot)

    def confirmed_path(self, slot: str) -> Path:
        """Public accessor for the confirmed path."""
        return self._component_path(slot)

    # ---------- queries ----------

    def list_confirmed_slots(self) -> list[str]:
        if not self._components_dir.is_dir():
            return []
        return sorted(
            filename_to_slot(p.name)
            for p in self._components_dir.iterdir()
            if p.is_file() and p.name.endswith(SLOT_SUFFIX)
        )

    def list_staged_slots(self) -> list[str]:
        if not self._staged_dir.is_dir():
            return []
        return sorted(
            filename_to_slot(p.name)
            for p in self._staged_dir.iterdir()
            if p.is_file() and p.name.endswith(SLOT_SUFFIX)
        )

    def read(self, slot: str) -> str | None:
        """Return the confirmed value for a slot, or None if no confirmed file exists."""
        path = self._component_path(slot)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_staged(self, slot: str) -> str | None:
        path = self._staged_path(slot)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ---------- mutations ----------

    def write(self, slot: str, content: str, *, clear_staged: bool = True) -> Path:
        """Write a confirmed value, atomically replacing any existing one.

        ``clear_staged`` defaults to True because the normal write path is
        ``gepa components set`` / ``gepa components confirm``, where the user
        is intentionally promoting / overwriting a slot and any staged stub is
        superseded. Pass ``clear_staged=False`` from re-seeding paths (e.g.
        ``gepa init --force``) so a user's in-progress staged edits survive.
        """
        self._components_dir.mkdir(parents=True, exist_ok=True)
        path = self._component_path(slot)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        if clear_staged:
            staged = self._staged_path(slot)
            if staged.exists():
                staged.unlink()
        return path

    def stage(self, slot: str, content: str) -> Path:
        """Write a stub under `.gepa/staged/` awaiting `gepa components confirm`."""
        self._staged_dir.mkdir(parents=True, exist_ok=True)
        path = self._staged_path(slot)
        path.write_text(content, encoding="utf-8")
        return path

    def confirm_staged(self, slot: str, override_text: str | None = None) -> Path:
        """Promote a staged stub to a confirmed value, optionally overriding the seed text."""
        staged = self._staged_path(slot)
        if not staged.exists():
            raise FileNotFoundError(
                f"No staged stub for slot {slot!r} at {staged}. "
                f"Run `gepa eval` first to detect new slots, or use `gepa components set` for an existing slot."
            )
        content = (
            override_text
            if override_text is not None
            else staged.read_text(encoding="utf-8")
        )
        return self.write(slot, content)

    def delete(self, slot: str) -> bool:
        """Delete both confirmed and staged copies of a slot. Returns True if anything was deleted."""
        deleted = False
        for path in (self._component_path(slot), self._staged_path(slot)):
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    # ---------- lifecycle composition ----------

    def slot_records(self, agent: AbstractAgent[Any, Any]) -> list[SlotRecord]:
        """Compose a per-slot view across introspection + on-disk state."""
        introspected = introspect_agent(agent)
        introspected_names = set(introspected)
        confirmed = set(self.list_confirmed_slots())
        staged = set(self.list_staged_slots())

        all_names = sorted(introspected_names | confirmed | staged)

        records: list[SlotRecord] = []
        for name in all_names:
            seed = introspected.get(name)
            confirmed_text = self.read(name)
            staged_text = self.read_staged(name)

            if name not in introspected_names:
                status = SlotStatus.ORPHAN
            elif confirmed_text is not None:
                status = SlotStatus.CONFIRMED
            elif staged_text is not None:
                status = SlotStatus.STAGED
            else:
                status = SlotStatus.INTROSPECTED_ONLY

            records.append(
                SlotRecord(
                    name=name,
                    status=status,
                    introspected_seed=seed,
                    confirmed_text=confirmed_text,
                    staged_text=staged_text,
                )
            )
        return records

    def detect_new_slots(self, agent: AbstractAgent[Any, Any]) -> list[str]:
        """Stage stubs for any introspected slots that lack a confirmed file. Returns the newly-staged names."""
        introspected = introspect_agent(agent)
        confirmed = set(self.list_confirmed_slots())
        staged_now: list[str] = []
        for name, seed in introspected.items():
            if name in confirmed:
                continue
            if self._staged_path(name).exists():
                # Already staged — leave it; only `gepa components confirm` advances the state.
                continue
            self.stage(name, seed or "")
            staged_now.append(name)
        return sorted(staged_now)

    def effective_candidate(self, agent: AbstractAgent[Any, Any]) -> dict[str, str]:
        """Return the candidate dict that should be applied for the current baseline.

        Resolution order per slot:
          1. confirmed file at `.gepa/components/<slot>.md`
          2. introspected seed from the live agent
          3. empty string (slot exists but no text yet)

        Staged-only slots are *not* included — baseline `gepa eval` is supposed to block on them.
        Orphan slots (file exists but not introspected) are *not* included either.
        """
        introspected = introspect_agent(agent)
        candidate: dict[str, str] = {}
        for slot, seed in introspected.items():
            confirmed = self.read(slot)
            if confirmed is not None:
                candidate[slot] = confirmed
            elif seed is not None:
                candidate[slot] = seed
            else:
                candidate[slot] = ""
        return candidate


# ---------- agent introspection ----------


def introspect_agent(agent: AbstractAgent[Any, Any]) -> dict[str, str]:
    """Return ``{slot_name: introspected_seed_text}`` for the optimizable surface of an agent.

    Slot kinds covered:
      * ``instructions`` — literal instructions on the agent or its wrapped target.
      * ``tool:<name>:description`` — function tool descriptions.
      * ``tool:<name>:param:<path>`` — function tool parameter descriptions.
      * ``output:<name>:description`` — output tool descriptions (when present).
      * ``output:<name>:param:<path>`` — output tool parameter descriptions.
      * Signature input field descriptions (when the agent is a ``SignatureAgent``).

    This function ensures the tool optimization manager is installed so its
    catalog has seen the registered tools. It does NOT execute the agent.
    """
    from ..components import extract_seed_candidate_with_input_type
    from ..signature_agent import SignatureAgent
    from ..tool_components import (
        get_or_create_output_tool_optimizer,
        get_or_create_tool_optimizer,
    )

    # Installing the optimizers pre-seeds their catalogs from registered tools
    # (and registers the `PrepareTools` capability so future runs ingest dynamic
    # tools too). For introspection alone we just need the pre-seed.
    get_or_create_tool_optimizer(agent)
    optimize_output_type = False
    try:
        get_or_create_output_tool_optimizer(agent)
        optimize_output_type = True
    except Exception:
        # Output tool optimization is best-effort — some agents won't have an
        # output toolset, and that's fine.
        optimize_output_type = False

    input_type = None
    if isinstance(agent, SignatureAgent):
        input_type = getattr(agent, "input_type", None)

    candidate_map = extract_seed_candidate_with_input_type(
        agent,
        input_type=input_type,
        optimize_output_type=optimize_output_type,
    )

    # CandidateMap is dict[str, ComponentValue]; flatten to dict[str, str].
    return {slot: value.text for slot, value in candidate_map.items()}
