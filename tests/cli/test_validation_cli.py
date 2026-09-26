"""Actionable diagnostics when validation isolation cannot be verified."""

import subprocess

import pytest
import typer

from pydantic_ai_gepa.cli.validation import validation_dataset_path
from pydantic_ai_gepa.cli.safe_git import Repository


@pytest.fixture(autouse=True)
def discovered_repository(monkeypatch):
    monkeypatch.setattr(
        Repository,
        "discover",
        lambda path: Repository(path, path / ".git", path / ".git"),
    )


@pytest.mark.parametrize("failure", ["discovery", "read", "hash-object", "cat-file"])
def test_validation_verification_refusals_withhold_path_and_explain_remediation(
    tmp_path, monkeypatch, failure
):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    if failure != "read":
        validation.write_text('{"name": "private-case", "inputs": "secret"}\n')

    if failure == "discovery":

        def discover(path):
            raise OSError("verification failed")

        monkeypatch.setattr(Repository, "discover", discover)

    def run(repository, operation, *args, **kwargs):
        command = [operation, *args]
        if operation == failure:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="" if kwargs.get("text") else b"",
                stderr="verification failed",
            )
        if operation == "discovery":
            return subprocess.CompletedProcess(
                command, 0, stdout=str(project), stderr=""
            )
        assert operation == "hash-object"
        return subprocess.CompletedProcess(command, 0, stdout=b"123456\n", stderr=b"")

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.safe_run_git", run)
    with pytest.raises(typer.BadParameter) as error:
        validation_dataset_path("../validation.jsonl", project_root=project)

    message = str(error.value)
    assert str(validation.resolve()) not in message
    assert "outside every checkout and GEPA_DIR" in message
    assert "Git history that never contained" in message
    assert "GEPA_HELDOUT_DATASET" in message
    assert "private-case" not in message


def test_validation_discovery_does_not_depend_on_locale(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")

    def discover(path):
        raise FileNotFoundError("not a git repository")

    monkeypatch.setattr(Repository, "discover", discover)
    assert (
        validation_dataset_path(str(validation), project_root=project)
        == validation.resolve()
    )


def test_validation_missing_git_is_clean_refusal(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()

    def missing(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.safe_run_git", missing)
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")
    with pytest.raises(
        typer.BadParameter, match="Cannot verify validation dataset isolation"
    ) as error:
        validation_dataset_path(str(validation), project_root=project)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert str(validation.resolve()) not in str(error.value)
    assert "Git history that never contained" in str(error.value)


@pytest.mark.parametrize("operation", ["discovery", "hash-object", "cat-file"])
def test_validation_git_os_error_is_clean_refusal(tmp_path, monkeypatch, operation):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")

    if operation == "discovery":

        def discover(path):
            raise OSError("cannot discover git")

        monkeypatch.setattr(Repository, "discover", discover)

    def run(repository, command, *args, **kwargs):
        if command == operation:
            raise OSError("cannot execute git")
        if command == "discovery":
            return subprocess.CompletedProcess(
                command, 0, stdout=str(project), stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"123456\n", stderr=b"")

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.safe_run_git", run)
    with pytest.raises(
        typer.BadParameter, match="Cannot verify validation dataset isolation"
    ) as error:
        validation_dataset_path(str(validation), project_root=project)
    assert str(validation.resolve()) not in str(error.value)
    assert "outside every checkout and GEPA_DIR" in str(error.value)
