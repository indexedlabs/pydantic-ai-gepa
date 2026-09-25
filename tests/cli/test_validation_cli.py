"""Actionable diagnostics when validation isolation cannot be verified."""

import subprocess

import pytest
import typer

from pydantic_ai_gepa.cli.validation import validation_dataset_path


@pytest.mark.parametrize("failure", ["rev-parse", "read", "hash-object", "cat-file"])
def test_validation_verification_refusals_name_resolved_path_and_remediation(
    tmp_path, monkeypatch, failure
):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    if failure != "read":
        validation.write_text('{"name": "private-case", "inputs": "secret"}\n')

    def run(command, **kwargs):
        operation = command[3]
        if operation == failure:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="" if kwargs.get("text") else b"",
                stderr="verification failed",
            )
        if operation == "rev-parse":
            return subprocess.CompletedProcess(
                command, 0, stdout=str(project), stderr=""
            )
        assert operation == "hash-object"
        return subprocess.CompletedProcess(command, 0, stdout=b"123456\n", stderr=b"")

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.subprocess.run", run)
    with pytest.raises(typer.BadParameter) as error:
        validation_dataset_path("../validation.jsonl", project_root=project)

    message = str(error.value)
    assert str(validation.resolve()) in message
    assert (
        "Keep the validation dataset outside the GEPA workspace/repository" in message
    )
    assert (
        "Start the workspace from Git history that never contained the validation dataset"
        in message
    )
    assert "validation_dataset" in message
    assert "--validation-dataset" in message
    assert "private-case" not in message


def test_validation_git_discovery_uses_c_locale(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")

    def run(command, **kwargs):
        locale = kwargs.get("env", {}).get("LC_ALL")
        stderr = "not a git repository" if locale == "C" else "Kein Git-Repository"
        return subprocess.CompletedProcess(command, 128, stdout="", stderr=stderr)

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.subprocess.run", run)
    assert (
        validation_dataset_path(str(validation), project_root=project)
        == validation.resolve()
    )


def test_validation_missing_git_is_clean_refusal(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("PATH", str(project))
    validation = tmp_path / "validation.jsonl"
    with pytest.raises(
        typer.BadParameter, match="Cannot verify validation dataset isolation"
    ) as error:
        validation_dataset_path(str(validation), project_root=project)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert str(validation.resolve()) in str(error.value)
    assert "Git history that never contained" in str(error.value)


@pytest.mark.parametrize("operation", ["rev-parse", "hash-object", "cat-file"])
def test_validation_git_os_error_is_clean_refusal(tmp_path, monkeypatch, operation):
    project = tmp_path / "project"
    project.mkdir()
    validation = tmp_path / "validation.jsonl"
    validation.write_text("{}\n")

    def run(command, **kwargs):
        if command[3] == operation:
            raise OSError("cannot execute git")
        if command[3] == "rev-parse":
            return subprocess.CompletedProcess(
                command, 0, stdout=str(project), stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"123456\n", stderr=b"")

    monkeypatch.setattr("pydantic_ai_gepa.cli.validation.subprocess.run", run)
    with pytest.raises(
        typer.BadParameter, match="Cannot verify validation dataset isolation"
    ) as error:
        validation_dataset_path(str(validation), project_root=project)
    assert str(validation.resolve()) in str(error.value)
    assert "outside the GEPA workspace/repository" in str(error.value)
