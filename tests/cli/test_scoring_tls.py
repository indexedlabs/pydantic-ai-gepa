"""Real TLS over the scoring CONNECT proxy, with a throwaway local CA."""

import json
import socketserver
import ssl
import subprocess
import sys
import threading

import pytest

from pydantic_ai_gepa.cli import scoring_sandbox as sandbox
from pydantic_ai_gepa.cli.scoring_proxy import connect_proxy
from tests.cli.test_scoring_sandbox import (
    clean_environment,
    real_backend,
    git_repo,
    private,
    private_run,
    commit_evaluator,
    score,
    protocol_backend,
)

__all__ = [
    "clean_environment",
    "real_backend",
    "git_repo",
    "private",
    "private_run",
    "protocol_backend",
]


@pytest.fixture
def tls_server(tmp_path):
    config = tmp_path / "leaf.cnf"
    config.write_text(
        "subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n"
    )
    commands = [
        [
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Test CA",
            "-keyout",
            "ca.key",
            "-out",
            "ca.pem",
        ],
        [
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-keyout",
            "leaf.key",
            "-out",
            "leaf.csr",
        ],
        [
            "x509",
            "-req",
            "-in",
            "leaf.csr",
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-days",
            "1",
            "-extfile",
            "leaf.cnf",
            "-out",
            "leaf.pem",
        ],
    ]
    for command in commands:
        subprocess.run(
            ["openssl", *command], cwd=tmp_path, capture_output=True, check=True
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "leaf.pem", tmp_path / "leaf.key")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                self.request.settimeout(5)
                with context.wrap_socket(self.request, server_side=True) as connection:
                    if connection.recv(16384):
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
                        )
            except OSError:
                pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        try:
            yield tmp_path / "ca.pem", server.server_address[1]
        finally:
            server.shutdown()
            thread.join()


def probe_paths(tmp_path):
    private = tmp_path / "private"
    scratch, checkout = private / "scratch", private / "checkout"
    scratch.mkdir(parents=True)
    checkout.mkdir()
    return private, checkout, scratch


def test_real_sandbox_allows_proxy_https_scoring(
    real_backend, tls_server, git_repo, private, private_run, monkeypatch
):
    ca, port = tls_server
    monkeypatch.setattr(sandbox.certifi, "where", lambda: str(ca))
    monkeypatch.setenv("GEPA_HARNESS_ALLOWED_HOSTS", f"localhost:{port}")
    sha = commit_evaluator(
        git_repo,
        f"""import httpx2
async def evaluate(case):
    async with httpx2.AsyncClient(timeout=5) as client:
        response = await client.get("https://localhost:{port}/")
        assert response.status_code == 200
        assert response.text == "ok"
    return "good"
""",
    )
    assert score(git_repo, sha)[0].score == 1


def test_smoke_refuses_before_scorer_import_or_rollout(
    git_repo, private, private_run, protocol_backend, monkeypatch
):
    sha = commit_evaluator(git_repo, 'raise AssertionError("scorer must not import")')
    from pydantic_ai_gepa.cli.spend import EvalSpendMeter

    def no_rollout(*args):
        pytest.fail("rollout admitted before smoke check")

    monkeypatch.setattr(EvalSpendMeter, "rollout", no_rollout)

    def refuse(*args):
        raise sandbox.ScoringSandboxError("TLS smoke refusal")

    monkeypatch.setattr(sandbox, "check_tls", refuse)
    with pytest.raises(sandbox.ScoringSandboxError, match="TLS smoke refusal"):
        score(git_repo, sha)


@pytest.mark.parametrize(
    "mode",
    [
        "httpx2_bundle",
        "httpx2_without_bundle",
        "truststore_bundle",
        "truststore_explicit_bundle",
    ],
)
def test_real_seatbelt_httpx2_and_truststore(
    real_backend, tls_server, tmp_path, monkeypatch, mode
):
    ca, target_port = tls_server
    monkeypatch.setattr(sandbox.certifi, "where", lambda: str(ca))
    private, checkout, scratch = probe_paths(tmp_path)
    code = """
import asyncio, json, os, ssl, sys
import httpx2, truststore
async def run():
    options = {}
    if sys.argv[1].startswith("truststore"):
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if sys.argv[1] == "truststore_explicit_bundle":
            context.load_verify_locations(cafile=os.environ["SSL_CERT_FILE"])
        options["verify"] = context
    try:
        async with httpx2.AsyncClient(timeout=5, **options) as client:
            response = await client.get(sys.argv[2])
            print(json.dumps({"status": response.status_code, "body": response.text}))
    except Exception as error:
        errors = []
        while error is not None:
            errors.append({"class": type(error).__name__, "message": str(error)})
            error = error.__cause__
        print(json.dumps({"errors": errors}))
asyncio.run(run())
"""
    with connect_proxy(frozenset({("localhost", target_port)})) as port:
        env = sandbox.child_environment(scratch, port)
        if mode == "httpx2_without_bundle":
            env.pop("SSL_CERT_FILE")
        result = subprocess.run(
            sandbox.sandbox_command(
                sandbox.seatbelt_profile(private, checkout, scratch, port),
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    code,
                    mode,
                    f"https://localhost:{target_port}/",
                ],
            ),
            cwd=scratch,
            env=env,
            capture_output=True,
            timeout=20,
            check=True,
        )
    outcome = json.loads(result.stdout)
    print(mode, json.dumps(outcome))
    if mode == "httpx2_bundle":
        assert outcome == {"status": 200, "body": "ok"}
    else:
        assert "errors" in outcome
        assert "OSStatus -26276" in json.dumps(
            outcome
        ) or "CERTIFICATE_VERIFY_FAILED" in json.dumps(outcome)


def test_real_seatbelt_smoke_success_once(
    real_backend, tls_server, tmp_path, monkeypatch
):
    ca, target_port = tls_server
    monkeypatch.setattr(sandbox.certifi, "where", lambda: str(ca))
    monkeypatch.setenv(
        "GEPA_HARNESS_ALLOWED_HOSTS", f"localhost:{target_port},unreachable.invalid:443"
    )
    monkeypatch.setattr(sandbox, "_tls_checked", set())
    private, checkout, scratch = probe_paths(tmp_path)
    with connect_proxy(frozenset({("localhost", target_port)})) as port:
        profile = sandbox.seatbelt_profile(private, checkout, scratch, port)
        env = sandbox.child_environment(scratch, port)
        sandbox.check_tls(profile, scratch, port, env)
        monkeypatch.setattr(
            sandbox.subprocess, "run", lambda *a, **kw: pytest.fail("second probe")
        )
        sandbox.check_tls(profile, scratch, port, env)


@pytest.mark.parametrize("variable", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"])
def test_ca_environment_override_and_path_refusal(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, "/unpassed.pem")
    assert (
        sandbox.child_environment(tmp_path, 1234)[variable] == sandbox.certifi.where()
    )
    monkeypatch.setenv("GEPA_HARNESS_PASS_ENV", variable)
    assert sandbox.child_environment(tmp_path, 1234)[variable] == "/unpassed.pem"
    monkeypatch.setenv("GEPA_HELDOUT_DATASET", "/heldout.jsonl")
    monkeypatch.setenv(variable, "/heldout.jsonl")
    with pytest.raises(sandbox.ScoringSandboxError, match="held-out path"):
        sandbox.child_environment(tmp_path, 1234)


def test_smoke_refusal_and_empty_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "_tls_checked", set())
    monkeypatch.setattr(sandbox, "sandbox_command", lambda profile, command: command)
    env = sandbox.child_environment(tmp_path, 1234)
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            [], 1, b'{"error_class":"SSLCertVerificationError"}', b"private stderr"
        )

    monkeypatch.setattr(sandbox.subprocess, "run", fail)
    sandbox.check_tls("profile", tmp_path, 1234, env)
    assert not calls
    monkeypatch.setenv(
        "GEPA_HARNESS_ALLOWED_HOSTS", "first.example:443,second.example:443"
    )
    with pytest.raises(sandbox.ScoringSandboxError) as error:
        sandbox.check_tls("profile", tmp_path, 1234, env)
    message = str(error.value)
    assert "SSLCertVerificationError" in message
    assert "system trust store is blocked" in message
    assert "SSL_CERT_FILE" in message
    assert "ssl.create_default_context(cafile=certifi.where())" in message
    assert "private stderr" not in message
    assert calls[0][0][0][-3:] == ["first.example", "443", "1234"]
    assert calls[0][1]["env"] == env
    assert not sandbox._tls_checked
