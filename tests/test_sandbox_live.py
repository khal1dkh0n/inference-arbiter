"""Opt-in live test of the dispatch gate against a real `sbx` MicroVM.

Skipped unless `sbx` is on PATH and INFERENCE_ARBITER_LIVE_SANDBOX=1:

    INFERENCE_ARBITER_LIVE_SANDBOX=1 uv run pytest tests/test_sandbox_live.py -v

It creates sandboxes named `inference-arbiter-<12 hex>` only, and the gate removes each one.
It makes no model call. The canary variables it sets hold made-up strings, never a key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from inference_arbiter import sandbox

pytestmark = pytest.mark.skipif(
    shutil.which("sbx") is None or os.environ.get("INFERENCE_ARBITER_LIVE_SANDBOX") != "1",
    reason="live sandbox test: needs sbx on PATH and INFERENCE_ARBITER_LIVE_SANDBOX=1",
)

CANARY_VALUE = "canary-value-that-must-not-cross"
CANARY_NAMES = ("OPENROUTER_API_KEY", "AWS_SECRET_ACCESS_KEY", "INFERENCE_ARBITER_CANARY")

ISOLATION_PROBE = r'''
import json, os, socket, ssl, urllib.request

report = {"uid": os.getuid(), "cwd": os.getcwd(), "environment_names": sorted(os.environ)}
report["openrouter_key_in_env"] = "OPENROUTER_API_KEY" in os.environ
report["canary_in_env_values"] = any("canary-value" in value for value in os.environ.values())

host_paths = {}
for path in ("/Users", "__HOST_HOME__", "/home/__HOST_USER__", "/host", "/mnt", "/Volumes", "/private"):
    try:
        host_paths[path] = sorted(os.listdir(path))[:5]
    except OSError as error:
        host_paths[path] = "absent: " + type(error).__name__
report["host_paths"] = host_paths
report["cwd_listing"] = sorted(os.listdir("."))
try:
    open("/proc/1/environ", "rb").read()
    report["pid1_environ_readable"] = True
except OSError:
    report["pid1_environ_readable"] = False

network = {}
try:
    network["dns example.com"] = "RESOLVED " + socket.getaddrinfo("example.com", 443)[0][4][0]
except OSError as error:
    network["dns example.com"] = "blocked: " + type(error).__name__
for name, ip in (("pypi.org", "151.101.0.223"), ("openrouter.ai", "104.18.2.115")):
    try:
        raw = socket.create_connection((ip, 443), timeout=6)
        ssl.create_default_context().wrap_socket(raw, server_hostname=name)
        network["tls " + name] = "CONNECTED"
    except OSError as error:
        network["tls " + name] = "blocked: " + type(error).__name__
for host in ("host.docker.internal", "gateway.docker.internal"):
    try:
        raw = socket.create_connection((host, 11434), timeout=6)
        raw.sendall(b"GET /api/tags HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        reply = raw.recv(200)
        network["ollama via " + host] = "ANSWERED" if reply else "blocked: no bytes returned"
    except OSError as error:
        network["ollama via " + host] = "blocked: " + type(error).__name__
for proxy in ("http://gateway.docker.internal:3128", None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy} if proxy else {}))
    label = "https pypi via " + ("the sbx proxy" if proxy else "no proxy")
    try:
        network[label] = "FETCHED %d" % opener.open("https://bootstrap.pypa.io/", timeout=8).status
    except Exception as error:
        network[label] = "blocked: " + type(error).__name__
report["network"] = network
print(json.dumps(report))
'''


def sandbox_names() -> set[str]:
    listed = subprocess.run(["sbx", "ls", "--quiet"], capture_output=True, text=True, timeout=30)
    return set(listed.stdout.split())


@pytest.fixture
def host_canaries(monkeypatch):
    for name in CANARY_NAMES:
        monkeypatch.setenv(name, CANARY_VALUE)


def test_a_program_finds_no_host_key_no_host_files_and_no_network(host_canaries):
    before = sandbox_names()

    host_home = os.path.expanduser("~")
    probe = ISOLATION_PROBE.replace("__HOST_HOME__", host_home).replace("__HOST_USER__", os.path.basename(host_home))
    result = sandbox.run_program(probe)

    assert result.ok, (result.error_type, result.error_message, result.stderr[-500:])
    report = json.loads(result.stdout)
    print("\nisolation probe:", json.dumps(report, indent=2))

    # 1. No host credential: the environment is exactly the one the gate built.
    assert report["environment_names"] == ["HOME", "LANG", "PATH"]
    assert report["openrouter_key_in_env"] is False
    assert report["canary_in_env_values"] is False
    assert CANARY_VALUE not in result.stdout and CANARY_VALUE not in result.stderr
    assert report["uid"] == 65534 and report["cwd"] == "/sandbox-run"
    assert report["pid1_environ_readable"] is False

    # 2. No host files: nothing of the host's home is mounted, and the run directory holds the script only.
    assert all(str(listing).startswith("absent") for listing in report["host_paths"].values()), report["host_paths"]
    assert report["cwd_listing"] == ["program.py"]

    # 3. No network: DNS, TLS to a host the global policy allows, the host's Ollama, and the proxy.
    assert all(outcome.startswith("blocked") for outcome in report["network"].values()), report["network"]

    assert result.network_policy == "deny_all"
    assert result.sandbox_removed is True
    assert sandbox_names() == before, "the gate must leave no sandbox behind"


def test_typed_failures_against_the_real_cli_and_cleanup_after_each():
    before = sandbox_names()
    limits = sandbox.SandboxLimits(timeout_s=3, max_output_bytes=5000)

    clean = sandbox.run_program("print(17 * 23)", limits=limits)
    failed = sandbox.run_program("import sys; print('partial'); sys.exit(3)", limits=limits)
    slow = sandbox.run_program("import time\nprint('start', flush=True)\ntime.sleep(60)", limits=limits)
    loud = sandbox.run_program("print('x' * 2_000_000)", limits=limits)

    assert (clean.status, clean.exit_code, clean.stdout) == ("ok", 0, "391\n")
    assert (failed.error_type, failed.exit_code, failed.stdout) == ("nonzero_exit", 3, "partial\n")
    assert (slow.error_type, slow.timed_out, slow.stdout) == ("timeout", True, "start\n")
    assert loud.error_type == "output_too_large" and len(loud.stdout) == 5000 and loud.stdout_truncated
    for result in (clean, failed, slow, loud):
        assert result.sandbox_removed is True and result.sandbox_name.startswith("inference-arbiter-")
    assert len({result.sandbox_name for result in (clean, failed, slow, loud)}) == 4
    assert sandbox_names() == before
