# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import subprocess
import sys

import pytest

from vllm.env_override import _apply_ipv4_only_getaddrinfo_patch
from vllm.utils.network_utils import resolve_ipv4_host

pytestmark = pytest.mark.skip_global_cleanup


def test_ipv4_only_getaddrinfo_constrains_only_unspecified_family(monkeypatch):
    requested_families = []

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        requested_families.append(family)
        return [(family, type, proto, "", (host, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    _apply_ipv4_only_getaddrinfo_patch()
    patched_getaddrinfo = socket.getaddrinfo

    assert socket.getaddrinfo("worker", 1234)[0][0] == socket.AF_INET
    assert socket.getaddrinfo("worker", 1234, socket.AF_UNSPEC)[0][0] == socket.AF_INET
    assert socket.getaddrinfo("worker", 1234, socket.AF_INET6)[0][0] == socket.AF_INET6
    assert requested_families == [socket.AF_INET, socket.AF_INET, socket.AF_INET6]

    _apply_ipv4_only_getaddrinfo_patch()
    assert socket.getaddrinfo is patched_getaddrinfo


def test_ipv4_only_resolution_preserves_default_and_ip_literals(monkeypatch):
    requested_families = []

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        requested_families.append(family)
        return [(family, type, proto, "", ("192.0.2.10", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.delenv("VLLM_FORCE_IPV4", raising=False)
    assert resolve_ipv4_host("worker", 1234) == "worker"

    monkeypatch.setenv("VLLM_FORCE_IPV4", "1")
    assert resolve_ipv4_host("worker", 1234) == "192.0.2.10"
    assert resolve_ipv4_host("192.0.2.20", 1234) == "192.0.2.20"
    assert resolve_ipv4_host("2001:db8::20", 1234) == "2001:db8::20"
    assert requested_families == [socket.AF_INET]


@pytest.mark.parametrize(
    ("flag", "is_same_function", "requested_family"),
    [(None, True, socket.AF_UNSPEC), ("1", False, socket.AF_INET)],
)
def test_ipv4_only_getaddrinfo_is_applied_during_isolated_import(
    flag, is_same_function, requested_family
):
    script = """
import json
import socket

requested_families = []

def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    requested_families.append(family)
    return [(family, type, proto, '', (host, port))]

socket.getaddrinfo = fake_getaddrinfo
original_getaddrinfo = socket.getaddrinfo
import vllm
socket.getaddrinfo('worker', 1234, socket.AF_UNSPEC)
print(json.dumps({
    'is_same_function': socket.getaddrinfo is original_getaddrinfo,
    'requested_family': requested_families[-1],
}))
"""
    env = os.environ.copy()
    if flag is None:
        env.pop("VLLM_FORCE_IPV4", None)
    else:
        env["VLLM_FORCE_IPV4"] = flag
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    observed = json.loads(result.stdout.rsplit("\n", 2)[-2])
    assert observed == {
        "is_same_function": is_same_function,
        "requested_family": requested_family,
    }


def test_ipv4_only_resolution_reaches_tcp_store():
    script = """
import json
import socket
from datetime import timedelta

requested_families = []

def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    assert host == 'c05.invalid'
    requested_families.append(family)
    return [(
        socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        '',
        ('127.0.0.1', port),
    )]

socket.getaddrinfo = fake_getaddrinfo
import vllm
from vllm.distributed.utils import create_tcp_store

store = create_tcp_store(
    'c05.invalid',
    0,
    world_size=1,
    is_master=True,
    wait_for_workers=False,
    timeout=timedelta(seconds=1),
)
print(json.dumps({
    'port': store.port,
    'requested_families': requested_families,
}))
"""
    env = os.environ.copy()
    env["VLLM_FORCE_IPV4"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    observed = json.loads(result.stdout.rsplit("\n", 2)[-2])
    assert observed["port"] > 0
    assert observed["requested_families"] == [socket.AF_INET]
