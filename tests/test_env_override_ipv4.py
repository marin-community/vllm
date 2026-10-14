# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import subprocess
import sys


from vllm.utils.network_utils import resolve_ipv4_host



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
