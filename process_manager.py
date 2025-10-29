#!/usr/bin/env python3
"""Shared utilities to orchestrate ROS task pipelines from Python."""

import atexit
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import List


@dataclass
class ManagedProcess:
    label: str
    process: subprocess.Popen


def _compose_env_prefix(sources: List[str]) -> str:
    parts = [f"source {src}" for src in sources if src]
    return " && ".join(parts)


class ProcessGroup:
    """Launches long-running commands and cleans them up on exit."""

    def __init__(self, env_sources: List[str]):
        self._env_prefix = _compose_env_prefix(env_sources)
        self._processes: List[ManagedProcess] = []
        atexit.register(self.shutdown)

    def _wrap_command(self, command: str) -> str:
        if not self._env_prefix:
            return command
        return f"{self._env_prefix} && {command}"

    def start(self, label: str, command: str, delay: float = 0.0) -> subprocess.Popen:
        full_command = self._wrap_command(command)
        proc = subprocess.Popen(["bash", "-lc", full_command])
        self._processes.append(ManagedProcess(label=label, process=proc))
        if delay > 0.0:
            time.sleep(delay)
        return proc

    def shutdown(self) -> None:
        # Attempt graceful shutdown
        for managed in self._processes:
            proc = managed.process
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    continue
        time.sleep(1.0)
        # Escalate if still alive
        for managed in self._processes:
            proc = managed.process
            if proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    continue
        time.sleep(1.0)
        for managed in self._processes:
            proc = managed.process
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    continue


def wait_for_ros_master(env_sources: List[str], retries: int = 30, interval: float = 0.5) -> None:
    """Poll rosnode list until roscore responds."""
    env_prefix = _compose_env_prefix(env_sources)
    command = f"{env_prefix} && rosnode list" if env_prefix else "rosnode list"
    for _ in range(retries):
        result = subprocess.run(["bash", "-lc", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return
        time.sleep(interval)
    raise RuntimeError("ROS master did not become available within the expected time window")


def run_once(env_sources: List[str], command: str) -> None:
    env_prefix = _compose_env_prefix(env_sources)
    full_command = f"{env_prefix} && {command}" if env_prefix else command
    subprocess.run(["bash", "-lc", full_command], check=True)
