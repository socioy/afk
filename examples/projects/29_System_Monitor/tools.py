"""
---
name: System Monitor — Tools
description: System monitoring tool definitions for CPU, memory, disk, and network checks.
tags: [tools]
---
---
Tool definitions for the system monitoring agent. Each tool returns simulated system
metrics. In production, these would call psutil, /proc, or platform APIs.
---
"""

from pydantic import BaseModel  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


class EmptyArgs(BaseModel):
    pass


@tool(args_model=EmptyArgs, name="check_cpu", description="Check current CPU usage and load averages.")
def check_cpu(args: EmptyArgs) -> str:
    return (
        "CPU Status:\n"
        "  Overall utilization: 34.2%\n"
        "  Core 0: 42.1% | Core 1: 28.7% | Core 2: 31.5% | Core 3: 34.6%\n"
        "  Load average (1m/5m/15m): 2.14 / 1.87 / 1.53\n"
        "  Running processes: 247\n"
        "  Status: HEALTHY"
    )


@tool(args_model=EmptyArgs, name="check_memory", description="Check current memory (RAM) usage.")
def check_memory(args: EmptyArgs) -> str:
    return (
        "Memory Status:\n"
        "  Total: 16.0 GB\n"
        "  Used: 10.4 GB (65.0%)\n"
        "  Available: 5.6 GB\n"
        "  Cached: 3.2 GB\n"
        "  Swap total: 4.0 GB | Swap used: 0.8 GB (20.0%)\n"
        "  Status: HEALTHY"
    )


@tool(args_model=EmptyArgs, name="check_disk", description="Check disk usage across mounted volumes.")
def check_disk(args: EmptyArgs) -> str:
    return (
        "Disk Status:\n"
        "  /dev/sda1 (/):\n"
        "    Total: 512 GB | Used: 287 GB (56.1%) | Free: 225 GB\n"
        "    Read IOPS: 1,240 | Write IOPS: 856\n"
        "  /dev/sdb1 (/data):\n"
        "    Total: 1 TB | Used: 412 GB (40.2%) | Free: 612 GB\n"
        "    Read IOPS: 320 | Write IOPS: 180\n"
        "  Status: HEALTHY"
    )


@tool(args_model=EmptyArgs, name="check_network", description="Check network interface status and traffic.")
def check_network(args: EmptyArgs) -> str:
    return (
        "Network Status:\n"
        "  Interface: eth0 (UP)\n"
        "    RX: 142.5 MB/s | TX: 38.7 MB/s\n"
        "    Packets RX: 98,432 | Packets TX: 45,210\n"
        "    Errors: 0 | Dropped: 2\n"
        "  Active connections: 1,847\n"
        "    TCP ESTABLISHED: 1,623 | TCP TIME_WAIT: 224\n"
        "  DNS resolution: 2.3ms avg\n"
        "  Status: HEALTHY"
    )
