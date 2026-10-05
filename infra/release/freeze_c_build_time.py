"""Emit C date macros from the frozen source timestamp for Rust dependencies."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def c_date_header(timestamp: int) -> str:
    source_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    date = f"{source_date:%b} {source_date.day:2d} {source_date.year}"
    return (
        '#pragma GCC diagnostic push\n'
        '#pragma GCC diagnostic ignored "-Wbuiltin-macro-redefined"\n'
        '#undef __DATE__\n'
        '#undef __TIME__\n'
        f'#define __DATE__ "{date}"\n'
        f'#define __TIME__ "{source_date:%H:%M:%S}"\n'
        '#pragma GCC diagnostic pop\n'
    )


def main() -> None:
    print(c_date_header(int(os.environ["SOURCE_DATE_EPOCH"])), end="")


if __name__ == "__main__":
    main()
