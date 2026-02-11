from __future__ import annotations

from pathlib import Path
from typing import Any


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - exercised in environments without pyarrow
        raise RuntimeError(
            "Parquet support requires `pyarrow`. "
            "Install dependencies with the project training/runtime environment."
        ) from exc
    return pa, pq, ds


def write_parquet_rows(rows: list[dict[str, Any]], output_path: Path) -> int:
    pa, pq, _ = require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.Table.from_pylist([])
    else:
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path))
    return int(table.num_rows)

