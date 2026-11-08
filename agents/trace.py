"""Thread-safe JSONL traces for graph nodes, search tools, and structured LLM calls."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4
from pydantic import BaseModel


def jsonable(value):
    """Serialize typed messages without traversing model clients or credentials."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class TraceWriter:
    """Write per-bid traces and an aggregate example trace, redacting known secrets."""

    def __init__(self, output_dir: Path, bid_id: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / f'{bid_id}.trace.jsonl'
        self.aggregate = output_dir / 'trace.jsonl'
        self.path.write_text('')
        self.run_id = uuid4().hex
        self.bid_id = bid_id
        self.lock = Lock()

    def event(self, agent: str, input: object, output: object, latency_ms: float = 0,
              tool_calls: list | None = None, tokens: dict | None = None,
              status: str = 'ok') -> None:
        """Append a complete step; API keys never enter serialized prompts or state."""
        payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), run_id=self.run_id,
            bid_id=self.bid_id, agent=agent, input=jsonable(input), tool_calls=jsonable(tool_calls or []),
            output=jsonable(output), tokens=tokens or {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
            latency_ms=round(latency_ms, 2), status=status)
        line = json.dumps(payload, ensure_ascii=False)
        for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            secret = os.getenv(name)
            if secret:
                line = line.replace(secret, '[REDACTED]')
        with self.lock:
            for path in (self.path, self.aggregate):
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(line + '\n')
