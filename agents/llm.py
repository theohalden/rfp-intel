"""Provider adapter for schema-constrained extraction; model IDs stay in .env."""
from __future__ import annotations
import os
from config import Settings, llm_environment


def create_llm(settings: Settings):
    """Build the selected chat client without exposing credentials to graph state/logs."""
    key, model = llm_environment()
    provider = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
    if provider != 'openai':
        raise ValueError('This extraction adapter currently supports LLM_PROVIDER=openai')
    from langchain_openai import ChatOpenAI
    # This model defaults to no reasoning at the provider. Source interpretation
    # and consistency checks benefit from an explicit, configurable effort.
    default_effort = 'medium' if model.startswith('gpt-5.4-mini') else ''
    effort = os.getenv('OPENAI_REASONING_EFFORT', default_effort).strip()
    if effort and effort not in {'none', 'low', 'medium', 'high', 'xhigh'}:
        raise ValueError('Invalid OPENAI_REASONING_EFFORT')
    return ChatOpenAI(model=model, api_key=key, timeout=settings.agents.timeout_seconds,
                      max_retries=0, max_tokens=6000, use_responses_api=True,
                      reasoning={'effort': effort} if effort else None)
