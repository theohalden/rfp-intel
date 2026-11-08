"""Provider adapters for schema-constrained extraction; keys and model IDs stay in .env."""
from __future__ import annotations
import os
from config import Settings, llm_environment


class _SchemaAdapter:
    """Give a chat client the structured-output call the graph uses, dropping options it lacks."""

    def __init__(self, client, drop: tuple[str, ...]):
        self.client, self.drop = client, drop

    def with_structured_output(self, schema, **kwargs):
        return self.client.with_structured_output(schema, **{k: v for k, v in kwargs.items() if k not in self.drop})

    def __getattr__(self, name):
        return getattr(self.client, name)


def create_llm(settings: Settings):
    """Build the selected chat client without exposing credentials to graph state/logs."""
    key, model = llm_environment()
    provider = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
    # No fixed output cap by default: the model uses what it needs. Set LLM_MAX_OUTPUT_TOKENS to bound cost.
    max_tokens = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '0')) or None
    if provider == 'anthropic':
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ValueError('Install langchain-anthropic to use LLM_PROVIDER=anthropic') from exc
        client = ChatAnthropic(model=model, api_key=key, timeout=settings.agents.timeout_seconds,
                               max_retries=0, max_tokens=max_tokens or 16000)  # the Anthropic API requires a cap
        return _SchemaAdapter(client, drop=('method', 'strict'))
    from langchain_openai import ChatOpenAI
    # This model defaults to no reasoning at the provider. Source interpretation
    # and consistency checks benefit from an explicit, configurable effort.
    default_effort = 'medium' if model.startswith('gpt-5.4-mini') else ''
    effort = os.getenv('OPENAI_REASONING_EFFORT', default_effort).strip()
    if effort and effort not in {'none', 'low', 'medium', 'high', 'xhigh'}:
        raise ValueError('Invalid OPENAI_REASONING_EFFORT')
    return ChatOpenAI(model=model, api_key=key, timeout=settings.agents.timeout_seconds,
                      max_retries=0, max_tokens=max_tokens, use_responses_api=True,
                      reasoning={'effort': effort} if effort else None)
