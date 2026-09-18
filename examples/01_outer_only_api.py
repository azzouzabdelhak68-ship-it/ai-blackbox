"""Outer on closed API (no weights, no GPU).

Spec section EXAMPLES 01 contract:
  In: prompt string.
  Out: outer.json (~3 KB: prompt, output tokens+order, timing,
    seed/settings, watchlist_ref=outer_only) + seal.json;
    `blackbox view RUN-xxx` shows prompt/output + `Outer-only` notice;
    overhead <0.5%.

Runnable CPU-only: uses a fake closed-API client plus ``bb.wrap`` for
capture and ``BlackBox.watch`` for sealing. No weights, no GPU, no
network calls.
"""

from __future__ import annotations

from blackbox import BlackBox
from blackbox.tap.wrapper import wrap


class LlmOutput:
    """Minimal typed stand-in for a closed-API response."""

    def __init__(self, text: str) -> None:
        self.text = text


class LlmClient:
    """Minimal typed stand-in for a closed-API client."""

    def generate(self, prompt: str, seed: int = 42) -> LlmOutput:
        return LlmOutput(text=f"echo:{prompt} (seed {seed})")


def run_outer_only(prompt: str, seed: int = 42) -> str:
    """Run Outer-only capture; returns output text."""
    llm_client = LlmClient()
    wrapped = wrap(llm_client)
    try:
        with BlackBox.watch("watchlists/outer_only.yaml"):
            out = wrapped.generate(prompt, seed=seed)
    finally:
        wrapped.detach()
    print(out.text)
    return out.text


def main() -> None:
    run_outer_only("Explain why the sky is blue", seed=42)


if __name__ == "__main__":
    main()
