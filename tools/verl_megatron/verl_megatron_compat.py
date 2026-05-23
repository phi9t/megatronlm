"""Container startup compatibility shims for the verl Megatron validation image."""

from transformers import PreTrainedTokenizerBase


def _all_special_tokens_extended(self: PreTrainedTokenizerBase) -> list[str]:
    return list(getattr(self, "all_special_tokens", []))


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        _all_special_tokens_extended
    )
