"""Runtime compatibility patches for local tooling."""

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return list(getattr(self, "all_special_tokens", []))

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
except Exception:
    pass
