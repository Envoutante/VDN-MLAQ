from importlib import import_module


_EXPORTS = {
    "LLMResponseParser": "parser",
    "FeedbackManager": "feedback",
    "TextFeedbackManager": "feedback",
    "ModelPrompter": "model_prompter",
    "ModelTextPrompter": "model_text_prompter",
}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
