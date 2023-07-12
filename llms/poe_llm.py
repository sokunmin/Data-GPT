"""
Tutorial:
    https://python.langchain.com/docs/modules/model_io/models/llms/how_to/custom_llm
"""
import poe
from typing import Any, List, Mapping, Optional, Dict, Mapping, Optional, Tuple

from pydantic import BaseModel, Extra, root_validator
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env


# from langchain.callbacks.manager import

class PoeLLM(LLM, BaseModel):
    client: Any
    api_token: str
    model_id: str
    request_timeout: int = 60
    max_retries: int = 6
    model_kwargs: Optional[dict] = None
    model_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.ignore

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_token = get_from_dict_or_env(
            values, "model_api_token", "MODEL_API_TOKEN"
        )
        try:
            client = poe.Client(api_token)
            values['client'] = client
        except ImportError:
            raise ValueError(
                "Could not import poe package. "
                "Please it install it with `pip install poe`."
            )
        return values

    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs}
        }

    @property
    def _llm_type(self) -> str:
        return "gpt-3.5-turbo"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        _model_kwargs = self.model_kwargs or {}
        response = self.client(input=prompt, params=_model_kwargs)
        if "error" in response:
            raise ValueError(f"Error raised by inference API: {response}")

        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(None, stop)
            # text = enforce_stop_tokens(text, stop)
        return text
