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


DEFAULT_MODEL_CODE = "chinchilla"


class PoeLLM(LLM, BaseModel):
    client: Any
    model_code: str = DEFAULT_MODEL_CODE
    request_timeout: int = 60
    max_retries: int = 6
    model_kwargs: Optional[dict] = None
    poe_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.ignore

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_token = get_from_dict_or_env(
            values, "poe_api_token", "POE_API_TOKEN"
        )
        try:
            client = poe.Client(api_token)
            values['client'] = client
        except ImportError:
            raise ValueError(
                "Could not import poe package. "
                "Please it install it with `pip install poe-api`."
            )
        return values

    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_code": self.model_code},
            **{"model_kwargs": _model_kwargs}
        }

    @property
    def _llm_type(self) -> str:
        return self.client.bot_names[self.model_code]

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        _model_kwargs = self.model_kwargs or {}
        for response in self.client.send_message(self.model_code, prompt):
            pass
        response = self.client(input=prompt, params=_model_kwargs)
        return response['text']
