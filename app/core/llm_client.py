"""LLM Client — универсальный клиент для разных LLM-провайдеров.
Поддерживает Ollama (локально), OpenAI API и OpenAI-compatible (vLLM, TGI, etc.).
"""

from abc import ABC, abstractmethod

import httpx
from loguru import logger

from app.core.config import settings


class LLMClientBase(ABC):
    """Абстрактный базовый класс для LLM-клиентов."""

    @abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        """Отправить промпт и получить ответ."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Проверить доступность сервиса."""
        ...


class OllamaClient(LLMClientBase):
    """Клиент для локального Ollama."""

    def __init__(self, base_url: str = None, model_name: str = None):
        self.base_url = (base_url or settings.llm_api_base_url).rstrip("/")
        self.model_name = model_name or settings.llm_model_name
        self._client: httpx.Client | None = None
        self._available: bool | None = None

    def _get_client(self) -> httpx.Client | None:
        if self._client is None:
            try:
                self._client = httpx.Client(base_url=self.base_url, timeout=30)
                response = self._client.get("/api/tags")
                self._available = response.status_code == 200
                if self._available:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    logger.info(f"Ollama connected. Available models: {model_names}")
                else:
                    logger.warning(f"Ollama status: {response.status_code}")
            except Exception as e:
                logger.warning(f"Ollama not available at {self.base_url}: {e}")
                self._available = False
        return self._client if self._available else None

    def is_available(self) -> bool:
        self._get_client()
        return bool(self._available)

    def query(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        if not client:
            return ""

        options = {
            "temperature": kwargs.get("temperature", 0.1),
            "num_predict": kwargs.get("max_tokens", 2048),
        }

        try:
            response = client.post(
                "/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": options,
                },
                timeout=kwargs.get("timeout", 60),
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            logger.warning(f"Ollama generate failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
        return ""


class OpenAIClient(LLMClientBase):
    """Клиент для OpenAI API и OpenAI-compatible серверов (vLLM, TGI, Together, etc.)."""

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_api_base_url).rstrip("/")
        self.model_name = model_name or settings.llm_model_name
        self._client: httpx.Client | None = None
        self._available: bool | None = None

    def _get_client(self) -> httpx.Client | None:
        if self._client is None:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=60)
                # Проверка доступности через список моделей
                response = self._client.get("/v1/models")
                self._available = response.status_code == 200
                if self._available:
                    logger.info(f"OpenAI-compatible API connected at {self.base_url}")
            except Exception as e:
                logger.warning(f"OpenAI API not available: {e}")
                self._available = False
        return self._client if self._available else None

    def is_available(self) -> bool:
        self._get_client()
        return bool(self._available)

    def query(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        if not client:
            return ""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False,
        }

        try:
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                timeout=kwargs.get("timeout", 60),
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            logger.warning(f"OpenAI API failed: {response.status_code} {response.text[:200]}")
        except Exception as e:
            logger.warning(f"OpenAI query failed: {e}")
        return ""


class HuggingFaceClient(LLMClientBase):
    """Клиент для HuggingFace Transformers (локальный инференс)."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.llm_model_name
        self._model = None
        self._tokenizer = None
        self._available: bool | None = None

    def _load(self) -> bool:
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, device_map="auto", torch_dtype="auto"
                )
                self._available = True
                logger.info("HuggingFace model loaded")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {e}")
                self._available = False
        return self._available

    def is_available(self) -> bool:
        return self._load()

    def query(self, prompt: str, **kwargs) -> str:
        if not self._load():
            return ""

        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.1),
                    do_sample=False,
                )
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response[len(prompt) :].strip()
            return response
        except Exception as e:
            logger.warning(f"HuggingFace inference failed: {e}")
        return ""


def create_llm_client(provider: str = None) -> LLMClientBase:
    """Factory: создаёт LLM-клиент по указанному провайдеру."""
    provider = provider or settings.llm_provider

    providers = {
        "ollama": OllamaClient,
        "openai": OpenAIClient,
        "huggingface": HuggingFaceClient,
    }

    client_class = providers.get(provider)
    if not client_class:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to ollama")
        client_class = OllamaClient

    client = client_class()
    logger.info(f"LLM client created: {provider} ({client.__class__.__name__})")
    return client
