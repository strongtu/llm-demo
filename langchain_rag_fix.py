# openai.py中代码有点问题，需子类化并覆盖要修改一下才能正常运行
from importlib.metadata import version

import logging
from typing import (
    Any,
    Callable,
    List,
    Optional,
)

import numpy as np
from packaging.version import Version, parse
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings.openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    # Wait 2^x * 1 second between each retry starting with
    # retry_min_seconds seconds, then up to retry_max_seconds seconds,
    # then retry_max_seconds seconds afterwards
    # retry_min_seconds and retry_max_seconds are optional arguments of
    # OpenAIEmbeddings
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=embeddings.retry_min_seconds,
            max=embeddings.retry_max_seconds,
        ),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: OpenAIEmbeddings) -> Any:
    import openai

    # Wait 2^x * 1 second between each retry starting with
    # retry_min_seconds seconds, then up to retry_max_seconds seconds,
    # then retry_max_seconds seconds afterwards
    # retry_min_seconds and retry_max_seconds are optional arguments of
    # OpenAIEmbeddings
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=embeddings.retry_min_seconds,
            max=embeddings.retry_max_seconds,
        ),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


# https://stackoverflow.com/questions/76469415/getting-embeddings-of-length-1-from-langchain-openaiembeddings
def _check_response(response: dict, skip_empty: bool = False) -> dict:
    if any(len(d["embedding"]) == 1 for d in response["data"]) and not skip_empty:
        import openai

        raise openai.error.APIError("OpenAI API returned an empty embedding")
    return response


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    if _is_openai_v1():
        return embeddings.client.create(**kwargs)
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        response = embeddings.client.create(**kwargs)
        return _check_response(response, skip_empty=embeddings.skip_empty)

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""

    if _is_openai_v1():
        return await embeddings.async_client.create(**kwargs)

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        response = await embeddings.client.acreate(**kwargs)
        return _check_response(response, skip_empty=embeddings.skip_empty)

    return await _async_embed_with_retry(**kwargs)


def _is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version >= Version("1.0.0")


class FixOpenAIEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, respecting the
        set embedding context length and chunk size. It supports both tiktoken
        and HuggingFace tokenizer based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        _chunk_size = chunk_size or self.chunk_size

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed in order to for OpenAIEmbeddings without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk = tokenized[j : j + self.embedding_ctx_length]

                    # Convert token IDs back to a string
                    chunk_text = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "Could not import tiktoken python package. "
                    "This is needed in order to for OpenAIEmbeddings. "
                    "Please install it with `pip install tiktoken`."
                )

            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                encoding = tiktoken.get_encoding(model)
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                token = encoding.encode(
                    text=text,
                    allowed_special=self.allowed_special,
                    disallowed_special=self.disallowed_special,
                )

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter = tqdm(range(0, len(tokens), _chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), _chunk_size)
        else:
            _iter = range(0, len(tokens), 1)        # notice: 此处有修改

        batched_embeddings: List[List[float]] = []
        for i in _iter:
            response = embed_with_retry(
                self,
                input=encoding.decode(tokens[i : i + _chunk_size][0]),  # notice: 此处有修改
                **self._invocation_params,
            )
            if not isinstance(response, dict):
                response = response.dict()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average_embedded = embed_with_retry(
                    self,
                    input="",
                    **self._invocation_params,
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.dict()
                average = average_embedded["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings


    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Asynchronously generate length-safe embeddings for a list of texts.

        This method handles tokenization and asynchronous embedding generation,
        respecting the set embedding context length and chunk size. It supports both
        `tiktoken` and HuggingFace `tokenizer` based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        _chunk_size = chunk_size or self.chunk_size

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "This is needed in order to for OpenAIEmbeddings without "
                    " `tiktoken`. Please install it with `pip install transformers`."
                )

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk = tokenized[j : j + self.embedding_ctx_length]

                    # Convert token IDs back to a string
                    chunk_text = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "Could not import tiktoken python package. "
                    "This is needed in order to for OpenAIEmbeddings. "
                    "Please install it with `pip install tiktoken`."
                )

            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                encoding = tiktoken.get_encoding(model)
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                token = encoding.encode(
                    text=text,
                    allowed_special=self.allowed_special,
                    disallowed_special=self.disallowed_special,
                )

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = await async_embed_with_retry(
                self,
                input=encoding.decode(tokens[i : i + _chunk_size][0]),  # notice: 此处有修改
                **self._invocation_params,
            )

            if not isinstance(response, dict):
                response = response.dict()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average_embedded = await async_embed_with_retry(
                    self,
                    input="",
                    **self._invocation_params,
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.dict()
                average = average_embedded["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings
