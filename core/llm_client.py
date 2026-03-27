'''
Author: danielwangow daomiao.wang@live.com
Description: LLM client adapter for ProEngOpt.
             Provides a unified interface for DashScope / Qwen model calls,
             supporting both streaming and non-streaming modes.
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''

from typing import Generator, Optional
import dashscope
from dashscope import Generation
from http import HTTPStatus


class LLMClientError(Exception):
    """Raised when the LLM API call fails."""
    pass


def call_llm(
    user_content: str,
    system_prompt: str,
    api_key: str,
    model: str = "qwen-plus",
) -> str:
    """
    Non-streaming LLM call. Returns the full response text.

    Parameters
    ----------
    user_content  : The user message (rendered prompt with metrics).
    system_prompt : The system role instruction.
    api_key       : DashScope API key (from environment variable).
    model         : Model identifier.

    Returns
    -------
    Full response text as a string.

    Raises
    ------
    LLMClientError if the API call fails.
    """
    if not api_key:
        raise LLMClientError(
            "DASHSCOPE_API_KEY is not set. "
            "Please add it to your .env file."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    response = Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format="message",
    )

    if response.status_code != HTTPStatus.OK:
        raise LLMClientError(
            f"LLM API error {response.status_code}: "
            f"{response.message}"
        )

    return response.output.choices[0].message.content


def stream_llm(
    user_content: str,
    system_prompt: str,
    api_key: str,
    model: str = "qwen-plus",
) -> Generator[str, None, None]:
    """
    Streaming LLM call. Yields text chunks incrementally.

    Suitable for use with Streamlit's st.write_stream().

    Parameters
    ----------
    Same as call_llm().

    Yields
    ------
    Incremental text chunks from the LLM response.

    Raises
    ------
    LLMClientError if the API call fails.
    """
    if not api_key:
        raise LLMClientError(
            "DASHSCOPE_API_KEY is not set. "
            "Please add it to your .env file."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    responses = Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format="message",
        stream=True,
        incremental_output=True,
    )

    full_text = ""
    for chunk in responses:
        if chunk.status_code != HTTPStatus.OK:
            raise LLMClientError(
                f"LLM stream error {chunk.status_code}: {chunk.message}"
            )
        delta = chunk.output.choices[0].message.content
        if delta:
            full_text += delta
            yield delta
