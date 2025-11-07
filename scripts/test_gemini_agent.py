#!/usr/bin/env python3
"""Simple test script to call Google Generative (Gemini) API and optionally expose a tiny LangChain LLM wrapper.

Usage:
  - Put your GEMINI_API_KEY in the project's `.env` file (you said you already have this).
  - Optionally set GEMINI_MODEL (default: models/text-bison-001).
  - Run: python3 scripts/test_gemini_agent.py

This script avoids extra dependencies by using the standard library for HTTP.
If `langchain` is installed, a minimal `GeminiLLM` subclass of LangChain's LLM is exposed
and a small `LLMChain` demo run is attempted.
"""
import os
import json
from urllib import request, error

try:
    # prefer python-dotenv if available
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # simple fallback: load KEY=VALUE lines from .env in repo root
    def _fallback_load_dotenv(path: str = ".env") -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except FileNotFoundError:
            # no .env present — that's fine, environment may be set elsewhere
            return

    _fallback_load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")


def call_gemini(
    prompt: str, model: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY
) -> str:
    """Call the Google Generative Language REST API (simple wrapper).

    Uses the API key as query parameter (key=...). The endpoint is:
      POST https://generativelanguage.googleapis.com/v1/{model}:generate?key=API_KEY

    Returns the primary text output or raises on HTTP error.
    """
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment. Put it in .env and try again."
        )

    base = "https://generativelanguage.googleapis.com/v1"
    # model should be like "models/text-bison-001"
    url = f"{base}/{model}:generate?key={api_key}"

    payload = {
        "prompt": {"text": prompt},
        # keep defaults small for a quick test
        "temperature": 0.2,
        "maxOutputTokens": 256,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        msg = e.read().decode("utf-8") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTP error calling Gemini API: {e.code} {e.reason}\n{msg}")
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")

    j = json.loads(body)

    # Typical response contains 'candidates' (list) with 'output' or 'content'
    # Fall back to a few known locations.
    # Example: { 'candidates': [ {'output': '...'} ] }
    # or chat-like: { 'candidates': [ {'content': '...'} ] }
    text = None
    if isinstance(j, dict):
        if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
            cand = j["candidates"][0]
            text = cand.get("output") or cand.get("content") or cand.get("text")
        # Some API variants return 'output' top-level
        if not text and "output" in j:
            # output may be a dict or string
            if isinstance(j["output"], dict):
                text = j["output"].get("text")
            else:
                text = str(j["output"])

    if not text:
        # If we couldn't find expected keys, dump JSON for debugging
        text = json.dumps(j, indent=2)

    return text


def main():
    print("Running Gemini test script")

    # Basic direct test
    test_prompt = (
        "Write a very short friendly greeting and one-sentence description of yourself."
    )
    try:
        out = call_gemini(test_prompt)
        print("\n=== Direct Gemini call result ===")
        print(out)
    except Exception as e:
        print(f"Direct Gemini call failed: {e}")

    # Prefer LangChain's Google GenAI integration (langchain_google_genai) when available.
    # Falls back to a minimal LangChain LLM wrapper if that integration isn't installed.
    try:
        # try the official LangChain Google GenAI integration
        from langchain_google_genai import ChatGoogleGenerativeAI

        print("\nlangchain_google_genai detected — running ChatGoogleGenerativeAI demo")
        model_name = os.getenv("GEMINI_MODEL") or os.getenv(
            "GEMINI_MODEL", "gemini-2.5-pro"
        )
        # prefer an explicit key if provided, fall back to GOOGLE_API_KEY
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

        # simple chat-style invocation
        messages = [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Write a very short friendly greeting and one-sentence description of yourself.",
            ),
        ]
        ai_msg = llm.invoke(messages)
        # ai_msg may be an AIMessage-like object or a simple object with .content
        content = getattr(ai_msg, "content", ai_msg)
        print("\n=== langchain_google_genai Chat result ===")
        print(content)

    except Exception as e1:
        # If the specialized integration isn't available, try a minimal LangChain LLM wrapper
        try:
            from langchain.llms.base import LLM
            from typing import Optional, Mapping, Any
            from langchain import LLMChain, PromptTemplate

            class GeminiLLM(LLM):
                """Minimal LangChain LLM wrapper around `call_gemini`.

                This implements the required methods for a synchronous LLM.
                """

                def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
                    # ignore stop for this minimal example
                    return call_gemini(prompt)

                @property
                def _identifying_params(self) -> Mapping[str, Any]:
                    return {"model": GEMINI_MODEL}

                @property
                def _llm_type(self) -> str:
                    return "gemini"

            print(
                "\nLangChain core detected — running a small LLMChain demo using fallback GeminiLLM"
            )
            llm = GeminiLLM()
            prompt = PromptTemplate.from_template("Summarize in one sentence: {text}")
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run({"text": "LangChain + Gemini integration test"})
            print("\n=== LangChain LLMChain result (fallback) ===")
            print(result)

        except Exception as e2:
            print("\nLangChain demo skipped (langchain integration missing or error):")
            print("langchain_google_genai error:", e1)
            print("langchain core fallback error:", e2)


if __name__ == "__main__":
    main()
