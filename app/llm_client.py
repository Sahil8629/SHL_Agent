# from __future__ import annotations

# import os
# from typing import Any

# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# from pydantic import BaseModel, ValidationError

# load_dotenv()


# class Recommendation(BaseModel):
#     name: str
#     url: str
#     test_type: str


# class ChatResponse(BaseModel):
#     reply: str
#     recommendations: list[Recommendation]
#     end_of_conversation: bool


# class LLMClient:

#     def __init__(self, model: str = "gemini-2.5-flash"):
#         api_key = os.getenv("GOOGLE_API_KEY")

#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not found.")

#         self.client = genai.Client(api_key=api_key)
#         self.model = model

#     def respond(
#         self,
#         system_prompt: str,
#         conversation_messages: list[dict[str, str]],
#         max_tokens: int = 4096,  # raised — thinking tokens eat into this budget
#     ) -> dict[str, Any]:

#         history = ""
#         for m in conversation_messages:
#             history += f"{m['role'].upper()}:\n{m['content']}\n\n"

#         prompt = f"""
# {system_prompt}

# Conversation:

# {history}

# Return ONLY the JSON object.

# Keep the "reply" field concise (2-4 sentences max).

# Never use markdown.

# Never explain anything.

# Return exactly one object.
# """

#         # Try up to 2 times, doubling the token budget if we get truncated
#         last_error: Exception | None = None
#         current_max_tokens = max_tokens

#         for attempt in range(2):
#             response = self.client.models.generate_content(
#                 model=self.model,
#                 contents=prompt,
#                 config=types.GenerateContentConfig(
#                     temperature=0.2,
#                     max_output_tokens=current_max_tokens,
#                     response_mime_type="application/json",
#                     response_schema=ChatResponse,
#                     # This is the key fix: gemini-2.5-flash spends output
#                     # tokens on internal "thinking" by default, which can
#                     # consume the whole budget before it writes the JSON.
#                     # Disabling it frees the full max_output_tokens for
#                     # the actual response.
#                     thinking_config=types.ThinkingConfig(thinking_budget=0),
#                 ),
#             )

#             print("=" * 80)
#             print(f"attempt={attempt} max_tokens={current_max_tokens}")
#             print(response.text)
#             print("finish_reason:", response.candidates[0].finish_reason if response.candidates else None)
#             print("=" * 80)

#             # Preferred path — SDK already parsed it for us
#             if getattr(response, "parsed", None):
#                 return response.parsed.model_dump()

#             # Fallback: parse manually
#             try:
#                 return ChatResponse.model_validate_json(response.text).model_dump()
#             except (ValidationError, ValueError) as e:
#                 last_error = e
#                 current_max_tokens *= 2  # give it more room and retry once
#                 continue

#         raise RuntimeError(
#             f"Gemini returned invalid/truncated JSON after retries: {last_error}\n\n"
#             f"Raw text: {response.text if response else 'N/A'}"
#         )


from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()


# -----------------------------------------------------
# Output Schema
# -----------------------------------------------------

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponseSchema(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# -----------------------------------------------------
# Gemini Client
# -----------------------------------------------------

class LLMClient:

    def __init__(self, model: str = "gemini-2.5-flash"):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not found in .env"
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    # -------------------------------------------------

    def respond(
        self,
        system_prompt: str,
        conversation_messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:

        history = ""

        for message in conversation_messages:

            history += (
                f"{message['role'].upper()}:\n"
                f"{message['content']}\n\n"
            )

        prompt = f"""
{system_prompt}

Conversation

{history}

IMPORTANT

Return ONLY the JSON object.

Do not use markdown.

Do not explain.

Keep reply short (2-4 sentences).

Never include anything before or after the JSON.
"""

        last_error = None

        token_budget = max_tokens

        for attempt in range(2):

            response = self.client.models.generate_content(

                model=self.model,

                contents=prompt,

                config=types.GenerateContentConfig(

                    temperature=0.2,

                    max_output_tokens=token_budget,

                    response_mime_type="application/json",

                    response_schema=ChatResponseSchema,

                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    ),

                ),
            )

            print("\n")
            print("=" * 80)
            print(f"Attempt {attempt+1}")
            print("=" * 80)

            print(response.text)

            print("=" * 80)

            # ---------------------------------------------------------
            # BEST CASE
            # SDK already validated the JSON
            # ---------------------------------------------------------

            if getattr(response, "parsed", None):

                return response.parsed.model_dump()

            # ---------------------------------------------------------
            # FALLBACK
            # ---------------------------------------------------------

            try:

                parsed = ChatResponseSchema.model_validate_json(
                    response.text
                )

                return parsed.model_dump()

            except Exception as e:

                last_error = e

                token_budget *= 2

                continue

        raise RuntimeError(
            f"Gemini failed to return valid JSON.\n\n{last_error}"
        )