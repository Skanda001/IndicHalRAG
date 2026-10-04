"""
llm_clients.py
==============

Shared LangChain LLM wrappers + prompt for Gemini and Sarvam-105B.

This file is NOT run directly. It is imported by generate_responses.py only.

MODELS
------
1. Gemini 2.5 Flash Lite  (Google)
   - Free tier: 20 req/day (very limited). Use pay-as-you-go for full runs.
   - Enable billing at: https://aistudio.google.com → Get API key → Set up billing
   - Cost: ~$0.01 total for 500 questions (negligible)
   - Model string: "gemini-2.5-flash-lite"

2. Sarvam-105B  (Sarvam AI)
   - Indian company, model trained natively on Kannada
   - Free credits on signup at dashboard.sarvam.ai
   - reasoning_effort=None disables hidden thinking tokens (important!)

SETUP
-----
Create .env in project root:
    GEMINI_API_KEY=your_gemini_key_from_aistudio.google.com
    SARVAM_API_KEY=your_key_from_dashboard.sarvam.ai
"""

import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ── Output token budget ────────────────────────────────────────────────
# 300 tokens = enough for 2-4 complete Kannada sentences.
MAX_OUTPUT_TOKENS = 300

# ── Rate limit safety delay (seconds between each question) ───────────
# Gemini free tier: 15 RPM = 1 request per 4 seconds minimum.
# We use 5 seconds to stay comfortably under the limit.
# If you still get 429 errors, change this to 8 or 10.
DELAY_BETWEEN_REQUESTS = 5   # used in generate_responses.py


# ── Shared prompt (written in Kannada) ────────────────────────────────
# Same prompt sent to BOTH models so the comparison is fair.
# All instructions in Kannada — mixing English caused code-switching.
QA_PROMPT = PromptTemplate.from_template(
    """ನೀವು ಕೆಳಗಿನ ಪ್ಯಾಸೇಜ್ ಅನ್ನು ಮಾತ್ರ ಆಧರಿಸಿ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಬೇಕು.

ನಿಯಮಗಳು:
1. ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಇರುವ ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ. ಹೊರಗಿನ ಜ್ಞಾನ ಅಥವಾ ಊಹೆಯನ್ನು ಬಳಸಬೇಡಿ.
2. ಉತ್ತರವನ್ನು ಸಂಪೂರ್ಣ ವಾಕ್ಯ(ಗಳಲ್ಲಿ) ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ. ಇಂಗ್ಲಿಷ್ನಲ್ಲಿ ಉತ್ತರಿಸಬೇಡಿ.
3. ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಉತ್ತರ ಇಲ್ಲದಿದ್ದರೆ, ನಿಖರವಾಗಿ ಈ ಪದಗಳನ್ನು ಬರೆಯಿರಿ: "ಈ ಮಾಹಿತಿ ಪ್ಯಾಸೇಜ್ನಲ್ಲಿ ಇಲ್ಲ."

ಪ್ಯಾಸೇಜ್:
{passage}

ಪ್ರಶ್ನೆ:
{question}

ಉತ್ತರ:"""
)


# ── Gemini 2.5 Flash Lite ─────────────────────────────────────────────
# Free tier: 20 req/day. Enable pay-as-you-go for unlimited (~$0.01 total).
def get_gemini_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


# ── Sarvam-105B ────────────────────────────────────────────────────────
class SarvamLLM(LLM):
    model: str = "sarvam-105b"
    api_key: Optional[str] = None
    max_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "sarvam"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        from sarvamai import SarvamAI

        client = SarvamAI(
            api_subscription_key=self.api_key or os.getenv("SARVAM_API_KEY")
        )
        response = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # IMPORTANT: disables hidden "thinking" tokens.
            # Without this, the token budget gets eaten by reasoning
            # and the visible answer comes back empty.
            reasoning_effort=None,
        )
        return response.choices[0].message.content.strip()


def get_sarvam_llm() -> SarvamLLM:
    return SarvamLLM()
