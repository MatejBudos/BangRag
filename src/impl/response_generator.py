from typing import List

from interface.base_response_generator import BaseResponseGenerator
from util.invoke_ai import invoke_ai


SYSTEM_PROMPT = """
You answer questions about the card game Bang! using only the provided rules context.
Prefer the custom Bang rules and rule interpretations contained in the context.
If multiple rules are relevant, combine them into one coherent answer.
General rules and glossary definitions have priority over individual card blurbs when they clarify terminology.
Strictly distinguish between a card and an effect:
- if the rules mention a specific card, apply the rule only to that named card
- do not automatically extend a card restriction to all effects with the same outcome
- if a different card or character only creates the same effect, that does not make it the same card
When the question is about whether something is allowed, answer from these distinctions explicitly.
Reply in the same language as the user's question.
If the answer is not supported by the context, say that the provided rules do not specify it.
Do not make up information outside the supplied Bang rules.
"""


class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query: str, context: List[str]) -> str:
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)
