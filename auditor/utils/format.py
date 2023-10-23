from typing import Optional

def construct_llm_input(
        prompt: str,
        pre_context: Optional[str],
        post_context: Optional[str],
        delimiter: str = " ",
    ) -> str:
        if pre_context is not None:
            full_prompt = pre_context + delimiter + prompt
        else:
            full_prompt = prompt
        if post_context is not None:
            full_prompt += delimiter + post_context
        return full_prompt