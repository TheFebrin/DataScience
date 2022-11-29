import os
import openai


class GPT3API:
    
    def __init__(self, max_length_tokens: int) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._max_length_tokens = max_length_tokens

    def send_prompt(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature = 0.7,
            top_p=1,
            max_tokens=self._max_length_tokens,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text