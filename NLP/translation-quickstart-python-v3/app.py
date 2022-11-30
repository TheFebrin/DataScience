from google.cloud import translate


def translate_sentence(text: str, from_lang: str = "en-US", to_lang: str = "pl"):

    client = translate.TranslationServiceClient()
    project_id="firestore-dev-nomagic-ai"
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": from_lang,
            "target_language_code": to_lang,
        }
    )
    return response.translations[0].translated_text


print(translate_sentence("Who is the president of the USA?"))