from openai import OpenAI

def do_test_openai_endpoint(endpoint, key, model,version):
    from openai import AzureOpenAI

    assert model is not None
    assert endpoint is not None
    assert key is not None
    assert version is not None
    client = AzureOpenAI(
        api_version = version,
        azure_endpoint = endpoint,
        api_key = key,
        )
    response = client.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": "Hello"}]
        )
    assert response is not None
    assert len(choices := response.choices) > 0
    assert (choice := choices[0]) is not None
    assert (message := choice.message) is not None
    assert message.content is not None
