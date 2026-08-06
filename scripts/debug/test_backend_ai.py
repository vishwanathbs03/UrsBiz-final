from app.config.settings import get_settings
from openai import OpenAI

settings = get_settings()

print("--- CONFIGURATION CHECK ---")
print(f"Provider: {settings.ai_provider}")
print(f"Base URL: {settings.ai_base_url}")
print(f"Model:    {settings.ai_model}")

# Initialize OpenAI-compatible client using your settings
client = OpenAI(
    api_key=settings.ai_api_key,
    base_url=settings.ai_base_url,
)

print("\n--- SENDING TEST REQUEST TO BACKEND CONFIG ---")
try:
    response = client.chat.completions.create(
        model=settings.ai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "State your exact model name and version clearly."
            }
        ],
    )

    print("\n--- RESPONSE RECEIVED ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nError communicating with AI provider: {e}")