from google import genai

# 1. Initialize the client 
# (This automatically reads the GEMINI_API_KEY environment variable you set in Step 2)
client = genai.Client()

# 2. Call the generate_content method
response = client.models.generate_content(
    model="gemini-3.6-flash",  # Specifies the Gemini 3.6 Flash model name
    contents="Explain how asynchronous processing works in backend systems.",
)

# 3. Print out the model's text response
print(response.text)