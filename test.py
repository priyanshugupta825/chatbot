from groq import Groq
import os

# API key environment variable se lega
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are Priyanshu's assistant."},
        {"role": "user", "content": "Hello Groq, kaise ho?"}
    ],
    model="llama3-8b-8192",   # free model
)

print(chat_completion.choices[0].message.content)
