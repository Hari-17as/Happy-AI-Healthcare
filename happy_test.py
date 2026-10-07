import json
import urllib.request

url = "http://localhost:11434/api/chat"

messages = [
    {
        "role": "system",
        "content": (
            "You are Baymax, a gentle, calm and caring healthcare companion. "
            "Speak simply and kindly. You are an AI companion and not a replacement "
            "for a human doctor. For emergencies, advise the user to seek local "
            "emergency medical help."
        )
    }
]

print("================================")
print("       BAYMAX HEALTHCARE AI")
print("================================")
print("Type 'exit' to close Baymax.\n")

while True:
    user_text = input("You: ")

    if user_text.lower() == "exit":
        print("Baymax: Goodbye. Please take care of yourself. 💛")
        break

    messages.append({
        "role": "user",
        "content": user_text
    })

    data = {
        "model": "qwen3:8b",
        "messages": messages,
        "stream": False
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        baymax_reply = result["message"]["content"]

        print("\nBaymax:", baymax_reply, "\n")

        messages.append({
            "role": "assistant",
            "content": baymax_reply
        })

    except Exception as error:
        print("\nBaymax: I am having trouble connecting to my local brain.")
        print("Technical information:", error, "\n")