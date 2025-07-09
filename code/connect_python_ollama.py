import requests

payload = {
    "model": "llama3",
    "prompt": "¿Qué es la mononucleosis?",
    "stream": False
}

try:
    response = requests.post("http://147.96.81.71:11434/api/generate", json=payload)
    data = response.json()

    if "response" in data:
        print("🧠 Respuesta del modelo:")
        print(data["response"])
    else:
        print("⚠️ No se encontró la clave 'response' en la respuesta:")
        print(data)

except requests.exceptions.ConnectionError:
    print("❌ No se pudo conectar al servidor Ollama en 147.96.81.71:11434.")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
