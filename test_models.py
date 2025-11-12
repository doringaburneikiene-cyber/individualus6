import openai
import requests

def test_openrouter_models():
    # Read API key
    with open('api_key_openrouter.txt', 'r') as f:
        api_key = f.read().strip()
    
    # Test models list
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get('https://openrouter.ai/api/v1/models', headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models.get('data', [])[:10]:  # Show first 10 models
                print(f"- {model.get('id', 'Unknown')}")
        else:
            print(f"Error getting models: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

    # Test a simple API call
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Try with a very basic model
    test_models = [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3-8b-instruct",
        "google/gemma-7b-it",
        "mistralai/mistral-7b-instruct"
    ]
    
    for model in test_models:
        try:
            print(f"\nTesting model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✓ {model} works!")
            print(f"Response: {response.choices[0].message.content}")
            break  # Stop at first working model
        except Exception as e:
            print(f"✗ {model} failed: {e}")

if __name__ == "__main__":
    test_openrouter_models()