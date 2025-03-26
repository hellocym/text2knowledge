import os
import json
import requests

# BASE_URL = os.environ.get('OPENAI_API_BASE', 'http://localhost:1234/v1')

def generate(model_name, prompt, system=None, temperature=0.7, max_tokens=-1, stream=False, base_url='http://localhost:1234/v1'):
    try:
        url = f"{base_url}/chat/completions"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        if stream:
            return response.iter_lines()
        else:
            result = response.json()
            return result["choices"][0]["message"]["content"], None
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None
