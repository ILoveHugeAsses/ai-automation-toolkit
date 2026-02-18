"""
AI Text Generator - Production Ready
Author: v0id-lab
License: MIT

Supports multiple free/cheap AI providers:
- Ollama (local, free)
- HuggingFace Inference API (free tier)
- Google Gemini (free tier)
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from datetime import datetime

class AITextGenerator:
    def __init__(self, provider: str = "ollama", model: Optional[str] = None):
        """
        Initialize AI text generator
        
        Args:
            provider: "ollama", "huggingface", or "gemini"
            model: Model name (provider-specific)
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self.history = []
        
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            "ollama": "llama3.2",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
            "gemini": "gemini-pro"
        }
        return defaults.get(self.provider, "llama3.2")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum response length
            temperature: Creativity (0.0-1.0)
            
        Returns:
            Generated text
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama (local)"""
        url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "")
            
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": text,
                "provider": "ollama",
                "model": self.model
            })
            
            return text
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace Inference API"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set")
        
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
            else:
                text = result.get("generated_text", "")
            
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": text,
                "provider": "huggingface",
                "model": self.model
            })
            
            return text
        except Exception as e:
            raise Exception(f"HuggingFace generation failed: {e}")
    
    def _generate_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Google Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": text,
                "provider": "gemini",
                "model": self.model
            })
            
            return text
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}")
    
    def chat(self, message: str, max_tokens: int = 500) -> str:
        """
        Chat with conversation history
        
        Args:
            message: User message
            max_tokens: Max response length
            
        Returns:
            AI response
        """
        # Build context from history
        context = "\n".join([
            f"User: {h['prompt']}\nAssistant: {h['response']}"
            for h in self.history[-5:]  # Last 5 exchanges
        ])
        
        prompt = f"{context}\nUser: {message}\nAssistant:"
        return self.generate(prompt, max_tokens)
    
    def save_history(self, filepath: str = "ai_history.json"):
        """Save conversation history to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def load_history(self, filepath: str = "ai_history.json"):
        """Load conversation history from file"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)


# ========== CLI USAGE ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Text Generator")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "huggingface", "gemini"])
    parser.add_argument("--model", help="Model name (optional)")
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = AITextGenerator(provider=args.provider, model=args.model)
    
    if args.chat:
        print(f"Chat mode - {args.provider} ({generator.model})")
        print("Type 'exit' to quit\n")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            response = generator.chat(user_input, max_tokens=args.max_tokens)
            print(f"AI: {response}\n")
        
        generator.save_history()
        print("History saved to ai_history.json")
    
    elif args.prompt:
        response = generator.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)
    
    else:
        print("Error: Provide --prompt or use --chat mode")
        parser.print_help()
