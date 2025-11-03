"""LLM handler for Mistral models.

This module provides a MistralLLMHandler class and a module-level
get_llm() function that returns a singleton instance. It searches
for common model file names under the `models/` directory and gives
clear messages when a model or llama-cpp-python is missing.
"""

import os
from typing import Optional, List

try:
    from llama_cpp import Llama  # type: ignore
    _HAS_LLAMA = True
except Exception:
    Llama = None
    _HAS_LLAMA = False


class MistralLLMHandler:
    """Handles loading a local Mistral .gguf model and simple text calls."""

    DEFAULT_POSSIBLE_PATHS: List[str] = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'),
    ]
    
    # Model parameters
    MODEL_PARAMS = {
        "n_ctx": 2048,        # Context window
        "n_threads": 4,       # Number of CPU threads to use
        "n_gpu_layers": 0,    # Number of layers to offload to GPU (0 for CPU-only)
        "seed": 42,          # RNG seed for reproducibility
        "verbose": True      # Enable verbose logging
    }

    def __init__(self, model_path: Optional[str] = None):
        self.llm = None
        self.model_path = model_path

        # Determine model path if not provided
        if not self.model_path:
            self.model_path = self._find_model_file(self.DEFAULT_POSSIBLE_PATHS)

        if not self.model_path:
            # No model found; print helpful diagnostics and leave llm as None
            self._print_model_help()
            return

        if not _HAS_LLAMA or Llama is None:
            print("llama-cpp-python not installed or failed to import. Install with: pip install llama-cpp-python")
            return

        # Load model
        try:
            print(f"Loading Mistral model from: {self.model_path}")
            print(f"Model file exists: {os.path.exists(self.model_path)}")
            print(f"Model file size: {os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 'N/A'} bytes")
            print(f"Loading model with parameters: {self.MODEL_PARAMS}")
            self.llm = Llama(
                model_path=self.model_path,
                **self.MODEL_PARAMS
            )
            print("✓ Mistral model loaded successfully")
        except Exception as e:
            print(f"Error loading Mistral model: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            self.llm = None

    def _find_model_file(self, possible_paths: List[str]) -> Optional[str]:
        """Return the first existing path from possible_paths, or search models/ for any .gguf file."""
        # Check common explicit paths first
        for p in possible_paths:
            if os.path.exists(p):
                return p

        # Then search recursively under models/
        models_dir = 'models'
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for fname in files:
                    if fname.lower().endswith('.gguf'):
                        return os.path.join(root, fname)

        return None

    def _print_model_help(self) -> None:
        print("\n❌ No Mistral .gguf model found.")
        print("Please download a compatible model and place it under the 'models' directory.")
        print("Suggested filenames to look for:\n  - mistral-7b-instruct-v0.2.Q4_K_M.gguf\n  - mistral-7b-q4.gguf")
        if os.path.exists('models'):
            print("\nFiles currently in models/:")
            for root, _, files in os.walk('models'):
                for f in files:
                    print(f"  - {os.path.join(root, f)}")

    def generate_health_advisory(self, prediction: int, confidence: float) -> str:
        """Generate a short advisory using the LLM, or return a fallback message."""
        if self.llm is None:
            return self._default_advisory(prediction, confidence)

        condition = "MALNOURISHED" if prediction == 1 else "HEALTHY"
        prompt = (
            f"A child has been detected as {condition} with {confidence:.1f}% confidence. "
            "Provide a brief health advisory (2-3 sentences)."
        )

        try:
            resp = self.llm(prompt, max_tokens=150, temperature=0.7)
            # llama-cpp-python returns a dict-like with choices -> text
            text = None
            if isinstance(resp, dict) and 'choices' in resp:
                text = resp['choices'][0].get('text')
            elif hasattr(resp, 'choices'):
                text = resp.choices[0].text
            return (text or self._default_advisory(prediction, confidence)).strip()
        except Exception:
            return self._default_advisory(prediction, confidence)

    def answer_question(self, question: str) -> str:
        """Answer a user question using the LLM or return a helpful fallback."""
        if self.llm is None:
            return "LLM not available. Install llama-cpp-python and place a .gguf model in the models/ directory."

        try:
            resp = self.llm(question, max_tokens=200, temperature=0.7)
            if isinstance(resp, dict) and 'choices' in resp:
                return resp['choices'][0].get('text', '').strip()
            if hasattr(resp, 'choices'):
                return resp.choices[0].text.strip()
            return str(resp)
        except Exception as e:
            return f"Error generating response: {e}"

    def _default_advisory(self, prediction: int, confidence: float) -> str:
        if prediction == 1:
            return f"Child shows signs of malnutrition (Confidence: {confidence:.1f}%). Please consult a healthcare professional."
        return f"Child appears healthy (Confidence: {confidence:.1f}%). Continue regular monitoring."


# Module-level singleton
_LLM_INSTANCE: Optional[MistralLLMHandler] = None

def get_llm() -> MistralLLMHandler:
    """Return a singleton MistralLLMHandler instance."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = MistralLLMHandler()
    return _LLM_INSTANCE


