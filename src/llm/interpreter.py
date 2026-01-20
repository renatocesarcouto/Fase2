from typing import Dict, Any, Optional
import requests
import json
from utils.logger import model_logger

class LLMInterpreter:
    """
    Generates natural language explanations for medical diagnoses.
    Supports local Ollama or falls back to rule-based templates.
    """
    
    def __init__(self, use_ollama: bool = False, ollama_url: str = "http://localhost:11434/api/generate", model_name: str = "llama3"):
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.model_name = model_name

    def generate_explanation(self, prediction: int, probability: float, shap_features: Dict[str, float]) -> str:
        """
        Generates an explanation for the prediction.
        """
        diagnosis = "MALIGNO" if prediction == 1 else "BENIGNO"
        confidence = f"{probability:.1%}"
        
        # Sort features by importance (absolute SHAP value)
        sorted_features = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_3 = sorted_features[:3]
        
        context = f"""
        PACIENTE: Diagnóstico {diagnosis} (Confiança: {confidence}).
        PRINCIPAIS FATORES (SHAP):
        - {top_3[0][0]}: {top_3[0][1]:.4f}
        - {top_3[1][0]}: {top_3[1][1]:.4f}
        - {top_3[2][0]}: {top_3[2][1]:.4f}
        """

        if self.use_ollama:
            try:
                return self._call_ollama(context)
            except Exception as e:
                model_logger.warning(f"Ollama failed ({e}). Falling back to template.")
                return self._template_based_explanation(diagnosis, confidence, top_3)
        else:
            return self._template_based_explanation(diagnosis, confidence, top_3)

    def _call_ollama(self, context: str) -> str:
        """Calls local Ollama API."""
        prompt = f"""
        Você é um assistente médico especialista. Explique o diagnóstico abaixo para um médico, 
        citando as evidências (features) mais importantes. Seja direto e profissional.
        
        {context}
        
        Explicação:
        """
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        model_logger.info(f"Connecting to Ollama at {self.ollama_url}")
        response = requests.post(self.ollama_url, json=payload, timeout=10)
        response.raise_for_status()
        model_logger.info("Ollama response received successfully")
        return response.json()['response']

    def _template_based_explanation(self, diagnosis: str, confidence: str, top_features: list) -> str:
        """Fallback rule-based explanation."""
        
        # Determine strictness based on confidence
        try:
            conf_val = float(confidence.strip('%'))
        except ValueError:
            conf_val = 0.0 # Default if parsing fails
            
        tone = "sugere fortemente" if conf_val > 90 else "indica probabilidade de"
        
        text = f"A análise {tone} um diagnóstico {diagnosis} com {confidence} de confiança. "
        text += "Esta conclusão é baseada principalmente nos seguintes marcadores citológicos: "
        
        parts = []
        for feature, val in top_features:
            # Interpreting SHAP sign roughly
            impact = "elevando o risco" if val > 0 else "reduzindo o risco" if diagnosis == "BENIGNO" else "atenuando o sinal"
            parts.append(f"{feature} ({impact})")
            
        text += ", ".join(parts) + "."
        return text
