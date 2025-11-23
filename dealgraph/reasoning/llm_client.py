# ABOUTME: LLM client integration for deal reasoning with Cerebras API and DSPy support.
# ABOUTME: Provides JSON response parsing and error handling for deal analysis.

from typing import Dict, Any, Optional
import json
import logging
import importlib
import openai

settings_module = importlib.import_module("dealgraph.config.settings")


class LLMClientError(Exception):
    """Raised when LLM client operations fail."""
    pass


class LLMClient:
    """
    LLM client for deal reasoning with JSON response parsing.
    
    Supports both Cerebras API (primary) and OpenAI-compatible APIs.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (defaults to settings.DSPY_MODEL)
            api_key: API key (defaults to settings.CEREBRAS_API_KEY)
            base_url: Base URL (defaults to settings.CEREBRAS_BASE_URL)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        settings = settings_module.settings
        self.model = model or settings.DSPY_MODEL
        self.api_key = api_key or settings.CEREBRAS_API_KEY
        self.base_url = base_url or settings.CEREBRAS_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("API key not provided and not found in settings")
        
        # Initialize OpenAI-compatible client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.logger = logging.getLogger(__name__)
    
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        Complete a prompt and parse JSON response.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with the actual request
            temperature: Override default temperature
            max_tokens: Override default max tokens
            retries: Number of retry attempts on failure
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            LLMClientError: If completion fails or response is invalid JSON
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        for attempt in range(retries):
            try:
                self.logger.debug(
                    f"LLM completion attempt {attempt + 1}/{retries} "
                    f"for model {self.model}"
                )
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                
                if not content:
                    raise LLMClientError("Empty response from LLM")
                
                # Parse JSON response
                try:
                    parsed = json.loads(content)
                    self.logger.debug(f"Successfully parsed JSON response: {parsed}")
                    return parsed
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {content}")
                    if attempt == retries - 1:
                        raise LLMClientError(f"Invalid JSON response: {e}")
                    continue
                    
            except Exception as e:
                self.logger.warning(f"LLM completion attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise LLMClientError(f"LLM completion failed: {e}")
                continue
        
        raise LLMClientError("All retry attempts failed")
    
    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3
    ) -> str:
        """
        Complete a prompt and return text response.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt with the actual request
            temperature: Override default temperature
            max_tokens: Override default max tokens
            retries: Number of retry attempts on failure
            
        Returns:
            Text response as string
            
        Raises:
            LLMClientError: If completion fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok
                )
                
                content = response.choices[0].message.content.strip()
                
                if not content:
                    raise LLMClientError("Empty response from LLM")
                
                self.logger.debug(f"Successfully completed text response: {content[:100]}...")
                return content
                    
            except Exception as e:
                self.logger.warning(f"LLM text completion attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise LLMClientError(f"LLM text completion failed: {e}")
                continue
        
        raise LLMClientError("All text completion retry attempts failed")
    
    def validate_json_response(
        self,
        response: Dict[str, Any],
        required_fields: list[str]
    ) -> bool:
        """
        Validate that JSON response contains required fields.
        
        Args:
            response: Parsed JSON response
            required_fields: List of required field names
            
        Returns:
            True if all required fields are present and valid
        """
        for field in required_fields:
            if field not in response:
                self.logger.warning(f"Missing required field: {field}")
                return False
            
            # Basic validation for common fields
            value = response[field]
            if field == "precedents" and not isinstance(value, list):
                self.logger.warning(f"Field {field} should be a list")
                return False
            elif field == "playbook_levers" and not isinstance(value, list):
                self.logger.warning(f"Field {field} should be a list")
                return False
            elif field == "risk_themes" and not isinstance(value, list):
                self.logger.warning(f"Field {field} should be a list")
                return False
            elif field == "narrative_summary" and not isinstance(value, str):
                self.logger.warning(f"Field {field} should be a string")
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured model."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": str(self.temperature),
            "max_tokens": str(self.max_tokens)
        }
    
    def __repr__(self) -> str:
        """String representation of the LLM client."""
        return f"LLMClient(model={self.model}, base_url={self.base_url})"


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    
    if _llm_client is None:
        _llm_client = LLMClient()
    
    return _llm_client


def set_llm_client(client: LLMClient) -> None:
    """Set the global LLM client instance."""
    global _llm_client
    _llm_client = client
