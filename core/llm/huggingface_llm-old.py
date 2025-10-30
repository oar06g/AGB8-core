from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from langchain_core.runnables import Runnable
from typing import Optional, Dict, Any, List, Union
import torch
import logging
from pathlib import Path

# -----------------------------
# Logger setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceLLM(Runnable):
    """
    Advanced HuggingFace LLM Integration for local or Hub models.

    Features:
    - Supports local and remote models
    - Automatic 4-bit/8-bit quantization
    - Error handling
    - Full generation parameters support
    - Batching support
    - Memory optimization
    """

    def __init__(
        self,
        model_id: str,
        use_quantization: bool = False,
        quantization_bits: int = 4,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs
    ):
        """
        Args:
            model_id: Model name on HuggingFace Hub or local path
            use_quantization: Enable automatic quantization
            quantization_bits: 4 or 8 bits
            device_map: Device placement ("auto", "cpu", "cuda:0")
            torch_dtype: Torch data type (torch.float16, torch.bfloat16)
            trust_remote_code: Allow executing custom code from the model
            load_in_8bit: Load in 8-bit mode
            load_in_4bit: Load in 4-bit mode
            max_memory: Max memory per device
            cache_dir: Cache directory
        """
        self.model_id = model_id
        self.device = self._get_device()

        logger.info(f"🚀 Loading model: {model_id}")
        logger.info(f"🔧 Using device: {self.device}")

        try:
            # Setup quantization if needed
            quantization_config = None
            if use_quantization or load_in_4bit or load_in_8bit:
                quantization_config = self._setup_quantization(
                    load_in_4bit=load_in_4bit or (use_quantization and quantization_bits == 4),
                    load_in_8bit=load_in_8bit or (use_quantization and quantization_bits == 8)
                )

            # Set default torch dtype
            if torch_dtype is None and self.device != "cpu":
                torch_dtype = torch.float16

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_load_kwargs = {
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
                "cache_dir": cache_dir,
                **model_kwargs
            }
            if quantization_config:
                model_load_kwargs["quantization_config"] = quantization_config
            elif torch_dtype:
                model_load_kwargs["torch_dtype"] = torch_dtype
            if max_memory:
                model_load_kwargs["max_memory"] = max_memory

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)

            # Setup pipeline
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise

    def _get_device(self) -> str:
        """Detect available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_quantization(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration"""
        if not load_in_4bit and not load_in_8bit:
            return None

        try:
            config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else False,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None
            )
            logger.info(f"⚡ Enabled {'4-bit' if load_in_4bit else '8-bit'} quantization")
            return config
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup quantization: {str(e)}")
            return None

    def invoke(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        remove_prompt: bool = True,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from the model

        Args:
            prompt: Input string or list of strings
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-K sampling
            repetition_penalty: Repetition penalty
            do_sample: Enable sampling
            num_return_sequences: Number of outputs
            remove_prompt: Remove prompt from output

        Returns:
            Generated string or list of strings
        """
        try:
            gen_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **generation_kwargs
            }

            results = self.pipe(prompt, **gen_config)

            if isinstance(prompt, str):
                outputs = self._process_single_result(results, prompt if remove_prompt else None)
            else:
                outputs = [
                    self._process_single_result(res, prompt[i] if remove_prompt else None)
                    for i, res in enumerate(results)
                ]

            return outputs

        except Exception as e:
            logger.error(f"❌ Generation error: {str(e)}")
            raise

    def _process_single_result(
        self,
        result: Union[List[Dict], Dict],
        prompt: Optional[str] = None
    ) -> Union[str, List[str]]:
        """Process a single generation result"""
        if isinstance(result, dict):
            result = [result]

        texts = []
        for res in result:
            text = res["generated_text"]
            if prompt and text.startswith(prompt):
                text = text[len(prompt):]
            texts.append(text.strip())

        return texts[0] if len(texts) == 1 else texts
    def __call__(self, prompt: str, **kwargs):
        return self.invoke(prompt, **kwargs)

    def batch_invoke(self, prompts: List[str], batch_size: int = 4, **generation_kwargs) -> List[str]:
        """Generate text in batches"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.invoke(batch, **generation_kwargs)
            if isinstance(batch_results, str):
                batch_results = [batch_results]
            results.extend(batch_results)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "vocab_size": len(self.tokenizer),
            "model_type": self.model.config.model_type,
            "hidden_size": self.model.config.hidden_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.model.dtype),
        }

    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("🧹 Memory cache cleared")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    llm = HuggingFaceLLM(
        model_id="gpt2",
        use_quantization=True,
        quantization_bits=4
    )

    # Single generation
    response = llm.invoke("Hello, how can I")
    print("Output:", response)

    # Batch generation
    prompts = ["Artificial intelligence is", "Programming means", "The future will be"]
    batch_results = llm.batch_invoke(prompts, batch_size=2)
    for prompt, result in zip(prompts, batch_results):
        print(f"\n{prompt}: {result}")

    # Model info
    info = llm.get_model_info()
    print("\nModel Info:", info)
