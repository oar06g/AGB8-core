# from core.llm.huggingface_llm import HuggingFaceLLM

# # -----------------------------
# # Step 1: Create the LLM
# # -----------------------------
# llm = HuggingFaceLLM(
#     model_id="EleutherAI/gpt-neo-125M",
#     use_quantization=True,
#     quantization_bits=4
# )

# # -----------------------------
# # Step 2: Define the prompt
# # -----------------------------
# prompt = "Write a short paragraph about Artificial Intelligence."

# # -----------------------------
# # Step 3: Generate text
# # -----------------------------
# response = llm.invoke(
#     prompt,
#     max_new_tokens=100,
#     temperature=0.7
# )

# print("Generated Text:\n", response)

from core.config.logging_config import log_info, log_error
from core.llm.huggingface_llm import ModelManager

# NOTE: in real usage, use local cached models or HF tokens if private models.
manager = ModelManager(max_models_loaded=2, result_cache_size=512, result_ttl=3600)

# load small model for quick testing (use a small HF model name you have)
model_id = "gpt2"  # replace with a model you have/allow
try:
    # load model (first time)
    m = manager.load_model(model_id, model_type="causal", use_quantization=False, cache_dir=None)

    # single generation
    out = manager.generate(model_id, "Hello, how are you?", max_new_tokens=40)
    print("OUT:", out)

    # batch generation
    prompts = ["Artificial intelligence is", "Programming means", "The future will be"]
    outs = manager.generate(model_id, prompts, max_new_tokens=20, num_return_sequences=1)
    print("BATCH OUTS:", outs)

    # list loaded
    print("Loaded models:", manager.list_loaded())

    # use result cache: same prompt should be cached
    out2 = manager.generate(model_id, "Hello, how are you?", max_new_tokens=40)
    print("OUT (cached):", out2)

    # unload
    manager.unload_model(model_id)
    print("After unload, loaded:", manager.list_loaded())

except Exception as e:
    log_error(f"Example run failed: {e}")