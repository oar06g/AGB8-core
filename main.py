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
