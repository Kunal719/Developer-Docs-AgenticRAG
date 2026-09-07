from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_PATH = "reranking/onnx/bge-reranker-base"

query = "How does LangGraph checkpoint recovery work?"
document = "LangGraph checkpoints persist graph state and allow execution to resume after failures."

print("Loading PyTorch tokenizer...")
cross_encoder = CrossEncoder(MODEL_NAME)

print("\nLoading ONNX tokenizer...")
onnx_tokenizer = AutoTokenizer.from_pretrained(ONNX_PATH)

print("\nPyTorch tokenizer:")
print(cross_encoder.tokenizer.__class__)
print(cross_encoder.tokenizer.name_or_path)

print("\nONNX tokenizer:")
print(onnx_tokenizer.__class__)
print(onnx_tokenizer.name_or_path)

print("\nSpecial tokens:")
print("PyTorch:", cross_encoder.tokenizer.special_tokens_map)
print("ONNX:   ", onnx_tokenizer.special_tokens_map)

print("\nTokenizer configs:")
print("PyTorch model max length:", cross_encoder.max_length)
print("ONNX tokenizer max length:", onnx_tokenizer.model_max_length)

# Compare token IDs
pt_tokens = cross_encoder.tokenizer(
    query,
    document,
    truncation=True,
)

onnx_tokens = onnx_tokenizer(
    query,
    document,
    truncation=True,
)

print("\nToken IDs identical:")
print(pt_tokens["input_ids"] == onnx_tokens["input_ids"])

print("\nPyTorch token count:", len(pt_tokens["input_ids"]))
print("ONNX token count:   ", len(onnx_tokens["input_ids"]))