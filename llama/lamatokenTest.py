from transformers import AutoTokenizer

tokenizers = AutoTokenizer.from_pretrained("/net/llm-compiles-high-perf/project-apex/mutable/llama3/Meta-Llama-3-8B-Instruct")

text = tokenizers.encode("Hello, my dog is cute")
print(text)
print(tokenizers.decode(text))