from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./resources/models/ostrich1")
print(tokenizer)
print(tokenizer.bos_token)
text = "你好啊"
input_ids = tokenizer(text)
print(input_ids)