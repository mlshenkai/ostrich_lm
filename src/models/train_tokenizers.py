from tokenizers import Tokenizer, models, decoders, pre_tokenizers, trainers
import os
from scripts.tokenizers.tokenizers_models import create_tokenizer_config, read_texts_from_jsonl
from transformers import AutoTokenizer

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int=8192) -> None:
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(model=models.BPE(unk_token="<unk>"))
    # tokenizer.normalizer = NFKC() # type: ignore
    # 使用 Whitespace 预处理器替代 ByteLevel，更适合中文
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # type: ignore

    # 移除 ByteLevel 解码器，使用默认解码
    tokenizer.decoder = decoders.ByteLevel() # type: ignore
    
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"       
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print(f"Training tokenzier with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))
    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")


def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "你好吗?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "真实一个好天气！"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)


if __name__ == "__main__":
    train_tokenizer(data_path="./seq_monkey_datawhale.jsonl", save_dir="./resources/models/ostrich1")
    # eval_tokenizer(tokenizer_path="./resources/models/ostrich2")