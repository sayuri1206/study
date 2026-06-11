#20260528利用
#○○とは何かと出題する
#Original.pyでファインチューニングしたモデルを使用
import os
import torch
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("モデル初期化中...")

model_id = "line-corporation/japanese-large-lm-1.7b-instruction-sft"
adapter_path = "./../final-domain-model_20260514"

# 1. Tokenizerのロード (left指定を確実にする)
# legacy=Falseに加えて、clean_up_tokenization_spacesを設定
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# 2. 4bitロード設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 3. モデルロード
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 4. LoRA適用と【強制設定上書き】
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 警告を消すための呪文
model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id
# 念のためモデル全体の設定を確認
model.config.use_cache = True 

device = next(model.parameters()).device
print(f"準備完了！ (デバイス: {device})")

def chat(user_input):
    # 改行 \n が <unk> になるのを防ぐため、シンプルなスペース区切りに変更
    # 学習時のフォーマットが「指示: {input} 回答:」のような形式であればこちらが正解です
    prompt = f"### 指示: {user_input} ### 回答:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False, # 特殊トークンの自動挿入をオフにする
        return_token_type_ids=False
    ).to(device)
    
    print("生成中...", end="", flush=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2, # 少し強めに設定してループを防ぐ
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]] # <unk>を出さないように制約
        )
    
    print("完了！")
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # デバッグ用：もし空ならトークンIDを表示
    if len(generated_tokens) == 0:
        return "トークンが生成されませんでした。"

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if not response.strip():
        return f"デコード結果が空です (トークンID: {generated_tokens[:5].tolist()})"
        
    return response.strip()

print("チャット開始（exitで終了）")

while True:
    user_input = input("\nあなた: ").strip()
    if user_input.lower() in ["exit", "quit", "終了"]:
        print("終了")
        break
    
    if user_input:
        response = chat(user_input)
        print(f"AI: {response}\n")
