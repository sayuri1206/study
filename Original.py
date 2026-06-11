#20260528
#モデルのファインチューニングを行う
#→Inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

model_id = "line-corporation/japanese-large-lm-1.7b-instruction-sft"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4bit設定（重要）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # 精度高い
    bnb_4bit_compute_dtype=torch.bfloat16,  # H100ならこれ
)

# モデルロード（ここで量子化される）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},  # ← これが本命
)

# kbit用準備
model = prepare_model_for_kbit_training(model)

# モデルの構造を確認して、LoRAのターゲットモジュールを特定
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(name)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],   # ← ここ直す
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset

# 3.1. CSVファイルの読み込み
# data_filesにはパスを指定します。
dataset = load_dataset("csv", data_files="/home/student/t123122/jupyter/sample/app/results_find_1.csv", split="train")

# 3.2. プロンプトテンプレートの適用
# LLMが学習しやすい形式（Instruction形式など）に整形します。
def format_prompts(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        # モデルの学習フォーマットに合わせて調整（以下は一例）
        text = f"### 指示: {examples['instruction'][i]}\n### 回答: {examples['output'][i]}"
        output_texts.append(text)
    return {"text": output_texts}

# カラム名が 'instruction' と 'output' の場合のみ実行
if "instruction" in dataset.column_names:
    dataset = dataset.map(format_prompts, batched=True)

# 3.3. トークナイズ処理
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512, 
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. トレーニング実行
training_args = TrainingArguments(
    output_dir="./../output-domain-llm_20260528",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=False,
    bf16=False,
    optim="paged_adamw_32bit",
    remove_unused_columns=True,
    # 追加：モデルをデバイスに自動的に移動させる処理をスキップする（古いバージョン用）
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# 5. モデルの保存
model.save_pretrained("./../final-domain-model_20260528")
