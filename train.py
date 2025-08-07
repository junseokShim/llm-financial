#pip install transformers datasets peft trl accelerate bitsandbytes

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load FinShibainu dataset
dataset = load_dataset("aiqwe/FinShibainu", name="qa", split="train")

# 2. Load Meta-LLaMA-3-8B tokenizer and model (4bit)
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# 3. Formatting dataset for instruction tuning
def generate_prompt(example):
    prompt = f"### 질문: {example['question']}\n### 답변: {example['answer']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(generate_prompt)

# 4. LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]  # LLaMA 계열
)

model = get_peft_model(model, peft_config)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./llama3-finshibainu-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./model/llama3-finshibainu-lora")
tokenizer.save_pretrained("./model/llama3-finshibainu-lora")