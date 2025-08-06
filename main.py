import re
import os
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.utils import *
from src.inference  import extract_answer_only
from huggingface_hub import login

#set device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

login(token="")

if __name__ == "__main__":
    # Load test data
    test = pd.read_csv('./data/test.csv')

    # Define the model name

    # mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B
    # beomi/KoAlpaca-Polyglot-5.8B : 0.19
    #model_name = "mistralai/Mistral-7B-v0.1"

    model_name = "openai/gpt-oss-20b"

    # Tokenizer 및 모델 로드 (4bit)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,       # float16 사용
        device_map="auto",               # GPU/CPU 자동 분산
        low_cpu_mem_usage=True,          # 로딩 시 CPU 메모리 절약
        use_safetensors=True             # safetensors 파일 사용 시
    )

    # Inference pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
        
    preds = []

    for q in tqdm(test['Question'], desc="Inference"):
        prompt = make_prompt_auto(q)
        output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
        pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
        preds.append(pred_answer)

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission['Answer'] = preds
    sample_submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')