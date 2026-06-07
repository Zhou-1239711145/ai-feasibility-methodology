# AI批量测试脚本模板
# 说明：此为通用脚本框架，不包含具体用例和system_prompt

import requests
import numpy as np
import pandas as pd
import time

url_chat = "https://api.siliconflow.cn/v1/chat/completions"
url_emb = "https://api.siliconflow.cn/v1/embeddings"
api_key = "your_api_key_here"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def ask_ai(question, system_prompt):
    data = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(url_chat, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def cosine_similarity_manual(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    data = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text
    }
    response = requests.post(url_emb, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def semantic_similarity(text1, text2):
    emb1 = np.array(get_embedding(text1))
    emb2 = np.array(get_embedding(text2))
    return cosine_similarity_manual(emb1, emb2)

def batch_test(csv_file, system_prompt, threshold=0.7):
    df = pd.read_csv(csv_file)
    results = []
    
    for idx, row in df.iterrows():
        question = row['question']
        expected = row['expected']
        ai_answer = ask_ai(question, system_prompt)
        score = semantic_similarity(expected, ai_answer)
        passed = score >= threshold
        
        results.append({
            "问题": question,
            "期望": expected,
            "AI回答": ai_answer,
            "相似度": round(score, 4),
            "是否通过": "通过" if passed else "失败"
        })
        
        time.sleep(0.5)
    
    return results

if __name__ == "__main__":
    # 使用方法：请自行准备CSV文件（包含question和expected列）
    # 以及system_prompt
    print("请配置CSV文件路径和system_prompt后使用")
