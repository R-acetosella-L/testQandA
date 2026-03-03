import os
import time
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

API_TOKEN = os.getenv("API_TOKEN")
client = OpenAI()

db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# レート制限（簡易：IPごと1分5回）
request_log = {}

def rate_limit(ip):
    now = time.time()
    request_log.setdefault(ip, [])
    request_log[ip] = [t for t in request_log[ip] if now - t < 60]

    if len(request_log[ip]) >= 5:
        return False

    request_log[ip].append(now)
    return True

# 個人情報検知
PII_PATTERN = r"(住所|電話|メール|@|氏名)"

@app.route("/chat", methods=["POST"])
def chat():
    # トークン認証
    token = request.headers.get("X-API-TOKEN")
    if token != API_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    # レート制限
    ip = request.remote_addr
    if not rate_limit(ip):
        return jsonify({"answer": "利用回数が多すぎます。時間をおいてください。"}), 429

    data = request.json
    question = data.get("question", "")

    # 個人情報ブロック
    if re.search(PII_PATTERN, question):
        return jsonify({"answer": "個人情報は入力しないでください。"}), 400

    docs = db.similarity_search(question, k=3)

    context = ""
    sources = set()
    for d in docs:
        context += d.page_content + "\n"
        sources.add(d.metadata["source"])

    prompt = f"""
以下の資料に基づいて回答してください。
資料にない場合は「資料に記載がありません」と答えてください。

{context}

質問：{question}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return jsonify({
        "answer": answer,
        "sources": list(sources)
    })

if __name__ == "__main__":
    app.run()
