from fastapi import FastAPI
from config import APP_PORT
from utils.logging_config import setup_logging
import uvicorn

# 导入检索和生成回答的函数
from retrieval import retrieve_docs
from llm_service import generate_answer
from vector_store.setup import get_vector_store

# 设置日志
setup_logging()

# FastAPI应用
app = FastAPI()

# 初始化向量存储
index = get_vector_store()

@app.get("/index")
async def show_index():
    return {"message": "这是首页!"}


@app.get("/hello/{query}")
async def query_llm(query: str):
    # 打印查询
    print("Query:", query)
    
    retrieved_docs, retrieved_chunks = retrieve_docs(query, index)
    
    # 打印检索到的文档
    print("检索到的文档：")
    for doc in retrieved_docs:
        print(doc)
    
    # 打印检索到的文本块
    print("\n检索到的相关文本块：")
    for doc_id, chunk in retrieved_chunks:
        print(f"[文档{doc_id}] {chunk}")
    
    answer = generate_answer(query, retrieved_docs, retrieved_chunks)
    print("\nAnswer:", answer)
 
    return {"message": f"{query} 的回答是: {answer}"}


if __name__ == "__main__":
    # 使用配置的端口
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)