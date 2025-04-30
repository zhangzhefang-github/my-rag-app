import requests
import json
import time
import uuid

API_URL = "http://localhost:8000"

def test_health():
    """测试健康检查端点"""
    response = requests.get(f"{API_URL}/health")
    print(f"健康检查: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.status_code == 200

def test_conversation():
    """测试会话创建和消息发送"""
    # 创建会话
    conv_response = requests.post(
        f"{API_URL}/conversations",
        json={"title": f"测试会话 {uuid.uuid4()}"}
    )
    
    if conv_response.status_code != 200:
        print(f"创建会话失败: {conv_response.status_code}")
        return False
    
    conversation = conv_response.json()
    conversation_id = conversation["id"]
    print(f"会话创建成功: {conversation['title']}, ID: {conversation_id}")
    
    # 发送测试消息
    message_text = "你好，请介绍一下这个RAG系统的功能"
    message_response = requests.post(
        f"{API_URL}/conversations/{conversation_id}/messages",
        json={
            "conversation_id": conversation_id,
            "content": message_text,
            "role": "user"
        },
        stream=True  # 启用流式接收
    )
    
    if message_response.status_code != 200:
        print(f"发送消息失败: {message_response.status_code}")
        return False
    
    print(f"消息发送成功，正在接收流式响应...")
    
    # 接收流式消息
    full_response = ""
    for line in message_response.iter_lines():
        if line:
            # 解析SSE格式
            line_text = line.decode('utf-8')
            if line_text.startswith('data: '):
                try:
                    data = json.loads(line_text[6:])
                    if 'token' in data:
                        full_response += data['token']
                        print(data['token'], end='', flush=True)
                except json.JSONDecodeError:
                    print(f"无法解析JSON: {line_text}")
    
    print("\n\n完整响应:", full_response)
    return True

def get_all_conversations():
    """获取所有会话"""
    response = requests.get(f"{API_URL}/conversations")
    if response.status_code == 200:
        conversations = response.json().get("conversations", [])
        print(f"找到 {len(conversations)} 个会话:")
        for i, conv in enumerate(conversations):
            print(f"{i+1}. {conv['title']} (ID: {conv['id']})")
        return conversations
    else:
        print(f"获取会话失败: {response.status_code}")
        return []

def standard_query(query_text="什么是RAG技术?"):
    """测试标准查询"""
    print(f"发送标准查询: '{query_text}'")
    response = requests.post(
        f"{API_URL}/query",
        json={"query": query_text, "top_k": 3}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n标准查询结果:")
        print(result["answer"])
        print("\n来源:")
        for source in result["sources"]:
            print(f"- {source}")
        return True
    else:
        print(f"标准查询失败: {response.status_code}")
        return False

if __name__ == "__main__":
    # 测试API是否正常工作
    if not test_health():
        print("健康检查失败，退出测试")
        exit(1)
    
    print("\n1. 测试标准查询")
    standard_query()
    
    print("\n2. 测试会话和消息")
    test_conversation()
    
    print("\n3. 查看所有会话")
 