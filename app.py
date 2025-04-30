import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import datetime
import uuid
import asyncio
import threading
from typing import Dict, List, Any
import os
from sseclient import SSEClient
import pandas as pd

# API地址配置
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_TIMEOUT = 30  # 增加超时时间到30秒

# 创建一个带有重试机制的会话
def create_retry_session():
    """创建带有重试机制的会话，兼容新旧版本urllib3"""
    session = requests.Session()
    
    # 尝试创建Retry对象，兼容不同版本API
    try:
        # 新版urllib3使用allowed_methods
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=['GET', 'POST', 'PUT', 'DELETE', 'HEAD']
        )
    except TypeError:
        try:
            # 旧版urllib3使用method_whitelist
            retry = Retry(
                total=3,
                connect=3,
                read=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                method_whitelist=['GET', 'POST', 'PUT', 'DELETE', 'HEAD']
            )
        except TypeError:
            # 最旧版本可能没有method_whitelist
            retry = Retry(
                total=3,
                connect=3,
                read=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            )
    
    # 挂载适配器到会话
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 创建一个全局请求会话
try:
    http_session = create_retry_session()
except Exception as e:
    st.warning(f"无法创建带重试功能的会话，将使用标准会话: {e}")
    http_session = requests.Session()

# 页面配置
st.set_page_config(
    page_title="RAG对话助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 主容器调整 */
    .main .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 10rem; /* 为固定底部输入框留出空间 */
    }
    
    /* 隐藏Streamlit默认页脚 */
    footer {
        visibility: hidden;
    }
    
    /* 固定在底部的输入区域 - 关键CSS */
    #fixed-input-container {
        position: fixed;
        bottom: 0;
        right: 0;
        width: calc(100% - 22%); /* 调整宽度以适应侧边栏 */
        background-color: white;
        border-top: 1px solid #e0e0e0;
        padding: 1rem;
        z-index: 9999; /* 确保在最上层 */
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* 消息区域与样式 */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    /* 用户消息样式 */
    .user-message {
        align-self: flex-end;
        background-color: #e3f2fd;
        color: #0d47a1;
        border-radius: 1rem 1rem 0 1rem;
        padding: 0.8rem 1rem;
        max-width: 80%;
        margin-left: auto;
    }
    
    /* AI消息样式 */
    .ai-message {
        align-self: flex-start;
        background-color: #f5f5f5;
        color: #333;
        border-radius: 1rem 1rem 1rem 0;
        padding: 0.8rem 1rem;
        max-width: 80%;
    }
    
    /* 发送者标签 */
    .sender-label {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    
    /* 统计信息样式 */
    .stats-info {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.3rem;
        text-align: right;
    }
    
    /* 覆盖Streamlit默认边距 */
    .stTextArea div[data-baseweb="textarea"] {
        margin-bottom: 0 !important;
    }
    
    /* 清除浮动 */
    .clear-float {
        clear: both;
    }
    
    /* 适应移动设备 */
    @media (max-width: 768px) {
        #fixed-input-container {
            width: 100%;
            padding: 0.5rem;
        }
    }
</style>

<script>
// 监听DOM变化，确保滚动到最新消息
document.addEventListener('DOMContentLoaded', function() {
    // 创建观察器并配置
    const observer = new MutationObserver(function() {
        // 滚动到底部函数
        const scrollToBottom = function() {
            window.scrollTo(0, document.body.scrollHeight);
        };
        
        // 延迟执行，确保DOM已完全更新
        setTimeout(scrollToBottom, 200);
    });
    
    // 观察整个body元素的变化
    const config = { childList: true, subtree: true };
    observer.observe(document.body, config);
    
    // 初始滚动
    setTimeout(function() {
        window.scrollTo(0, document.body.scrollHeight);
    }, 500);
});
</script>
""", unsafe_allow_html=True)

# 初始化会话状态
if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = {}

if "response_times" not in st.session_state:
    st.session_state.response_times = {}

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "user_input_value" not in st.session_state:
    st.session_state.user_input_value = ""

if "api_status" not in st.session_state:
    st.session_state.api_status = "未知"

if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = time.time() - 61  # 确保首次运行立即执行健康检查

# 健康检查函数
def check_api_health():
    """检查API服务器是否在线，如果离线尝试重连"""
    try:
        # 首先尝试健康检查端点
        response = http_session.get(f"{API_URL}/health", timeout=5)  # 短超时
        if response.status_code == 200:
            st.session_state.api_status = "在线"
            return True
        elif response.status_code == 404:
            # 如果健康检查端点不存在，尝试检查根端点
            root_response = http_session.get(f"{API_URL}/", timeout=5)
            if root_response.status_code in [200, 404]:  # 即使是404也表示服务器在线
                st.session_state.api_status = "在线 (无健康检查端点)"
                return True
        
        # 如果状态码不是200也不是404，标记为错误
        st.session_state.api_status = f"错误 (状态码: {response.status_code})"
        return False
    except requests.exceptions.RequestException:
        try:
            # 健康检查失败，尝试访问根路径
            root_response = http_session.get(f"{API_URL}/", timeout=5)
            if root_response.status_code in [200, 404]:  # 即使是404也表示服务器在线
                st.session_state.api_status = "在线 (无健康检查端点)"
                return True
        except:
            pass
        
        st.session_state.api_status = "离线"
        return False

# 去掉定期健康检查的后台线程实现
def background_health_check():
    """后台定期检查API状态 - 已移除，改为在主UI中执行"""
    pass

# API调用函数
def get_conversations():
    """获取所有会话"""
    try:
        response = requests.get(f"{API_URL}/conversations", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()["conversations"]
        else:
            st.error(f"获取会话失败: {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return []
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return []
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return []

def create_conversation(title="新会话"):
    """创建新会话"""
    try:
        data = {"title": title}
        response = requests.post(f"{API_URL}/conversations", json=data, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"创建会话失败: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return None
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return None
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return None

def get_messages(conversation_id):
    """获取会话的消息历史"""
    try:
        response = requests.get(f"{API_URL}/conversations/{conversation_id}/messages", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()["messages"]
        else:
            st.error(f"获取消息失败: {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return []
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return []
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return []

def update_conversation(conversation_id, title):
    """更新会话标题"""
    try:
        data = {"title": title}
        response = requests.put(f"{API_URL}/conversations/{conversation_id}", json=data, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"更新会话失败: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return None
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return None
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return None

def delete_conversation(conversation_id):
    """删除会话"""
    try:
        response = requests.delete(f"{API_URL}/conversations/{conversation_id}", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return True
        else:
            st.error(f"删除会话失败: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return False
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return False
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return False

def reload_documents():
    """重新加载文档"""
    try:
        with st.spinner("正在重新加载文档..."):
            response = requests.post(f"{API_URL}/reload", timeout=API_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                st.success(f"文档加载成功，共 {result['doc_count']} 个文档")
                return True
            else:
                st.error(f"文档加载失败: {response.status_code}")
                return False
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查API服务器状态")
        return False
    except requests.exceptions.ConnectionError:
        st.error("连接错误，请确保API服务器正在运行")
        return False
    except Exception as e:
        st.error(f"API调用错误: {e}")
        return False

def send_message_stream(conversation_id, content):
    """发送消息并获取流式响应"""
    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""
    
    # 最大重试次数
    max_retries = 2
    retry_count = 0
    
    try:
        # 准备请求数据
        data = {
            "conversation_id": conversation_id,
            "content": content,
            "role": "user"
        }
        
        # 设置生成状态
        st.session_state.is_generating = True
        
        # 创建容器用于更新响应
        response_container = st.empty()
        time_container = st.empty()
        
        # 将用户消息添加到消息列表中
        if conversation_id not in st.session_state.conversation_messages:
            st.session_state.conversation_messages[conversation_id] = []
        
        # 添加用户消息
        st.session_state.conversation_messages[conversation_id].append({
            "role": "user",
            "content": content,
            "created_at": datetime.datetime.now().isoformat()
        })
        
        # 使用SSE客户端进行流式请求 (带重试逻辑)
        url = f"{API_URL}/conversations/{conversation_id}/messages"
        headers = {"Content-Type": "application/json"}
        
        # 重试逻辑
        while retry_count <= max_retries:
            try:
                # 检查API状态
                if not check_api_health():
                    raise ConnectionError("API服务器不可用，请检查连接")
                
                # 使用会话进行请求
                with http_session.post(url, json=data, headers=headers, stream=True, timeout=API_TIMEOUT) as response:
                    # 检查响应状态
                    if response.status_code != 200:
                        error_msg = f"API错误: 状态码 {response.status_code}"
                        if retry_count < max_retries:
                            retry_count += 1
                            response_container.markdown(f"正在重试... ({retry_count}/{max_retries})")
                            time.sleep(1)  # 等待1秒再重试
                            continue
                        else:
                            st.error(error_msg)
                            st.session_state.conversation_messages[conversation_id].append({
                                "role": "assistant",
                                "content": f"发生错误: {error_msg}",
                                "created_at": datetime.datetime.now().isoformat(),
                                "error": True
                            })
                            return None
                    
                    # 成功获取响应，处理SSE流
                    client = SSEClient(response)
                    
                    # 处理流式响应
                    for event in client.events():
                        if event.data:
                            try:
                                data = json.loads(event.data)
                                
                                # 检查事件类型
                                if event.event == "error":
                                    error_detail = data.get("detail", "未知错误")
                                    st.error(error_detail)
                                    break
                                
                                if event.event == "end":
                                    break
                                
                                # 正常数据处理
                                token = data.get("token", "")
                                token_count += 1
                                
                                if token_count == 1 and first_token_time is None:
                                    first_token_time = time.time()
                                
                                full_response += token
                                
                                # 更新UI
                                response_container.markdown(full_response)
                                
                                # 实时更新计时器
                                current_time = time.time() - start_time
                                time_container.markdown(f"⏱️ 已用时: {current_time:.2f}秒", unsafe_allow_html=True)
                                
                            except json.JSONDecodeError:
                                continue
                    
                    # 成功处理完流，跳出重试循环
                    break
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ConnectionError) as e:
                if retry_count < max_retries:
                    retry_count += 1
                    error_msg = f"连接错误，正在重试 ({retry_count}/{max_retries})..."
                    response_container.markdown(error_msg)
                    time.sleep(1)  # 等待1秒再重试
                else:
                    error_msg = f"多次连接失败，请检查API服务器状态: {str(e)}"
                    st.error(error_msg)
                    full_response = f"发生错误: {error_msg}"
                    break
        
        # 计算最终时间指标
        end_time = time.time()
        total_time = end_time - start_time
        first_token_latency = (first_token_time - start_time) if first_token_time else 0
        tokens_per_second = token_count / total_time if total_time > 0 and total_time > 0 else 0
        
        # 保存响应时间
        st.session_state.response_times[f"{conversation_id}_{len(st.session_state.conversation_messages.get(conversation_id, []))}"] = {
            "total_time": total_time,
            "first_token_latency": first_token_latency,
            "tokens_per_second": tokens_per_second,
            "start_time": datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        # 显示最终时间信息
        time_info = (
            f"⏱️ 总耗时: {total_time:.2f}秒 | "
            f"首token延迟: {first_token_latency:.2f}秒 | "
            f"速度: {tokens_per_second:.1f}token/秒 | "
            f"开始于: {datetime.datetime.now().strftime('%H:%M:%S')}"
        )
        time_container.markdown(time_info, unsafe_allow_html=True)
        
        # 将AI回复添加到消息列表中 (不再从后端获取，避免重复)
        st.session_state.conversation_messages[conversation_id].append({
            "role": "assistant",
            "content": full_response,
            "created_at": datetime.datetime.now().isoformat(),
            "response_time": time_info
        })
        
        return full_response
        
    except Exception as e:
        error_msg = f"消息发送错误: {str(e)}"
        st.error(error_msg)
        
        # 添加错误消息到对话中
        if conversation_id in st.session_state.conversation_messages:
            st.session_state.conversation_messages[conversation_id].append({
                "role": "assistant",
                "content": f"发生错误: {error_msg}",
                "created_at": datetime.datetime.now().isoformat(),
                "error": True
            })
        
        return None
    finally:
        st.session_state.is_generating = False

# UI 组件函数
def new_chat():
    """创建新会话并切换到它"""
    new_conv = create_conversation()
    if new_conv:
        st.session_state.conversations.insert(0, new_conv)
        st.session_state.current_conversation = new_conv["id"]
        st.session_state.conversation_messages[new_conv["id"]] = []
        st.rerun()

def select_conversation(conv_id):
    """选择特定会话"""
    st.session_state.current_conversation = conv_id
    
    # 如果尚未加载消息，则加载消息
    if conv_id not in st.session_state.conversation_messages:
        messages = get_messages(conv_id)
        st.session_state.conversation_messages[conv_id] = messages
    
    st.rerun()

def edit_title(conv_id, new_title):
    """编辑会话标题"""
    result = update_conversation(conv_id, new_title)
    if result:
        # 更新本地会话列表
        for i, conv in enumerate(st.session_state.conversations):
            if conv["id"] == conv_id:
                st.session_state.conversations[i]["title"] = new_title
                break
    
def remove_conversation(conv_id):
    """删除会话"""
    if delete_conversation(conv_id):
        # 从本地状态中移除
        st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != conv_id]
        if conv_id in st.session_state.conversation_messages:
            del st.session_state.conversation_messages[conv_id]
        
        # 如果删除的是当前会话，选择新的当前会话
        if st.session_state.current_conversation == conv_id:
            if st.session_state.conversations:
                st.session_state.current_conversation = st.session_state.conversations[0]["id"]
            else:
                st.session_state.current_conversation = None
        
        st.rerun()

def filter_conversations():
    """根据搜索词过滤会话"""
    if not st.session_state.search_query:
        return st.session_state.conversations
    
    query = st.session_state.search_query.lower()
    return [c for c in st.session_state.conversations 
            if query in c.get("title", "").lower()]

def load_conversations():
    """加载会话列表"""
    conversations = get_conversations()
    if conversations:
        st.session_state.conversations = conversations
        # 如果尚未选择会话，选择第一个
        if not st.session_state.current_conversation and conversations:
            st.session_state.current_conversation = conversations[0]["id"]

# 刷新文档索引
def refresh_documents_action():
    reload_documents()

# 回调函数
def clear_input():
    st.session_state.user_input_value = ""

def process_input():
    if st.session_state.user_input and not st.session_state.is_generating:
        # 获取输入值
        user_input = st.session_state.user_input
        # 清空输入
        st.session_state.user_input_value = ""
        # 处理消息
        if st.session_state.current_conversation:
            send_message_stream(st.session_state.current_conversation, user_input)

# 处理回车键发送
def handle_enter():
    if st.session_state.user_input and not st.session_state.is_generating:
        process_input()
        return True
    return False

# 主UI布局
def main():
    # 处理定期健康检查 - 每60秒检查一次
    current_time = time.time()
    if current_time - st.session_state.last_health_check > 60:
        check_api_health()
        st.session_state.last_health_check = current_time
    
    # 先检查API健康状态
    api_ready = st.session_state.api_status in ["在线", "在线 (无健康检查端点)"]
    
    # 加载会话列表（如果尚未加载）
    if not st.session_state.conversations and api_ready:
        load_conversations()
    
    # 左侧导航栏
    with st.sidebar:
        st.title("RAG对话助手")
        
        # 显示API状态
        st.markdown(f"API状态: **{st.session_state.api_status}**")
        
        # 新建会话按钮
        if st.button("➕ 新建会话", key="new_chat_btn", use_container_width=True):
            if api_ready:
                new_chat()
            else:
                st.error("API服务器未就绪，无法创建会话")
        
        # 刷新文档按钮
        if st.button("🔄 刷新文档", key="refresh_docs", use_container_width=True):
            if api_ready:
                refresh_documents_action()
            else:
                st.error("API服务器未就绪，无法刷新文档")
                
        # 刷新连接按钮
        if st.button("🔌 刷新连接", key="refresh_connection", use_container_width=True):
            if check_api_health():
                st.success("API连接正常")
                load_conversations()
            else:
                st.error("API服务器未就绪，请检查服务器状态")
        
        # 搜索框
        st.text_input("🔍 搜索会话", key="search_query", 
                      value=st.session_state.search_query,
                      on_change=lambda: None)
        
        # 会话列表
        st.subheader("会话列表")
        
        filtered_conversations = filter_conversations()
        
        if not filtered_conversations:
            st.info("没有找到会话，点击'新建会话'开始对话")
        
        for conv in filtered_conversations:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # 会话项，点击选择会话
                if st.button(
                    conv.get("title", "未命名会话"), 
                    key=f"conv_{conv['id']}", 
                    use_container_width=True,
                    help="点击加载此会话"
                ):
                    select_conversation(conv["id"])
            
            with col2:
                # 删除按钮
                if st.button("🗑️", key=f"delete_{conv['id']}", help="删除此会话"):
                    remove_conversation(conv["id"])

    # 主聊天区域
    if st.session_state.current_conversation:
        conv_id = st.session_state.current_conversation
        
        # 找到当前会话
        current_conv = next((c for c in st.session_state.conversations if c["id"] == conv_id), None)
        
        if current_conv:
            # 1. 会话标题
            st.markdown(f"## {current_conv.get('title', '新会话')}")
            
            # 标题编辑区
            new_title = st.text_input(
                "编辑会话标题", 
                value=current_conv.get("title", "新会话"), 
                key=f"title_input_{conv_id}"
            )
            if new_title != current_conv.get("title", ""):
                edit_title(conv_id, new_title)
            
            # 2. 消息区域
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if conv_id in st.session_state.conversation_messages:
                messages = st.session_state.conversation_messages[conv_id]
                
                if not messages:
                    st.info("发送消息开始对话")
                
                for i, msg in enumerate(messages):
                    if msg["role"] == "user":
                        # 用户消息
                        st.markdown(f'''
                        <div class="user-message">
                            <div class="sender-label">您</div>
                            <div>{msg["content"]}</div>
                        </div>
                        <div class="clear-float"></div>
                        ''', unsafe_allow_html=True)
                    else:
                        # AI消息
                        stats_html = f'''<div class="stats-info">{msg.get("response_time", "")}</div>''' if "response_time" in msg else ""
                        st.markdown(f'''
                        <div class="ai-message">
                            <div class="sender-label">AI</div>
                            <div>{msg["content"]}</div>
                            {stats_html}
                        </div>
                        <div class="clear-float"></div>
                        ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 3. 固定在底部的输入区域 - 使用st.markdown创建容器
            st.markdown('<div id="fixed-input-container">', unsafe_allow_html=True)
            
            # 使用列布局
            cols = st.columns([5, 1])
            with cols[0]:
                # 输入框
                user_input = st.text_area(
                    "输入您的问题", 
                    key="user_input", 
                    height=80,
                    placeholder="在这里输入您的问题...（按Shift+Enter发送）",
                    value=st.session_state.user_input_value,
                    on_change=process_input
                )
            
            with cols[1]:
                # 按钮区
                st.write("")  # 空行用于垂直居中
                send_btn = st.button("发送", key="send_btn", use_container_width=True, on_click=process_input)
                clear_btn = st.button("清空", key="clear_btn", use_container_width=True, on_click=clear_input)
                
            # 关闭固定容器
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 无选中会话时显示欢迎信息
        st.markdown("## 欢迎使用RAG对话助手")
        st.markdown("从左侧边栏选择已有会话或创建新会话以开始")

# 运行应用
if __name__ == "__main__":
    main() 