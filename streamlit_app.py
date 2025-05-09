import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from utils.env_helper import load_env_config # Reuse env loading
import aiohttp
import logging
import re # <<< Add import re
from dotenv import load_dotenv
from typing import Dict, Any, List

# --- Basic Logging Configuration ---
log_level_str = os.environ.get("FRONTEND_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Log to console
logger = logging.getLogger("StreamlitApp")
logger.info("Streamlit application starting/reloading...")
# ---------------------------------

# --- Configuration ---
# 使用 session_state 确保环境变量只加载一次
if 'env_loaded' not in st.session_state or not st.session_state.env_loaded:
    load_env_config()
    st.session_state.env_loaded = True
    # 可以在这里加一个日志，只打印一次
    logger.info("已加载环境变量 (首次加载)。") 

# Load environment variables
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") # Default for local dev

# Setup logging
logging.basicConfig(level=logging.INFO)

# --- Helper Functions --- #
def get_api_url(endpoint: str) -> str:
    """Constructs the full API URL."""
    return f"{API_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"

def check_api_health() -> bool:
    """Checks if the backend API is available."""
    health_url = get_api_url("health")
    logger.info(f"Attempting health check for backend API at: {health_url}")
    try:
        response = requests.get(health_url, timeout=3)
        is_healthy = response.status_code == 200 and response.json().get("status") == "ok"
        if is_healthy:
            logger.info(f"API health check successful for {health_url}.")
        else:
            logger.warning(f"API health check for {health_url} failed with status {response.status_code}. Response: {response.text[:200]}")
        return is_healthy
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed for {health_url}. Error: {e}", exc_info=True)
        return False

# Function to create a new conversation via API
def create_conversation(title: str) -> Dict[str, Any] | None:
    """Calls the backend API to create a new conversation."""
    create_url = get_api_url('/conversations')
    payload = {"title": title}
    logger.info(f"Attempting to create conversation with title: '{title}' at {create_url}")
    try:
        response = requests.post(create_url, json=payload, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        new_conv_data = response.json()
        logger.info(f"Successfully created conversation: {new_conv_data.get('id')}")
        return new_conv_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create conversation at {create_url}. Error: {e}", exc_info=True)
        st.error(f"创建新对话失败: {e}") # Show error to user
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse create conversation response from {create_url}. Status: {response.status_code}, Response: {response.text[:200]}. Error: {e}")
        st.error("创建新对话时收到无效的响应。")
        return None

# Function to get the list of conversations via API
def get_conversations() -> List[Dict[str, Any]]:
    """Calls the backend API to get the list of conversations."""
    list_url = get_api_url('/conversations')
    logger.info(f"Attempting to fetch conversation list from {list_url}")
    try:
        response = requests.get(list_url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses
        response_data = response.json()
        # Extract the list from the 'conversations' key
        conversations = response_data.get('conversations', []) 
        
        # Ensure it's actually a list after extraction
        if isinstance(conversations, list):
             logger.info(f"Successfully fetched {len(conversations)} conversations.")
             return conversations
        else:
             logger.error(f"Extracted 'conversations' key did not contain a list. Type: {type(conversations)}, Response Data: {str(response_data)[:200]}")
             st.error("获取对话列表时收到无效的数据结构。")
             return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch conversations from {list_url}. Error: {e}", exc_info=True)
        # Avoid showing error directly here as it might be called frequently
        # st.error(f"获取对话列表失败: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conversation list response from {list_url}. Status: {response.status_code}, Response: {response.text[:200]}. Error: {e}")
        # st.error("获取对话列表时收到无效的响应。")
        return []

# Function to parse LLM output on the frontend (similar to backend)
def parse_llm_output_frontend(raw_output: str) -> tuple[str | None, str]:
    """Parses raw LLM output string, extracting <think> block and main answer."""
    think_content = None
    answer_content = raw_output

    match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        answer_content = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        answer_content = re.sub(r"^\s*\n", "", answer_content) 

    return think_content, answer_content

# --- Streamlit App UI --- #
st.set_page_config(
    page_title="智源对话",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("💬 智源对话")
st.caption("采用检索增强生成 (RAG) 架构：基于 FastAPI 构建，集成 Sentence Transformers 与 FAISS 实现高效语义检索，由大语言模型提供支持。")

# Check API status
api_available = check_api_health()
if not api_available:
    st.error("🚨 Backend API is not reachable. Please ensure the FastAPI server is running.")
    st.stop()
else:
    st.success("✅ Backend API is connected.")

# Initialize session state for conversation history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = None # Store sources for the last response
# Add initialization for conversation_list
if 'conversation_list' not in st.session_state:
    st.session_state.conversation_list = []
# Add initialization for current_conversation_id
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
# Add initialization for pending_delete_id
if 'pending_delete_id' not in st.session_state:
    st.session_state.pending_delete_id = None
# Add initialization for current_citations (Sprint 2)
if 'current_citations' not in st.session_state:
    st.session_state.current_citations = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif message["role"] == "assistant":
            # Use the frontend parser to clean potential <think> tags
            _, answer_content = parse_llm_output_frontend(message["content"])
            st.markdown(answer_content)
            
            # --- SPRINT 2: Display Citations for historical messages --- 
            if "citations" in message and message["citations"]:
                with st.expander("查看引用", expanded=False):
                    for i, citation in enumerate(message["citations"]):
                        st.markdown(f"**[{i+1}] 引用自:** {citation.get('doc_source_name', '未知来源')}")
                        st.markdown(f"> {citation.get('text_quote', '...')}")
                        # Optionally add a way to see the full chunk_text
                        with st.popover("查看完整来源块", use_container_width=True):
                            st.markdown(f"##### 来源: {citation.get('doc_source_name', 'N/A')} (块 ID: {citation.get('chunk_id', 'N/A')})")
                            st.markdown(f"```\n{citation.get('chunk_text', '无内容')}\n```")
                        st.markdown("---")
            # --- End Sprint 2 Citation Display ---
            
            # --- SPRINT 1: Remove old source display --- 
            # if "sources" in message and message["sources"]:
            #      with st.expander("查看来源 (原始数据)"):
            #          st.json(message["sources"])
            # --- End Sprint 1 Removal ---

# --- User Input Area ---

# Using st.chat_input as per Streamlit 1.45.0 documentation provided by the user.
# It can return None, a string (if accept_file=False), or a dict-like object
# with .text and .files attributes.
prompt = st.chat_input(
    "Say something and/or attach an image",
    accept_file=True,  # Allows file uploads
    file_type=["jpg", "jpeg", "png"], # Specifies allowed file types
)

if prompt:
    user_text = ""
    uploaded_files = []

    # prompt is a dict-like object here because accept_file is True
    # Access .text and .files attributes
    if hasattr(prompt, 'text'):
        user_text = prompt.text if prompt.text is not None else ""
    
    if hasattr(prompt, 'files'):
        uploaded_files = prompt.files if prompt.files is not None else []

    logger.info(f"User submitted: text='{user_text}', files_count={len(uploaded_files)}")

    # Proceed if there's either text or at least one file
    if user_text or uploaded_files:
        query_for_backend = user_text # Use the text part for the backend query for now

        # --- Start: Original Input Handling Logic (adapted) ---
        current_cid = st.session_state.get("current_conversation_id")
        
        if current_cid is None:
            st.info("创建新对话中...")
            new_conv_title = f"对话: {user_text[:20]}..." if user_text else "新对话"
            if uploaded_files:
                new_conv_title += f" (含 {len(uploaded_files)} 个附件)"
            
            new_conv = create_conversation(title=new_conv_title)
            if new_conv and new_conv.get("id"):
                current_cid = new_conv["id"]
                st.session_state.current_conversation_id = current_cid
                if "conversation_list" in st.session_state:
                    st.session_state.conversation_list.insert(0, new_conv) 
                else:
                    st.session_state.conversation_list = [new_conv]
                st.success(f"新对话已创建: {new_conv.get('title')}")
            else:
                st.error("无法创建新对话，请检查后端连接。")
                st.stop() # Stop if conversation creation fails
        
        # Construct user message content for display
        user_message_display_content = user_text
        if uploaded_files:
            if user_text: # Add a separator if there's also text
                user_message_display_content += f"\\n\\n--- (附带 {len(uploaded_files)} 个文件) ---"
            else: # Only files were uploaded
                user_message_display_content = f"(用户上传了 {len(uploaded_files)} 个文件)"

        st.session_state.messages.append({"role": "user", "content": user_message_display_content})
        with st.chat_message("user"):
            # Display text first, then images
            if user_text:
                st.markdown(user_text) # Display the original text part
            
            if uploaded_files:
                if not user_text: # If only files, print the placeholder message
                    st.markdown(user_message_display_content)
                for idx, uploaded_file_item in enumerate(uploaded_files):
                    try:
                        # Display the image using st.image
                        st.image(uploaded_file_item, caption=f"附件 {idx + 1}: {uploaded_file_item.name}", width=250)
                    except Exception as img_e:
                        st.warning(f"无法显示附件 {idx + 1} ({uploaded_file_item.name}): {img_e}")
        
        # --- Backend call logic (if there's text or a policy to process files) ---
        # For now, backend is called primarily if there's text.
        # If only files are uploaded, we've shown them. The RAG backend might not act on file-only submissions
        # unless specifically designed for it.
        
        if query_for_backend or (uploaded_files and not query_for_backend): # Proceed if text OR only files (to show assistant ack for files)
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                citations_placeholder = st.empty()
                time_info_placeholder = st.empty()

                full_answer = ""
                st.session_state.current_citations = [] # Reset citations
                error_occurred = False
                start_time = time.time()
                first_token_time = None
                token_count = 0
                start_datetime = datetime.now().strftime("%H:%M:%S")

                try:
                    # If only files were uploaded and no text, assistant can just acknowledge.
                    if not query_for_backend and uploaded_files:
                        with st.chat_message("assistant"):
                            ack_message = f"已收到您上传的 {len(uploaded_files)} 个文件。"
                            st.markdown(ack_message)
                        st.session_state.messages.append({"role": "assistant", "content": ack_message, "citations": []})
                        # No further backend call needed for simple acknowledgment of file upload without text.
                    
                    elif query_for_backend: # If there is text, call the backend
                        with status_placeholder.status("Assistant is thinking...", expanded=False) as status:
                            logger.info(f"向对话 {current_cid[:8]} 发送消息: '{query_for_backend}'")
                            
                            message_payload = {
                                "conversation_id": current_cid,
                                "content": query_for_backend, 
                                "role": "user"
                            } 
                            message_stream_url = get_api_url(f'/conversations/{current_cid}/messages') 
                            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
                            
                            with requests.post(message_stream_url, json=message_payload, headers=headers, stream=True, timeout=180) as response:
                                response.raise_for_status()
                                status.update(label="接收回复...", state="running")
                                
                                current_event_type = None
                                for line in response.iter_lines(decode_unicode=True):
                                    if line.startswith('event:'):
                                        current_event_type = line.split('event:', 1)[1].strip()
                                        continue
                                    elif line.startswith('data:'):
                                        data_str = line.split('data:', 1)[1].strip()
                                        if not data_str: continue
                                        
                                        event_to_process = current_event_type if current_event_type else 'chunk'
                                        try:
                                            data = json.loads(data_str)
                                            if event_to_process == 'citations':
                                                extracted_citations = data.get('data', data.get('citations'))
                                                if isinstance(extracted_citations, list):
                                                    st.session_state.current_citations = extracted_citations
                                                elif isinstance(data, list):
                                                    st.session_state.current_citations = data
                                                else:
                                                    logger.warning(f"Citations event: failed to extract list from {data}")
                                            elif event_to_process == 'error':
                                                detail = data.get('detail', data.get('data', '未知流错误'))
                                                answer_placeholder.error(f"流处理错误: {detail}")
                                                logger.error(f"SSE Error Event: {detail}")
                                                error_occurred = True
                                                status.update(label="流处理出错", state="error", expanded=True)
                                            elif event_to_process == 'end':
                                                logger.info(f"SSE End Event received with data: {data}")
                                                status.update(label="流处理完成.", state="complete", expanded=False)
                                                break
                                            elif event_to_process == 'chunk':
                                                token = data.get("token", data.get("data"))
                                                if token:
                                                    token_count += 1
                                                    if first_token_time is None: first_token_time = time.time()
                                                    full_answer += token
                                                    answer_placeholder.markdown(full_answer + "▌")
                                                else:
                                                    logger.warning(f"Chunk/data event, but no token/data: {data}")
                                            elif event_to_process == 'debug':
                                                logger.debug(f"Debug info from stream: {data.get('data', data)}")
                                            else:
                                                logger.warning(f"Received unhandled SSE event '{event_to_process}' with data: {data}")
                                        except json.JSONDecodeError as e:
                                            if event_to_process == 'chunk': 
                                                token = data_str
                                                token_count += 1
                                                if first_token_time is None: first_token_time = time.time()
                                                full_answer += token
                                                answer_placeholder.markdown(full_answer + "▌")
                                            else:
                                                logger.warning(f"Could not parse JSON for event '{event_to_process}': {line}. Error: {e}")
                                        except Exception as parse_e:
                                            logger.error(f"Error processing SSE data line: {line}. Error: {parse_e}", exc_info=True)
                                    elif line.strip() == "":
                                        current_event_type = None
                                    else:
                                        logger.warning(f"Received non-SSE formatted line (ignoring): {line}")
                        
                        end_time = time.time()
                        total_elapsed = end_time - start_time
                        tokens_per_second = token_count / total_elapsed if total_elapsed > 0 and token_count > 0 else 0
                        first_token_latency = (first_token_time - start_time) if first_token_time else total_elapsed
                        
                        time_info_text = (
                            f"⏱️ 总耗时: {total_elapsed:.2f}秒 | 首token: {first_token_latency:.2f}秒 | "
                            f"速度: {tokens_per_second:.1f} t/s ({token_count} t) | 开始于: {start_datetime}"
                        ) if first_token_time else f"⏱️ 总耗时: {total_elapsed:.2f}秒 (无答案 token)"

                        if st.session_state.current_citations and not error_occurred:
                            with citations_placeholder.expander("查看引用", expanded=True):
                                for i, citation in enumerate(st.session_state.current_citations):
                                    details = citation.get('source_details', [{}])[0]
                                    doc_name = details.get('doc_source_name', '未知来源')
                                    text_quote = citation.get('text_quote', '...')
                                    chunk_id = details.get('chunk_id', 'N/A')
                                    chunk_text = details.get('chunk_text', '无内容')
                                    st.markdown(f"**[{i+1}] 引用自:** {doc_name}")
                                    st.markdown(f"> {text_quote}")
                                    with st.popover("查看完整来源块", use_container_width=True):
                                        st.markdown(f"##### 来源: {doc_name} (块 ID: {chunk_id})")
                                        st.markdown(f"```\\n{chunk_text}\\n```")
                                    st.markdown("---")
                        
                        if not error_occurred:
                            if full_answer:
                                answer_placeholder.markdown(full_answer) # Final answer display
                                time_info_placeholder.caption(time_info_text)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": full_answer,
                                    "citations": st.session_state.current_citations, "response_time": time_info_text 
                                })
                                st.session_state.conversation_list = get_conversations() # Refresh conv list
                            elif st.session_state.current_citations: # Only citations, no text answer
                                answer_placeholder.info("已找到相关引用信息。")
                                time_info_placeholder.caption(time_info_text)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": "", "citations": st.session_state.current_citations,
                                    "response_time": time_info_text 
                                })
                                st.session_state.conversation_list = get_conversations()
                            else: # No answer, no citations
                                answer_placeholder.warning("收到空回复。")
                                time_info_placeholder.caption(time_info_text)
                                st.session_state.messages.append({
                                    "role": "assistant", "content": "", "citations": [], "response_time": time_info_text 
                                })
                        status_placeholder.empty()

                except requests.exceptions.HTTPError as e:
                    with st.chat_message("assistant"):
                        error_msg = f"API 错误 (状态 {e.response.status_code}): {e.response.text[:300]}"
                        st.error(error_msg)
                    logger.error(f"{error_msg} from URL: {e.request.url}", exc_info=True)
                    st.session_state.messages.append({"role": "assistant", "content": f"错误: {error_msg}"})
                except requests.exceptions.RequestException as e:
                    with st.chat_message("assistant"):
                        error_msg = f"连接错误: 无法连接到 API ({message_stream_url}). 详情: {e}"
                        st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                    st.session_state.messages.append({"role": "assistant", "content": f"错误: {error_msg}"})
                except Exception as e: # General exception
                    with st.chat_message("assistant"):
                        error_msg = f"发生意外错误: {e}"
                        st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
                    st.session_state.messages.append({"role": "assistant", "content": f"错误: {error_msg}"})
                finally:
                    logger.debug("Input processing block finished.")
                    # Ensure correct indentation for the conditional rerun logic below.
                    if uploaded_files and not query_for_backend: # If only files were processed (assistant ack)
                        st.rerun() # Ensure UI updates after assistant ack for files.
                    elif query_for_backend: # If text was processed (API call made)
                        # Let Streamlit's default rerun on session_state.messages change handle it.
                        pass # pass is sufficient if no explicit action is needed here.

# Sidebar for conversation management
with st.sidebar:
    st.header("对话管理")

    # Button to create a new conversation
    if st.button("➕ 新建对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.current_sources = None
        st.session_state.current_citations = []
        # Optionally, immediately create on backend or wait for first message:
        # For simplicity, let's clear current_cid and a new one will be made on first message.
        logger.info("New conversation started by user button.")
        st.rerun() # Rerun to clear main chat area and reflect new state

    st.markdown("---")
    st.subheader("历史对话")

    # Fetch and display list of conversations
    # Ensure conversation_list is initialized in session state
    if 'conversation_list' not in st.session_state:
        st.session_state.conversation_list = [] # Initialize if not present
    
    # Attempt to load conversations if the list is empty or forced refresh
    # This might be called frequently, consider if get_conversations() is expensive
    # For now, assume it's acceptable or has its own caching/efficiency.
    if not st.session_state.conversation_list: # Load if empty
        st.session_state.conversation_list = get_conversations()
        if not st.session_state.conversation_list:
            st.caption("暂无历史对话。")
        # No rerun here, display will happen naturally

    if st.session_state.conversation_list:
        # Create a list of conversation titles for display, handling potential None or missing titles
        conv_options = {}
        for conv_data in st.session_state.conversation_list:
            conv_id = conv_data.get("id")
            conv_title = conv_data.get("title", f"对话 {conv_id[:8]}...") if conv_id else "未知对话"
            if conv_id:
                conv_options[conv_id] = conv_title

        # Use a selectbox or radio buttons for switching conversations
        # For a large number of conversations, a selectbox is better.
        # Let's use buttons for now as it's common in chat UIs.
        
        selected_conv_id_sidebar = None
        current_conv_id_main_area = st.session_state.get("current_conversation_id")

        for conv_id_iter, conv_title_iter in conv_options.items():
            # Highlight the selected conversation
            button_type = "primary" if conv_id_iter == current_conv_id_main_area else "secondary"
            if st.button(f"{conv_title_iter}", key=f"conv_btn_{conv_id_iter}", use_container_width=True, type=button_type):
                selected_conv_id_sidebar = conv_id_iter
        
        if selected_conv_id_sidebar and selected_conv_id_sidebar != current_conv_id_main_area:
            logger.info(f"User selected conversation ID: {selected_conv_id_sidebar}")
            st.session_state.current_conversation_id = selected_conv_id_sidebar
            # Fetch messages for the selected conversation
            try:
                conv_messages_url = get_api_url(f'/conversations/{selected_conv_id_sidebar}/messages')
                response = requests.get(conv_messages_url, timeout=10)
                response.raise_for_status()
                messages_data = response.json()
                
                # Backend returns {"messages": [...]} where each message is {"id", "conversation_id", "role", "content", "created_at", "updated_at"}
                # We need to transform this to the format expected by st.session_state.messages: {"role": ..., "content": ...}
                # And potentially handle citations if they are stored with messages on backend.
                
                formatted_messages = []
                raw_messages_from_api = messages_data.get("messages", [])
                for msg in raw_messages_from_api:
                    formatted_msg = {"role": msg.get("role"), "content": msg.get("content")}
                    # If backend stores citations with messages, extract them here.
                    # For now, assuming basic structure.
                    # if msg.get("citations"):
                    #    formatted_msg["citations"] = msg.get("citations") 
                    formatted_messages.append(formatted_msg)
                
                st.session_state.messages = formatted_messages
                st.session_state.current_sources = None # Clear previous sources/citations
                st.session_state.current_citations = []
                logger.info(f"Loaded {len(st.session_state.messages)} messages for conversation {selected_conv_id_sidebar}")
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"加载对话失败: {e}")
                logger.error(f"Failed to load messages for conv {selected_conv_id_sidebar}: {e}", exc_info=True)
            except json.JSONDecodeError as e:
                st.error("加载对话时收到无效响应。")
                logger.error(f"Failed to parse messages for conv {selected_conv_id_sidebar}. Status: {response.status_code if 'response' in locals() else 'N/A'}. Error: {e}")

    else:
        st.caption("无历史对话或无法加载。")

    # Refresh button for conversations
    if st.button("🔄 刷新列表", use_container_width=True):
        st.session_state.conversation_list = get_conversations()
        st.rerun()

    # Placeholder for delete functionality (can be added later)
    st.markdown("---")
    if st.session_state.get("current_conversation_id"):
        current_conv_title = "当前对话"
        for conv in st.session_state.get("conversation_list", []):
            if conv.get("id") == st.session_state.current_conversation_id:
                current_conv_title = conv.get("title", f"对话 {st.session_state.current_conversation_id[:8]}...")
                break
        
        if st.button(f"🗑️ 删除对话: {current_conv_title}", use_container_width=True):
            # Confirmation step would be good here
            conv_to_delete = st.session_state.current_conversation_id
            delete_url = get_api_url(f'/conversations/{conv_to_delete}')
            try:
                response = requests.delete(delete_url, timeout=10)
                response.raise_for_status() # If not 2xx, raises HTTPError
                logger.info(f"Successfully deleted conversation ID: {conv_to_delete}")
                st.session_state.current_conversation_id = None
                st.session_state.messages = []
                st.session_state.conversation_list = get_conversations() # Refresh list
                st.success(f"对话 '{current_conv_title}' 已成功删除。")
                st.rerun()
            except requests.exceptions.HTTPError as e:
                # --- MODIFICATION START (Refined) ---
                cid_to_log = conv_to_delete # Use the correct variable defined in the try block
                title_to_log = current_conv_title # This should also be available from the outer scope

                if e.response.status_code == 404:
                    logger.warning(f"Attempted to delete conversation {cid_to_log}, but it was not found (404). Assuming already deleted.")
                    st.warning(f"对话 '{title_to_log}' 未找到或已被删除。")
                    # Treat as success for UI update
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                    st.session_state.conversation_list = get_conversations() # Directly update the conversation list
                    st.rerun()
                else:
                    logger.error(f"Failed to delete conversation {cid_to_log} due to HTTPError: {e.response.status_code} - {e.response.text[:100]}")
                    st.error(f"删除对话 '{title_to_log}' 失败: 服务器错误 {e.response.status_code}。")
                # --- MODIFICATION END (Refined) ---
            except requests.exceptions.RequestException as e:
                # More specific error message for network/request level issues
                logger.error(f"Network or request error when trying to delete conversation {conv_to_delete}: {e}", exc_info=True)
                st.error(f"删除对话 '{current_conv_title}' 失败: 网络请求错误。")
