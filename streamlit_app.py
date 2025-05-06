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

# Get user input
# user_query = st.chat_input("向 智源对话 提问...") # <-- REMOVE THIS

# if user_query: # <-- REMOVE THIS BLOCK
    # logger.info(f"User query: '{user_query}'")
    # # Add user message to chat history and display it
    # st.session_state.messages.append({"role": "user", "content": user_query})
    # with st.chat_message("user"):
    #     st.markdown(user_query)
    #
    # # Prepare API request data
    # request_data = {
    #     "query": user_query,
    #     "stream": True,
    #     "top_k": None
    # }
    #
    # # Display assistant response placeholder
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     full_response = ""
    #     st.session_state.current_sources = None # Reset sources for the new query
    #     sources_container = st.expander("查看来源 (原始数据)", expanded=False) # Pre-create expander
    #     sources_placeholder = sources_container.empty() # Placeholder within expander
    #
    #     try:
    #         if True:
    #             logger.debug(f"Sending streaming request to {get_api_url('/query')}")
    #             stream_response = requests.post(get_api_url('/query'), json=request_data, stream=True)
    #             stream_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    #
    #             # Process Server-Sent Events (SSE)
    #             for line in stream_response.iter_lines(decode_unicode=True):
    #                 if line.startswith("data:"):
    #                     try:
    #                         message_data = json.loads(line[len("data:"):])
    #                         message_type = message_data.get("type")
    #                         data_content = message_data.get("data")
    #
    #                         #logger.debug(f"Received stream data: type='{message_type}', data='{str(data_content)[:100]}...'")
    #
    #                         if message_type == "chunk": # Assuming LLM token chunks
    #                             full_response += data_content
    #                             message_placeholder.markdown(full_response + "▌")
    #                         elif message_type == "final_answer": # Handle case where pipeline sends a single final answer
    #                             full_response = data_content
    #                             message_placeholder.markdown(full_response)
    #                         elif message_type == "sources":
    #                             # **SPRINT 1**: Store the raw sources list
    #                             st.session_state.current_sources = data_content
    #                             sources_placeholder.json(st.session_state.current_sources)
    #                             logger.info(f"Received sources data (count: {len(data_content) if isinstance(data_content, list) else 'N/A'})")
    #                         elif message_type == "error":
    #                             full_response += f"\n\n**Error:** {data_content}"
    #                             message_placeholder.error(full_response)
    #                             logger.error(f"Stream reported error: {data_content}")
    #                         elif message_type == "debug":
    #                             logger.debug(f"Debug info from stream: {data_content}")
    #                             # Optionally display debug info in a separate area
    #                         # Add handling for other types like 'status' if needed
    #
    #                     except json.JSONDecodeError:
    #                         logger.warning(f"Received non-JSON data line: {line}")
    #                     except Exception as stream_parse_e:
    #                          logger.error(f"Error parsing stream message '{line}': {stream_parse_e}", exc_info=True)
    #                          full_response += "\n\n*(Error parsing stream data)*"
    #                          message_placeholder.warning(full_response)
    #
    #             message_placeholder.markdown(full_response) # Final update without cursor
    #             logger.info("Streaming response processing complete.")
    #
    #         else: # Non-streaming request
    #             logger.debug(f"Sending non-streaming request to {get_api_url('/query')}")
    #             response = requests.post(get_api_url('/query'), json=request_data)
    #             response.raise_for_status()
    #             result = response.json()
    #
    #             # **SPRINT 1**: Expecting {"answer": ..., "sources": [raw_chunks...]}
    #             full_response = result.get("answer", "*No answer received*")
    #             st.session_state.current_sources = result.get("sources", []) # Get raw sources
    #             debug_info = result.get("debug_info")
    #
    #             message_placeholder.markdown(full_response)
    #             if st.session_state.current_sources:
    #                 sources_placeholder.json(st.session_state.current_sources)
    #             if debug_info:
    #                  logger.info(f"Non-streaming debug info: {debug_info}")
    #                  # st.sidebar.json(debug_info) # Optionally display debug info
    #             logger.info("Non-streaming response received and processed.")
    #
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"API request failed: {e}", exc_info=True)
    #         message_placeholder.error(f"Error communicating with backend: {e}")
    #     except Exception as e:
    #          logger.error(f"An unexpected error occurred in Streamlit app: {e}", exc_info=True)
    #          message_placeholder.error(f"An unexpected error occurred: {e}")
    #
    # # Add assistant response (and sources) to chat history
    # assistant_message = {
    #     "role": "assistant",
    #     "content": full_response,
    #     "sources": st.session_state.current_sources # Add sources here
    # }
    # st.session_state.messages.append(assistant_message)

# Optional: Clear history button
st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.update(messages=[], current_sources=None))

# --- Sidebar --- 
st.sidebar.title("导航与设置")

# --- New Chat Button ---
if st.sidebar.button("➕ 新建对话", use_container_width=True):
    st.session_state.current_conversation_id = None
    st.session_state.messages = [] # Clear messages for new chat
    st.rerun() # Rerun the app to reflect the change

st.sidebar.markdown("## 对话历史")

# --- Conversation List --- 
conversations = st.session_state.conversation_list

# Sort conversations by updated_at timestamp (most recent first)
# Assuming backend provides 'created_at' or 'updated_at'
if conversations:
    try:
        conversations = sorted(
            conversations,
            # Use updated_at if available, otherwise created_at, fallback to empty string
            key=lambda x: x.get("updated_at", x.get("created_at", "")) or "",
            reverse=True
        )
        st.session_state.conversation_list = conversations # Update sorted list in state
    except Exception as e:
        # Handle potential sorting errors (e.g., missing keys)
        st.sidebar.warning(f"无法排序对话列表: {e}")

# Display conversation buttons
for i, conv in enumerate(conversations):
    conv_id = conv.get("id")
    conv_title = conv.get("title", "Untitled")
    
    col1, col2 = st.sidebar.columns([0.85, 0.15]) 
    
    is_selected = (st.session_state.current_conversation_id == conv_id)

    # --- Start: Remove background styling and Re-add prefix ---
    # Remove wrapper div logic
    # if is_selected:
    #    st.sidebar.markdown('<div class="selected-conversation-col">', unsafe_allow_html=True)
    
    # Column 1: Conversation Title Button
    with col1:
        # Re-add blue dot prefix logic
        prefix = "🔵 " if is_selected else ""
        display_title = f"{prefix}{conv_title}" # Add prefix back
        
        # Use the display_title. Button has no type.
        if st.button(display_title, key=f"conv_{conv_id}", use_container_width=True):
            if not is_selected:
                st.session_state.current_conversation_id = conv_id
                st.session_state.messages = get_messages(conv_id)
                st.session_state.pending_delete_id = None 
                st.rerun()

    # Remove closing wrapper div logic
    # if is_selected:
    #    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    # --- End: Remove background styling and Re-add prefix ---

    # Column 2: Delete Button
    with col2:
        if st.session_state.pending_delete_id != conv_id:
           if st.button("🗑️", key=f"del_{conv_id}", help=f"删除对话: {conv_title}", use_container_width=True):
                st.session_state.pending_delete_id = conv_id
                st.rerun() 

    # --- Confirmation Controls (Displayed below the item if pending) ---
    if st.session_state.pending_delete_id == conv_id:
        st.sidebar.warning(f"确认删除 \'{conv_title}\'?")
        confirm_col1, confirm_col2 = st.sidebar.columns(2)
        with confirm_col1:
            if st.button("确认", key=f"confirm_del_{conv_id}", use_container_width=True):
                success = delete_conversation(conv_id)
                if success:
                    st.success(f"对话 \'{conv_title}\' 已删除。")
                    # Update state
                    st.session_state.conversation_list = [c for c in st.session_state.conversation_list if c.get("id") != conv_id]
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                    st.session_state.pending_delete_id = None
                    st.rerun() 
                else:
                    # Error message shown by delete_conversation
                    st.session_state.pending_delete_id = None
                    st.rerun()
        with confirm_col2:
            if st.button("取消", key=f"cancel_del_{conv_id}", use_container_width=True):
                st.session_state.pending_delete_id = None
                st.rerun()

st.sidebar.markdown("--- ") # Separator before settings

# --- Remove File Uploader from Sidebar ---
# st.sidebar.markdown("## 知识库管理") 
# with st.sidebar.expander("管理知识库文档", expanded=False): 
#    ...

st.sidebar.caption("当前设置：")
st.sidebar.write(f"- API Host: {API_BASE_URL}")

# --- Main Page --- 
# REMOVED: Duplicate chat history rendering loop
# # --- Display Chat History ---
# # Make sure current_conversation_id is valid before trying to display
# if st.session_state.current_conversation_id and st.session_state.messages:
#     for message in st.session_state.messages:
#         role = message.get("role", "unknown") # Use .get for safety
#         content = message.get("content", "") # Use .get for safety
#         with st.chat_message(role):
#             if role == "user":
#                 st.markdown(content)
#             elif role == "assistant":
#                 # Parse the historical assistant message content before displaying
#                 think_content, answer_content = parse_llm_output_frontend(content)
#                 st.markdown(answer_content) # Display the cleaned content
#                 # Optionally, display saved statistics if available
#                 # response_time_info = message.get("response_time", "")
#                 # if response_time_info:
#                 #     st.caption(f"响应时长: {response_time_info}")
#             else:
#                 # Handle potential unknown roles gracefully
#                 st.markdown(f"*{role}*: {content}")

# --- Uploader Area (Above Chat Input) ---
# Button to toggle the file uploader visibility
upload_col, _ = st.columns([0.1, 0.9]) # Make button column narrower (more left)
with upload_col:
    if st.button("📎", key="toggle_uploader", help="上传文档以添加到知识库"):
        st.session_state.show_uploader = not st.session_state.show_uploader # Toggle visibility
        st.rerun()

# Conditionally display the uploader and its logic
if st.session_state.get("show_uploader", False):
    with st.container(border=True): 
        uploaded_files = st.file_uploader(
            "", # <-- Remove label
            accept_multiple_files=True, 
            type=['txt', 'md', 'pdf', 'docx'], 
            # help="上传 TXT, Markdown, PDF 或 DOCX 文件以添加到知识库。", # <-- Remove help
            key="main_uploader" 
        )

        if uploaded_files:
            # Remove the separately displayed file list for minimalism
            # st.markdown("**已选择文件:**")
            # for file in uploaded_files:
            #     st.write(f"- {file.name} ({file.size} bytes)")
            
            if st.button("处理上传的文件", key="process_upload_main", use_container_width=True):
                # --- Upload Logic (remains the same) --- 
                upload_url = f"http://{API_BASE_URL.split('//')[1]}/upload-documents" 
                files_to_upload = []
                for file in uploaded_files:
                    files_to_upload.append(("files", (file.name, file, file.type)))
                    
                if files_to_upload:
                    with st.status("正在上传和处理文件...", expanded=True) as upload_status:
                        try:
                            upload_status.update(label=f"正在上传 {len(files_to_upload)} 个文件...")
                            response = requests.post(upload_url, files=files_to_upload, timeout=300) 
                            
                            if response.status_code == 200:
                                result = response.json()
                                added_count = result.get("added_count", 0)
                                skipped_count = result.get("skipped_count", 0)
                                errors = result.get("errors", [])
                                upload_status.update(label=f"处理完成！新增 {added_count}, 跳过 {skipped_count} 个文件。", state="complete", expanded=False)
                                if errors:
                                    st.error("处理部分文件时出错:")
                                    for error in errors:
                                        st.error(f"- {error}")
                                # Hide uploader after successful processing
                                st.session_state.show_uploader = False
                                st.rerun()
                            else:
                                error_msg = get_backend_error_message(response)
                                upload_status.update(label=f"上传失败: {error_msg}", state="error")
                                logger.error(f"文件上传失败: {error_msg}")

                        except requests.exceptions.RequestException as e:
                            upload_status.update(label=f"连接错误: {e}", state="error")
                            logger.error(f"文件上传时连接错误: {e}")
                        except Exception as e:
                            upload_status.update(label=f"发生意外错误: {e}", state="error")
                            logger.error(f"文件上传时意外错误: {e}")
                else:
                    st.warning("没有有效的文件可供上传。")
                # --- End Upload Logic ---
        # Add a cancel button maybe?
        if st.button("完成上传", key="close_uploader"):
            st.session_state.show_uploader = False 
            st.rerun() 
        
        # Track if uploader has files
        st.session_state.main_uploader_has_files = True
    
    # Logic to hide uploader if files are cleared (Commented out to fix syntax/indent issues)
    # If no files are currently selected in the uploader widget...
    # else: 
    #     # ... and we previously tracked that files *were* selected...
    #     if st.session_state.get("main_uploader_has_files", False):
    #         # ... it means the user just cleared the selection.
    #         st.session_state.show_uploader = False  # Hide the uploader
    #         st.session_state.main_uploader_has_files = False # Reset the tracker
    #         st.rerun() # Rerun to reflect the hidden state
    pass # Add pass to avoid empty block if needed

# --- User Input Area ---
query = st.chat_input("向 智源对话 提问...") # Updated placeholder text

if query:
    # --- Start: Modified Input Handling ---
    current_cid = st.session_state.get("current_conversation_id")
    
    # 1. Create a new conversation if none exists
    if current_cid is None:
        st.info("创建新对话中...")
        new_conv = create_conversation(title=f"对话: {query[:20]}...") # Use first 20 chars of query as title
        if new_conv and new_conv.get("id"):
            current_cid = new_conv["id"]
            st.session_state.current_conversation_id = current_cid
            # Add the new conversation to the beginning of the list for immediate display
            if "conversation_list" in st.session_state:
                st.session_state.conversation_list.insert(0, new_conv) 
            else:
                 st.session_state.conversation_list = [new_conv]
            st.success(f"新对话已创建: {new_conv.get('title')}")
            # No need to rerun here, will continue to process the message
        else:
            st.error("无法创建新对话，请检查后端连接。")
            st.stop() # Stop processing if conversation creation fails
            
    # 2. Add user message to session state AND RENDER IT IMMEDIATELY
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    # Render the user message immediately after adding it to state
    with st.chat_message("user"):
        st.markdown(query)

    # 3. Send message and handle streaming response
    # Setup placeholders BEFORE the try block
    # Use columns to place status and time_info potentially to the right or below
    status_placeholder = st.empty()
    answer_placeholder = st.empty()
    citations_placeholder = st.empty() 
    time_info_placeholder = st.empty()

    # Initialize response variables
    full_answer = ""
    st.session_state.current_citations = [] # Reset citations for new response
    error_occurred = False
    start_time = time.time()
    first_token_time = None
    token_count = 0
    start_datetime = datetime.now().strftime("%H:%M:%S")

    # REMOVED: Explicit rendering of assistant message container here
    # with st.chat_message("assistant"):
    # Status and placeholders are handled outside this removed block now

    try:
        with status_placeholder.status("Assistant is thinking...", expanded=False) as status:
            logger.info(f"向对话 {current_cid[:8]} 发送消息...")
            
            message_payload = {
                "conversation_id": current_cid,
                "content": query, 
                "role": "user"
            } 
            message_stream_url = get_api_url(f'/conversations/{current_cid}/messages') 
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            
            with requests.post(message_stream_url, json=message_payload, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()
                status.update(label="接收回复...", state="running")
                
                # --- Stream processing loop starts here --- 
                current_event_type = None # Track the event type
                for line in response.iter_lines(decode_unicode=True):
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Process SSE lines (event:, data:, or empty lines)
                    if line.startswith('event:'):
                        current_event_type = line.split('event:', 1)[1].strip()
                        # logger.debug(f"SSE Event Type: {current_event_type}")
                        continue # Move to next line (should be data:)
                    elif line.startswith('data:'):
                        data_str = line.split('data:', 1)[1].strip()
                        if not data_str: # Skip empty data lines
                             continue
                        
                        # Process data based on the tracked event type or default ('message' or 'chunk')
                        event_to_process = current_event_type if current_event_type else 'chunk' # Default to chunk if no event specified
                        
                        try:
                            data = json.loads(data_str) # Assume data is always JSON
                            
                            if event_to_process == 'citations':
                                extracted_citations = None
                                # Try extracting from {"data": [...]} structure first
                                if isinstance(data, dict):
                                     potential_list = data.get('data')
                                     if isinstance(potential_list, list):
                                         extracted_citations = potential_list
                                         logger.debug("Extracted citations using 'data' key.")
                                # Try extracting from {"citations": [...]} structure
                                if extracted_citations is None and isinstance(data, dict):
                                    potential_list_alt = data.get('citations')
                                    if isinstance(potential_list_alt, list):
                                        extracted_citations = potential_list_alt
                                        logger.debug("Extracted citations using 'citations' key.")
                                
                                # If not found or not a list, check if the data itself is the list
                                if extracted_citations is None and isinstance(data, list):
                                     extracted_citations = data
                                     logger.debug("Extracted citations directly from data object.")
                                
                                # If we got a list one way or another
                                if extracted_citations is not None:
                                    st.session_state.current_citations = extracted_citations
                                    logger.info(f"Stored citations in session state: {st.session_state.current_citations}")
                                else:
                                    logger.warning(f"Received citations event, but could not extract a valid list from data: {data}")
                            elif event_to_process == 'error':
                                detail = data.get('detail', data.get('data', '未知流错误')) # Backend might yield {'type': 'error', 'data': ...}
                                answer_placeholder.error(f"流处理错误: {detail}")
                                logger.error(f"SSE Error Event: {detail}")
                                error_occurred = True
                                status.update(label="流处理出错", state="error", expanded=True)
                            elif event_to_process == 'end': # Handle potential end event with data?
                                logger.info(f"SSE End Event received with data: {data}")
                                status.update(label="流处理完成.", state="complete", expanded=False)
                                break
                            elif event_to_process == 'chunk': # Default data processing
                                token = data.get("token", data.get("data")) # Backend might send {'token':...} or {'type':'chunk', 'data':...}
                                if token:
                                    token_count += 1
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    full_answer += token
                                    answer_placeholder.markdown(full_answer + "▌")
                                else:
                                     logger.warning(f"Received chunk/data event, but no 'token' or 'data' key found: {data}")
                            # Handle other event types like 'debug' if needed
                            elif event_to_process == 'debug':
                                 logger.debug(f"Debug info from stream: {data.get('data', data)}")
                                 # Optionally display debug info?
                                 # time_info_placeholder.caption(f"Debug: {data.get('data', data)}")
                            else:
                                logger.warning(f"Received unhandled SSE event type '{event_to_process}' with data: {data}")

                        except json.JSONDecodeError as e:
                            # Handle cases where data might not be JSON (e.g., simple text chunk without event type)
                            # If it wasn't explicitly typed, assume it's a text chunk
                            if event_to_process == 'chunk': 
                                token = data_str # Treat the raw string as the token
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = time.time()
                                full_answer += token
                                answer_placeholder.markdown(full_answer + "▌")
                            else:
                                logger.warning(f"Could not parse JSON for event '{event_to_process}': {line}. Error: {e}")
                        except Exception as parse_e:
                            logger.error(f"Error processing SSE data line: {line}. Error: {parse_e}", exc_info=True)
                            # Maybe display a generic processing error?
                            # answer_placeholder.warning("处理回复时发生错误。")
                            # error_occurred = True # Consider setting error flag

                    elif line.strip() == "": # Empty line separates messages in SSE
                        current_event_type = None # Reset event type after a message
                    else:
                        # Handle lines that don't conform to SSE format? Maybe log?
                        logger.warning(f"Received non-SSE formatted line (ignoring): {line}")
                        
                # --- Stream processing loop ends here --- 

            # --- After loop, before saving state --- 
            end_time = time.time()
            total_elapsed = end_time - start_time
            tokens_per_second = token_count / total_elapsed if total_elapsed > 0 and token_count > 0 else 0
            first_token_latency = (first_token_time - start_time) if first_token_time else total_elapsed 
            
            if first_token_time:
                 time_info_text = (
                     f"⏱️ 总耗时: {total_elapsed:.2f}秒 | "
                     f"首token延迟: {first_token_latency:.2f}秒 | "
                     f"速度: {tokens_per_second:.1f} token/秒 ({token_count} tokens) | "
                     f"开始于: {start_datetime}"
                 )
            else:
                 time_info_text = f"⏱️ 总耗时: {total_elapsed:.2f}秒 (未收到答案 token)"

            # --- ADD DEBUG LOG --- 
            logger.info(f"Checking citations before rendering: {st.session_state.current_citations}")
            # --- END DEBUG LOG ---

            # Render citations using the placeholder if received
            if st.session_state.current_citations and not error_occurred:
                 with citations_placeholder.expander("查看引用", expanded=True):
                     for i, citation in enumerate(st.session_state.current_citations):
                        # Access CitationSourceDetail correctly
                        details = citation.get('source_details', [{}])[0]
                        doc_name = details.get('doc_source_name', '未知来源')
                        text_quote = citation.get('text_quote', '...')
                        chunk_id = details.get('chunk_id', 'N/A')
                        chunk_text = details.get('chunk_text', '无内容')

                        st.markdown(f"**[{i+1}] 引用自:** {doc_name}")
                        st.markdown(f"> {text_quote}")
                        with st.popover("查看完整来源块", use_container_width=True):
                            st.markdown(f"##### 来源: {doc_name} (块 ID: {chunk_id})")
                            st.markdown(f"```\n{chunk_text}\n```")
                        st.markdown("---")

            # Update final UI elements and save assistant message to state
            if not error_occurred:
                if full_answer:
                    answer_placeholder.markdown(full_answer) # Final answer update
                    time_info_placeholder.caption(time_info_text)
                    # Append assistant message to session state HERE
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_answer,
                        "citations": st.session_state.current_citations,
                        "response_time": time_info_text 
                    })
                    st.session_state.conversation_list = get_conversations() # Update list after successful interaction
                # Handle empty answer case if needed (e.g., only citations returned?)
                elif st.session_state.current_citations: # If only citations, maybe show a note?
                    answer_placeholder.info("已找到相关引用信息。")
                    time_info_placeholder.caption(time_info_text)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "", # No text answer
                        "citations": st.session_state.current_citations,
                        "response_time": time_info_text 
                    })
                    st.session_state.conversation_list = get_conversations()
                else: # No answer, no citations
                    answer_placeholder.warning("收到空回复。")
                    time_info_placeholder.caption(time_info_text)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "",
                        "citations": [],
                        "response_time": time_info_text 
                    })
            # Clear the status placeholder at the very end if it still exists
            status_placeholder.empty()

    except requests.exceptions.RequestException as e:
        # ... (keep existing exception handling) ...
        error_msg = f"连接错误: 无法连接到 API ({message_stream_url}). 详情: {e}"
        st.error(error_msg)
        answer_placeholder.empty()
        status_placeholder.empty() # Ensure status is cleared on error
        # Optionally add an error message to session state?
        # st.session_state.messages.append({"role": "assistant", "content": f"错误: {error_msg}"})
        logger.error(error_msg)
    except Exception as e:
        # ... (keep existing exception handling) ...
        error_msg = f"发生意外错误: {e}"
        st.error(error_msg)
        answer_placeholder.empty()
        status_placeholder.empty()
        # st.session_state.messages.append({"role": "assistant", "content": f"错误: {error_msg}"})
        logger.error(error_msg, exc_info=True)
        
    # --- End: Modified Input Handling (No explicit rerun needed here) ---
    



