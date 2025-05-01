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

# --- Logger Setup ---
# Configure logging (optional, basic config shown)
# logging.basicConfig(level=logging.INFO) # You can adjust level
logger = logging.getLogger(__name__) # <<< Get logger instance

# --- Configuration ---
# 使用 session_state 确保环境变量只加载一次
if 'env_loaded' not in st.session_state or not st.session_state.env_loaded:
    load_env_config()
    st.session_state.env_loaded = True
    # 可以在这里加一个日志，只打印一次
    logger.info("已加载环境变量 (首次加载)。") 

API_HOST = os.environ.get("API_HOST", "localhost") # Allow overriding host
# --- 读取 APP_PORT --- 
# Get port directly from environment or use default
API_PORT = int(os.getenv("APP_PORT", 8000))
# --- 结束读取 --- 

API_STREAM_URL = f"http://{API_HOST}:{API_PORT}/query/stream"
API_QUERY_URL = f"http://{API_HOST}:{API_PORT}/query" # For fetching sources later if needed
API_CONVERSATIONS_URL = f"http://{API_HOST}:{API_PORT}/conversations"

# --- Frontend Parser Function (copied from api.py) ---
def parse_llm_output_frontend(raw_output: str) -> tuple[str | None, str]:
    """Parses raw LLM output, extracting <think> block and main answer.
    
    Args:
        raw_output: The raw string output from the LLM (potentially a chunk).
        
    Returns:
        A tuple containing: (think_content, answer_content)
        think_content is None if no <think> block is found.
        answer_content is the part of the raw_output outside the think block.
    """
    think_content = None
    answer_content = raw_output # Default to the full output
    
    # Use regex to find and extract <think> block (non-greedy)
    match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
    if match:
        think_content = match.group(1).strip() # Get content inside tags
        # Remove the think block and surrounding whitespace from the answer
        answer_content = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        # Optional: remove leading newline if present after removal
        answer_content = re.sub(r"^\s*\n", "", answer_content) 
        
    return think_content, answer_content
# --- End Parser Function ---

# --- API Client Functions ---
def get_backend_error_message(response: requests.Response) -> str:
    """Extracts error message from backend response."""
    try:
        detail = response.json().get("detail", "Unknown error")
        return f"Backend Error ({response.status_code}): {detail}"
    except json.JSONDecodeError:
        return f"Backend Error ({response.status_code}): {response.text}"

def get_conversations():
    """Fetch the list of conversations from the backend."""
    try:
        response = requests.get(API_CONVERSATIONS_URL, timeout=10)
        response.raise_for_status()
        return response.json().get("conversations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching conversations: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching conversations: {e}")
        return []

def create_conversation(title: str = "New Conversation"):
    """Create a new conversation on the backend."""
    try:
        payload = {"title": title}
        response = requests.post(API_CONVERSATIONS_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json() # Returns the created conversation object
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating conversation: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while creating conversation: {e}")
        return None

def get_messages(conversation_id: str):
    """Fetch messages for a specific conversation."""
    try:
        url = f"{API_CONVERSATIONS_URL}/{conversation_id}/messages"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching messages for conversation {conversation_id}: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching messages: {e}")
        return []

def delete_conversation(conversation_id: str):
    """Delete a conversation on the backend."""
    try:
        url = f"{API_CONVERSATIONS_URL}/{conversation_id}"
        response = requests.delete(url, timeout=10)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        # Check if the response indicates success (usually 200 OK or 204 No Content)
        # Some APIs return a body on DELETE, some don't. Status code is reliable.
        logger.info(f"对话 {conversation_id} 删除成功，状态码: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        error_msg = get_backend_error_message(e.response) if e.response else str(e)
        st.error(f"删除对话 {conversation_id} 失败: {error_msg}")
        logger.error(f"删除对话 {conversation_id} 时出错: {error_msg}")
        return False
    except Exception as e:
        st.error(f"删除对话 {conversation_id} 时发生意外错误: {e}")
        logger.error(f"删除对话 {conversation_id} 时发生意外错误: {e}")
        return False

# Note: Sending message needs careful adaptation of the streaming logic later
# Placeholder for now
def send_message_in_conversation(conversation_id: str, message_content: str):
    """Sends a message within a specific conversation and handles the stream."""
    # This function will replace the direct call to API_STREAM_URL
    # It needs to call POST /conversations/{conversation_id}/messages
    # and handle the streaming response similar to how it was done before.
    # We will implement this in the next step.
    pass

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="智源对话",
    layout="wide",
    initial_sidebar_state="auto" 
)

# --- Custom CSS Injection for Selected Conversation ---
# Remove the st.markdown CSS injection below
# st.markdown('''...''', unsafe_allow_html=True)

# --- Session State Initialization (Moved Up) ---
# Initialize state variables early, before accessing them in UI elements.
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None 
if "conversation_list" not in st.session_state:
    # Fetch initial list only once here
    st.session_state.conversation_list = get_conversations() 
if "pending_delete_id" not in st.session_state:
    st.session_state.pending_delete_id = None
if "show_uploader" not in st.session_state: # <<< Initialize show_uploader state
    st.session_state.show_uploader = False
if "think_displayed" not in st.session_state: # <<< Add flag for think block
    st.session_state.think_displayed = False
# <<< Add state for parsing think block >>>
if "parsing_think_block" not in st.session_state:
    st.session_state.parsing_think_block = False
if "current_think_content" not in st.session_state:
    st.session_state.current_think_content = ""

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
st.sidebar.write(f"- API Host: {API_HOST}")
st.sidebar.write(f"- API Port: {API_PORT}")
# 未来可以在这里添加更多设置，例如 top_k 滑块
# top_k_slider = st.sidebar.slider("检索文档数 (top_k)", 1, 10, 3)

# --- Main Page --- 
st.title("💬 智源对话")
st.caption("采用检索增强生成 (RAG) 架构：基于 FastAPI 构建，集成 Sentence Transformers 与 FAISS 实现高效语义检索，由大语言模型提供支持。")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果消息包含响应时间信息，显示它
        if "response_time" in message:
            st.caption(f"响应时长: {message['response_time']}")

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
                upload_url = f"http://{API_HOST}:{API_PORT}/upload-documents" 
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
            
    # 2. Add user message to chat history (both session state and UI)
    user_message = {"role": "user", "content": query} # Backend format might differ slightly
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(query)

    # 3. Send message and handle streaming response
    with st.chat_message("assistant"):
        status = st.status("Assistant is thinking...", expanded=False)
        answer_placeholder = st.empty()
        time_info = st.empty()
        full_answer = ""
        # Reset states for the new response stream
        st.session_state.think_displayed = False 
        st.session_state.parsing_think_block = False 
        st.session_state.current_think_content = "" 
        error_occurred = False
        start_time = time.time()
        first_token_time = None
        token_count = 0
        start_datetime = datetime.now().strftime("%H:%M:%S")

        try:
            status.update(label=f"向对话 {current_cid[:8]} 发送消息...", state="running")
            
            message_payload = {
                "conversation_id": current_cid,
                "content": query, 
                "role": "user"
            } 
            message_stream_url = f"{API_CONVERSATIONS_URL}/{current_cid}/messages"
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            
            with requests.post(message_stream_url, json=message_payload, headers=headers, stream=True, timeout=180) as response:
                response.raise_for_status()
                status.update(label="接收回复...", state="running")
                
                for line in response.iter_lines(decode_unicode=True):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    if line.startswith("event: error"):
                        try:
                            error_data_str = line.split("data: ", 1)[1] # Get data part after "data: "
                            error_data = json.loads(error_data_str)
                            detail = error_data.get('detail', '未知流错误')
                            # Display error in the answer placeholder temporarily
                            answer_placeholder.error(f"流处理错误: {detail}")
                            logger.error(f"SSE Error Event: {detail}")
                        except (IndexError, json.JSONDecodeError) as e:
                             error_msg = f"流处理错误，无法解析: {line}"
                             answer_placeholder.error(error_msg)
                             logger.error(error_msg + f" | Exception: {e}")
                        error_occurred = True
                        status.update(label="流处理出错", state="error", expanded=True)
                        # Don't break here yet, wait for potential 'end' event or stream close
                        
                    elif line.startswith("event: end"):
                        logger.info(f"SSE End Event received: {line}")
                        status.update(label="流处理完成.", state="complete", expanded=False)
                        break # Simply break the loop on end event

                    elif line.startswith("data:"):
                        try:
                            data_str = line.split("data: ", 1)[1]
                            data = json.loads(data_str)
                            raw_token = data.get("token", "") 
                            
                            if raw_token:
                                token_count += 1 
                                if first_token_time is None:
                                    first_token_time = current_time
                                
                                processed_chunk_for_answer = ""
                                remaining_token_part = raw_token
                                
                                while remaining_token_part:
                                    if not st.session_state.parsing_think_block:
                                        start_tag_pos = remaining_token_part.find("<think>")
                                        if start_tag_pos == -1:
                                            processed_chunk_for_answer += remaining_token_part
                                            remaining_token_part = "" 
                                        else:
                                            processed_chunk_for_answer += remaining_token_part[:start_tag_pos]
                                            # <<< Log state change >>>
                                            logger.debug("Entering think block parsing state.")
                                            st.session_state.parsing_think_block = True
                                            st.session_state.current_think_content = "" 
                                            remaining_token_part = remaining_token_part[start_tag_pos + len("<think>"):]
                                    else: # Inside think block
                                        end_tag_pos = remaining_token_part.find("</think>")
                                        if end_tag_pos == -1:
                                            st.session_state.current_think_content += remaining_token_part
                                            # <<< Log accumulation >>>
                                            logger.debug(f"Accumulated think content chunk: {repr(remaining_token_part)}")
                                            remaining_token_part = "" 
                                        else:
                                            think_chunk_before_end = remaining_token_part[:end_tag_pos]
                                            st.session_state.current_think_content += think_chunk_before_end
                                            # <<< Log end found and content before display >>>
                                            logger.debug(f"Found </think>. Final accumulated content: {repr(st.session_state.current_think_content)}")
                                            
                                            # Display expander directly (only once)
                                            if not st.session_state.think_displayed:
                                                logger.debug("Attempting to display expander directly.")
                                                try:
                                                    # Render expander directly in the chat message area
                                                    with st.expander("思考过程", expanded=False):
                                                        st.markdown(f'<div style="color: gray; font-size: 0.8em;">{st.session_state.current_think_content}</div>', unsafe_allow_html=True)
                                                    st.session_state.think_displayed = True
                                                    logger.debug("Expander displayed directly and think_displayed set to True.")
                                                except Exception as display_e:
                                                    logger.error(f"Error displaying expander: {display_e}", exc_info=True)
                                            else:
                                                 logger.debug("Expander already displayed, skipping.")
                                                 
                                            st.session_state.parsing_think_block = False # Exit think state
                                            remaining_token_part = remaining_token_part[end_tag_pos + len("</think>"):] 
                                            # <<< Log state change and remaining part >>>
                                            logger.debug(f"Exiting think block parsing state. Remaining token part: {repr(remaining_token_part)}")
                                
                                # Append the processed answer part 
                                if processed_chunk_for_answer:
                                    full_answer += processed_chunk_for_answer
                                    # Only update the answer placeholder here
                                    answer_placeholder.markdown(full_answer + "▌") 
                                
                        except (IndexError, json.JSONDecodeError) as e:
                            logger.warning(f"无法解析流数据: {line}. 错误: {e}")
                            continue 
                            
                # --- End of loop ---
                # Final check: If stream ends while still parsing think block (error?)
                if st.session_state.parsing_think_block and not st.session_state.think_displayed:
                     logger.warning("Stream ended while inside a think block without closing tag.")
                     # Optionally display incomplete think block?
                     # with think_container.container():
                     #    with st.expander("思考过程 (未结束)", expanded=True):
                     #        st.markdown(f'<div style="color: orange; font-size: 0.9em;">{st.session_state.current_think_content}</div>', unsafe_allow_html=True)
                     pass # Decide how to handle this

            # Calculate final timings
            end_time = time.time()
            total_elapsed = end_time - start_time
            tokens_per_second = token_count / total_elapsed if total_elapsed > 0 and token_count > 0 else 0
            first_token_latency = (first_token_time - start_time) if first_token_time else total_elapsed 
            
            if first_token_time:
                 time_info_text = (
                     f"⏱️ 总耗时: {total_elapsed:.2f}秒 | "
                     f"首token延迟: {first_token_latency:.2f}秒 | "
                     f"速度: {tokens_per_second:.1f} token/秒 ({token_count} tokens) | " # Added token count
                     f"开始于: {start_datetime}"
                 )
            else:
                 # Case where no tokens were received (maybe only think or error)
                 time_info_text = f"⏱️ 总耗时: {total_elapsed:.2f}秒 (未收到答案 token)"

            # Final UI update and state saving
            if not error_occurred:
                if full_answer:
                    answer_placeholder.markdown(full_answer) # Final answer to placeholder
                    time_info.caption(time_info_text) 
                    # Append final answer to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_answer, # Saved answer is already clean
                        "response_time": time_info_text 
                    })
                    # Fetch updated list, but don't rerun immediately
                    st.session_state.conversation_list = get_conversations() 
                elif st.session_state.think_displayed: # If only think was displayed
                    answer_placeholder.info("模型进行了思考，但未生成最终回复。") # Use placeholder
                    time_info.caption(time_info_text)
                else: # No error, no think, no answer
                    answer_placeholder.warning("收到空回复。") # Use placeholder
                    time_info.caption(time_info_text)

            # If error occurred, the message should already be in answer_placeholder
            # Do not append error message to history automatically unless desired

        except requests.exceptions.RequestException as e:
            error_msg = f"连接错误: 无法连接到 API ({message_stream_url}). 详情: {e}"
            st.error(error_msg)
            answer_placeholder.empty()
            status.update(label="连接失败.", state="error", expanded=True)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"发生意外错误: {e}"
            st.error(error_msg)
            answer_placeholder.empty()
            status.update(label="处理错误.", state="error", expanded=True)
            logger.error(error_msg, exc_info=True) # Log stack trace for unexpected errors
        
    # --- End: Modified Input Handling ---
    


