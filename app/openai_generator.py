from openai import OpenAI
import os
import asyncio
from typing import AsyncGenerator, Union, Optional, Dict, Any
import logging
import requests
import json
import time
import aiohttp

class OpenAIGenerator:
    def __init__(self, 
                 model=None,
                 api_key=None,
                 base_url=None):
        """
        初始化OpenAI生成器
        
        Args:
            model: 使用的模型名称，如未提供则使用环境变量OPENAI_MODEL或默认值
            api_key: OpenAI API密钥，如未提供则使用环境变量OPENAI_API_KEY
            base_url: API基础URL，如未提供则使用环境变量OPENAI_BASE_URL或默认值
        """
        # 使用参数或环境变量
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
        
        if not self.api_key:
            logging.warning("未提供API密钥，请设置OPENAI_API_KEY环境变量或在.env文件中配置")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logging.info(f"OpenAI生成器初始化完成，使用模型: {self.model}")
        logging.info(f"API基础URL: {self.base_url}")

    async def generate(self, prompt: str, max_length: Union[int, None] = None, stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        使用OpenAI API生成文本，支持常规和流式响应。
        
        Args:
            prompt: 输入提示。
            max_length: 生成的最大令牌数（可选）。
            stream: 如果为True，返回一个异步生成器，产生令牌。否则，返回完整文本。
        
        Returns:
            如果stream为False：生成的文本作为字符串。
            如果stream为True：产生文本块（令牌）的异步生成器。
        """
        logging.info(f"开始生成响应，流式模式={stream}，最大长度={max_length}")
        logging.debug(f"提示词前50个字符: {prompt[:50]}...")
        
        try:
            if stream:
                logging.info("使用流式生成模式")
                # 不要await异步生成器，直接返回它
                return self._generate_streaming(prompt, max_tokens=max_length)
            else:
                logging.info("使用标准生成模式")
                # 使用OpenAI客户端的常规API调用
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=max_length
                )
                content = response.choices[0].message.content
                logging.info(f"成功生成响应，长度: {len(content)}")
                return content

        except Exception as e:
            error_msg = f"OpenAI API调用失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            
            # 返回错误消息
            if stream:
                async def error_generator():
                    yield error_msg
                return error_generator()
            else:
                return error_msg

    async def _generate_streaming(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """
        使用直接HTTP请求从OpenAI API流式传输响应，以获得更好的控制。
        
        Args:
            prompt: 发送到OpenAI API的提示。
            temperature: 控制输出的随机性。
            max_tokens: 生成的最大令牌数（可选）。
            
        Yields:
            来自API响应流的文本块。
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True
        }
        
        # 如果提供了max_tokens，添加到请求中
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        logging.debug(f"设置流式请求到 {url}")
        
        # 检测是否使用fe8.cn API（可能需要特殊处理）
        is_api_fe8 = "fe8.cn" in self.base_url
        if is_api_fe8:
            logging.info("检测到fe8.cn API，将使用特殊处理逻辑")
            
            # 对于fe8.cn API，我们直接使用非流式请求然后模拟流式返回
            try:
                modified_data = data.copy()
                modified_data["stream"] = False  # 关闭流式
                
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    logging.info("对fe8.cn API使用非流式请求，然后模拟流式返回")
                    
                    async with session.post(url, headers=headers, json=modified_data, timeout=60) as response:
                        elapsed = time.time() - start_time
                        logging.info(f"HTTP请求: POST {url} \"{response.status} {response.reason}\" (耗时: {elapsed:.2f}秒)")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            logging.error(f"API错误: {response.status} - {error_text}")
                            yield f"错误: API返回状态码 {response.status}"
                            return
                        
                        # 读取完整响应
                        response_json = await response.json()
                        logging.debug(f"收到非流式响应: {json.dumps(response_json)[:200]}...")
                        
                        # 提取内容
                        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        if not content:
                            logging.error("无法从响应中提取内容")
                            yield "无法从API响应中提取内容"
                            return
                        
                        # 模拟流式返回，每次返回一小块内容
                        logging.info(f"开始模拟流式返回，总内容长度: {len(content)}")
                        
                        # 将内容分成小块（默认每块10个字符）
                        chunk_size = 25
                        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                        
                        for i, chunk in enumerate(chunks):
                            logging.debug(f"模拟流式块 #{i+1}/{len(chunks)}: '{chunk}'")
                            yield chunk
                            # 添加一个小延迟使其看起来更像流式
                            await asyncio.sleep(0.05)
                        
                        logging.info(f"完成模拟流式返回，共 {len(chunks)} 个块")
                        return
                        
            except Exception as e:
                logging.error(f"fe8.cn API请求失败: {str(e)}", exc_info=True)
                yield f"API请求失败: {str(e)}"
                return
        
        # 对于非fe8.cn API或上面的方法失败，尝试标准流式处理
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                logging.debug("开始HTTP流式请求 (aiohttp)")
                
                async with session.post(url, headers=headers, json=data, timeout=60) as response:
                    elapsed = time.time() - start_time
                    logging.info(f"HTTP请求: POST {url} \"{response.status} {response.reason}\" (耗时: {elapsed:.2f}秒)")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"流式API错误: {response.status} - {error_text}")
                        yield f"错误: API返回状态码 {response.status}"
                        return
                        
                    # 处理流式响应
                    chunk_count = 0
                    current_content = ""
                    buffer = ""
                    
                    try:
                        # 读取和处理响应内容
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if not line:
                                continue
                                
                            logging.debug(f"原始流行: '{line}'")
                            
                            # 处理不同的响应格式
                            if line.startswith('data:'):
                                data_line = line[5:].strip()
                                
                                # 处理[DONE]标记
                                if data_line == "[DONE]":
                                    logging.debug("收到[DONE]标记")
                                    break
                                    
                                # 尝试解析JSON
                                if data_line:
                                    try:
                                        json_data = json.loads(data_line)
                                        
                                        # 提取内容增量
                                        delta = json_data.get('choices', [{}])[0].get('delta', {})
                                        content_delta = delta.get('content', '')
                                        
                                        if content_delta:
                                            current_content += content_delta
                                            chunk_count += 1
                                            logging.debug(f"产生内容块 #{chunk_count}: '{content_delta}'")
                                            yield content_delta
                                    except json.JSONDecodeError as json_err:
                                        logging.warning(f"流中的JSON解析错误: {json_err} - 行: '{data_line}'")
                                        # 累积到缓冲区
                                        buffer += data_line
                            elif line and not line.startswith(':'):  # 跳过空行和注释行
                                # 累积到缓冲区
                                buffer += line
                        
                        logging.info(f"完成OpenAI流迭代，共 {chunk_count} 个块。")
                        
                        # 如果我们没有收到块但没有错误，尝试备用方法
                        if chunk_count == 0:
                            logging.warning("尽管响应成功，但未收到内容块。尝试备用方法。")
                            backup_result = await self._fallback_generation(prompt)
                            yield backup_result
                            
                    except Exception as stream_error:
                        logging.error(f"流处理过程中出错: {stream_error}", exc_info=True)
                        yield f"\n流处理错误: {str(stream_error)}"
                        # 尝试备用方法
                        backup_result = await self._fallback_generation(prompt)
                        yield backup_result
        
        except aiohttp.ClientError as aio_err:
            logging.error(f"aiohttp请求错误: {str(aio_err)}", exc_info=True)
            
            # 尝试备用方法
            yield "连接错误，尝试备用方法..."
            backup_result = await self._fallback_generation(prompt)
            yield backup_result
        
        except Exception as e:
            logging.error(f"流式请求中的致命错误: {str(e)}", exc_info=True)
            yield f"请求中的致命错误: {str(e)}"
            # 尝试备用方法
            backup_result = await self._fallback_generation(prompt)
            yield backup_result

    async def _fallback_generation(self, prompt: str) -> str:
        """当流式生成失败时的备用方法"""
        logging.info("使用备用生成方法")
        try:
            # 使用同步客户端进行非流式请求
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                stream=False  # 确保不使用流式
            )
            content = response.choices[0].message.content
            logging.info(f"备用生成成功，响应长度: {len(content)}")
            return content
        except Exception as e:
            logging.error(f"备用生成失败: {str(e)}", exc_info=True)
            return f"生成失败: {str(e)}" 