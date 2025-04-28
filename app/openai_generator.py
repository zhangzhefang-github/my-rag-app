from openai import OpenAI
import os

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
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
        
        if not api_key:
            print("警告: 未提供API密钥，请设置OPENAI_API_KEY环境变量或在.env文件中配置")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print(f"OpenAI生成器初始化完成，使用模型: {self.model}")
        print(f"API基础URL: {base_url}")

    def generate(self, prompt, max_length=None):
        """
        使用OpenAI API生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大长度（此参数在OpenAI API中不直接支持，但保留参数以兼容原接口）
        
        Returns:
            生成的文本
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                # 如果提供了max_length，则设置max_tokens
                max_tokens=max_length if max_length else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用失败: {str(e)}")
            return f"生成失败: {str(e)}" 