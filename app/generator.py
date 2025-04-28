from transformers import pipeline
import torch

class Generator:
    def __init__(self, model_name='gpt2', use_gpu=True):
        # 自动检测是否有可用的GPU
        device = -1  # 默认使用CPU
        
        if use_gpu and torch.cuda.is_available():
            # 获取GPU显存信息
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB单位
            print(f"GPU总显存: {free_mem:.2f}GB")
            
            # 如果显存较小，使用更小的模型
            if free_mem < 3 and model_name == 'gpt2':
                print("检测到低显存GPU，建议使用distilgpt2模型")
            
            device = 0  # 使用GPU
        
        self.generator = pipeline('text-generation', model=model_name, device=device)
        
        # 如果使用GPU，打印信息
        if device == 0:
            print(f"Generator使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Generator使用CPU")

    def generate(self, prompt, max_length=100):
        result = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['text'] 