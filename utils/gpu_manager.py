"""
GPU资源管理器 - 统一管理GPU检测、内存分配和资源使用
"""

import torch
import logging

logger = logging.getLogger(__name__) # Initialize logger

class GPUManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.gpu_available = torch.cuda.is_available()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else None
        
    def _get_gpu_info(self):
        """获取GPU信息"""
        info = {}
        try:
            device_count = torch.cuda.device_count()
            info['device_count'] = device_count
            info['devices'] = []
            
            for i in range(device_count):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                }
                info['devices'].append(device_info)
                
            info['current_device'] = torch.cuda.current_device()
        except Exception as e:
            info['error'] = str(e)
            
        return info
    
    def should_use_gpu(self, min_memory_gb=1.0, task_name=""):
        """智能决定是否应使用GPU"""
        if not self.gpu_available:
            logging.info(f"{task_name or '任务'} 使用CPU (GPU不可用)")
            return False
            
        # 获取当前设备内存信息
        device_id = self.gpu_info.get('current_device', 0)
        device = self.gpu_info['devices'][device_id]
        total_memory_gb = device['total_memory_gb']
        
        # 内存是否满足要求
        if total_memory_gb < min_memory_gb:
            logging.info(f"{task_name or '任务'} 使用CPU (GPU显存不足: {total_memory_gb:.2f}GB < {min_memory_gb:.2f}GB)")
            return False
            
        logging.info(f"{task_name or '任务'} 使用GPU: {device['name']}，显存: {total_memory_gb:.2f}GB")
        return True
        
    def get_device(self, use_gpu=True, min_memory_gb=1.0, task_name=""):
        """获取任务应使用的设备"""
        if use_gpu and self.should_use_gpu(min_memory_gb, task_name):
            return "cuda"
        return "cpu"
    
    def print_gpu_info(self):
        """打印GPU信息"""
        if not self.gpu_available:
            print("未检测到GPU")
            return
            
        for device in self.gpu_info['devices']:
            print(f"GPU #{device['id']}: {device['name']}, 显存: {device['total_memory_gb']:.2f}GB")
            
    def get_available_gpu_ids(self) -> list[int]:
        """Returns a list of available GPU device IDs."""
        if not self.gpu_available or not self.gpu_info or 'devices' not in self.gpu_info:
            return []
        return [device['id'] for device in self.gpu_info['devices']]

    def get_torch_device_for_model(self, requested_device: str = "auto") -> torch.device:
        """
        Determines the appropriate torch.device for loading a model.

        Args:
            requested_device: The desired device ("cuda", "cpu", "auto").

        Returns:
            A torch.device instance.
        """
        logger.debug(f"Determining torch device. Requested: '{requested_device}', GPU available: {self.gpu_available}")
        if requested_device == "cuda":
            if self.gpu_available:
                logger.info("Requested CUDA and GPU is available. Using CUDA.")
                return torch.device("cuda")
            else:
                logger.warning("Requested CUDA, but GPU is not available. Falling back to CPU.")
                return torch.device("cpu")
        elif requested_device == "cpu":
            logger.info("Requested CPU. Using CPU.")
            return torch.device("cpu")
        elif requested_device == "auto":
            if self.gpu_available:
                # You might add more sophisticated logic here, e.g., checking VRAM
                logger.info("Requested AUTO and GPU is available. Using CUDA.")
                return torch.device("cuda")
            else:
                logger.info("Requested AUTO and GPU is not available. Using CPU.")
                return torch.device("cpu")
        else:
            logger.warning(f"Invalid requested_device: '{requested_device}'. Defaulting to CPU.")
            return torch.device("cpu")

    def __str__(self):
        if not self.gpu_available:
            return "GPU: 不可用"
            
        devices = self.gpu_info['devices']
        if not devices:
            return "GPU: 无可用设备"
            
        return f"GPU: {devices[0]['name']}, 显存: {devices[0]['total_memory_gb']:.2f}GB" 