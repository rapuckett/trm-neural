import torch
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages device selection and tensor movement for TRM components.
    
    Just as a radio telescope array needs a control system to manage its different
    receiver components, our TRM system needs careful management of its computing
    resources across different types of hardware accelerators.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the device manager with optional specific device selection.
        
        Args:
            device: Optional device specification. If None, will auto-select
                   the best available device.
        """
        self.device = self._get_best_device() if device is None else device
        logger.info(f"Initialized DeviceManager using device: {self.device}")
        
    def _get_best_device(self) -> str:
        """
        Determine the best available device for computation.
        
        Returns:
            String identifying the selected device ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            logger.info("CUDA device detected and selected")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS device detected and selected")
            return 'mps'
        else:
            logger.info("No GPU detected, using CPU")
            return 'cpu'
            
    def move_to_device(self, 
                      tensor_or_module: Union[torch.Tensor, torch.nn.Module]
                      ) -> Union[torch.Tensor, torch.nn.Module]:
        """
        Move a tensor or module to the selected device.
        
        Args:
            tensor_or_module: The tensor or module to move
            
        Returns:
            The tensor or module on the appropriate device
        """
        return tensor_or_module.to(self.device)
        
    def get_device_properties(self) -> dict:
        """
        Get properties of the current device.
        
        Returns:
            Dictionary of device properties
        """
        properties = {
            'device_type': self.device,
            'memory_allocated': 0,
            'memory_cached': 0
        }
        
        if self.device == 'cuda':
            properties.update({
                'name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
                'compute_capability': torch.cuda.get_device_capability()
            })
        elif self.device == 'mps':
            # MPS doesn't provide as much device info as CUDA
            properties.update({
                'name': 'Apple Silicon GPU',
                # Note: MPS doesn't currently provide memory stats
            })
            
        return properties
        
    def is_gpu_available(self) -> bool:
        """
        Check if any GPU (CUDA or MPS) is available.
        
        Returns:
            Boolean indicating GPU availability
        """
        return self.device in ('cuda', 'mps')
        
    def synchronize(self):
        """
        Synchronize the device if needed.
        This is important for accurate timing and memory management.
        """
        if self.device == 'cuda':
            torch.cuda.synchronize()
        # Note: MPS doesn't currently need explicit synchronization
        
    def empty_cache(self):
        """
        Clear unused memory cache on the device.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # Note: MPS handles memory management differently and doesn't
        # need explicit cache clearing