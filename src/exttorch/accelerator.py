class TPUScope:
    @property
    def num_tpu_devices(self):
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    
    @property
    def install(self):
        import os
        os.system("""
            pip install torchvision torch~=2.6.0 'torch_xla[tpu]~=2.6.0' \ 
            -f https://storage.googleapis.com/libtpu-releases/index.html 
            -f https://storage.googleapis.com/libtpu-wheels/index.html
        """)
        
    def __enter__(self):
        from exttorch._env import _ENV
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.amp as amp
                
        _ENV["EXTTORCH_TPU"] = xm.xla_device()
        _ENV["EXTTORCH_XMP"] = xmp
        _ENV["EXTTORCH_PL"] = pl
        _ENV["EXTTORCH_XM"] = xm
        _ENV["EXTTORCH_AMP"] = amp
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        from exttorch._env import _ENV
        del _ENV["EXTTORCH_TPU"]
        del _ENV["EXTTORCH_XMP"]
        del _ENV["EXTTORCH_PL"]
        del _ENV["EXTTORCH_XM"]
        del _ENV["EXTTORCH_AMP"]