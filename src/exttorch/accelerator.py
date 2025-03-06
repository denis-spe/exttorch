class TpuScope:
    def __enter__(self):
        import os
        from exttorch._env import _ENV
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.distributed.parallel_loader as pl
                
        _ENV["EXTTORCH_TPU"] = xm.xla_device()
        _ENV["EXTTORCH_XMP"] = xmp
        _ENV["EXTTORCH_PL"] = pl
        _ENV["EXTTORCH_XM"] = xm
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(exc_type, exc_value, traceback)