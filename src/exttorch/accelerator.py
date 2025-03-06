class TpuScope:
    def __enter__(self):
        import os
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.distributed.parallel_loader as pl
        
        # Instantiate the ENV const
        ENV = os.environ
        
        ENV["EXTTORCH_TPU"] = xm.xla_device()
        ENV["EXTTORCH_XMP"] = xmp
        ENV["EXTTORCH_PL"] = pl
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass