from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from src.utils import build_elementwise_launch_config, build_matmul_launch_config


class CudaEngine:
    def __init__(
        self,
        kernel_dir: Optional[Path | str] = None,
        nvcc_options: Optional[list[str]] = None,
        compile_on_init: bool = True,
    ) -> None:
        self.kernel_dir = (
            Path(kernel_dir).resolve()
            if kernel_dir is not None
            else Path(__file__).resolve().parents[1] / "kernels"
        )
        if not self.kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found: {self.kernel_dir}")

        self.nvcc_options = nvcc_options or ["-std=c++11"]
        self._modules: Dict[str, SourceModule] = {}
        self._kernel_cache: Dict[Tuple[str, str], cuda.Function] = {}
        self._memory_pool: Dict[str, Tuple[cuda.DeviceAllocation, int]] = {}

        if compile_on_init:
            self.compile_all()

    def compile_all(self) -> None:
        for cu_file in sorted(self.kernel_dir.glob("*.cu")):
            self.compile_kernel_file(cu_file.name)

    def compile_kernel_file(self, file_name: str) -> SourceModule:
        source_path = self.kernel_dir / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {source_path}")

        source = source_path.read_text(encoding="utf-8")
        options = [*self.nvcc_options, "-I", str(self.kernel_dir)]
        module = SourceModule(source, options=options, no_extern_c=True)
        self._modules[file_name] = module

        stale_keys = [key for key in self._kernel_cache if key[0] == file_name]
        for key in stale_keys:
            del self._kernel_cache[key]
        return module

    def get_kernel(self, kernel_name: str, module_name: Optional[str] = None) -> cuda.Function:
        if module_name is not None:
            cache_key = (module_name, kernel_name)
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]

            if module_name not in self._modules:
                self.compile_kernel_file(module_name)
            module = self._modules[module_name]
            function = module.get_function(kernel_name)
            self._kernel_cache[cache_key] = function
            return function

        for current_module_name, module in self._modules.items():
            cache_key = (current_module_name, kernel_name)
            if cache_key in self._kernel_cache:
                return self._kernel_cache[cache_key]
            try:
                function = module.get_function(kernel_name)
            except cuda.LogicError:
                continue
            self._kernel_cache[cache_key] = function
            return function

        raise KeyError(f"Kernel function '{kernel_name}' was not found in compiled modules.")

    def alloc(self, name: str, nbytes: int, reuse: bool = True) -> cuda.DeviceAllocation:
        if name in self._memory_pool:
            allocation, current_nbytes = self._memory_pool[name]
            if reuse and current_nbytes >= nbytes:
                return allocation
            allocation.free()
            del self._memory_pool[name]

        allocation = cuda.mem_alloc(nbytes)
        self._memory_pool[name] = (allocation, nbytes)
        return allocation

    def free(self, name: str) -> None:
        if name in self._memory_pool:
            allocation, _ = self._memory_pool.pop(name)
            allocation.free()

    def free_all(self) -> None:
        for allocation, _ in self._memory_pool.values():
            allocation.free()
        self._memory_pool.clear()

    def upload(self, host_array: np.ndarray, name: Optional[str] = None) -> cuda.DeviceAllocation:
        contiguous = np.ascontiguousarray(host_array)
        if name is None:
            device_allocation = cuda.mem_alloc(contiguous.nbytes)
        else:
            device_allocation = self.alloc(name, contiguous.nbytes)
        cuda.memcpy_htod(device_allocation, contiguous)
        return device_allocation

    def download(
        self, device_allocation: cuda.DeviceAllocation, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        host_array = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(host_array, device_allocation)
        return host_array

    def run_matmul(
        self,
        a_gpu: cuda.DeviceAllocation,
        b_gpu: cuda.DeviceAllocation,
        c_gpu: cuda.DeviceAllocation,
        m: int,
        k: int,
        n: int,
        kernel: str = "naive",
        block: tuple[int, int, int] = (16, 16, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> None:
        kernel_map = {"naive": "matmul_naive", "tiled": "matmul_tiled", "vectorized":"matmul_vectorized"}
        if kernel not in kernel_map:
            raise ValueError(f"Unsupported matmul kernel: {kernel}")

        function = self.get_kernel(kernel_map[kernel], module_name="matmul.cu")
        grid, block = build_matmul_launch_config(m, n, block=block)

        function(
            a_gpu,
            b_gpu,
            c_gpu,
            np.int32(m),
            np.int32(k),
            np.int32(n),
            block=block,
            grid=grid,
            stream=stream,
        )

    def run_activation(
        self,
        activation: str,
        x_gpu: cuda.DeviceAllocation,
        y_gpu: cuda.DeviceAllocation,
        numel: int,
        block: tuple[int, int, int] = (256, 1, 1),
        stream: Optional[cuda.Stream] = None,
    ) -> None:
        activation_map = {
            "relu": "relu_forward",
            "sigmoid": "sigmoid_forward",
            "tanh": "tanh_forward",
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        function = self.get_kernel(activation_map[activation], module_name="activations.cu")
        grid, block = build_elementwise_launch_config(numel, block=block)
        function(
            x_gpu,
            y_gpu,
            np.int32(numel),
            block=block,
            grid=grid,
            stream=stream,
        )

