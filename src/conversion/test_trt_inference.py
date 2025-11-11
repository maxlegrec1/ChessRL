import os
import sys
from typing import List, Tuple, Union

import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401  # Creates CUDA context
import pycuda.driver as cuda
import torch
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.small.fen_encoder import batch_fen_to_tensor
from src.models.small.vocab import policy_index

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _load_engine(engine_path: str) -> trt.ICudaEngine:
    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine at: {engine_path}")

    return engine


def _allocate_buffers(
    engine: trt.ICudaEngine, context: trt.IExecutionContext, input_shape: Tuple[int, ...]
) -> Tuple[
    List[Tuple[np.ndarray, cuda.DeviceAllocation, Tuple[int, ...], Union[int, str]]],
    List[Tuple[np.ndarray, cuda.DeviceAllocation, Tuple[int, ...], Union[int, str]]],
    List[int],
    cuda.Stream,
    bool,
]:
    inputs = []
    outputs = []
    bindings: List[int] = []
    stream = cuda.Stream()

    use_tensor_api = hasattr(engine, "num_io_tensors") and callable(getattr(engine, "get_tensor_name", None))

    if not use_tensor_api:
        for binding_idx in range(engine.num_bindings):
            binding_name = engine.get_binding_name(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
            shape = tuple(engine.get_binding_shape(binding_idx))

            if engine.binding_is_input(binding_idx):
                if -1 in shape:
                    # Dynamic shape; supply the runtime dimensions.
                    context.set_binding_shape(binding_idx, input_shape)
                else:
                    # Ensure context sees the static dimensions.
                    context.set_binding_shape(binding_idx, shape)
            actual_shape = tuple(context.get_binding_shape(binding_idx))

            size = trt.volume(actual_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            target_list = inputs if engine.binding_is_input(binding_idx) else outputs
            target_list.append((host_mem, device_mem, actual_shape, binding_idx))

            print(
                f"[TensorRT] Binding '{binding_name}': {'input' if engine.binding_is_input(binding_idx) else 'output'} "
                f"shape={actual_shape} dtype={dtype.__name__}"
            )
        return inputs, outputs, bindings, stream, False

    # Fallback for TensorRT 10.x and later (I/O tensor API)
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            runtime_shape = input_shape if -1 in shape else shape
            context.set_input_shape(name, runtime_shape)

    if hasattr(context, "infer_shapes"):
        context.infer_shapes()

    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        runtime_shape = tuple(context.get_tensor_shape(name))
        if -1 in runtime_shape:
            if mode == trt.TensorIOMode.INPUT:
                runtime_shape = input_shape
            else:
                raise RuntimeError(f"Unable to resolve dynamic shape for output tensor '{name}'.")

        size = trt.volume(runtime_shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        context.set_tensor_address(name, int(device_mem))

        target_list = inputs if mode == trt.TensorIOMode.INPUT else outputs
        target_list.append((host_mem, device_mem, runtime_shape, name))

        print(
            f"[TensorRT] Tensor '{name}': {'input' if mode == trt.TensorIOMode.INPUT else 'output'} "
            f"shape={runtime_shape} dtype={dtype.__name__}"
        )

    return inputs, outputs, bindings, stream, True


def run_inference(engine_path: str, input_data: np.ndarray, temperature: float = 0.1) -> List[str]:
    engine = _load_engine(engine_path)
    with engine.create_execution_context() as context:
        start_time = time.time()
        inputs, outputs, bindings, stream, use_tensor_api = _allocate_buffers(engine, context, tuple(input_data.shape))

        # Copy host input to device.
        host_input, device_input, _, input_id = inputs[0]
        np.copyto(host_input, input_data.astype(host_input.dtype).ravel())
        cuda.memcpy_htod(device_input, host_input)

        # Execute inference.
        if use_tensor_api:
            context.execute_async_v3(stream.handle)
        else:
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy device output to host.
        host_output, device_output, output_shape, output_id = outputs[0]
        cuda.memcpy_dtoh(host_output, device_output)
        stream.synchronize()

        # Update output shape from context in case of dynamic dimensions.
        if use_tensor_api:
            runtime_shape = tuple(context.get_tensor_shape(output_id))
        else:
            runtime_shape = tuple(context.get_binding_shape(output_id))
        if np.prod(runtime_shape) != np.prod(output_shape):
            output_shape = runtime_shape

        logits = np.array(host_output, copy=False).reshape(output_shape)
    end_time = time.time()
    print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    logits_tensor = torch.from_numpy(logits).to(torch.float32)
    probs = torch.softmax(logits_tensor / temperature, dim=-1)

    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
    moves = [policy_index[idx] for idx in sampled_indices.tolist()]
    end_time = time.time()
    print(f"Time taken to sample moves: {end_time - start_time:.2f} seconds")
    return moves


def main():
    engine_path = "model.trt"

    fen = "r3k2r/ppp2ppp/2P5/8/2p1q1b1/B1P2N2/P4PPP/R2Q2K1 b kq - 0 14"
    fens = [fen] * 1024
    start_time = time.time()
    input_data = batch_fen_to_tensor(fens).cpu().numpy()
    print(f"Input tensor shape: {input_data.shape}")
    end_time = time.time()
    print(f"Time taken to encode: {end_time - start_time:.2f} seconds")
    sampled_moves = run_inference(engine_path, input_data, temperature=0.1)
    print(f"Sampled UCI moves: {sampled_moves}")


if __name__ == "__main__":
    main()

