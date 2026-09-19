import torch


def dense_memory_bytes(rows, columns, dtype=torch.float32, copies=1):
    if not all(isinstance(value, int) and value > 0 for value in (rows, columns, copies)):
        raise ValueError("Matrix dimensions and copies must be positive integers")
    return rows * columns * torch.empty((), dtype=dtype).element_size() * copies


def guard_dense_materialization(rows, columns, dtype=torch.float32, copies=1, limit_bytes=1_000_000_000):
    required = dense_memory_bytes(rows, columns, dtype, copies)
    if required > limit_bytes:
        raise MemoryError(f"Dense materialization requires {required} bytes, exceeding {limit_bytes}")
    return required


def exact_chunked_frobenius_mse(observed, activity, dictionary, feature_chunk_size=4096):
    if observed.ndim != 2 or activity.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("Observed, activity and dictionary must be matrices")
    if observed.shape[0] != activity.shape[0] or observed.shape[1] != dictionary.shape[1] or activity.shape[1] != dictionary.shape[0]:
        raise ValueError("Factorization dimensions do not align")
    if not isinstance(feature_chunk_size, int) or feature_chunk_size < 1:
        raise ValueError("feature_chunk_size must be positive")
    x_squared = observed.new_zeros(())
    cross = observed.new_zeros(())
    hh = observed.new_zeros((activity.shape[1], activity.shape[1]))
    for start in range(0, observed.shape[1], feature_chunk_size):
        stop = min(start + feature_chunk_size, observed.shape[1])
        x_chunk = observed[:, start:stop]
        h_chunk = dictionary[:, start:stop]
        x_squared = x_squared + x_chunk.square().sum()
        cross = cross + ((activity.T @ x_chunk) * h_chunk).sum()
        hh = hh + h_chunk @ h_chunk.T
    gram = activity.T @ activity
    squared_error = x_squared - 2 * cross + (gram * hh).sum()
    return squared_error / observed.numel()
