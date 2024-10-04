import torch
import torch.nn as nn

class BuffersDict(nn.Module):
    def __init__(self, stats_dict=None):
        super(BuffersDict, self).__init__()
        if stats_dict is not None:
            for name, tensor in stats_dict.items():
                self.register_buffer(name, tensor)

    def __getitem__(self, key):
        if key in self._buffers:
            return self._buffers[key]
        else:
            raise KeyError(f"'{key}' not found in BuffersDict")

    def __setitem__(self, key, value):
        self.register_buffer(key, value)

    def __delitem__(self, key):
        if key in self._buffers:
            del self._buffers[key]
        else:
            raise KeyError(f"'{key}' not found in BuffersDict")

    def __iter__(self):
        return iter(self._buffers)

    def __len__(self):
        return len(self._buffers)

    def keys(self):
        return self._buffers.keys()

    def items(self):
        return self._buffers.items()

    def values(self):
        return self._buffers.values()

    def __contains__(self, key):
        return key in self._buffers

    def __repr__(self):
        # Customize how tensors are represented
        def tensor_repr(tensor):
            if tensor.numel() > 10:
                # Show only the shape and dtype for large tensors
                return f"tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"
            else:
                # Use default tensor representation
                return repr(tensor)

        items_repr = ", ".join(
            f"'{key}': {tensor_repr(value)}"
            for key, value in self.items()
        )
        return f"BuffersDict({{{items_repr}}})"

    def __str__(self):
        return self.__repr__()
