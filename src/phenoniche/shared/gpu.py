"""Explicit physical-to-logical GPU mapping, without initializing CUDA or touching jobs."""
import os
import subprocess
import sys


def _devices():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    return sorted((int(index.strip()), uuid.strip()) for index, uuid in
                  (line.split(',') for line in text.strip().splitlines()))


def bind_physical_devices(physical_gpu, expected_uuid):
    """Expose all devices in physical-index order, using UUIDs to make CUDA indexes unambiguous."""
    devices = _devices()
    if dict(devices).get(physical_gpu) != expected_uuid:
        raise ValueError('The requested physical GPU UUID changed')
    visibility = ','.join(uuid for _, uuid in devices)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') != visibility:
        raise RuntimeError('CUDA was already initialized with another mapping; use a fresh notebook kernel')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = visibility
    return f'cuda:{[index for index, _ in devices].index(physical_gpu)}'


def verify_physical_device(device, physical_gpu, expected_uuid):
    if dict(_devices()).get(physical_gpu) != expected_uuid:
        raise ValueError('The physical GPU UUID does not match the experiment')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    logical = int(str(device).split(':')[1])
    if logical >= len(visible) or visible[logical] != expected_uuid:
        raise ValueError('Initialize the notebook with its explicit UUID GPU mapping before training')
