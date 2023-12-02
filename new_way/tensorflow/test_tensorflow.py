from tensorflow.python.client import device_lib

local_devices = device_lib.list_local_devices()
gpu_devices = [device for device in local_devices if device.device_type == 'GPU']

if gpu_devices:
    print("TensorFlow is using GPU.")
    print("GPU Devices:", gpu_devices)
else:
    print("TensorFlow is not using GPU.")
