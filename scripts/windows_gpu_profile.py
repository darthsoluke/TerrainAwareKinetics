"""Opt-in Direct3D development profiling scope; never enables Developer Mode."""

import ctypes as c
from ctypes import wintypes as w
import uuid
import winreg


def guid(value):
    return (c.c_ubyte*16).from_buffer_copy(uuid.UUID(value).bytes_le)


def method(pointer, index, result, *arguments):
    address = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents[index]
    return c.WINFUNCTYPE(result, c.c_void_p, *arguments)(address)


def require_ok(result, operation):
    if result < 0:
        raise OSError(f'{operation} failed: HRESULT0x{result & 0xffffffff:08x}')


class AdapterDescription(c.Structure):
    _fields_ = [('description', w.WCHAR*128), ('vendor', w.UINT), ('device', w.UINT),
        ('subsystem', w.UINT), ('revision', w.UINT), ('video_memory', c.c_size_t),
        ('system_memory', c.c_size_t), ('shared_memory', c.c_size_t),
        ('luid_low', w.DWORD), ('luid_high', w.LONG), ('flags', w.UINT)]


class StableGpuProfile:
    def __init__(self):
        self.device = c.c_void_p()
        self.enabled = False

    def __enter__(self):
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r'SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock') as key:
            if winreg.QueryValueEx(key, 'AllowDevelopmentWithoutDevLicense')[0] != 1:
                raise RuntimeError('Stable profiling requires Developer Mode already enabled')
        dxgi, d3d = c.WinDLL('dxgi'), c.WinDLL('d3d12')
        factory = c.c_void_p()
        dxgi.CreateDXGIFactory1.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
        dxgi.CreateDXGIFactory1.restype = w.LONG
        factory_iid = guid('770aae78-f26f-4dba-a829-253c83d1b387')
        require_ok(dxgi.CreateDXGIFactory1(c.byref(factory_iid), c.byref(factory)), 'CreateDXGIFactory1')
        adapter = c.c_void_p()
        try:
            for index in range(16):
                require_ok(method(factory, 12, w.LONG, w.UINT, c.POINTER(c.c_void_p))(
                    factory, index, c.byref(adapter)), 'EnumAdapters1')
                description = AdapterDescription()
                require_ok(method(adapter, 10, w.LONG, c.POINTER(AdapterDescription))(
                    adapter, c.byref(description)), 'GetDesc1')
                if description.vendor == 0x10de:
                    break
                method(adapter, 2, w.ULONG)(adapter)
                adapter = c.c_void_p()
            if not adapter:
                raise RuntimeError('The profiling scope needs the explicit NVIDIA adapter')
            d3d.D3D12CreateDevice.argtypes = [c.c_void_p, w.UINT, c.c_void_p, c.POINTER(c.c_void_p)]
            d3d.D3D12CreateDevice.restype = w.LONG
            device_iid = guid('189819f1-1db6-4b57-be54-1821339b85f7')
            require_ok(d3d.D3D12CreateDevice(adapter, 0xb000, c.byref(device_iid), c.byref(self.device)),
                'D3D12CreateDevice')
            self.adapter_name = description.description
            require_ok(method(self.device, 40, w.LONG, w.BOOL)(self.device, True), 'SetStablePowerState')
            self.enabled = True
            return self
        except BaseException:
            self.close()
            raise
        finally:
            if adapter:
                method(adapter, 2, w.ULONG)(adapter)
            method(factory, 2, w.ULONG)(factory)

    def close(self):
        if self.device:
            try:
                if self.enabled:
                    require_ok(method(self.device, 40, w.LONG, w.BOOL)(self.device, False),
                        'Restore dynamic power state')
            finally:
                method(self.device, 2, w.ULONG)(self.device)
                self.device = c.c_void_p()
                self.enabled = False

    def __exit__(self, *exc):
        self.close()
