"""Serialize cooperating experiment workers across Windows login sessions."""
import ctypes
import os
import time
from contextlib import contextmanager
from ctypes import wintypes


@contextmanager
def gpu_study_lock(timeout_seconds=18000):
    if os.name != 'nt':
        raise RuntimeError('This experiment lock requires Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, 'Global\\CodexRFStudyGPU172_18_214_79')
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    next_log = 0.
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result in (0, 0x80):
                acquired = True
                print(f'GPU LOCK ACQUIRED pid={os.getpid()}', flush=True)
                break
            if result != 0x102:
                raise ctypes.WinError(ctypes.get_last_error())
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError('GPU experiment lock timed out')
            if now >= next_log:
                print(f'WAITING FOR GPU LOCK pid={os.getpid()}', flush=True)
                next_log = now + 30
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
