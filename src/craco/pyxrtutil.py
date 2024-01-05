import pyxrt
import numpy as np
import logging
import time

log = logging.getLogger(__name__)

class Buffer:
    def __init__(self, shape, dtype, device, group_id:int, flags=pyxrt.bo.flags.normal):
        if isinstance(flags, str):
            flags = getattr(pyxrt.bo.flags, flags)
        itemsize = np.dtype(dtype).itemsize
        nbytes = int(np.prod(shape)*itemsize)
        # allocate buffer 
        self.buf = pyxrt.bo(device, nbytes, flags, group_id)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.nbytes = nbytes
        self.itemsize = itemsize
        self.size = self.nbytes
        
        log.info(f'Allocated {nbytes} bytes flags={flags} groupid={group_id} address={self.buf.address():#x}')
        # make numpy view from mappying the buffer. Doesn't use anymore memory
        # Any writes to teh np array go directly to the host pointer allocated above
        if flags == pyxrt.bo.flags.device_only:
            self.nparr = None
        else:
            self.nparr = np.frombuffer(self.buf.map(), dtype=dtype)
            self.nparr.shape = shape

    def clear(self):
        if self.nparr is not None:
            self.nparr[:] = 0
            self.copy_to_device()
        return self
         
    def copy_to_device(self):
        if self.nparr is not None:
            self.buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,  self.nparr.nbytes, 0)

        return self
                
    def copy_from_device(self):
        if self.nparr is not None:
            self.buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self.nparr.nbytes, 0)

        return self

    def saveto(self, fname):
        if self.nparr is not None:
            self.copy_from_device()
            log.info(f'Saving buffer {fname}')
            np.save(fname, self.nparr)

        return self

def convert_buffer(b):
    if isinstance(b, Buffer):
        return b.buf
    else:
        return b
      
class Kernel:
    def __init__(self, device, xbin, name, flags=pyxrt.kernel.exclusive, icu=None):
        if isinstance(flags, str):
            flags = getattr(pyxrt.kernel, flags)

        # new XRT wants extra brackets
        try:
            if icu is None:
                newname = name
            else:
                newname = f'{name}:{{{name}_{icu+1}}}'
            print(f'Opening kernel with new name {newname}')
            # new XRT doesn't like flags as the final argument, but we probably don't mind either.
            self.krnl = pyxrt.kernel(device, xbin.get_uuid(), newname)#, flags)
        except RuntimeError as r:
            raise r
            if icu is None:
                icu = 1
            newname = f'{name}:{name}_{icu+1}'
            self.krnl = pyxrt.kernel(device, xbin.get_uuid(), newname, flags)

        self.name = newname
        self.print_groups()
        self.device = device
        self.xbin = xbin
        
    def print_groups(self):
        self.groups = []
        i = 0
        print(f'Kernel {self.name} has groups')
        while True:
            try:
                groupid = self.krnl.group_id(i)
                self.groups.append(groupid)
                print(f'GID={i}={groupid}')
                i += 1
            except IndexError:
                break
            
        
        
    def __call__(self,*args):
        newargs = list(map(convert_buffer, args))
        raw_start = self.krnl(*newargs)
        return KernelStart(self, raw_start)

    def read_register(self, address):
        # krnl.read_reigster()  available in later versions of pyxrt
        # pyxrt.ip isnt in there either
        # F**K
        #return self.krnl.read_register(address)
        raise NotImplementedError('Not available in later versions of xrt')

    def read_status_register(self):
        return self.read_register(0x00)
        
    def group_id(self, gid):
        return self.krnl.group_id(gid)


class KernelStart:
    def __init__(self, kernel, raw_start):
        self.kernel = kernel
        self.raw_start = raw_start

        assert kernel is not None
        assert raw_start is not None

    def wait(self, timeout_ms:int=0):
        # We've had a problem where wait() times out.
        # I'm going to see if it improves if we check the status first and only call
        # wait() if it hasn't finished
        state = self.raw_start.state()
        if state == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            return state
        
        state = self.raw_start.wait(timeout_ms)
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            status = 0 # self.kernel.read_status_register()
            isdone = status & 0x04 == 0x04
            raise ValueError(f'Wait on start={self.start} failed with {state} status={status:02x} isok={isdone}')

        return state

def wait_for_starts(starts, call_start, timeout_ms: int=1000):
    '''
    Wait for all the runs.
    call_start is a timestamp so we can debug how long it took to run
    timeout_ms is a timeout in milliseconds (int)
    '''

    log.info('Waiting for %d starts', len(starts))
    # I don't know why this helps, but it does, and I don't like it!
    # It was really reliable when it was in there, lets see if its still ok when we remove it.
    #time.sleep(0.1)

    wait_start = time.perf_counter()
    for istart, start in enumerate(starts):
        log.debug(f'Waiting for istart={istart} start={start}')
        # change to wait2 as this is meant to throw a command_error execption
        # https://xilinx.github.io/XRT/master/html/xrt_native.main.html?highlight=wait#classxrt_1_1run_1ab1943c6897297263da86ef998c2e419c
        # see Also CRACO-128
        # Ah, but wait2 doesn't exist in PYXRT
        state = start.wait(timeout_ms) # 0 means wait forever
        wait_end = time.perf_counter()
        log.debug(f'Call: {wait_start - call_start} Wait:{wait_end - wait_start}: Total:{wait_end - call_start} state={state}')
        

class KernelStarts:
    def __init__(self):
        self.starts = []
        self.call_start = time.perf_counter()

    def append(self, s):
        self.starts.append(s)

    def wait(self, timeout:int=1000):
        '''
        Wait for all the starts
        :timeout: time to wait in milliseconds
        '''
        return wait_for_starts(self.starts, self.call_start, timeout)

    def __len__(self):
        return len(self.starts)

    def __str__(self):
        return str(self.starts)

 
