import pyxrt
import numpy as np
import logging

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
        
        print(f'Allocated {nbytes} bytes flags={flags} groupid={group_id} address={self.buf.address():#x}')
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
    def __init__(self, device, xbin, name, flags=pyxrt.kernel.exclusive):
        if isinstance(flags, str):
            flags = getattr(pyxrt.kernel, flags)
        self.krnl = pyxrt.kernel(device, xbin.get_uuid(), name, flags)
        self.name = name
        #self.print_groups()
        
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
        return self.krnl(*newargs)
        
    def group_id(self, gid):
        return self.krnl.group_id(gid)

 
