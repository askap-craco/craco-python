import numpy as np
from multiprocessing import shared_memory

class SharedBlockError(Exception):
    def __init__(self, msg, lvl = 0):
        self.msg = msg
        self.lvl = lvl
        
    def __str__(self):
        return self.msg
        
    def __repr__(self):
        return self. __str__()



class SharedBlock(shared_memory.SharedMemory):
    '''
    Available to acquire = 0
    Acquired but not engaged = 1
    Acquired and engaged = 2
    '''
    
    WRITER_BYTES_START_POS = 0
    WRITER_NBYTES = 8
    WRITER_BYTES_END_POS = WRITER_NBYTES
    READER_BYTES_START_POS = WRITER_NBYTES
    READER_NBYTES = 8
    READER_BYTES_END_POS = READER_BYTES_START_POS + READER_NBYTES
    HDR_START_POS = WRITER_NBYTES + READER_NBYTES
    HDR_SIZE = 16384
    HDR_END_POS = HDR_START_POS + HDR_SIZE
     
    def __init__(self, name, create, block_size):
        self.extra_bytes = self.WRITER_NBYTES + self.READER_NBYTES + self.HDR_SIZE
        super().__init__(name, create, block_size + self.extra_bytes)
        if create:
            self.reset_write_bytes()
            self.reset_read_bytes()
        
    def reset_write_bytes(self):
        self.buf[self.WRITER_BYTES_START_POS:self.WRITER_BYTES_END_POS] = np.zeros(self.WRITER_NBYTES, dtype='uint8').tobytes()
        
    def reset_read_bytes(self):
        self.buf[self.READER_BYTES_START_POS:self.READER_BYTES_END_POS] = np.zeros(self.READER_NBYTES, dtype='uint8').tobytes()
        
    def acquire_for_reading(self, ireader = None):
        '''
        Attaches a reader to the shared block at ireader position.
        If ireader is None, it attaches it at the first available position
        
        Returns the position at which the reader was attached
        
        Raises SharedBlockError if ireader is malformed/unavailable, or all slots are already attached to
        '''
        if ireader is not None:
            if ireader >= self.READER_NBYTES or ireader < 0:
                raise SharedBlockError(f"Max number of readers allowed = {self.READER_NBYTES}")
            if self.buf[self.READER_BYTES_START_POS + ireader] == 0:
                self.buf[self.READER_BYTES_START_POS + ireader] = 1
                return ireader
            else:
                raise SharedBlockError(f"There is already a reader attached at position {ireader}")
        else:
            for ipos in range(self.READER_BYTES_START_POS, self.READER_BYTES_END_POS):
                if self.buf[ipos] == 0:
                    self.buf[ipos] = 1
                    return ipos - self.READER_BYTES_START_POS
                raise SharedBlockError("All slots for reader seem to be full already! Try again later")

    def acquire_for_writing(self, iwriter = None):
        '''
        Attaches a writer to the shared block at iwriter position.
        If iwriter is None, it attaches it at the first available position
    
        Returns the position at which the writer was attached
    
        Raises SharedBlockError if iwriter is malformed/unavailable, or all slots are already attached to
        '''
        if iwriter is not None:
             if iwriter >= self.WRITER_NBYTES or iwriter < 0:
                 raise SharedBlockError(f"Max number of writer allowed = {self.WRITER_NBYTES}")
             if self.buf[self.WRITER_BYTES_START_POS + iwriter] == 0:
                 self.buf[self.WRITER_BYTES_START_POS + iwriter] = 1
                 return iwriter
             else:
                 raise SharedBlockError(f"There is already a writer attached at position {iwriter}")
        else:
             for ipos in range(self.WRITER_BYTES_START_POS, self.WRITER_BYTES_END_POS):
                 if self.buf[ipos] == 0:
                     self.buf[ipos] = 1
                     return ipos - self.WRITER_BYTES_START_POS
                 raise SharedBlockError("All slots for writer seem to be full already! Try again later")

    def engage_for_writing(self, iwriter):
        '''
        Sets the writer flag to 2 for iwriter acquired at iwriter

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if iwriter >= self.WRITER_NBYTES or iwriter < 0:
            raise SharedBlockError(f"Max number of writers allowed = {self.WRITER_NBYTES}, given = {iwriter}")
        
        if self.buf[self.WRITER_BYTES_START_POS + iwriter] != 1:
            raise SharedBlockError(f"Cannot engage the shared block at writer pos {iwriter} as it is either not yet acquired (status = 0), or already engaged (status = 2). Current status = {self.buf[self.WRITER_BYTES_START_POS + iwriter]}")
        else:
            self.buf[self.WRITER_BYTES_START_POS + iwriter] = 2
            return 0


    def engage_for_reading(self, ireader):
        '''
        Sets the reader flag to 2 for reader acquired at ireader

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if ireader >= self.READER_NBYTES or ireader < 0:
            raise SharedBlockError(f"Max number of readers allowed = {self.READER_NBYTES}, given = {ireader}")
        
        if self.buf[self.READER_BYTES_START_POS + ireader] != 1:
            raise SharedBlockError(f"Cannot engage the shared block at reader pos {ireader} as it is either not yet acquired (status = 0), or already engaged (status = 2). Current status = {self.buf[self.READER_BYTES_START_POS + ireader]}")
        else:
            self.buf[self.READER_BYTES_START_POS + ireader] = 2
            return 0

    def deengage_from_writing(self, iwriter):
        '''
        Sets the writer flag to 1 for writer acquired at iwriter

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if iwriter >= self.WRITER_NBYTES or iwriter < 0:
            raise SharedBlockError(f"Max number of writers allowed = {self.WRITER_NBYTES}, given = {iwriter}")
        
        if self.buf[self.WRITER_BYTES_START_POS + iwriter] != 2:
            raise SharedBlockError(f"Cannot deengage the shared block at writer pos {iwriter} as it is either not yet acquired (status = 0), or not yet engaged (status = 1). Current status = {self.buf[self.WRITER_BYTES_START_POS + iwriter]}")
        else:
            self.buf[self.WRITER_BYTES_START_POS + iwriter] = 1
            return 0
        

    def deengage_from_reading(self, ireader):
        '''
        Sets the reader flag to 1 for reader acquired at ireader

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if ireader >= self.READER_NBYTES or ireader < 0:
            raise SharedBlockError(f"Max number of readers allowed = {self.READER_NBYTES}, given = {ireader}")
        
        if self.buf[self.READER_BYTES_START_POS + ireader] != 2:
            raise SharedBlockError(f"Cannot deengage the shared block at reader pos {ireader} as it is either not yet acquired (status = 0), or not yet engaged (status = 1). Current status = {self.buf[self.READER_BYTES_START_POS + ireader]}")
        else:
            self.buf[self.READER_BYTES_START_POS + ireader] = 1
            return 0
        
    def release_from_writer(self, iwriter):
        '''
        Releases a slot in the shared block from writing by setting it to 0

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if iwriter >= self.WRITER_NBYTES or iwriter < 0:
            raise SharedBlockError(f"Max number of writers allowed = {self.WRITER_NBYTES}, given = {iwriter}")
        
        if self.buf[self.WRITER_BYTES_START_POS + iwriter] != 1:
            raise SharedBlockError(f"Cannot release writer at {iwriter}. Current status = {self.buf[self.WRITER_BYTES_START_POS + iwriter]}, expected 1")
        else:
            self.buf[self.WRITER_BYTES_START_POS + iwriter] = 0
            return 0

    def release_from_reader(self, ireader):
        '''
        Releases a slot in the shared block from reading by setting it to 0

        Returns 0 if all OK
        Raises SharedBlockError if unsuccesfull
        '''
        if ireader >= self.READER_NBYTES or ireader < 0:
            raise SharedBlockError(f"Max number of readers allowed = {self.READER_NBYTES}, given = {ireader}")
        
        if self.buf[self.READER_BYTES_START_POS + ireader] != 1:
            raise SharedBlockError(f"Cannot release reader at {ireader}. Current status = {self.buf[self.READER_BYTES_START_POS + ireader]}, expected 1")
        else:
            self.buf[self.READER_BYTES_START_POS + ireader] = 0
            return 0

    def destroy(self, force = False):
        '''
        Destroys the shared memory after making sure all slots are released
        If force = True, then it just directly unlinks the shared_memory without checking for anything
        '''
        for ireader in range(self.READER_BYTES_START_POS, self.READER_BYTES_END_POS):
            if self.buf[ireader] != 0:
                if force:
                    print(f"Clearing a reader at position {ireader}")
                    self.buf[ireader] = 0
                else:
                    raise SharedBlockError("A reader process may still be attached at pos {ireader}. Current status = {self.buf[ireader]}")
        for iwriter in range(self.WRITER_BYTES_START_POS, self.WRITER_BYTES_END_POS):
            if self.buf[iwriter] != 0:
                if force:
                    print(f"Clearing a writer at position {iwriter}")
                    self.buf[iwriter] = 0
                else:
                    raise SharedBlockError("A writer process may still be attached at pos {iwriter}. Current status = {self.buf[iwriter]}")
            
        self.unlink()
        
    @property
    def reader_status(self):
        '''
        Returns a list of status for each available reader slot
        '''
        statuses = []
        for ipos in range(self.READER_BYTES_START_POS, self.READER_BYTES_END_POS):
            statuses.append(self.buf[ipos])

        return statuses

    @property
    def writer_status(self):
        '''
        Returns a list of status for each available writer slot
        '''
        statuses = []
        for ipos in range(self.WRITER_BYTES_START_POS, self.WRITER_BYTES_END_POS):
            statuses.append(self.buf[ipos])

        return statuses