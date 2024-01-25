from multiprocessing import Process
from multiprocessing import Queue

class SharedWorker(Process):
    def __init__(self, input_block, output_block, target_function, function_args, reader_pos = None, writer_pos = None, daemonize = False):
        '''
        input_block: str, Name of the shared_block instance. Provide None if not needed
        output_block: str, Name of the shared_block instance. Provide None if not needed
        target_function: str, Name of the actual worker function
        function_args: tuple, List of arguments needed by the target_function
        reader_pos: int, Position at which the reader should be attached to the input_block, leave unspecified to be assigned automatically
        writer_pos: int, Position at which the writer should be attached to the output_block, leave unspecified to be assigned automatically
        daemonize: bool, Whether to daemonize the process or not
        '''
        self.input_block = input_block
        self.output_block = output_block
        if self.input_block:
            self.reader_pos = self.input_block.acquire_for_reading(reader_pos)
        if self.output_block:
            self.writer_pos = self.output_block.acquire_for_writing(writer_pos)

        