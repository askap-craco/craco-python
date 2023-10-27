
class SharedWorker:
    def __init__(self, input_block, output_block, reader_pos = None, writer_pos = None):
        '''
        input_block: str, Name of the shared_block instance. Provide None if not needed
        output_block: str, Name of the shared_block instance. Provide None if not needed
        reader_pos: int, Position at which the reader should be attached to the input_block, leave unspecified to be assigned automatically
        writer_pos: int, Position at which the writer should be attached to the output_block, leave unspecified to be assigned automatically
        '''
        self.input_block = input_block
        self.output_block = output_block
        if self.input_block:
            self.reader_pos = self.input_block.acquire_for_reading(reader_pos)
        if self.output_block:
            self.writer_pos = self.output_block.acquire_for_writing(writer_pos)

        