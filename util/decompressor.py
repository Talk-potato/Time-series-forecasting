import zipfile
from concurrent import futures
import os

DEFAULT_THREAD_NUM = 1

class Decompressor:
    def __init__(self, thread_num=DEFAULT_THREAD_NUM):
        if thread_num < 1:
            raise Exception()
        
        self.thread_num = thread_num

    def decompress(self, source_path_list, dest_folder_list, thread_num=None):
        if thread_num is None:
         thread_num = self.thread_num
        elif thread_num < 1:
            raise Exception()

        with futures.ThreadPoolExecutor(max_workers=thread_num) as thread_pool:
            task = [thread_pool.submit(self._decompress, src, dest, i) for i, (src, dest) in enumerate(zip(source_path_list, dest_folder_list))]
            futures.wait(task)

    def _decompress(self, src, dest, pos):
        if os.path.exists(src):
            with zipfile.ZipFile(src, "r") as zip_ref:
                print(f'[Decompress Started]: {src} to {dest}')
                zip_ref.extractall(dest)
                print(f'[Decompress Finished]: {src} to {dest}')