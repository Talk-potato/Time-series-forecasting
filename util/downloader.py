import requests
from tqdm import tqdm
from concurrent import futures
import math

DEFAULT_DOWNLOAD_UNIT = 1024*32
DEFAULT_THREAD_NUM = 1

class FileDownloader:
    def __init__(self, thread_num=DEFAULT_THREAD_NUM, download_unit=DEFAULT_DOWNLOAD_UNIT):
        if thread_num < 1 or download_unit < 4096:
            raise Exception()
        
        self.thread_num = thread_num
        self.download_unit = download_unit

    def download(self, filepath_list, url_list, thread_num=None):
        if thread_num is None:
            thread_num = self.thread_num
        elif thread_num < 1:
            raise Exception()

        with futures.ThreadPoolExecutor(max_workers=thread_num) as thread_pool:
            task = [thread_pool.submit(self._download, filepath, url, i) for i, (filepath, url) in enumerate(zip(filepath_list, url_list))]
            futures.wait(task)

    def _download(self, filepath, url, pos):
        try:
            res = requests.get(url, stream=True, allow_redirects=False, timeout=20)
        except requests.exceptions.HTTPError as e:
            print(f'Http Error: {e}')
            return False
        except requests.exceptions.RequestException as e:
            print(f'Exception: {e}')
            return False
        with open(filepath, 'wb') as output_file:  
            total_len = int(res.headers.get('content-length', 0))

            with tqdm(desc=filepath, total=math.ceil(total_len/self.download_unit), position=pos) as progress:
                for data in res.iter_content(chunk_size=self.download_unit):
                    progress.update(1)
                    output_file.write(data) 

        return True