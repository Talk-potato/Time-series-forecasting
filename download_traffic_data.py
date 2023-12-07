import util.downloader as downloader
import util.decompressor as decompressor
import os

path = './data/raw_data/traffic/'
target_list = [f'202310{i:02}_5Min.zip' for i in range(20, 23)]

class its_downloader(downloader.FileDownloader):
    def __init__(self, thread_num=2, download_unit=1024*32):
        super(its_downloader, self).__init__(thread_num=thread_num, download_unit=download_unit)
        self.baseurl = 'https://www.its.go.kr/opendata/fileDownload/traffic/2023/'
        self.decompressor = decompressor.Decompressor(thread_num=thread_num)

    def get_data(self, srcpath, destpath, target_list):
        path_list = [srcpath + i for i in target_list]
        url_list = [self.baseurl + i for i in target_list]

        self.download(path_list, url_list)
        self.decompressor.decompress(path_list, [destpath]*len(target_list))

        for i in path_list:
            if os.path.exists(i):
                os.remove(i)

t = its_downloader()
t.get_data(path, path, target_list)