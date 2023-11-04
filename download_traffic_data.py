import util.downloader as downloader
import util.decompressor as decompressor
import os

path = './data/raw_data/traffic/'
url = 'https://www.its.go.kr/opendata/fileDownload/traffic/2023/'
target_list = [f'202309{i:02}_5Min.zip' for i in range(1, 31)]

path_list = [path + i for i in target_list]
url_list = [url + i for i in target_list]

print(url_list)

dwn = downloader.FileDownloader(thread_num=2)
dwn.download(path_list, url_list)

dcmp = decompressor.Decompressor(thread_num=4)
dcmp.decompress(path_list, [path]*len(path_list))

# remove zip
for i in path_list:
    os.remove(i)