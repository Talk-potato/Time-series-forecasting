import util.downloader as downloader
import util.decompressor as decompressor
import os


dwn = downloader.FileDownloader(thread_num=1)
dwn.download(['./data/raw_data/traffic/20230910_5Min.zip'], ['https://www.its.go.kr/opendata/fileDownload/traffic/2023/20230910_5Min.zip'])

dcmp = decompressor.Decompressor(thread_num=1)
dcmp.decompress(['./data/raw_data/traffic/20230910_5Min.zip'],['./data/raw_data/traffic'])

# remove zip
os.remove('./data/raw_data/traffic/20230910_5Min.zip')