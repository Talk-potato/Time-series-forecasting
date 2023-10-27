import os
import util.downloader as downloader
import util.decompressor as decompressor

# make data directory
os.makedirs('./data/processed_data/node_link', exist_ok=True)
os.makedirs('./data/processed_data/traffic', exist_ok=True)
os.makedirs('./data/raw_data/node_link', exist_ok=True)
os.makedirs('./data/raw_data/traffic', exist_ok=True)

# download node link data
dwn = downloader.FileDownloader(thread_num=1)
dwn.download(['./data/raw_data/node_link/node_link.zip'], ['https://www.its.go.kr/opendata/nodelinkFileSDownload/DF_190/0'])

dcmp = decompressor.Decompressor(thread_num=1)
dcmp.decompress(['./data/raw_data/node_link/node_link.zip'],['./data/raw_data/node_link'])

os.remove('./data/raw_data/node_link/node_link.zip')

