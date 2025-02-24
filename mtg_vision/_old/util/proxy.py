#  MIT License
#
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from tqdm import tqdm
import json
import time
import random
import requests
import io
import os
import urllib.request
import zipfile
from multiprocessing.pool import ThreadPool, Pool

# ============================================================================ #
# Proxy Downloader                                                             #
# ============================================================================ #

class Proxy:
    def __init__(self, cache_folder, default_threads=16, default_attempts=10, force_scrape=False, days=10, scrape_limit=10000, fileName="proxies.json", logger=tqdm.write):
        self.default_threads = default_threads
        self.default_attempts = default_attempts
        self.logger = logger
        self.proxies = []

        time_ms = int(round(time.time() * 1000))
        file = os.path.join(cache_folder, fileName)

        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        if not os.path.isfile(file) or force_scrape:
            self.logger("Updating proxies file: {}".format(file))
            self.proxies = self._scrapeAndDump(file, time_ms, limit=scrape_limit)
        else:
            try:
                with open(file, "r") as file_stream:
                    read_obj = json.load(file_stream)
                proxies = read_obj["proxies"]
                ms_time_created = read_obj["created"]
                if time_ms - ms_time_created > 1000 * 60 * 60 * 24 * days:
                    self.logger("Proxy file is old... Scraping proxies!")
                    self.proxies = self._scrapeAndDump(file, time_ms, limit=scrape_limit)
                else:
                    self.proxies = proxies
            except:
                self.logger("Invalid proxy file... Scraping proxies!")
                self.proxies = self._scrapeAndDump(file, time_ms, limit=scrape_limit)

    def randomProxy(self):
        index = random.randint(0, len(self.proxies) - 1)
        return self.proxies[index]

    def _scrapeAndDump(self, file, time_ms, limit=100):
        proxies = self._downloadMorphProxies(num=limit)
        with open(file, "w") as file_stream:
            json.dump({"created": time_ms, "proxies": proxies}, file_stream)
        self.logger("Saved: {} proxies to: {}".format(len(proxies), file))
        return proxies

    def _downloadMorphProxies(self, morph_api_key="vvZhwW9dGzZUcnjAO4mZ", num=100):
        morph_api_url = "https://api.morph.io/CookieMichal/us-proxy/data.json"
        # morph_api_url = "https://api.morph.io/ftballpack/us-proxy/data.json"

        r = requests.get(morph_api_url, params={
            'key': morph_api_key,
            'query': "select * from 'data' where anonymity='elite proxy' and https='yes' limit {}".format(num)
        })

        proxies = []
        for row in r.json():
            proto = 'HTTPS' if row['https'] == 'yes' else 'HTTP'
            url = "{}://{}:{}".format(proto, row['ip'], row['port'])
            proxies.append({proto: url})

        self.logger("Downloaded: {} proxies!".format(len(proxies)))

        return proxies

    def directDownload(self, url: str, folder: str, unzip=False):
        fileName = os.path.basename(url)
        path = os.path.join(folder, fileName)
        urllib.request.urlretrieve(url, path)
        if unzip:
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(folder)
            zip_ref.close()

    def downloadThreaded(self, url_file_tuples, threads=None, attempts=None):
        if len(url_file_tuples) < 0:
            return
        if threads is None: threads = self.default_threads
        if attempts is None: attempts = self.default_attempts
        args = [(u, f, attempts) for (u, f) in url_file_tuples]
        with ThreadPool(processes=threads) as pool:
            pool.starmap(self.download, args)


    def download(self, url, file, attempts=10):
        for i in range(attempts):
            try:
                proxy_handler = urllib.request.ProxyHandler(self.randomProxy())
                proxy_auth_handler = urllib.request.ProxyBasicAuthHandler()
                opener = urllib.request.build_opener(proxy_handler, proxy_auth_handler)
                read = opener.open(url).read() #add timeout
                f = io.FileIO(file, "w")
                f.write(read)
                self.logger("Downloaded (try={}): {} : {}".format(i, url, file))
                return
            except:
                continue
        self.logger("Failed (tries={}): {} : {}".format(attempts, url, file))
