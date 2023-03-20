from bs4 import BeautifulSoup
import requests
import os
import time

class company_finder():

    stock_search_cnt_default = {"NASDAQ":15440, "NYSE":14811}
    link_uri='https://www.prnewswire.com'
    release_uri = "https://www.prnewswire.com/news-releases/"

    def __init__(self,filename="default_search.txt", custom_kw_search_cnt=None):
        if not custom_kw_search_cnt:
            self.kw_search_cnt=self.stock_search_cnt_default
        else:
            self.kw_search_cnt = custom_kw_search_cnt

        self.fpath=filename
        if os.path.exists(self.fpath):
            self.company_uris = self.load_company_links()
        else:
            self.company_uris = []


    def run_search(self):
        self.f = open(self.fpath, 'w')
        self.company_uris =self.get_prnewswire_company()
        self.f.close()

    def load_company_links(self,fpath=""):
        if not fpath:
            f = open(self.fpath, 'r')
        else:
            f = open(fpath, 'r')
        valid_company_links = [l.rstrip() for l in f.readlines()]
        f.close()
        print('=> loaded {} companies.'.format(len(valid_company_links)))
        return valid_company_links


    def getSoupNoLogin(self,uri):
        html = requests.get(uri).text
        soup = BeautifulSoup(html, 'html.parser')
        return soup


    def get_prnewswire_company(self):
        company_uri=set()
        for stock,cnt in self.kw_search_cnt.items():
            valid_links= self.scrape_prnewswire_company_by_stock(stock, cnt)
            company_uri=company_uri.union(valid_links)
        return sorted(company_uri)


    def scrape_prnewswire_company_by_stock(self, stock,num_search_records):
        valid_links=set()
        start_time=time.time()
        npage=num_search_records//25+1
        for ppno in range(1,npage):
            main_uri = "https://www.prnewswire.com/search/news/?keyword="+stock+"&pagesize=25&page="+str(ppno)
            main_soup = self.getSoupNoLogin(main_uri)
            print(main_uri)
            list_of_urls = [l.get('href') for l in main_soup.find_all('a')]
            company_urls=set([url for url in list_of_urls if url if url.startswith('/news/')])
            valid_links=valid_links.union(company_urls)
            print("{}/{}, {} secs, {} valid companies. ".format(ppno, npage, int(time.time() - start_time), len(valid_links)))

        for found_link in valid_links:
            self.f.write( self.link_uri + found_link+'\n')

        print('{} valid companies are found.'.format(len(valid_links)))
        return valid_links

