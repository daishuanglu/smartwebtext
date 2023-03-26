"""
Web press news scrapper for prnewswire website (https://www.prnewswire.com). Example usage
python3 scrappers/prnews.py\
    --output_dir=path/to/output/folder\
    --company_url_path=path/to/company/list/file.txt\
    --kw_search_counts='{"NASDAQ":15440, "NYSE":14811}'

Runing this command will scrap all the companies from NASDAQ and NYSE company news url to
'path/to/company/list/file.txt' and scrap the news for each company url in the file.
News for each company will be saved in a .txt files to the path/to/output/folder, e.g. if 10 companies
are found in 'path/to/company/list/file.txt' 10 .txt files will be saved to path/to/output/folder.
"""

import argparse
import os
import re
import time
import warnings

import requests
from dateutil.parser import parse
from bs4 import BeautifulSoup
from p_tqdm import p_map

from scrappers import prnews_company
from utils import string_utils
from scrappers import manager


warnings.filterwarnings("ignore")
main_link_uri = 'https://www.prnewswire.com'


def extract_date(body):
    dateparts = string_utils.keep_alphanum(body.split("/PRNewswire/")[0])
    datestr_back = ' '.join(dateparts.split()[-3:])
    datestr_front = ' '.join(dateparts.replace('Share this article', '').split()[:3])
    datestr = ' '
    try:
        if string_utils.hasNumbers(datestr_back):
            dateform = parse(datestr_back, fuzzy_with_tokens=True)[0]
            datestr = str(dateform).split()[0]
        elif string_utils.hasNumbers(datestr_front):
            dateform = parse(datestr_front, fuzzy_with_tokens=True)[0]
            datestr = str(dateform).split()[0]
    except:
        pass
    return datestr


class websearch():

    reject_titleWords = [
        'financial result', 'quater', 'market', 'award', 'report', 'industry', 'universit',
        'hospital', 'survey', 'forum', 'conference', 'report', 'meeting', 'earning']

    def __init__(self, output_file):
        self.output_file = output_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Company\tStock\tdate\tTitle\tBody\tLink\n")
        self.titles = []

    def _containRule(self, sentence, center_word):
        if center_word not in sentence:
            return False
        return string_utils.findLocalWords(sentence, center_word)

    def scrap_rule0(self, uri, company_name, login=None, loginform=None):
        soup = self.get_soup(uri, login, loginform)
        if soup is None:
            return 'Not found'

        title = '-'.join(uri.split('/')[-1].split('-')[:-1])
        kw = string_utils.containkwOR(title.lower(), self.reject_titleWords)
        if kw:
            return 'Reject title name by keyword %s - ' % kw.upper()
        if title in self.titles:
            return 'Duplicated title'
        self.titles.append(title)
        sid = soup.text.find('Share this article')
        eid = soup.text.find('\n×\nModal title')
        content = soup.text[sid:eid].replace('Share this article', '')
        content = re.sub(r'\n\s*\n', ' ', content)
        stocks = string_utils.find_stock_code(content)
        content=content.replace('\n',';;;;')
        content = content.replace('\t', ' ')
        datestr = extract_date(content)
        sid=content.find("/PRNewswire/ --")
        sid+=len("/PRNewswire/ --")
        output_str = [company_name, ','.join(stocks), datestr, title, content[sid:], uri]
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(string_utils.ignore_unicode_char('\t'.join(output_str))+'\n')
        return soup

    def get_soup(self, uri, login=None, loginform=None):
        ntries = 10
        for i in range(ntries):
            try:
                if not login:
                    html = requests.get(uri).text
                else:
                    html = login(uri, loginform)
                soup = BeautifulSoup(html, 'html.parser')
                return soup
            except:
                print('Retry exceeded, URL:', uri)
                time.sleep(3)
                continue
        return

def get_company_name_from_link(link):
    return link[:-1].split('/')[-1]

def get_company_name_from_url(url):
    link = url.split('?page=')[0]
    return link[:-1].split('/')[-1]


def scrap_news_from_link_href(scrapper, link, company):
    #print(company, "-", link)
    uri = link.get('href')
    if not uri:
        return False
    if 'news-releases' in uri and ('.html' in uri):
        company_uri = main_link_uri + uri
        soup = scrapper.scrap_rule0(company_uri, company)
        if isinstance(soup, str):
            print(soup, company_uri)
            return False
        #print(company_uri)
        return True
    return False


def scrap_companies(output_dir, company_list):

    def scrap_by_company(scrapper, company_link):
        i_page = 1
        found = True
        while found:
            company_uri = company_link + '?page=' + str(i_page) + '&pagesize=25'
            company = get_company_name_from_link(company_link)
            main_soup = scrapper.get_soup(company_uri)
            if main_soup is None:
                continue
            found = [
                scrap_news_from_link_href(
                    scrapper, link, company) for link in main_soup.find_all('a')]
            found = any(found)
            i_page += 1
        return i_page
    fname = 'webtext_thread_*.txt'
    fpath = os.path.join(output_dir, fname.replace('*', '{:d}'))
    incompleted_ids = manager.incompleted(output_dir, fname, min_doc=5)
    incompleted_ids = incompleted_ids if incompleted_ids else list(range(len(company_list)))
    print('%d incompleted companies.' % len(incompleted_ids))
    scrappers = [websearch(fpath.format(i)) for i in incompleted_ids]
    incompleted_company_list = [company_list[i] for i in incompleted_ids]
    nvalid_pages = p_map(scrap_by_company, scrappers, incompleted_company_list)
    print('{} valid uri.'.format(sum(nvalid_pages)*25))
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for all the news texts.')
    parser.add_argument(
        '--company_url_path', required=True,
        help='Scrapped company search urls saved in a .txt file.')
    parser.add_argument(
        '--kw_search_counts', required=False, default=None,
        help='Keyword search result counts, e.g. {"analytics": 9876, "NASDAQ": 15367}, '
               'default is {"NASDAQ":15440, "NYSE":14811} as searched from the website.'
               'https://www.prnewswire.com')
    args = parser.parse_args()
    search_cnt = eval(args.kw_search_counts)
    kw_company = prnews_company.company_finder(
        filename=args.company_url_path,
        custom_kw_search_cnt=search_cnt)
    if not kw_company.company_uris:
        kw_company.run_search()
    scrap_companies(args.output_dir, kw_company.company_uris)
    return


if __name__ == "__main__":
    main()
