import scrapy
from bs4 import BeautifulSoup
from pprint import pprint
import csv


class FinalSpider(scrapy.Spider):
    name = 'final'

    def start_requests(self):
        urls = ['https://en.wikipedia.org/wiki/Wikipedia:10,000_most_common_passwords']
        return [scrapy.Request(url=url, callback=self.parse)
                for url in urls]

    def parse(self, response):
        url = response.url
        data = response.xpath('//*[@id="mw-content-text"]/div[1]/div[2]').extract()
        data = str(data)
        soup = BeautifulSoup(data, features='lxml')
        text = str(soup.find_all('li', text=True))
        passwords = []

        while len(text) >= 4:
            close = text.index('>')
            text = text[close + 1:]
            opens = text.index('<')

            passwords.append(text[0:opens])
            text = text[opens + 7:]

        data = response.xpath('//*[@id="mw-content-text"]/div[1]/div[3]').extract()
        data = str(data)
        soup = BeautifulSoup(data, features='lxml')
        text = str(soup.find_all('li', text=True))

        while len(text) >= 4:
            close = text.index('>')
            text = text[close + 1:]
            opens = text.index('<')

            passwords.append(text[0:opens])
            text = text[opens + 7:]

        with open('passwords.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for x in passwords:
                writer.writerow([x])

