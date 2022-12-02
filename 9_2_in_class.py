import re
import requests
from bs4 import BeautifulSoup
from protego import Protego
import json
import time

# wait 3 seconds between requests

trope = 'https://tvtropes.org/pmwiki/pmwiki.php/Main/TheAllegedComputer'
robots = 'https://tvtropes.org/robots.txt'
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.102 " \
             "Safari/537.36 "
html = requests.get(robots, user_agent)
parsed = Protego.parse(html.text)
fetched = parsed.can_fetch(trope, user_agent)
# print(fetched)
if fetched:
    prev_pages = []
    time.sleep(3)
    trope_request = requests.get(trope, user_agent)
    bs = BeautifulSoup(trope_request.text, 'html.parser')
    for link in bs.find_all('a', href=re.compile('^(/pmwiki/pmwiki.php/Main/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in prev_pages:
                newPage = link.attrs['href']
                prev_pages.append(newPage)
    # print(prev_pages)

    filename = 'lines.json'
    i = 1
    with open(filename, 'w') as file_object:
        d = dict({'text': bs.get_text(strip=True) + ' ' + str(i), 'links': prev_pages})
        json.dump(d, file_object)
    i += 1
    # Start page is done
    with open(filename, 'a') as file_object:
        file_object.write('\n')
    file_object.close()

    for x in prev_pages:
        set2_pages = []
        tr = requests.get("https://tvtropes.org" + x, user_agent)
        bs2 = BeautifulSoup(tr.text, 'html.parser')
        for link in bs2.find_all('a', href=re.compile('^(/pmwiki/pmwiki.php/Main/)')):
            if 'href' in link.attrs:
                if link.attrs['href'] not in set2_pages:
                    newPage = link.attrs['href']
                    set2_pages.append(newPage)

        d = dict({'text': bs2.get_text(strip=True) + ' ' + str(i), 'links': set2_pages})
        i += 1
        json_s = json.dumps(d)
        with open(filename, 'a') as file_object:
            file_object.write(json_s + '\n')
        file_object.close()
