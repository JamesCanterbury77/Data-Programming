import requests
from bs4 import BeautifulSoup
from protego import Protego
trope = 'https://tvtropes.org/pmwiki/pmwiki.php/Main/TheAllegedComputer'
robots = 'https://tvtropes.org/robots.txt'
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.102 " \
             "Safari/537.36 "
html = requests.get(robots, user_agent)
parsed = Protego.parse(html.text)
fetched = parsed.can_fetch(trope, user_agent)
print(fetched)
if fetched:
    trope_request = requests.get(trope, user_agent)
    bs = BeautifulSoup(trope_request.text, 'html.parser')
    # print(bs.prettify())
    print(bs.find_all(name='li'))
