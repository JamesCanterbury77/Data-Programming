import time
import json
import requests
from protego import Protego
from selenium import webdriver
from selenium.common import ElementNotVisibleException, ElementNotSelectableException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# Driver setup
chrome_options = Options()
chrome_options.add_argument('--headless')
service = Service(executable_path="./chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)

user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
robots = 'https://en.wikipedia.org/robots.txt'
start_url = 'https://en.wikipedia.org/wiki/Wikipedia:10,000_most_common_passwords'

# Check robots.txt file
html = requests.get(robots, user_agent)
parsed = Protego.parse(html.text)
fetched = parsed.can_fetch(start_url, user_agent)

if fetched:
    time.sleep(3)
    driver.get(start_url)
    url = driver.url
    print(url)
    title = driver.title
    print(title)

    # data = [
    #    {'url': url}, {'title': title}, {'venue': venue}, {'date': date}, {'event_type': event_type}
    #]
    #with open('data.jl', 'a') as file:
    #    json.dump(data, file)
    #    file.write('\n')


driver.quit()
