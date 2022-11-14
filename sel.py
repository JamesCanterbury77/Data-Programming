'''
James Canterbury
https://www.selenium.dev/documentation/overview/
https://www.guru99.com/accessing-forms-in-webdriver.html
Additional Work:
Use selenium to wait for a desired element to appear
Use selenium to scroll to a desired object
Use selenium to hover over an object
Use selenium to enter text
'''
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
firefox_options = Options()
firefox_options.add_argument('--headless')
service = Service(executable_path="./bin/geckodriver")
driver = webdriver.Firefox(service=service, options=firefox_options)
# driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

filter = "/events/"
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
robots = 'https://www.motorsportreg.com/robots.txt'
start_url = 'https://www.motorsportreg.com'
sitemap = 'https://www.motorsportreg.com/sitemap.cfm'

# Check robots.txt file
html = requests.get(robots, user_agent)
parsed = Protego.parse(html.text)
fetched = parsed.can_fetch(start_url, user_agent)

if fetched:
    time.sleep(3)
    driver.get(sitemap)
    locs = driver.find_elements(By.TAG_NAME, 'loc')
    l = len(locs)
    valid_urls = []
    for x in range(0, l):
        if filter in locs[x].text:
            valid_urls.append(locs[x].text)
        if len(valid_urls) == 100:
            locs = None
            break
    for i in range(0, len(valid_urls)):
        time.sleep(3)
        driver.implicitly_wait(5)
        driver.get(valid_urls[i])
        url = driver.current_url
        # print(url)
        title = driver.title
        # print(title)
        venue = driver.find_element(By.CSS_SELECTOR,
                                'div.hero__container > div.hero__content > a').text
        # print(venue)
        date = driver.find_element(By.XPATH, '//div[@class="hero__eyebrow"]').text
        # print(date)
        event_type = driver.find_element(By.CSS_SELECTOR, 'div.hero-organiser__body').text
        # print(event_type)

        data = [
            {'url': url}, {'title': title}, {'venue': venue}, {'date': date}, {'event_type': event_type}
        ]
        with open('data.jl', 'a') as file:
            json.dump(data, file)
            file.write('\n')
    # Extra stuff and final json entry
    time.sleep(3)
    driver.get(start_url + '/calendar')
    driver.implicitly_wait(5)
    search = driver.find_element(By.CSS_SELECTOR, '#calendarForm > div.event-search__fields > div > div > input')
    search.send_keys('HSCC' + Keys.ENTER)
    time.sleep(3)
    driver.execute_script("window.scrollTo(0, 1300);")
    wait = WebDriverWait(driver, timeout=10, poll_frequency=1,
                         ignored_exceptions=[ElementNotVisibleException, ElementNotSelectableException])
    element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, '2022 HSCC Night Series #11')))
    hover = ActionChains(driver).move_to_element(element).click()
    hover.perform()
    time.sleep(1)
    url = driver.current_url
    title = driver.title
    venue = driver.find_element(By.CSS_SELECTOR,
                                'div.hero__container > div.hero__content > a').text
    date = driver.find_element(By.XPATH, '//div[@class="hero__eyebrow"]').text
    event_type = driver.find_element(By.CSS_SELECTOR, 'div.hero-organiser__body').text

    data = [
        {'url': url}, {'title': title}, {'venue': venue}, {'date': date}, {'event_type': event_type}
    ]
    with open('data.jl', 'a') as file:
        json.dump(data, file)
        file.write('\n')

driver.quit()
