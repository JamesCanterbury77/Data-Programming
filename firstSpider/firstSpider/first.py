# James Canterbury
# Links: https://docs.scrapy.org/en/latest/intro/tutorial.html
#        https://docs.scrapy.org/en/latest/topics/selectors.html
# More than 10 attributes and scraped more than 100 items
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

filter = "/events/"


class firstSpider(CrawlSpider):
    name = 'first'
    allowed_domains = ['motorsportreg.com']
    start_urls = [
        'https://www.motorsportreg.com/calendar/']
    rules = [Rule(LinkExtractor(allow='.*'), callback='parse_items', follow=True)]

    def parse_items(self, response):
        url = response.url
        if filter in url:
            yield {
                'title': response.css('title::text').get().strip(),
                'venue': response.css(
                    '#event-main > div.event-main__content > div:nth-child(9) > div:nth-child(3) > div > h4::text').get().strip(),
                'location': response.css(
                    '#event-main > div.event-main__content > div:nth-child(9) > div:nth-child(3) > div > div::text').get().strip(),
                'date': response.css(
                    '#__layout > div > div > div:nth-child(2) > div > div > div.hero__container > div.hero__content > '
                    'div::text').get().strip(),
                'event_type': response.css(
                    '#event-main > div.event-main__content > div:nth-child(11) > '
                    'div.organiser.organiser_profile.organiser_profile-blurb > div.organiser__header > '
                    'div.organiser__header__body > div > a::text').get().strip(),
                'subheading1': response.css('#event-main > div.event-main__content > div:nth-child(2) > div > h2::text').get().strip(),
                'subheading2': response.css('#event-main > div.event-main__content > div:nth-child(4) > div > h2::text').get().strip(),
                'entries': response.css(
                    '#event-main > div.event-main__content > div:nth-child(7) > '
                    'div.section__head.section__head_justify > div:nth-child(1) > h2::text').get().strip(),
                'subheading3': response.css(
                    '#event-main > div.event-main__content > div:nth-child(9) > '
                    'div.section__head.section__head_justify > div:nth-child(1) > h2::text').get().strip(),
                'image': response.css(
                    '#__layout > div > div > div:nth-child(2) > div > div > div.hero__background > div > img::attr(src)').get().strip(),
                'registration ends': response.css(
                    '#event-main > div.event-main__sidebar.sub-nav-bar > div > div:nth-child(2) > '
                    'div.event-detail__body > div.event-detail__subtitle::text').get().strip(),
                'time': response.css(
                    '#event-main > div.event-main__sidebar.sub-nav-bar > div > div:nth-child(2) > '
                    'div.event-detail__body > div.event-detail__content::text').get().strip(),
            }
            '''
            title = response.css('title::text').get()
            event_location = response.css('#event-main > div.event-main__content > div:nth-child(9) > div:nth-child(3) > div > h4::text').get()
            location = response.css('#event-main > div.event-main__content > div:nth-child(9) > div:nth-child(3) > div > div::text').get()
            date = response.css('#__layout > div > div > div:nth-child(2) > div > div > div.hero__container > div.hero__content > div::text').get()
            event_type = response.css('#event-main > div.event-main__content > div:nth-child(11) > div.organiser.organiser_profile.organiser_profile-blurb > div.organiser__header > div.organiser__header__body > div > a::text').get()
            sh1 = response.css('#event-main > div.event-main__content > div:nth-child(2) > div > h2::text').get()
            sh2 = response.css('#event-main > div.event-main__content > div:nth-child(4) > div > h2::text').get()
            entries = response.css('#event-main > div.event-main__content > div:nth-child(7) > div.section__head.section__head_justify > div:nth-child(1) > h2::text').get()
            sh3 = response.css('#event-main > div.event-main__content > div:nth-child(9) > div.section__head.section__head_justify > div:nth-child(1) > h2::text').get()
            image = response.css('#__layout > div > div > div:nth-child(2) > div > div > div.hero__background > div > img::attr(src)').get()
            registration = response.css('#event-main > div.event-main__sidebar.sub-nav-bar > div > div:nth-child(2) > div.event-detail__body > div.event-detail__subtitle::text').get()
            time = response.css('#event-main > div.event-main__sidebar.sub-nav-bar > div > div:nth-child(2) > div.event-detail__body > div.event-detail__content::text').get()
            print('URL is: {}'.format(url))
            print('title is: {} '.format(title))
            print('event location is: {} '.format(str(event_location).strip()))
            print('location is: {} '.format(str(location).strip()))
            print('date is: {} '.format(str(date).strip()))
            print('event_type is: {} '.format(str(event_type).strip()))
            print('first subheading is: {} '.format(str(sh1).strip()))
            print('second subheading is: {} '.format(str(sh2).strip()))
            print('entries is: {} '.format(str(entries).strip()))
            print('third subheading is: {} '.format(str(sh3).strip()))
            print('image link is: https:{} '.format(str(image).strip()))
            print('registration ends {} '.format(str(registration).strip()))
            print('time registration closes: {}'.format(str(time).strip()))
            print('\n')
            '''
