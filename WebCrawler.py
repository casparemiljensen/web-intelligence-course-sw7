import urllib.parse
import urllib.robotparser
import urllib.request
from bs4 import BeautifulSoup

##### Global #####
rp = urllib.robotparser.RobotFileParser()
robots_string = '/robots.txt'


def parse_robots(url):
    robots_url = urllib.parse.urljoin(url, robots_string)
    try:
        rp.set_url(robots_url)
        rp.read()
        print(f'Successfully paresd the robots.txt for {url}')
    except Exception as e:
        print(f'Error fecthing robots.txt: {e}')
        return None
    
    user_agent = 'MyCrawler'
    return {
        'can_fetch': rp.can_fetch(user_agent, url),
        'crawl_delay': rp.crawl_delay(user_agent),
        'site_maps': rp.site_maps()
    }

def print_sitemaps(sitemaps):
    if sitemaps is None:
        print("No Sitemap entries found or invalid syntax.")
    elif len(sitemaps) == 0:
        print("No Sitemap entries available in robots.txt.")
    else:
        print("Sitemap(s) found:")
        for sitemap in sitemaps:
            print(f" - {sitemap}")


url_to_crawl = "https://dr.dk"
rules = parse_robots(url_to_crawl)

if rules:
    print(f"Can fetch: {rules['can_fetch']}")
    print(f"Crawl delay: {rules['crawl_delay']}")
    print_sitemaps(rules['site_maps'])
else:
    print("No rules fetched.")