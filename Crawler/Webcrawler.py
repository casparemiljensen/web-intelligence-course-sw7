import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time


class WebCrawler:
    def __init__(self, seed_urls):
        # 1. Initialize the crawler with a set of seed URLs.
        self.frontier = seed_urls  # Queue of URLs to crawl
        self.index = set()  # Set to keep track of indexed URLs
        self.crawled = set()  # Set to keep track of crawled URLs

    def normalize_url(self, url):
        # 4a. Normalize the URL by removing any trailing slashes.
        return url.rstrip('/')

    def obey_robots_txt(self, url):
        # 4b. Check if the URL complies with robots.txt (implementation needed).
        # Placeholder for robots.txt compliance logic.
        return True

    def is_valid_url(self, url):
        # 4b. Check that the URL passes certain filter tests, e.g., only crawl .dk domains.
        return url.endswith('.dk')

    def fetch_page(self, url):
        # 2. Fetch the page content for the given URL.
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def parse_page(self, url, content):
        # 3. Parse the page content and extract text and links.
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()  # Extract text from the page
        print(f"Extracted text from {url}")

        # Extract "link-to" URLs and add them to a set to avoid duplicates.
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            # tel is still extracted
            if not (href.startswith('mailto:') or href.startswith('tel:')):
                full_url = urljoin(url, href)  # Resolve relative URLs
                links.add(self.normalize_url(full_url))  # Normalize and add to links set

        return text, links  # Return extracted text and links

    def crawl(self):
        # 1. Begin crawling the URLs in the frontier.
        while self.frontier:
            current_url = self.frontier.pop(0)  # Fetch next URL from the frontier
            if current_url in self.crawled:
                continue  # Skip if already crawled

            print(f"Crawling: {current_url}")
            content = self.fetch_page(current_url)  # Fetch the page content

            if content:
                text, extracted_links = self.parse_page(current_url, content)  # Parse the page
                self.index.add(current_url)  # Add the current URL to the index
                self.crawled.add(current_url)  # Mark the URL as crawled

                for link in extracted_links:
                    # Normalize the extracted link URL.
                    # Check the link against filters and compliance.
                    if (link not in self.crawled):
                        # and
                        # self.is_valid_url(link) and
                        # self.obey_robots_txt(link)):
                        print(f"Adding to frontier: {link}")  # Debug: Show link being added
                        self.frontier.append(link)  # Add to frontier if it passes tests
                    else:
                        print(f"Skipping link: {link}")  # Debug: Show link being skipped

                # 5. Optionally, sleep to avoid overwhelming the server.
                time.sleep(1)
        print('Crawling completed! - Number of indexed URLs:', len(self.index))


if __name__ == "__main__":
    seed_urls = ["https://laesehesten.dk/"]  # Replace with your seed URLs
    crawler = WebCrawler(seed_urls)  # Create an instance of WebCrawler
    crawler.crawl()  # Start the crawling process
