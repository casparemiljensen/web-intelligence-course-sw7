from collections import defaultdict
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import heapq

class WebCrawler:
    def __init__(self, seed_urls):
        self.front_queues = defaultdict(list)  # Front queues based on priority
        self.back_queue_heap = []  # Heap to track when a host can be crawled again
        self.index = set()         # Set to keep track of indexed URLs
        self.crawled = set()       # Set to keep track of crawled URLs
        self.seed_host = self.get_host(seed_urls[0])  # Store the host of the seed URL
        self.initialize_front_queues(seed_urls)

    def initialize_front_queues(self, seed_urls):
        """Assign initial seed URLs to front queues based on priority."""
        print("Initializing front queues...")
        for url in seed_urls:
            priority = self.assign_priority(url)
            self.front_queues[priority].append(url)

    def assign_priority(self, url):
        """Assign priority to a URL based on heuristics (e.g., domain importance)."""
        print(f"assign_priority: {url}")
        if 'news' in url:
            return 1  # Higher priority for news sites
        return 5  # Default lower priority

    def normalize_url(self, url):
        """Normalize the URL by removing any trailing slashes."""
        return url.rstrip('/')

    def fetch_page(self, url):
        """Fetch the page content for the given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad responses
            return response.text
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def parse_page(self, url, content):
        """Parse the page content and extract text and links."""
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not (href.startswith('mailto:') or href.startswith('tel:')):
                full_url = urljoin(url, href)  # Resolve relative URLs
                links.add(self.normalize_url(full_url))  # Normalize and add to links set
        return text, links

    def is_same_domain(self, url):
        """Check if the URL belongs to the same domain as the seed."""
        host = self.get_host(url)
        same_domain = host == self.seed_host
        print(f"Checking if {url} is same domain as seed {self.seed_host}: {same_domain}")
        return same_domain

    def crawl(self):
        """Begin crawling process."""
        while True:
            # Check if any front queue has URLs to crawl
            if not any(self.front_queues[priority] for priority in self.front_queues):
                print("No more URLs to crawl. Exiting.")
                break

            # Select the highest priority front queue with URLs
            for priority in sorted(self.front_queues.keys()):
                if self.front_queues[priority]:  # Only consider non-empty queues
                    current_url = self.front_queues[priority].pop(0)
                    break
            else:
                # If no URLs were found in any priority queue, exit the loop
                print("All front queues are empty. Exiting.")
                break

            # Only proceed if the URL hasn't been crawled yet
            if current_url not in self.crawled:
                print(f"Crawling: {current_url}")
                content = self.fetch_page(current_url)
                if content:
                    self.index.add(current_url)
                    self.crawled.add(current_url)

                    # Parse the page and extract links
                    text, extracted_links = self.parse_page(current_url, content)

                    # Add links to appropriate front queues based on priority
                    for link in extracted_links:
                        if link not in self.crawled and self.is_same_domain(link):
                            priority = self.assign_priority(link)
                            print(f"Adding to priority {link}")
                            self.front_queues[priority].append(link)
                        else:
                            print(f"Skipping link: {link}")  # Debug: Show link being skipped
                else:
                    print(f"Failed to fetch {current_url}, marking as crawled.")
                    self.crawled.add(current_url)  # Mark as crawled even if fetch fails
            else:
                print(f"Already crawled: {current_url}")

        print(f"Crawling completed! Indexed {len(self.index)} URLs.")

    def get_host(self, url):
        """Extract the host from a URL."""
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            return parsed_url.netloc
        else:
            print(f"Invalid URL: {url}")
            return None

if __name__ == "__main__":
    seed_urls = ["https://q-distribution.dk"]
    crawler = WebCrawler(seed_urls)
    crawler.crawl()
