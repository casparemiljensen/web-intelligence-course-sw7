from collections import defaultdict, deque
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import heapq
import random


class WebCrawler:
    def __init__(self, seed_urls):
        self.front_queues = defaultdict(deque)  # Front queues based on priority
        self.back_queues = defaultdict(deque)  # Back queues, one per host
        self.back_queue_heap = []  # Heap for politeness, tracking the earliest time a host can be crawled
        self.index = set()  # Set to keep track of indexed URLs
        self.crawled = set()  # Set to keep track of crawled URLs
        self.host_to_backqueue = {}  # Map hosts to their respective back queues
        self.seed_host = self.get_host(seed_urls[0])  # Store the host of the seed URL
        self.initialize_front_queues(seed_urls)
        self.wait_time = 1  # Time to wait before crawling the same host again

    def initialize_front_queues(self, seed_urls):
        """Assign initial seed URLs to front queues based on priority."""
        print("Initializing front queues...")
        for url in seed_urls:
            priority = self.assign_priority(url)
            self.front_queues[priority].append(url)
        self.print_front_queues()  # Add this line

    def assign_priority(self, url):
        """Assign priority to a URL based on heuristics."""
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
            # print(f"Failed to fetch {url}: {e}")
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
        return host == self.seed_host

    def biased_front_queue_selector(self):
        """Select a front queue with bias towards higher priority queues."""
        total_queues = len(self.front_queues)
        priorities = sorted(self.front_queues.keys())
        weights = [1 / (i + 1) for i in range(total_queues)]  # Higher priority queues get higher weight
        selected_priority = random.choices(priorities, weights=weights, k=1)[0]
        return selected_priority

    # def add_to_back_queue(self, url):
    #     """Add a URL to its corresponding back queue based on the host."""
    #     host = self.get_host(url)
    #     if host not in self.host_to_backqueue:
    #         self.host_to_backqueue[host] = deque()  # Create a new back queue for the host
    #         heapq.heappush(self.back_queue_heap, (time.time(), host))  # Add to heap with current time

    # def add_to_back_queue(self, url):
    #     """Add a URL to its corresponding back queue based on the host."""
    #     host = self.get_host(url)
    #
    #     if host not in self.host_to_backqueue:
    #         self.host_to_backqueue[host] = deque()  # Create a new back queue for the host
    #         heapq.heappush(self.back_queue_heap, (time.time(), host))  # Add to heap with current time
    #     else:
    #         # The host is already in the back queue, so just append the URL to the existing queue
    #         self.host_to_backqueue[host].append(url)
    #
    #     self.back_queues[host].append(url)
    #     print("Added to back queue: ", url)

    def add_to_back_queue(self, url):
        """Add a URL to its corresponding back queue based on the host."""
        host = self.get_host(url)

        # Check if the host already exists in the back queues
        if host not in self.host_to_backqueue:
            self.host_to_backqueue[host] = deque()  # Create a new back queue for the host

        # Add URL to the back queue for the host
        self.host_to_backqueue[host].append(url)

        # Update or add the host in the back queue heap
        next_available_time = (time.time() + self.wait_time)

        # Instead of adding a new entry, we update the next available time for the host
        # by marking the old entry stale or removing it first

        # Create a new heap entry for the host
        updated_entry = (next_available_time, host)

        # Remove old entries for the same host (if any)
        self.back_queue_heap = [(time, h) for (time, h) in self.back_queue_heap if h != host]
        heapq.heapify(self.back_queue_heap)  # Re-heapify after removing old entries

        # Add the updated entry with the new next available time
        heapq.heappush(self.back_queue_heap, updated_entry)

    # def crawl(self):
    #     """Begin crawling process."""
    #     while True:
    #         # Extract the root of the heap to get the host to crawl
    #         if not self.back_queue_heap:
    #             print("No back queues available. Exiting.")
    #             break
    #
    #         # Get the host with the earliest available time
    #         next_time, host = heapq.heappop(self.back_queue_heap)
    #
    #         # Check if we can crawl this host
    #         if time.time() < next_time:
    #             # Reinsert the host if it is not yet ready to be crawled
    #             heapq.heappush(self.back_queue_heap, (next_time, host))
    #             print("Waiting to crawl host:", host)
    #             continue  # Wait until the host is available
    #
    #         # Use biased front queue selector to get a URL from front queues
    #         selected_priority = self.select_biased_front_queue()
    #         if selected_priority is None:
    #             print("No URLs in front queues. Exiting.")
    #             break
    #
    #         current_url = self.front_queues[selected_priority].pop()  # Get the next URL from the front queue
    #
    #         # Only proceed if the URL hasn't been crawled yet
    #         if current_url not in self.crawled:
    #             print(f"Crawling: {current_url}")
    #             content = self.fetch_page(current_url)
    #             if content:
    #                 self.index.add(current_url)
    #                 self.crawled.add(current_url)
    #
    #                 # Parse the page and extract links
    #                 text, extracted_links = self.parse_page(current_url, content)
    #
    #                 # Add links to appropriate front queues based on priority
    #                 for link in extracted_links:
    #                     if link not in self.crawled and self.is_same_domain(link):
    #                         priority = self.assign_priority(link)
    #                         print(f"Adding {link} to front queue with priority {priority}")
    #                         self.front_queues[priority].append(link)
    #
    #                 # Add the current URL to the back queue
    #                 self.add_to_back_queue(current_url)
    #
    #             else:
    #                 print(f"Failed to fetch {current_url}, marking as crawled.")
    #                 self.crawled.add(current_url)  # Mark as crawled even if fetch fails
    #         else:
    #             print(f"Already crawled: {current_url}")
    #
    #         self.print_back_queue_heap()

    def crawl(self):
        """Begin crawling process."""
        while True:
            # Check if there are any URLs in the front queue
            selected_priority = self.select_biased_front_queue()
            if selected_priority is None:
                print("No URLs in front queues. Exiting.")
                break

            # Get the next URL from the front queue
            current_url = self.front_queues[selected_priority].pop()

            # Extract the host and check if it can be crawled based on the back queue's timing
            host = self.get_host(current_url)
            next_time, _ = self.back_queue_heap[0] if self.back_queue_heap else (None, None)

            if not next_time or time.time() >= next_time:
                # Crawl the URL if it's time for this host
                print(f"Crawling: {current_url}")
                content = self.fetch_page(current_url)

                if content:
                    self.index.add(current_url)
                    self.crawled.add(current_url)

                    # Parse and extract links
                    text, extracted_links = self.parse_page(current_url, content)
                    for link in extracted_links:
                        if link not in self.crawled and self.is_same_domain(link):
                            priority = self.assign_priority(link)
                            print(f"Adding {link} to front queue with priority {priority}")
                            self.front_queues[priority].append(link)

                            # Also push to the back queue after assigning priority
                            self.add_to_back_queue(link)

                    # Move the current URL to the back queue for politeness
                    self.add_to_back_queue(current_url)

                else:
                    print(f"Failed to fetch {current_url}, marking as crawled.")
                    self.crawled.add(current_url)  # Mark as crawled even if fetch fails

                # Update the next crawl time for the host in the back queue heap
                heapq.heappush(self.back_queue_heap, (time.time() + self.wait_time, host))  # Add delay before next crawl
                self.print_back_queue_heap()

            else:
                print(f"Host {host} is not ready for crawling yet. Waiting...")
                heapq.heappush(self.back_queue_heap, (next_time, host))  # Reinsert the host for future crawling
                time.sleep(1)  # Optional: add sleep to avoid busy waiting

    def select_biased_front_queue(self):
        """Select a front queue biased towards higher priority queues."""
        # Get a list of available priorities (keys)
        available_priorities = [priority for priority in self.front_queues.keys() if self.front_queues[priority]]

        if not available_priorities:
            return None  # No available front queues

        # Example: Randomly choose from the highest two priorities
        biased_priority = sorted(available_priorities)[:2]  # Get the two highest priorities
        selected_priority = random.choice(biased_priority)  # Randomly select one of the highest priorities
        return selected_priority

    def get_host(self, url):
        """Extract the host from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc if parsed_url.scheme and parsed_url.netloc else None

    def print_front_queues(self):
        """Print the current state of the front queues."""
        print("Current Front Queues:")
        for priority, queue in self.front_queues.items():
            print(f"Priority {priority}: {list(queue)}")

    def print_back_queue_heap(self):
        """Print the current state of the back queue heap."""
        print("Current Back Queue Heap:")
        for entry in self.back_queue_heap:
            print(f" - Host: {entry[1]}, Next available time: {entry[0]}")


if __name__ == "__main__":
    seed_urls = ["https://laesehesten.dk"]
    crawler = WebCrawler(seed_urls)
    crawler.crawl()


# Gem html docs i hver sin fil json.
# kør k-iterationer