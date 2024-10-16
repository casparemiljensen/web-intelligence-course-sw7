from collections import defaultdict, deque
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json  # Import the json module
import os  # Import os for file handling
import robots


# Tasks
# change from allowed to disallowed
# perhaps empty disallowed list/robots.txt when changing host?
# When we see a new host, init robots.txt
# How do we actually know if we are allowed to crawl the seed url? We start by fetching robots.txt. from the host


class WebCrawler:
    def __init__(self, seed_urls, respect_robots=True, same_domain_only=True, user_agent="*"):
        self.seed_urls = seed_urls
        self.respect_robots = respect_robots  # Option to respect robots.txt
        self.same_domain_only = same_domain_only  # Option to restrict to same domain
        self.user_agent = user_agent
        self.index = set()  # Set to keep track of indexed URLs
        self.crawled = set()  # Set to keep track of crawled URLs
        self.disallowed_paths = {}  # Set to keep disallowed paths from robots.txt
        self.frontier = deque(seed_urls)  # Queue of URLs to crawl
        self.seed_host = self.get_host(seed_urls[0])  # Store the host of the seed URL
        self.initialize_robots()  # Initialize robots.txt rules

    def initialize_robots(self):
        """Fetch and parse robots.txt if respecting robots is enabled."""
        if self.respect_robots:
            robots_url = urljoin(self.seed_urls[0], "/robots.txt")
            response = requests.get(robots_url)
            if response.status_code == 200:
                self.disallowed_paths[self.get_host(robots_url)] = robots.parse_robots_txt(response.text, self.user_agent)

    # def parse_robots_txt(self, robots_txt):
    #     """Parse the robots.txt file and extract allowed paths."""
    #     user_agent = "*"
    #     rules = robots_txt.splitlines()
    #     for rule in rules:
    #         rule = rule.strip()
    #         if rule.startswith("User-agent:"):
    #             user_agent = rule.split(":")[1].strip()
    #         elif rule.startswith("Disallow:") and user_agent == "*":
    #             path = rule.split(":")[1].strip()
    #             self.allowed_paths.add(path)

    def is_allowed(self, url):
        """Check if the URL is allowed by robots.txt rules."""

        host = self.get_host(url)

        if not self.respect_robots:
            return True

        if host not in self.disallowed_paths:
            try:
                self.disallowed_paths[host] = robots.parse_robots_txt(requests.get(urljoin(url, "/robots.txt")).text, self.user_agent)
            except:
                print("Failed to fetch robots.txt")
                return True
        else:
            for path in self.disallowed_paths[host]:
                # Use urlparse to match against the path correctly
                if url.startswith(urljoin(url, path)):
                    return False
            return True

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
            if not (href.startswith('mailto:') or href.startswith('tel:')):  # Ignore mailto and tel links
                full_url = urljoin(url, href)  # Resolve relative URLs
                links.add(self.normalize_url(full_url))  # Normalize and add to links set
        return text, links

    def is_same_domain(self, url):
        """Check if the URL belongs to the same domain as the seed."""
        host = self.get_host(url)
        return host == self.seed_host

    def save_document_to_json(self, url, text, links):
        """Save the crawled document to a JSON file in a specified project directory."""
        # Ensure the output directory exists
        hostname = self.get_host(url)
        directory = os.path.join("crawled_pages", hostname)  # Create a subdirectory for the hostname

        # Ensure the output directory exists
        os.makedirs(directory, exist_ok=True)
        # Generate a filename based on the URL
        filename = self.get_filename_from_url(url)
        filepath = os.path.join(directory, filename)  # Combine output directory and filename

        data = {
            'url': url,
            'text': text,
            'links': list(links)
        }

        with open(filepath, 'w') as json_file:
            json.dump(data, json_file, indent=4)  # Save as JSON

    def get_filename_from_url(self, url):
        """Generate a filename from the URL."""
        parsed_url = urlparse(url)
        safe_url = parsed_url.netloc + parsed_url.path.replace('/', '_').replace(':',
                                                                                 '-')  # Replace / with _ and : with -
        return f"{safe_url}.json"

    def crawl(self):
        """Begin crawling process."""
        while self.frontier:
            url = self.frontier.popleft()  # Fetch the next URL from the frontier
            if url in self.crawled:
                continue

            # if self.respect_robots:
            #     if(self.crawled)
            # print("parsing robots.txt")

            print(f"Crawling: {url}")
            content = self.fetch_page(url)

            if content:
                self.index.add(url)
                self.crawled.add(url)

                # Parse and extract links

                text, extracted_links = self.parse_page(url, content)

                # Save the document to a JSON file
                self.save_document_to_json(url, text, extracted_links)

                for link in extracted_links:
                    if link not in self.crawled and self.is_allowed(link):
                        if not self.same_domain_only or self.is_same_domain(link):
                            print(f"Adding {link} to frontier")
                            self.frontier.append(link)  # Add to frontier for further crawling
                            self.index.add(link)  # Optionally add to index as well

                print(f"Crawled: {url}")
            else:
                print(f"Failed to fetch {url}, marking as crawled.")
                self.crawled.add(url)

    def get_host(self, url):
        """Extract the host from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc if parsed_url.scheme and parsed_url.netloc else None


if __name__ == "__main__":
    seed_urls = ["https://laesehesten.dk/"]  # Change to your desired seed URL
    crawler = WebCrawler(seed_urls, respect_robots=True, same_domain_only=True, user_agent="*")
    crawler.crawl()

    for agent, paths in crawler.disallowed_paths.items():

        print(f"Host: {agent}")
        for path in paths:
            print(f"  - {path}")

    user_agent = "*"
