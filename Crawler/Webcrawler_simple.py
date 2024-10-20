import re
from collections import defaultdict, deque
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json  # Import the json module
import os  # Import os for file handling
from Crawler import robots

# Tasks
# change from allowed to disallowed
# perhaps empty disallowed list/robots.txt when changing host?
# When we see a new host, init robots.txt
# How do we actually know if we are allowed to crawl the seed url? We start by fetching robots.txt. from the host
# How do we handle cookie consent? We need to click the button to accept cookies somehow???


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../"))


class WebCrawler:
    def __init__(self, seed_urls, respect_robots=True, same_domain_only=True, user_agent="*", verbose=False):
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
        self.verbose = verbose

    def initialize_robots(self):
        """Fetch and parse robots.txt if respecting robots is enabled."""
        if self.respect_robots:
            robots_url = urljoin(self.seed_urls[0], "/robots.txt")
            response = requests.get(robots_url)
            if response.status_code == 200:
                self.disallowed_paths[self.get_host(robots_url)] = robots.parse_robots_txt(response.text,
                                                                                           self.user_agent)

    def is_allowed(self, url):
        """Check if the URL is allowed by robots.txt rules."""

        host = self.get_host(url)

        if not self.respect_robots:
            return True

        if host not in self.disallowed_paths:
            try:
                self.disallowed_paths[host] = robots.parse_robots_txt(requests.get(urljoin(url, "/robots.txt")).text,
                                                                      self.user_agent)
            except:
                print("Failed to fetch robots.txt") if self.verbose else None
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
            print(f"Failed to fetch {url}: {e}") if self.verbose else None
            return None

    def parse_page(self, url, content):
        """Parse the page content and extract text and relevant links."""
        if not content:
            return "", set()  # If content is None or empty, return early

        # Ensure the content is cleaned and properly decoded before parsing
        cleaned_content = self.clean_text(content)

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(cleaned_content, 'html.parser')

        # Extract all text from the page
        text = soup.get_text()

        # Extract and normalize links, filtering out non-HTML files (e.g., images, scripts)
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Ignore mailto, tel, and file types like images, PDFs, etc.
            if href.startswith(('mailto:', 'tel:')):
                continue  # Skip mailto and telephone links

            # Only allow URLs that are likely to be HTML pages (ignores images, videos, etc.)
            if not href.endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.pdf', '.mp4', '.mp3', '.avi', '.mov', '.zip')):
                full_url = urljoin(url, href)  # Resolve relative URLs
                links.add(self.normalize_url(full_url))  # Normalize and add to links set

        return text, links

    def is_same_domain(self, url):
        """Check if the URL belongs to the same domain as the seed."""
        host = self.get_host(url)
        return host == self.seed_host

    def clean_text(self, text):
        """Clean up Unicode escape sequences and normalize text."""
        try:
            text = text.encode('utf-8').decode('unicode_escape')  # Normalize Unicode characters
        except UnicodeDecodeError:
            pass

        # Replace multiple consecutive newlines (\n) with a single newline
        # text = re.sub(r'\n+', '\n', text)
        text = " ".join(l.strip() for l in text.split("\n"))

        text = text.strip()

        return text  # In case the decoding fails, return the original text

    def save_document_to_json(self, url, text, links):
        """Save the crawled document to a JSON file with an incrementing integer ID in the specified directory."""

        # Get the hostname to create a subdirectory for the hostname
        hostname = self.get_host(url)

        directory = os.path.join(PROJECT_ROOT, "lib/crawled_pages", hostname)

        # Ensure the output directory exists
        os.makedirs(directory, exist_ok=True)

        # Get the list of existing JSON files in the directory to determine the next available ID
        existing_files = [f for f in os.listdir(directory) if f.endswith(".json")]

        # Determine the next ID by checking the number of files already present
        next_id = len(existing_files) + 1

        # Generate a filename using the next available ID
        filename = f"doc{next_id}.json"
        filepath = os.path.join(directory, filename)  # Combine output directory and filename

        # Prepare the data to be saved (simplified to include just URL and text)
        data = {
            'url': url,
            'text': text
        }

        # Save the data to a JSON file
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        print(f"Saved document to {filepath}")

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

            print(f"Crawling: {url}") if self.verbose else None
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
                            print(f"Adding {link} to frontier") if self.verbose else None
                            self.frontier.append(link)  # Add to frontier for further crawling
                            self.index.add(link)  # Optionally add to index as well

                print(f"Crawled: {url}") if self.verbose else None
            else:
                print(f"Failed to fetch {url}, marking as crawled.") if self.verbose else None
                self.crawled.add(url)

    def get_host(self, url):
        """Extract the host from a URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc if parsed_url.scheme and parsed_url.netloc else None


if __name__ == "__main__":
    seed_urls = ["https://laesehesten.dk/"]  # Change to your desired seed URL
    crawler = WebCrawler(seed_urls, respect_robots=True, same_domain_only=True, user_agent="*", verbose=True)
    crawler.crawl()

    for agent, paths in crawler.disallowed_paths.items():

        print(f"Host: {agent}") if crawler.verbose else None
        for path in paths:
            print(f"  - {path}")

    user_agent = "*"
