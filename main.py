import os.path

import helper
from Crawler.Webcrawler_simple import WebCrawler
import Indexer.Indexer as indexer


def invoke():
    seed_urls = ["https://laesehesten.dk"]  # Change to your desired seed URL
    print("Crawling...")
    crawler = WebCrawler(seed_urls, respect_robots=True, same_domain_only=True, user_agent="*", verbose=False)
    crawler.crawl()

    if helper.is_dir_empty(os.path.abspath("lib/crawled_pages")) == 0:
        print("No crawled pages...")
        return

    print(f"Crawling successful...")
    print("Beginning indexing...")
    indexer.run_indexer()

    if helper.is_dir_empty("lib/index_data") == 0:
        print("No index data...")
        return

    query = "dog AND doubl and drivetrain"
    print(f"Processing query({query})...")
    result = indexer.eval_query(query)
    print(f"Result: {result}")


if __name__ == "__main__":
    invoke()
