from urllib.parse import urljoin, urlparse

import helper

# I use a disallowed list instead of allowed, since i assume that the list of disallowed paths is shorter than the list of allowed paths


def parse_robots_txt(robots_txt, user_agent=None):
    """Parse the robots.txt file and extract disallowed paths for all user-agents."""
    disallowed_paths = {}
    current_user_agent = None

    rules = robots_txt.splitlines()
    for rule in rules:
        rule = rule.strip()
        if rule.startswith("User-agent:"):
            # Update the current user-agent
            current_user_agent = rule.split(":")[1].strip()
            # Initialize the disallowed paths list for this user-agent
            disallowed_paths[current_user_agent] = []
        elif rule.startswith("Disallow:") and current_user_agent:
            path = rule.split(":")[1].strip()
            # Add disallowed path for the current user-agent if path is not empty
            if path:
                disallowed_paths[current_user_agent].append(path)

    if user_agent:
        return disallowed_paths[user_agent]

    # If no user-agent is provided, return disallowed paths for all user-agents
    return disallowed_paths


def is_allowed(url, sites):
    # Removed the user-agent since we filter by agent in crawler init
    # Split at the first '/', and get the second part
    stripped_url = '/' + url.split('/', 1)[1]
    if stripped_url not in sites:
        return True
    return False


# def get_host(url):
#     """Extract the host from a URL."""
#     parsed_url = urlparse(url)
#     return parsed_url.netloc if parsed_url.scheme and parsed_url.netloc else None

# def is_allowed(url, sites):
#     """Check if the URL is allowed by drdk_robots.txt rules."""
#
#     host = get_host(url)
#     for path in sites["*"]:
#         # Use urlparse to match against the path correctly
#         if url.startswith(urljoin(url, path)):
#             return False
#     return True


if __name__ == "__main__":
    robots_txt = helper.read_text_file("../lib/laesehesten_robots.txt")
    disallowed_sites = parse_robots_txt(robots_txt)
    for agent, paths in disallowed_sites.items():

        print(f"User-agent: {agent}")
        for path in paths:
            print(f"  - {path}")

    user_agent = "*"

    url = "https://laesehesten.dk/bin/"
    is_allowed(url, disallowed_sites)
    # print(f"Is {url} allowed?: {is_allowed(url, disallowed_sites)}")
    # print(f"Is {url} allowed?: {is_allowed(url, disallowed_sites)}")



