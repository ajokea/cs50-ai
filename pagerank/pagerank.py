import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    num_pages = len(corpus)

    if not corpus[page]:
        for pg in corpus:
            distribution[pg] = 1 / num_pages
    else:
        num_links = len(corpus[page])
        for pg in corpus:
            distribution[pg] = (1 - damping_factor) / num_pages
            if pg in corpus[page]:
                distribution[pg] += damping_factor / num_links
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = random.choice(list(corpus.keys()))
    counts = {p: 0 for p in corpus}

    for _ in range(n):
        counts[page] += 1

        sample = transition_model(corpus, page, damping_factor)

        page = random.choices(list(sample.keys()), list(sample.values()))[0]

    return {p: counts[p] / n for p in corpus}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # handle pages with no outgoing links
    for page in corpus:
        if not corpus[page]:
            corpus[page] = {p for p in corpus}

    # find incoming links
    incoming_links = {}
    for page, links in corpus.items():
        if links:
            for link in links:
                incoming_links[link] = incoming_links.get(link, set()).union({page})

    num_pages = len(corpus)
    pr = {page: 1 / num_pages for page in corpus}

    delta = 0.001
    flag = True
    while flag:
        flag = False

        new_pr = {}
        for p in corpus:
            new_pr[p] = (1 - damping_factor) / num_pages

            for i in incoming_links.get(p, set()):
                new_pr[p] += damping_factor * (pr[i] / (len(corpus[i]) if corpus[i] else num_pages))

            if abs(new_pr[p] - pr[p]) > delta:
                flag = True

        pr = new_pr

    return pr


if __name__ == "__main__":
    main()
