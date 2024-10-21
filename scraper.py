import requests
from bs4 import BeautifulSoup
import pdfplumber
import os
from urllib.parse import urljoin, urlparse

# Create a directory to store scraped data
if not os.path.exists('scraped_data'):
    os.makedirs('scraped_data')

# File to store the combined content
combined_file = 'scraped_data/combined_content.txt'

# Function to scrape HTML pages
def scrape_html(url, file_name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main content from the page (modify as per page structure)
        paragraphs = soup.find_all('p')
        text_content = "\n".join([para.get_text() for para in paragraphs])

        # Combine all scraped content into one file
        with open(combined_file, 'a', encoding='utf-8') as file:
            file.write(f"\n\nContent from {url}:\n\n")
            file.write(text_content)
        print(f"Data saved for {url}")

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

# Function to extract text from PDF
def scrape_pdf(pdf_url, file_name):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Save PDF locally before reading
        pdf_path = f'scraped_data/{file_name}.pdf'
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text()

        # Combine all scraped content into one file
        with open(combined_file, 'a', encoding='utf-8') as file:
            file.write(f"\n\nContent from PDF {pdf_url}:\n\n")
            file.write(all_text)
        print(f"PDF text saved for {pdf_url}")

    except Exception as e:
        print(f"Failed to scrape PDF {pdf_url}: {e}")

# Function to filter important links (Wikipedia, etc.)
def filter_links(link):
    # Example filtering rules for Wikipedia
    href = link.get('href', '')

    # Skip citation, footnote, and external links
    if any(substring in href for substring in ['#cite_note', '#footnote', 'wikipedia.org/wiki/Help', '/w/index.php']):
        return False

    # Skip external links (outside of the domain)
    if '://' in href and 'wikipedia.org' not in href:
        return False

    return True

# Function to find and scrape subpages (limited to 10 subpages per URL)
def scrape_subpages(url, domain, level=0, subpage_limit=1):
    try:
        if level > 1:
            return  # Stop after one level of subpages

        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all internal links (subpages), apply filter
        links = soup.find_all('a', href=True)
        subpage_count = 0

        for link in links:
            if subpage_count >= subpage_limit:
                break  # Stop scraping after reaching the limit of 10 subpages
            if filter_links(link):  # Filter out unimportant links
                href = link['href']
                # Only consider internal links within the same domain
                if domain in href or href.startswith('/'):
                    subpage_url = urljoin(url, href)
                    file_name = urlparse(subpage_url).path.replace("/", "_").strip("_")
                    scrape_html(subpage_url, f"subpage_{file_name}")
                    subpage_count += 1
        
        # Find PDF links
        pdf_links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True) if link['href'].endswith('.pdf')]
        for pdf_link in pdf_links:
            if subpage_count >= subpage_limit:
                break  # Stop if the limit is reached
            file_name = urlparse(pdf_link).path.replace("/", "_").strip("_")
            scrape_pdf(pdf_link, file_name)
            subpage_count += 1

    except Exception as e:
        print(f"Failed to scrape subpages or PDFs from {url}: {e}")

# Main function to start scraping with subpages (limited to 1 subpages)
def main():
    # Example URLs for general info and history (expand this with subpages and more sources)
    urls = {
        "pittsburgh_wikipedia": "https://en.wikipedia.org/wiki/Pittsburgh",
        "history_pittsburgh_wikipedia": "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
        "cmu_wikipedia": "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
        "pittsburgh_city": "https://pittsburghpa.gov/index.html",
        "encyclopedia_britannica": "https://www.britannica.com/place/Pittsburgh"
    }

    # Clear the combined file before starting the scraping
    with open(combined_file, 'w', encoding='utf-8') as file:
        file.write("")

    for name, url in urls.items():
        # Scrape the main page
        scrape_html(url, name)

        # Automatically scrape subpages and PDFs within the same domain, limit to 1 subpages
        domain = urlparse(url).netloc
        scrape_subpages(url, domain, level=1, subpage_limit=5)  # Limit to 1 subpages

if __name__ == "__main__":
    main()
