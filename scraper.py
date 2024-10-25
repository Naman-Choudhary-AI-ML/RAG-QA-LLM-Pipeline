import requests
from bs4 import BeautifulSoup
import pdfplumber
import os
from urllib.parse import urljoin, urlparse

if not os.path.exists('scraped_data'):
    os.makedirs('scraped_data')

combined_file = 'scraped_data/combined_content.txt'


def scrape_html(url, file_name):
    """Scrapes the HTML content of a page and saves the paragraphs to a combined file."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        remove_unwanted_elements(soup)

        paragraphs = soup.find_all('p')
        text_content = "\n".join([para.get_text() for para in paragraphs])

        with open(combined_file, 'a', encoding='utf-8') as file:
            file.write(f"\n\nContent from {url}:\n\n")
            file.write(text_content)
        print(f"Data saved for {url}")

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

def scrape_pdf(pdf_url, file_name):
    """Downloads and extracts text from an online PDF."""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        pdf_path = f'scraped_data/{file_name}.pdf'
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"

        with open(combined_file, 'a', encoding='utf-8') as file:
            file.write(f"\n\nContent from PDF {pdf_url}:\n\n")
            file.write(all_text)
        print(f"PDF text saved for {pdf_url}")

    except Exception as e:
        print(f"Failed to scrape PDF {pdf_url}: {e}")


def scrape_subpages(url, domain, level=0):
    """Scrapes all subpages and PDFs from the given URL, limited to one level deep."""
    try:
        if level > 1:
            return  

        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        links = soup.find_all('a', href=True)

        for link in links:
            href = link['href']
            if filter_links(link, domain):  # Filter out unimportant links
                subpage_url = urljoin(url, href)
                file_name = urlparse(subpage_url).path.replace("/", "_").strip("_")
                scrape_html(subpage_url, f"subpage_{file_name}")

        pdf_links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True) if link['href'].endswith('.pdf')]
        for pdf_link in pdf_links:
            file_name = urlparse(pdf_link).path.split('/')[-1].replace('.pdf', '')
            scrape_pdf(pdf_link, file_name)

    except Exception as e:
        print(f"Failed to scrape subpages or PDFs from {url}: {e}")



def filter_links(link, domain):
    """Filters out links that are irrelevant or external based on predefined rules."""
    href = link.get('href', '')

    if any(substring in href for substring in [
        '#cite_note', '#footnote', 'mailto:', 'tel:', 'twitter.com', 'facebook.com', 
        'linkedin.com', 'instagram.com', 'youtube.com', 'pinterest.com', '/Site-Footer/', 
        'oc_lang=', '?oc_lang=', '/Contact-Us', '/Help:', '/wiki/Help:', '/wiki/File:', 
        '/wiki/Category:', '/wiki/Talk:', '/w/index.php', 'javascript:void(0)', 
        '#print', '#share', '#cite', '#feedback', 'maps', 'google.com/maps', 
        'bing.com/maps', 'apple.com/maps', 'yahoo.com/maps']):
        return False

    if '://' in href and domain not in href:
        return False

    return True

def remove_unwanted_elements(soup):
    """Removes unwanted elements like scripts and styles from the BeautifulSoup object."""
    for element in soup(["script", "style"]):
        element.decompose()

def scrape_local_pdf(pdf_path, file_name):
    """Scrapes a local PDF file and extracts text."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"

        with open(combined_file, 'a', encoding='utf-8') as file:
            file.write(f"\n\nContent from local PDF {file_name}:\n\n")
            file.write(all_text)
        print(f"Text extracted and saved from local PDF {file_name}")

    except Exception as e:
        print(f"Failed to scrape local PDF {pdf_path}: {e}")

def scrape_local_pdfs_in_folder(folder_path):
    """Scrapes all local PDFs in the specified folder."""
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            scrape_local_pdf(pdf_path, file_name)


# main code for scraping html and pdf
def main():
    urls = {
    "pittsburgh_wikipedia": "https://en.wikipedia.org/wiki/Pittsburgh",
    "history_pittsburgh_wikipedia": "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "cmu_wikipedia": "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
    "pittsburgh_city_home": "https://pittsburghpa.gov/Home",
    "pittsburgh_city_recycle": "https://www.pittsburghpa.gov/Resident-Services/Trash-Recycling",
    "pittsburgh_city_contact": "https://www.pittsburghpa.gov/Resident-Services/311/Online-Request-Form",
    "encyclopedia_britannica": "https://www.britannica.com/place/Pittsburgh",
    "visit_pittsburgh": "https://www.visitpittsburgh.com/",
    "tax_forms": "https://www.pittsburghpa.gov/City-Government/Finance/Taxes/Tax-Forms",
    "cmu_about": "https://www.cmu.edu/about/",
    "pittsburgh_events": "https://pittsburgh.events/",
    "downtown_pittsburgh_events": "https://downtownpittsburgh.com/events/",
    "pgh_city_paper_events": "https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d",
    "cmu_events": "https://events.cmu.edu/",
    "cmu_alumni_events": "https://www.cmu.edu/engage/alumni/events/campus/index.html",
    "pittsburgh_symphony": "https://www.pittsburghsymphony.org/",
    "pittsburgh_opera": "https://pittsburghopera.org/",
    "trust_arts": "https://trustarts.org/",
    "carnegie_museums": "https://carnegiemuseums.org/",
    "heinz_history_center": "https://www.heinzhistorycenter.org/",
    "frick_pittsburgh": "https://www.thefrickpittsburgh.org/",
    "list_of_museums_pittsburgh": "https://en.wikipedia.org/wiki/List_of_museums_in_Pittsburgh",
    "picklesburgh": "https://www.picklesburgh.com/",
    "pgh_taco_fest": "https://www.pghtacofest.com/",
    "pittsburgh_restaurant_week": "https://pittsburghrestaurantweek.com/",
    "little_italy_days": "https://littleitalydays.com/",
    "banana_split_fest": "https://bananasplitfest.com/",
    "visit_pittsburgh_sports": "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
    "mlb_pirates": "https://www.mlb.com/pirates",
    "steelers": "https://www.steelers.com/",
    "penguins": "https://www.nhl.com/penguins"
    }

    if os.path.exists(combined_file):
        with open(combined_file, 'w', encoding='utf-8') as file:
            file.write("")

    for name, url in urls.items():
        print(f"Starting scraping for: {name}")
        
        scrape_html(url, name)
        
        domain = urlparse(url).netloc
        
        scrape_subpages(url, domain, level=1)


if __name__ == "__main__":
    main()

