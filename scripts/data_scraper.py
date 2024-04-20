import os
import random
import requests
from bs4 import BeautifulSoup
import glob
import scipdf
import spacy
import json
import pandas as pd
import os
import argparse

def scrape_articles(data_dir, acl_links):
    '''
    Function to scrape ACL articles and create a data directory to store the PDFs and TXTs
    Params:
    - data_dir (str): The directory in which all files should be stored
    - acl_links (Dict[str]): a dictionary with key as conference name and value as URL of collection
    '''
    # utils
    os.system(f"mkdir {data_dir}/pdfs/")
    os.system(f"mkdir {data_dir}/jsons/")
    random.seed(42)

    # scraping using bs4
    pdf_links = []

    for id, page in acl_links.items():
        print('id',id, '  page  ', page)
        reqs = requests.get(page)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        div = soup.find('div', {'id': id})
        links = div.find_all('a')

        # Extract and print the href attribute (the URL) from each 'a' tag
        conf_links = [link['href'] for link in links if link['href'].endswith('.pdf')]
        conf_links = conf_links[1:11]
        print(f"PDFs from {id}: {conf_links}")
        pdf_links.extend(conf_links)

        os.system(f"mkdir {data_dir}/pdfs/{id}")
        os.system(f"mkdir {data_dir}/txts/{id}")
        for i, pdf_url in enumerate(conf_links):
            name = f"{data_dir}/pdfs/{id}/" + pdf_url.split('/')[-1]
            os.system("wget -O {} {}".format(name, pdf_url))
    print(f'Data has been scraped successfully to {data_dir} folder')

def exract_article_sections(grobid_URL, data_dir):
    '''
    Function to extract sections of articles and store them as JSONs and CSV files.

    Params:
    - local_URL (str): The GROBID server URL 
    - data_dir (str): The directory in which all articles have been stored
    '''
    # Set the GROBID server URL
    scipdf.grobid_url = grobid_URL

    # Get all subdirectories in data/pdfs
    subdirectories = [x[0] for x in os.walk(f'{data_dir}/pdfs')][1:]  # skip the root directory

    for subdir in subdirectories:
        pdfs = glob.glob(f"{subdir}/**/*.pdf", recursive=True)
        print(f"Processing {len(pdfs)} PDFs in {subdir}...")

        data = []  # Prepare to collect data for DataFrame

        for pdf in pdfs:
            print(f"================================================================================")
            print(f"Parsing {pdf}...")
            article_dict = scipdf.parse_pdf_to_dict(pdf, as_list=True)
            print("Title:", article_dict['title'])
            print("Authors:", article_dict['authors'])
            print("Abstract:", article_dict['abstract'])
            print("Pub_date:", article_dict['pub_date'])

            
            # Extract necessary details
            sections_list = [section['heading'] for section in article_dict['sections']]
            data.append({
                'title': article_dict['title'],
                'authors': article_dict['authors'],
                'abstract': article_dict['abstract'],
                'pub_date': article_dict['pub_date'],
                'sections_list': sections_list
            })

            # Determine JSON path based on PDF path
            json_subdir = subdir.replace('pdfs', 'jsons')  # Replace pdfs with json in the path
            json_path = os.path.join(json_subdir, os.path.basename(pdf).replace('.pdf', '.json'))
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(article_dict, f, indent=4)

        # Create DataFrame for current subdirectory and save to CSV
        df = pd.DataFrame(data)
        subdir_name = os.path.basename(subdir)  # Get the name of the subdirectory
        csv_path = os.path.join(data_dir, f"{subdir_name}_article_data.csv")  # Save CSV in the data folder
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print("DataFrame saved to CSV:", csv_path)

if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data directory path')
    parser.add_argument('grobid_server', type='str', help='The GROBID server URL')
    args = parser.parse_args()
    data_dir = args.data_dir
    grobid_URL = args.grobid_server

    # Scrape articles
    acl_links = {
        '2023acl-long': "https://aclanthology.org/events/acl-2023/",
        '2023emnlp-main': "https://aclanthology.org/events/emnlp-2023/",
        '2023conll-1': "https://aclanthology.org/events/conll-2023/"
    }
    scrape_articles(data_dir=data_dir, acl_links=acl_links)

    # Extract article information
    exract_article_sections(grobid_URL=grobid_URL, data_dir=data_dir)