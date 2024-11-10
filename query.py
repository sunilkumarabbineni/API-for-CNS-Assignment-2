import requests
from xml.etree import ElementTree as ET

# Function to fetch context from PubMed using E-utilities
def fetch_context_from_pubmed(query):
    # E-utilities API endpoint for searching PubMed
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # Parameters for the search request
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "api_key": "5e6702c9d9290e07dd5160f291a30af93b08"
    }
    
    # Perform the search request
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    
    # Extract the list of PubMed IDs (PMIDs)
    pmids = search_data.get("esearchresult", {}).get("idlist", [])
    
    if not pmids:
        return None
    
    # Parameters for the fetch request
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": "5e6702c9d9290e07dd5160f291a30af93b08"
    }
    
    # Perform the fetch request
    fetch_response = requests.get(fetch_url, params=fetch_params)
    
    # Parse the XML response to extract abstracts
    root = ET.fromstring(fetch_response.content)
    abstracts = []
    for article in root.findall(".//AbstractText"):
        if article.text:
            abstracts.append(article.text)
    
    # Combine abstracts as context, ensuring all items are strings
    context = " ".join([abstract for abstract in abstracts if abstract])
    return context

# Test the function with a sample query
query = "DNA"
response = fetch_context_from_pubmed(query)

# Print the response to check in Postman
print(response)
