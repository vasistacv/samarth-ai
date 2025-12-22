import requests
import os

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)

# --- THE NEW, RELIABLE, AND CORRECT LINKS ---

DIRECT_LINKS = {
    "crop_production": {
        "name": "District-wise Crop Production",
        "url": "https://raw.githubusercontent.com/srinivas-com/Data-Science-Projects/master/Crop_production_data/crop_production_data.csv",
        "output_file": "data/raw/crop_production.csv"
    },
    "district_rainfall": {
        "name": "District-wise Rainfall Normal",
        "url": "https://raw.githubusercontent.com/airwarriorg/rainfall-analysis/master/district_rainfall_normal.csv",
        "output_file": "data/raw/district_rainfall.csv"
    }
}

def download_file(name, url, output_file):
    """Downloads a file from a direct, stable URL."""
    print(f"Downloading: {name}...")
    print(f"  From: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win66; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        res_data = requests.get(url, headers=headers)
        res_data.raise_for_status() # Check for any download errors
        
        with open(output_file, 'wb') as f:
            f.write(res_data.content)
            
        print(f"  Successfully downloaded to {output_file}\n")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"  HTTP Error: {http_err} - Could not download the file. Check the URL.\n")
    except requests.exceptions.RequestException as e:
        print(f"  Download Error: {e}\n")

if __name__ == "__main__":
    print("Starting data download from new, reliable, DISTRICT-LEVEL sources...")
    for key, info in DIRECT_LINKS.items():
        download_file(info["name"], info["url"], info["output_file"])
    print("All downloads complete.")