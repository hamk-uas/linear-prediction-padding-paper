import os
import requests
import zipfile
import hashlib
from paths_config import paths_config

# A utility function for downloading and extracting the satellite image dataset
def download_and_extract_fairdata_dataset(dataset_permalink, temp_folder, dataset_folder):
    print(f"For manual download, if necessary, the dataset permalink is {dataset_permalink}")
    dataset_preferred_id = dataset_permalink.replace("https://doi.org/", "doi:")
    dataset_metadata_url = f"https://metax.fairdata.fi/rest/v2/datasets?preferred_identifier={dataset_preferred_id}"
    dataset_metadata_response = requests.get(dataset_metadata_url)
    dataset_metadata_response.raise_for_status()
    dataset_metadata = dataset_metadata_response.json()
    description = list(dataset_metadata["research_dataset"]["description"].values())[0]
    dataset_id = dataset_metadata["identifier"]
    mirror_url = list(filter(lambda line: line.startswith("There is a machine-downloadable mirror of the dataset at: "), description.split("\n")))[0].split(": ")[1]
    if mirror_url.endswith("/"):
        mirror_url = mirror_url[:-1]
    print(f"Found mirror {mirror_url}")
    dataset_file_metadata_url = f"https://metax.fairdata.fi/rest/v2/datasets/{dataset_id}/files"
    print(f"Obtaining file list of dataset {dataset_id}")
    dataset_file_metadata_response = requests.get(dataset_file_metadata_url)
    dataset_file_metadata_response.raise_for_status()
    dataset_file_metadata = dataset_file_metadata_response.json()
    file_checksums = {}
    for file_metadata in dataset_file_metadata:
        file_checksums[file_metadata["file_path"].split("/")[-1]] = file_metadata["checksum"]
    if not file_checksums:
        print(f"No files found in dataset {dataset_id}")
        return
    for filename, checksum in file_checksums.items():
        if filename.endswith(".zip"):
            download_url = f"{mirror_url}/{filename}"
            temp_filename = f"{temp_folder}/{filename}"
            print(f"Downloading {temp_filename} from {download_url}")
            download_response = requests.get(download_url, stream=True)
            download_response.raise_for_status()
            h = hashlib.new(checksum["algorithm"].replace("-", "").lower())
            with open(temp_filename, 'wb') as fd:
                for chunk in download_response.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)
                    h.update(chunk)
            if h.hexdigest() == checksum["value"]:
                print("Checksum OK")
            else:
                print(f"Checksum mismatch, found {h.hexdigest()}, expected {checksum["value"]}")
                exit()
            print(f"Extracting {temp_filename} to {dataset_folder}")
            os.makedirs(dataset_folder, exist_ok=True)
            with zipfile.ZipFile(temp_filename, 'r') as zip:
                zip.extractall(dataset_folder)
            print(f"Deleting {temp_filename}")
            os.remove(temp_filename)

if __name__ == "__main__":
    download_and_extract_fairdata_dataset("https://doi.org/10.23729/32a321ac-9012-4f17-a849-a4e7ed6b6c8c", paths_config["temp_folder"], paths_config["dataset_folder"])

