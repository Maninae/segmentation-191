
import requests

###################
# Code from:
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

# Credit to Andrew Hundt's code, to download from GDrive with a shareable link.

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

####################

if __name__ == "__main__":
    print("Usage: `python3 download_diamondback_weights.py`")
    # URL: https://drive.google.com/open?id=1Ll-F5VdG_TCx30uJ_S4hPE4XM-EbBJf4
    # ID from shareable link (The alphanumeric_- hash)
    file_id = "1Ll-F5VdG_TCx30uJ_S4hPE4XM-EbBJf4"

    # DESTINATION FILE ON YOUR DISK
    destination = 'diamondback_ep11-tloss=34423.1392-vloss=44111.4496-tIOU=0.7913-vIOU=0.7554.h5'
    
    download_file_from_google_drive(file_id, destination)