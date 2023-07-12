import os, time, requests
from bird_db import URLDatabase

# Set the User-Agent header
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

def google_image_urls(query, api_key, cx, delay = 7):
    """ Returns a list of URLs of images relating to the query
        For this function to work, a google api key is required that has the Custom Search API enabled.
            It also needs a Programmable Custom Search Engine ID that can access images.

    Args:
        query (str): Google Search Query to find images for
        api_key (str): API key that has Custom Search API enabled
        cx (str): Programmable Custom Google Search Engine ID
        delay (int, optional): Requests per two seconds. Defaults to 7.
    
    Raises:
        Exception: Raises an exception when a request cannot be made in 5 min
        
    Returns:
        lst : lst of image urls
    """  
    # Create the Google Custom Search API URL
    url = f"https://www.googleapis.com/customsearch/v1?searchType=image&key={api_key}&cx={cx}&q={query}"

    # Send a GET request to the API URL with the User-Agent header
    error_count = 0
    while True:
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            break  # Exit the loop if the request is successful
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            error_count +=1
            if error_count > 3:
                raise Exception("Wait until midnight PST for your Google API requests quota to reset.")
            if response.status_code == 429:
                # Handle rate limit exceeded error by waiting for a minute
                wait_time = delay * (2 ** error_count)
                print(f"Rate limit exceeded. Retrying in {wait_time} ...")
                time.sleep(wait_time)

    # Get the JSON response data/ Extract the image URLs from the JSON response
    json_data = response.json()
    image_urls = [item['link'] for item in json_data['items']]
    return image_urls

def download_urls(image_urls, save_directory, n=10, name="image", delay=7):
    """ Downloads the first n-number of images in the url list to the given directory.

    Args:
        image_urls (lst) : A list of image URLs from a Google request
        save_directory (str): Path to a directory to save the JPG images
        n (int, optional): Number of images to be downloaded. Defaults to 10.
        name (str, optional): Name of the queried images. Defaults to "image".
        delay (int, optional): Requests per two seconds. Defaults to 7.
        
    Returns:
        int : number of saved images
    """  
    # Save the specified number of images to the specified directory
    os.makedirs(save_directory, exist_ok=True)
    saved_images = []
    for i, image_url in enumerate(image_urls):
        if saved_images >= n:
            break
        try:
            response = requests.get(image_url, headers=HEADERS)
            response.raise_for_status()
            file_path = os.path.join(save_directory, f"{name}{len(saved_images)+1}.jpg")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            if (i + 1) % delay == 0: 
                time.sleep(2)
            saved_images.append(image_url)
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                print(f"Error: {str(e)}")
                wait_time = delay ** len(saved_images)  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return saved_images

def google_image_download(query, save_directory, api_key, cx, n=10, name="image", delay=7, db_name = ".\\bird_im_urls.db"):
    """ Downloads n-number of images to the given directory. Images are sourced from a goolge image query.
            Images are saved with the given name.
            After finding the urls, they are stored in the given database. This database can then be used to source the urls instead.
            In 2 seconds, the delay number of images will be downloaded.
            
        !! For this function to work, a google api key is required that has the Custom Search API enabled.
        !! It also needs a Programmable Custom Search Engine ID that can access images.
    Args:
        query (str): Google Search Query to find images for
        save_directory (str): Path to a directory to save the JPG images
        api_key (str): API key that has Custom Search API enabled
        cx (str): Programmable Custom Google Search Engine ID
        n (int, optional): Number of images to be downloaded. Defaults to 10.
        name (str, optional): Name of the queried images. Defaults to "image".
        delay (int, optional): Requests per two seconds. Defaults to 7.
        db_name (str, optional): Database path to store saved image urls. Defaults to ".\\bird_im_urls.db".
    """    
    db = URLDatabase(db_name)
    image_urls = []
    if name != "image":
        image_urls = db.query_urls(name, n)
    if len(image_urls) < n:
        image_urls = image_urls + google_image_urls(query, api_key, cx, delay)
    saved_urls = download_urls(image_urls, save_directory, n, name, delay)
    if name != "image":
        db.add_urls_by_name(name, saved_urls)