import os, time, requests, binascii
from bird_db import URLDatabase

# Set the User-Agent header
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

def google_image_urls(query, api_key, cx, num_images, exclude_urls=None):
    """ Returns a list of URLs of images relating to the query, excluding specified URLs.
        For this function to work, a Google API key is required that has the Custom Search API enabled.
        It also needs a Programmable Custom Search Engine ID that can access images.

    Args:
        query (str): Google Search Query to find images for.
        api_key (str): API key that has Custom Search API enabled.
        cx (str): Programmable Custom Google Search Engine ID.
        num_images (int): Number of images to retrieve.
        exclude_urls (list, optional): List of URLs to exclude from the results. Defaults to None.

    Raises:
        Exception: Raises an exception when a request cannot be made in 5 minutes.

    Returns:
        list: List of image URLs.
    """
    url = f"https://www.googleapis.com/customsearch/v1?searchType=image&key={api_key}&cx={cx}&q={query}"

    error_count = 0
    image_urls = []
    while len(image_urls) < num_images:
        try:
            response = requests.get(url, headers=HEADERS, params={'start': len(image_urls)+1})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            if response.status_code == 429:
                wait_time = 2 ** error_count
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                error_count += 1
                if error_count > 5:
                    raise Exception("Wait until midnight PST for your Google API requests quota to reset.")
            else:
                return []
            
        # Get the JSON response data and extract the image URLs
        json_data = response.json()
        items = json_data.get('items', [])
        new_urls = [item['link'] for item in items if item['link'] not in (exclude_urls or [])]
        image_urls += new_urls
        if 'nextPage' not in json_data['queries']:
            break  # Exit the loop if there are no more pages
    return image_urls[:num_images]

def download_urls(image_urls, save_directory, n=10, name = "image",delay=7):
    """ Downloads the first n-number of images in the url list to the given directory.
    
        SAVE FORMAT:
            {n}__{64 bit value of the URL}.jpg
        
    Args:
        image_urls (lst) : A list of image URLs from a Google request
        save_directory (str): Path to a directory to save the JPG images
        n (int, optional): Number of images to be downloaded. Defaults to 10.
        delay (int, optional): Requests per two seconds. Defaults to 7.
        
    Returns:
        int : number of saved images
    """  
    # Save the specified number of images to the specified directory
    os.makedirs(save_directory, exist_ok=True)
    saved_images = []
    for i, image_url in enumerate(image_urls):
        if len(saved_images) >= n:
            break
        try:
            response = requests.get(image_url, headers=HEADERS)
            response.raise_for_status()
            file_path = os.path.join(save_directory, f"{len(saved_images)+1}__{name}.jpg")
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
    Returns:
        lst: list of the saved images
    """ 
    db = URLDatabase(db_name)
    image_urls = []
    if name != "image":
        image_urls = db.query_urls(name, n)
    if len(image_urls) < n:
        image_urls = image_urls + google_image_urls(query, api_key, cx,n-len(image_urls), image_urls)
    saved_urls = download_urls(image_urls, save_directory, n, name, delay)
    if name != "image":
        db.add_urls_by_name(name, saved_urls)
    return saved_urls