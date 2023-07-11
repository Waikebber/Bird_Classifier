import os, time, requests

def google_image_download(query, save_directory, api_key, cx, n=10, name="image", delay=7):
    """ Downloads the first n-number of images regarding a query to the given directory.
        For this function to work, a google api key is required that has the Custom Search API enabled.
            It also needs a Programmable Custom Search Engine ID that can access images.

    Args:
        query (str): Google Search Query to find images for
        save_directory (str): Path to a directory to save the JPG images
        api_key (str): API key that has Custom Search API enabled
        cx (str): Programmable Custom Google Search Engine ID
        n (int, optional): Number of images to be downloaded. Defaults to 10.
        name (str, optional): Name of the queried images. Defaults to "image".
        delay (int, optional): Requests per two seconds. Defaults to 7.
    
    Raises:
        Exception: Raises an exception when a request cannot be made in 5 min
        
    Returns:
        int : number of saved images
    """  
    # Create the Google Custom Search API URL
    url = f"https://www.googleapis.com/customsearch/v1?searchType=image&key={api_key}&cx={cx}&q={query}"

    # Set the User-Agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }

    # Send a GET request to the API URL with the User-Agent header
    error_count = 0
    while True:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            break  # Exit the loop if the request is successful
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            if response.status_code == 429:
                # Handle rate limit exceeded error by waiting for a minute
                print("Rate limit exceeded. Retrying in 1 minute...")
                time.sleep(60)
                error_count +=1
            if error_count > 5:
                raise Exception("Wait until midnight PST for your Google API requests quota to reset.")

    # Get the JSON response data
    json_data = response.json()

    # Extract the image URLs from the JSON response
    image_urls = [item['link'] for item in json_data['items']]

    # Save the specified number of images to the specified directory
    os.makedirs(save_directory, exist_ok=True)
    saved_images = 0
    for i, image_url in enumerate(image_urls):
        if saved_images >= n:
            break
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            file_path = os.path.join(save_directory, f"{name}{saved_images+1}.jpg")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            if (i + 1) % delay == 0: 
                time.sleep(2)
            saved_images += 1
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                print(f"Error: {str(e)}")
                wait_time = delay ** saved_images  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    return saved_images
