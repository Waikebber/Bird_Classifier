import sqlite3

class URLDatabase():
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_tables(self):
        """Creates the name and url tables
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                tag_id INTEGER,
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')

    def insert_data(self, data):
        """Inserts new data into the database

        Args:
            data (lst): A list of dictionaries. Each entry in the dictionary is a 'name' tag that correlates to a its urls.
                            The 'urls' tag is a list of strings.
        """        
        self.connect()

        for item in data:
            self.cursor.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)', (item['name'],))
            tag_id = self.cursor.lastrowid

            for url in item['urls']:
                self.cursor.execute('INSERT OR IGNORE INTO urls (url, tag_id) VALUES (?, ?)', (url, tag_id))

        self.close()

    def query_urls(self, name, num_urls):
        """ Queries the database and returns the n-number of URLs when given a specific name.

        Args:
            name (str): Name of tag
            num_urls (int): Number of urls to return

        Returns:
            lst: A list of URLs regarding the name with a length of num_urls
        """
        if not self.check_tag_exists(name):
            return []
        self.connect()

        self.cursor.execute('''
            SELECT urls.url
            FROM urls
            JOIN tags ON tags.id = urls.tag_id
            WHERE tags.name = ?
            LIMIT ?
        ''', (name, num_urls))

        urls = self.cursor.fetchall()
        self.close()

        return [url[0] for url in urls]
    
    def add_urls_by_name(self, name, urls):
        """Adds a list of new URLs to the table with name.
                Doesn't add duplicates.
                Creates a tag for the name if it doesn't exist already

        Args:
            name (str): Name of the tag for the urls
            urls (lst): List of the urls related to the name to be added
        """        
        self.connect()

        # Retrieve the tag ID based on the name
        self.cursor.execute('SELECT id FROM tags WHERE name = ?', (name,))
        tag_id = self.cursor.fetchone()

        if not tag_id:
            # If the tag doesn't exist, insert it into the tags table
            self.cursor.execute('INSERT INTO tags (name) VALUES (?)', (name,))
            tag_id = self.cursor.lastrowid
        else:
            tag_id = tag_id[0]

        # Check for duplicate URLs before inserting
        for url in urls:
            try:
                self.cursor.execute('INSERT INTO urls (url, tag_id) VALUES (?, ?)', (url, tag_id))
            except sqlite3.IntegrityError:
                print(f"Skipping duplicate URL: {url}")
        self.close()
    
    def check_tag_exists(self, name):
        """Checks whether a given tag name exists in the database
        Args:
            name (str): Name of the tag
        Returns:
            bool: True when the name exists in the database. False otherwise.
        """        
        self.connect()
        self.cursor.execute('SELECT EXISTS(SELECT 1 FROM tags WHERE name = ? LIMIT 1)', (name,))
        exists = self.cursor.fetchone()[0]
        self.close()
        return exists