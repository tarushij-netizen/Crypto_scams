import importlib.util
import csv
import time
from datetime import datetime

praw_spec = importlib.util.find_spec("praw")
if praw_spec is None:
    raise ImportError("The praw module is required. Install it with `pip install praw`.")

praw = importlib.import_module("praw")

#API credentials
CLIENT_ID = "Oc8NGlvN8WkYujQmIT3_NQ"
CLIENT_SECRET = "FFVhVV3aKw-eHH7kG9Qkan0yOiWZrA"

# Reddit account 
USERNAME = "imaginearyan"
PASSWORD = "Ar@121002"

# A unique name for our script (JUST FOR FUN)
USER_AGENT = "CryptoScamScraper/0.2 by /u/" + USERNAME

#SEARCH
SUBREDDITS_TO_SCRAPE = ['cryptocurrency', 'Bitcoin', 'ethtrader', 'solana', 'CryptoMarkets', 'NFT', 'CryptoMoonShots', 'btc', 'CryptoIndia', 'Scams', 'CryptoScams']
KEYWORD_TO_SEARCH = 'Centra' #Keyword
POST_LIMIT_PER_SUB = 5000  # Number of Posts
OUTPUT_FILE = 'reddit_sentiment_data_Centra_2024.csv'

def setup_praw():
    print("Authenticating with Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD
        )
        # Test authentication
        print(f"Successfully authenticated as: {reddit.user.me()}")
        return reddit
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please check your CLIENT_ID, CLIENT_SECRET, USERNAME, and PASSWORD.")
        return None

# def scrape_data(reddit, writer):
#     total_posts = 0
#     total_comments = 0
    
#     print(f"\nStarting scraper for keyword: '{KEYWORD_TO_SEARCH}'...")
#     print(f"Targeting subreddits: {', '.join(SUBREDDITS_TO_SCRAPE)}\n")

#     for subreddit_name in SUBREDDITS_TO_SCRAPE:
#         try:
#             subreddit = reddit.subreddit(subreddit_name)
#             print(f"--- Scraping r/{subreddit_name} ---")
            
#             # Search for posts
#             search_results = subreddit.search(query=KEYWORD_TO_SEARCH, sort='top', time_filter='year', limit=POST_LIMIT_PER_SUB)
            
#             posts_found = 0
#             for post in search_results:
#                 posts_found += 1
#                 total_posts += 1
                
#                 post_text = post.title + " " + post.selftext
                
#                 # Write the POST to the CSV
#                 writer.writerow({
#                     'id': post.id,
#                     'subreddit': subreddit_name,
#                     'type': 'post',
#                     'author': post.author.name if post.author else '[deleted]',
#                     'timestamp_utc': datetime.utcfromtimestamp(post.created_utc).isoformat(),
#                     'text': post_text.strip(),
#                     'parent_id': None
#                 })
                
                
#                 post.comments.replace_more(limit=0)  
                
#                 for comment in post.comments.list():
#                     total_comments += 1
#                     writer.writerow({
#                         'id': comment.id,
#                         'subreddit': subreddit_name,
#                         'type': 'comment',
#                         'author': comment.author.name if comment.author else '[deleted]',
#                         'timestamp_utc': datetime.utcfromtimestamp(comment.created_utc).isoformat(),
#                         'text': comment.body.strip(),
#                         'parent_id': comment.parent_id
#                     })
#             ##THE ABOVE CODE IS TO WRITE THE COMMENTS/POSTS' CONTENT IN THE CSV
#             if posts_found == 0:
#                 print(f"No posts found for '{KEYWORD_TO_SEARCH}' in r/{subreddit_name}.")
#             else:
#                 print(f"Scraped {posts_found} posts and their comments from r/{subreddit_name}.")
                
#         except Exception as e:
#             print(f"Error scraping r/{subreddit_name}: {e}")
        
#         time.sleep(2)
import time as time_module

def scrape_data(reddit, writer):
    total_posts = 0
    total_comments = 0

    # ── Define your exact date range here ──
    START_DATE = datetime(2017, 1, 1)   # change to whatever start year you want
    END_DATE   = datetime(2025, 12, 31)

    start_ts = int(START_DATE.timestamp())
    end_ts   = int(END_DATE.timestamp())

    print(f"\nScraping from {START_DATE.date()} to {END_DATE.date()}")

    for subreddit_name in SUBREDDITS_TO_SCRAPE:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"--- Scraping r/{subreddit_name} ---")

            # Use 'all' to remove the 1-year cap, then filter by timestamp manually
            search_results = subreddit.search(
                query=KEYWORD_TO_SEARCH,
                sort='top',
                time_filter='all',          # ← removes the 1-year restriction
                limit=POST_LIMIT_PER_SUB
            )

            posts_found = 0
            for post in search_results:

                # Filter by your custom date range
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if not (START_DATE <= post_time <= END_DATE):
                    continue                 # skip posts outside your range

                posts_found += 1
                total_posts += 1
                post_text = post.title + " " + post.selftext

                writer.writerow({
                    'id': post.id,
                    'subreddit': subreddit_name,
                    'type': 'post',
                    'author': post.author.name if post.author else '[deleted]',
                    'timestamp_utc': post_time.isoformat(),
                    'text': post_text.strip(),
                    'parent_id': None
                })

                post.comments.replace_more(limit=0)

                for comment in post.comments.list():
                    comment_time = datetime.utcfromtimestamp(comment.created_utc)
                    if not (START_DATE <= comment_time <= END_DATE):
                        continue            # filter comments too

                    total_comments += 1
                    writer.writerow({
                        'id': comment.id,
                        'subreddit': subreddit_name,
                        'type': 'comment',
                        'author': comment.author.name if comment.author else '[deleted]',
                        'timestamp_utc': comment_time.isoformat(),
                        'text': comment.body.strip(),
                        'parent_id': comment.parent_id
                    })

            if posts_found == 0:
                print(f"No posts found in range for r/{subreddit_name}.")
            else:
                print(f"Scraped {posts_found} posts from r/{subreddit_name}.")

        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {e}")

        time_module.sleep(2)
        
    print(f"\n--- Scraping Complete ---")
    print(f"Total Posts Scraped: {total_posts}")
    print(f"Total Comments Scraped: {total_comments}")

#EXEC
if __name__ == "__main__":
    
    if "PASTE_YOUR_" in CLIENT_ID or "PASTE_YOUR_" in USERNAME:
        print("ERROR")
    else:
        reddit_instance = setup_praw()
        
        if reddit_instance:
            #CSV file headers
            fieldnames = ['id', 'subreddit', 'type', 'author', 'timestamp_utc', 'text', 'parent_id']
            
            try:
                with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()  
                    
                    #INITIATION
                    scrape_data(reddit_instance, writer)
                    
                print(f"Success! All data saved to {OUTPUT_FILE}")
                
            except IOError as e:
                print(f"Error: Could not write to file {OUTPUT_FILE}. {e}")