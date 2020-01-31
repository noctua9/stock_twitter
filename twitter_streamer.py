from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
 
import twitter_credentials



# # # # TWITTER AUTHENTICATOR # # # #
class TwitterAuthenticator:
	def authetificate_twitter_app(self):
		auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
		auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
		return auth

		
# # # # TWITTER CLIENT # # # #

class TwitterClient():
	def __init__(self, twitter_user=None):
		self.auth = TwitterAuthenticator().authetificate_twitter_app()
		self.twitter_client = API(self.auth)
		self.twitter_user = twitter_user
	
	def get_twitter_client_api(self):
		return self.twitter_client
		
	def get_user_timeline_tweets(self, num_tweets):
		tweets = []
		for tweet in Cursor(self.twitter_client.user_timeline, id = self.twitter_user).items(num_tweets):
			tweets.append(tweet)
		return tweets

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
	    self.twitter_authenticator = TwitterAuthenticator()
		
    def stream_tweets(self, tweets, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(tweets)
        auth = self.twitter_authenticator.authetificate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, tweets):
        self.tweets = tweets

    def on_data(self, data):
        try:
            print(data)
            with open(self.tweets, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          

    def on_error(self, status):
	    if status == 420:
		# stops if rate limit occurs
		    return false
	    print(status)

class TweetAnalyzer():
#analyzes tweets
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
	
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

 
if __name__ == '__main__':
 
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    tweets = api.user_timeline(screen_name="ICICIBank", count=100)

    #print(dir(tweets[0]))
    

    df = tweet_analyzer.tweets_to_data_frame(tweets)
    df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
    print(df.head(20))
    hash_tag_list = ("icici")
    tweets = "tweets.txt"
    #twitter_streamer = TwitterStreamer()
    #twitter_streamer.stream_tweets(tweets, hash_tag_list)
    
	# Get average length over all tweets:
    #print("Average lenght",np.mean(df['len']))

    # Get the number of likes for the most liked tweet:
    #print("number of likes for the most liked tweet",np.max(df['likes']))

    # Get the number of retweets for the most retweeted tweet:
    #print("number of retweets for the most retweeted tweet",np.max(df['retweets']))
    
    

    # Time Series
    #time_likes = pd.Series(data=df['len'].values, index=df['date'])
    #time_likes.plot(figsize=(16, 4), color='r')
    #plt.show()
    
    #time_favs = pd.Series(data=df['likes'].values, index=df['date'])
    #time_favs.plot(figsize=(16, 4), color='r')
    #plt.show()

    #time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
    #time_retweets.plot(figsize=(16, 4), color='r')
    #plt.show()

    # Layered Time Series:
    time_likes = pd.Series(data=df['likes'].values, index=df['date'])
    time_likes.plot(figsize=(16, 4), label="likes", legend=True)

    time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
    time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
    plt.show()
