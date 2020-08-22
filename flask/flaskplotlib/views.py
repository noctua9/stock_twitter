from __future__ import print_function
import sys
import os
import base64
import io
import traceback
from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from urllib.request import urlopen,Request
import urllib
import bs4
import re
import unicodedata
import nltk
from yahoo_fin import stock_info as si
from flask import (
    Blueprint,
    render_template,
    abort,
    current_app,
    make_response
    )
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from flask import jsonify, Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib import pyplot as PLT


# # # # TWITTER AUTHENTICATOR # # # #
class TwitterAuthenticator:
	def authetificate_twitter_app(self):
		auth = OAuthHandler("nfgNlUMW4k79psyuo5TxsR1Kx", "jJXGbqq4Qsbm0YPojItNYlZjQS3xVu8KDCEMXX5CkVF1gmbOxO")
		auth.set_access_token("1133933994146328577-rzBWQH0qEKQFmupsapvgbStjoqV3hq", "4mHLfgKu5cyIbOLrWCFNevQX2knI4iTx0tTdkvkl32y54")
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


class TweetAnalyzer():
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[[tweet.created_at.date(),tweet.text]for tweet in tweets],columns=['Date','Tweets'])
        return df



client = Blueprint('client', __name__, template_folder='templates', static_url_path='/static')
@client.route('/')
def start():
    return render_template('MAIN1.html')


@client.route('/', methods=['POST'])
def home():
    title = current_app.config['TITLE']
    nam = request.form['nam']
    title = current_app.config['TITLE']
    scores = {}
    maxdelta = 30
    delta = range(8, maxdelta)
    symbol = nam
    symbol = symbol.upper()	
    name = symbol+".NS"
    ds = yf.download(name,'2017-03-01','2020-04-12')
    datasets = [ds]
    finance = pd.concat(datasets)
    high_value = 365
    high_value = min(high_value, finance.shape[0] - 1)
    lags = range(high_value, 30)
    if 'symbol' in finance.columns:
        finance.drop('symbol', axis=1, inplace=True)
    finance = finance.fillna(finance.mean())
    finance.columns = [str(col.replace('&', '_and_')) for col in finance.columns]
    mean_squared_errors, r2_scores, c,pricess = performRegression(finance, 0.80, symbol)
    dset = finance[:10]
    scores[symbol] = [mean_squared_errors, r2_scores]
    result = r2_scores[0]
    result = int(result*100)
    if symbol == "TCS":
        descriptions = "Tata Consultancy Services Limited provides information technology (IT) and IT enabled services worldwide. It operates through Banking, Financial Services and Insurance; Manufacturing; Retail and Consumer Business; Communication, Media and Technology; and Others segments. The company offers CHROMA, a cloud-based talent management solution; ignio, a cognitive automation software product; iON, an assessment platform; TAP, a procurement offering; TCS MasterCraft, a platform to automate and manage IT processes; and Quartz, a blockchain solution. It also provides customer intelligence and insight solutions to deliver retail, banking, and communications experiences; Intelligent Urban Exchange, an integrated software to accelerate smart city programs; OPTUMERA, a digital merchandising suite; TCS BaNCS, a financial platform; and Jile, an agile DevOps product. In addition, the company offers advanced drug development and connected intelligent platforms; ERP on cloud, an enterprise solution that offers hosted ERP applications and services; HOBS, a platform for subscription based digital business. Further, it provides cognitive business, consulting, analytics, automation and artificial intelligence, Internet of Things, cloud applications, blockchain, cloud infrastructure, cyber security, interactive, industrial engineering, quality engineering, and enterprise services. The company serves banking, financial, and public services; consumer goods and distribution, insurance, life sciences and healthcare, manufacturing, retail, hi-tech, travel, transportation, and hospitality industries; communications, media, and technology industries; and energy, resource, and utility industries. It has strategic partnership with Posten Norge AS.; and a strategic alliance with Zendesk, Inc. to provide enterprise grade CRM solutions. The company was founded in 1968 and is headquartered in Mumbai, India. Tata Consultancy Services Limited is a subsidiary of Tata Sons Private Limited."
    name = 'TCS'
    url_str = 'https://in.finance.yahoo.com/quote/'+name+'.NS/'
    url = url_str
#    url = 'https://in.finance.yahoo.com/quote/TCS.NS/'
    page = urlopen(url)
    soup = bs4.BeautifulSoup(page,"html.parser")
    prev_close = soup.find('td',{'data-test': 'PREV_CLOSE-value'}).find().text	
    close = pricess[0]
    close = float(close)
    prev_close =re.sub('[,]', '', prev_close) 
    prev_close = float(prev_close)
    diff_value = prev_close - close
    diff_value = abs(diff_value)
    diff_percent = (diff_value/prev_close) * 100
    diff_percent = round(diff_percent,2)
    diff_percent = str(diff_percent)
    if close>prev_close : 
        change = "Stock will be increased by +"+diff_percent+"%"
    else : 
        change = "Stock will be decreased by -"+diff_percent+"%"

# ************************TWITTER_CODE*********************************
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()
    tweets = api.user_timeline(screen_name=symbol,count = 3000, since='2020-01-21')
    df = tweet_analyzer.tweets_to_data_frame(tweets)
    cdata=df[['Date','Tweets']].copy()	
    cdata=pd.DataFrame(columns=['Date','Tweets'])
    index=0
    for index,row in df.iterrows():
        stre=row["Tweets"]
        my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
        cdata.sort_index()
        cdata.set_value(index,'Date',row["Date"])
        cdata.set_value(index,'Tweets',my_new_string)
    index=index+1
    cdata["Comp"] = ''
    cdata["Negative"] = ''
    cdata["Neutral"] = ''
    cdata["Positive"] = ''
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiment_i_a = SentimentIntensityAnalyzer()
    for indexx, row in cdata.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', cdata.loc[indexx, 'Tweets'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            cdata.set_value(indexx, 'Comp', sentence_sentiment['compound'])
            cdata.set_value(indexx, 'Negative', sentence_sentiment['neg'])
            cdata.set_value(indexx, 'Neutral', sentence_sentiment['neu'])
            cdata.set_value(indexx, 'Positive', sentence_sentiment['pos'])
        except TypeError:
            print (stocks_dataf.loc[indexx, 'Tweets'])
            print (indexx)
    posi=0
    nega=0
    for i in range (0,len(cdata)):
        get_val=cdata.Comp[i]
        if(float(get_val)<(0)):
            nega=nega+1
        if(float(get_val>(0))):
            posi=posi+1
    posper=(posi/(len(cdata)))*100
    negper=(nega/(len(cdata)))*100
    arr=np.asarray([posper,negper], dtype=int)
    labels = ['positive', 'negative']
    values = [ posper , negper ]	
    colors = ["#00FF00", "#FF0000"]	
    render_template('index.html')
    return render_template('index.html', title=title,max=17000, posper=posper ,negper=negper, pricess=pricess, change=change, set=zip(values, labels, colors),name=name,nam=nam,dset=dset.to_html(classes="table table-striped"),c=c,  result=result)

def count_missing(dataframe):
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

def benchmark_model(model, train, test, features, output, *args, **kwargs):
    model_name = model.__str__().split('(')[0].replace('Regressor', ' Regressor')
    a = (si.get_data("tcs.ns"))
    open = (a.open.iloc[-1])
    high = (a.high.iloc[-1])
    low = (a.low.iloc[-1])
    model.fit(train[features].values, train[output].values, *args, **kwargs)
    predicted_value = model.predict(test[features].values)
    q = {'Open' : open , 'High' : [high] , 'Low' : [low]}
    qdf = pd.DataFrame(q, columns = ['Open','High','Low'])
    pricess = model.predict(qdf.values)
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(test[output].values, color='g', ls='-', label='Actual Value')
    ax.plot(predicted_value, color='b', ls='--', label='predicted_value Value')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(model_name)
    ax.grid(True)
    img = io.StringIO()
    fig.savefig(img, format='svg')
    svg_img = '<svg' + img.getvalue().split('<svg')[1]
    return predicted_value,svg_img,pricess	
	
def performRegression(dataset, split, symbol):
    features = dataset.columns[0:3]
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
    output = dataset.columns[3]
    predicted_values = []
    classifier = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    a,b,pricess = benchmark_model(classifier,train, test, features, output)
    predicted_values.append(a)
    maxiter = 1000
    batch = 150
    mean_squared_errors = []
    r2_scores = []
    for pred in predicted_values:
        mean_squared_errors.append(mean_squared_error(test[output], \
            pred))
        r2_scores.append(r2_score(test[output], pred))
    return mean_squared_errors, r2_scores, b, pricess

	

	
	
	
	
	
	
	
	
	