import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import requests
import json
import datetime

def getPushshiftData(after, goalSubCount, subreddit):
    """Returns 2D list of points
    Given epoch start time and goal number of subscribers, returns sub count at every day
    between start time and time which goal sub count was reached
    """
    count = 1
    totSubCounts = [[after, 0]]
    while (totSubCounts[-1][1] < goalSubCount):
        after = totSubCounts[-1][0]
        url = 'https://api.pushshift.io/reddit/search/submission/?after=' + str(after) + '&before='+ str(after + 86400)+ '&size=1&subreddit=' + subreddit
        r = requests.get(url)
        data = json.loads(r.text)
        if len(data['data']) > 0:
            totSubCounts.append(collectSubData(data['data'][0], after + count * 86400))
            count = 1
        else:
            totSubCounts.append([after + count * 86400, totSubCounts[-1][1]])
            count += 1
    return totSubCounts

def collectSubData(subm, created):
    """Returns array of time created and sub count at time
    """
    try:
        subCount = subm['subreddit_subscribers']
    except KeyError:
        subCount = 0
    return [created, subCount]

def augment(xold,yold,numsteps):
    """Returns two lists
    Given list, "fills in" gaps between points to make lines less jumpy.
    """
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difX = xold[i+1]-xold[i]
        stepsX = difX/numsteps
        difY = yold[i+1]-yold[i]
        stepsY = difY/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew,xold[i]+s*stepsX)
            ynew = np.append(ynew,yold[i]+s*stepsY)
    return xnew,ynew

def smoothListGaussian(listin,strippedXs=False,degree=5):
    """Return list smoothed
    Applies guassian function to average data. Removes sharp corners in lines.
    More info here: https://www.swharden.com/wp/2008-11-17-linear-data-smoothing-in-python/
    """
    window=degree*2-1
    weight=np.array([1.0]*window)
    weightGauss=[]
    for i in range(window):
        i=i-degree+1
        frac=i/float(window)
        gauss=1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight=np.array(weightGauss)*weight
    smoothed=[0.0]*(len(listin)-window)
    for i in range(len(smoothed)):
        smoothed[i]=sum(np.array(listin[i:i+window])*weight)/sum(weight)
    return smoothed

def manipulateData(totSubCounts):
    """Returns x list and y list of data points
    Extracts data from 2D list and converts epoch time to date time
    """
    x = []
    y = []
    for sub in totSubCounts:
        x.append(sub[0])
        y.append(sub[1])
    # OPTIONAL methods
    # x,y = augment(x,y,4)
    # x = smoothListGaussian(x)
    # y = smoothListGaussian(y)
    xFormatted = []
    for epochTime in x:
        xFormatted.append(datetime.date.fromtimestamp(epochTime))
    return xFormatted, y

def animate(i, x, y):
    """
    Animation function for matplotlib
    """
    p = sns.lineplot(x=x[:int(i+1)], y=y[:int(i+1)], color="r") # takes in range of data up to current point
    p.tick_params(labelsize=10)
    plt.setp(p.lines,linewidth=3)

def main():
    # get data
    totSubCounts = getPushshiftData(1577836800, 1000000, "coronavirus")

    # configuring animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

    # set graph style
    sns.set(rc={'axes.facecolor':'lightgrey', 'figure.facecolor':'lightgrey','figure.edgecolor':'black','axes.grid':False})

    # create graph
    fig, ax = plt.subplots(figsize=(10,6))

    # set x axis ticks to display once per week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())

    # format x axis labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # set x axis and y axis limits
    plt.xlim([datetime.date(2019, 12, 31), datetime.date(2020, 3, 16)]) # start date to date when goalSubCount was reached
    plt.ylim([0, 1000000]) # 0 to goalSubCount

    # rotate labels for x axis
    plt.xticks(rotation=75)

    # set y axis tick frequency
    plt.yticks(np.arange(0, 1100000, 100000))

    # labels
    plt.xlabel('Date',fontsize=15)
    plt.ylabel('Total Subscribers',fontsize=15)
    plt.title('r/coronavirus Road To 1,000,000',fontsize=20)

    # get layout
    plt.tight_layout()

    # get x, y and run animation
    x, y = manipulateData(totSubCounts)
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(x), fargs=[x, y], repeat=True)

    # save animation video
    ani.save('SubCount.mp4', writer=writer)


main()
