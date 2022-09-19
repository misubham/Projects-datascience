import requests
from bs4 import BeautifulSoup
import pandas as pd

url=['https://www.reuters.com/markets/companies/AAPL.OQ',
     'https://www.reuters.com/markets/companies/AMZN.OQ/',
     'https://www.reuters.com/markets/companies/MMM.N/',
     'https://www.reuters.com/markets/companies/RELI.NS/']
members=[]
for x in url:
    url_members=[]
    response=requests.get(x).text
    soup=BeautifulSoup(response,'html5lib')

    search_of_members=soup.find('div',{"class":"about-company-card__company-leadership__1mNWX"})
    
    try:
        for dt,dd in zip(search_of_members.find_all('dt'),search_of_members.find_all('dd')):
            url_members.append((dt.text.strip(),dd.text.strip()))
    except:pass
    members.append(url_members)

c=1
for member in members:
    df=pd.DataFrame(member,columns=['Member Name','Job Title'])
    print(df)
    df.to_csv(str(c)+'.csv',index=False)
    c+=1
print("data saved")

            
