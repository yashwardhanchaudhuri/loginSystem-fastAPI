import http.client
import json
from pymongo import MongoClient
import pymongo
from pymongo import MongoClient
import pandas
import io
import matplotlib
import webbrowser


def get_database():

    CONNECTION_STRING = "mongodb+srv://BitcoinBoii:heehaw@cluster0.1ihmubb.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(CONNECTION_STRING)
    return client['token_prices']

def day_entries(json_object, collection_name):
    
    counter = 1
    time = json_object['status']['timestamp'][:10]

    for i in json_object["data"]:
        k = collection_name["token_prices"].find_one({"name": i["name"] + "(" + i["symbol"] + ")"})
        if not bool(k):
            collection_name["token_prices"].insert_many([{"name" : i["name"] + "(" + i["symbol"] + ")", "price" :  i["quote"]["USD"]["price"], time: i["quote"]["USD"]["market_cap"] }])
            counter += 1
            print("new insertion")
            
            
        else:
            k = collection_name["token_prices"].find_one({"name": i["name"] + "(" + i["symbol"] + ")"})
            collection_name["token_prices"].delete_one(k)
            k[time] = i["quote"]["USD"]["market_cap"]
            collection_name["token_prices"].insert_many([k])
            counter += 1
        if i%50 == 0:
            print(i, " queries processed")
    return time 


def highlight_mean_greater(s):
    '''
    highlight yellow is value is greater than mean else red.
    '''
    is_max = s > s.mean()
    return ['background-color: yellow' if i else 'background-color: red' for i in is_max] 
          
def create_html(token_database, date = "2022-08-21"):
    pandas.set_option('display.float_format', lambda x: '%.10f' % x)
    cursor = token_database['token_prices'].find()
    mongo_docs = list(cursor)
    docs = pandas.DataFrame(columns=[])
    counter = 0
    k = 0
    for num, doc in enumerate(mongo_docs):

            doc["_id"] = str(doc["_id"])

            doc_id = doc["name"]

            series_obj = pandas.Series( doc, name= counter)

            docs = docs.append(series_obj)
            counter += 1
            k+=1

    reference = docs["2022-08-21"]

    docs = docs.reindex(sorted(docs.columns, reverse = True), axis=1)
    cols = docs.columns.tolist()
    cols = cols[:3][::-1] + cols[3:]
    docs = docs[cols]
    docs = docs.sort_values(by=[date], ascending=False)
    print(docs.shape)
    for i in range(docs.shape[0]):
        docs.loc[i] = docs.loc[i].fillna(value=docs.loc[i][3:].mean())
    print(docs)
    docs.style.background_gradient(cmap='RdYlGn',axis=1)

    html_str = io.StringIO()

    # export as HTML
    docs.to_html(
    buf=html_str,
    classes='table table-striped'
    )

    # print out the HTML table
    # print (html_str.getvalue())
    # save the MongoDB documents as an HTML table
    docs.to_html("object_rocket.html")
    docs.to_excel("changed.xlsx")
    # webbrowser.open('object_rocket.html')
    # print(reference)
    

if __name__ == "__main__":    

    conn = http.client.HTTPSConnection("coinmarketcap-cc.p.rapidapi.com")

    headers = {
        'X-RapidAPI-Key': "ac328979c5msh0809d2a0c8c2777p1d5286jsnd1d93714ec7e",
        'X-RapidAPI-Host': "coinmarketcap-cc.p.rapidapi.com"
        }

    conn.request("GET", "/listings/latest/1000?api_key=c9cc696b-c1db-40d8-9576-3eb498c8ac93", headers=headers)

    res = conn.getresponse()
    data = res.read()

    json_object = json.loads(data)

    """
    Intended dictionary structure:
    {name: "NAME(SYMBOL)",
    market_cap_date_a: "MARKET_CAP",
    market_cap_date_b: "MARKET_CAP",
    market_cap_date_c: "MARKET_CAP",
    ...
    }
    """

    token_database = get_database()
    # k = day_entries(json_object, token_database)
    create_html(token_database, "2022-08-24")

