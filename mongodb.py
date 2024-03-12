import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://conorjkeane01:65rKfTYY0kLojp4p@cluster0.ajxrfz8.mongodb.net/")

db = cluster["Sleep_Data"]
collection = db["UserData"]

#collection.insert_one({"_id":0, "sleep_direction":"Right", "time_spent":100})
#collection.delete_one({"_id":0})