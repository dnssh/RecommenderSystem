import collections
import json
import math
import sys
import time
from pyspark import SparkConf, SparkContext


# def FlatIt(dict_list):
#     result = collections.defaultdict(list)
#     for item in dict_list:
#         result[list(item.keys())[0]] = list(item.values())[0]
#     return result
def calCS(p1, p2):
    if len(p1) != 0 and len(p2) != 0:
        s1 = set(p1)
        s2 = set(p2)
        # num = len(s1.intersection(s2))
        # den = math.sqrt(len(s1)) * math.sqrt(len(s2))
        return (len(s1.intersection(s2))/(math.sqrt(len(s1)) * math.sqrt(len(s2))))
        # return num / den
    else:
        return 0.0

# def genModel(json_array, file_path):
#     with open(file_path, 'w+') as output_file:
#         for item in json_array:
#             output_file.writelines(json.dumps(item) + "\n")
#         output_file.close()

def genModel(json_array, file_path):
    print("Started Writing")
    f=open(file_path, 'w+')
    for item in json_array:
        f.writelines(json.dumps(item) + "\n")
    f.close()

if __name__ == '__main__':

    test_file_path = sys.argv[1]
    mpathpath = sys.argv[2]
    output_file_path = sys.argv[3]

    # def starter():
    #   SparkContext.setSystemProperty('spark.executor.memory', '4g')
    #   SparkContext.setSystemProperty('spark.driver.memory', '4g')
    #   sc = SparkContext('local[*]', 'task1')
    #   start = time.time()

    # starter()

    # USER_ID = 'user_id'
    # USER_PROFILE = 'user_profile'
    # USER_INDEX = "user_index"
    # BUSINESS_ID = 'business_id'
    # BUSINESS_PROFILE = 'business_profile'
    # BUSINESS_INDEX = "business_index"
    # TYPE = "type"

    SparkContext.setSystemProperty('spark.executor.memory', '4g')
    SparkContext.setSystemProperty('spark.driver.memory', '4g')
    sc = SparkContext('local[*]', 'task2')
    start = time.time()
    count=0

    mdata = sc.textFile(mpathpath).map(lambda row: json.loads(row))

    uidict=mdata.filter(lambda kv: kv['type'] == "user_index")
    uidict=uidict.map(lambda kv: {kv['user_id']: kv["user_index"]}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    count+=1
    ruidict = {v: k for k, v in uidict.items()}
    count+=1
    bidict = mdata.filter(lambda kv: kv["type"] =="business_index").map(lambda kv: {kv['business_id']: kv["business_index"]}) \
        .flatMap(lambda kv_items: kv_items.items()).collectAsMap()
    count+=1
    rbidict = {v: k for k, v in bidict.items()}

    upd=mdata.filter(lambda kv: kv["type"] == "user_profile").map(lambda kv: {kv['user_index']: kv["user_profile"]})
    count+=1
    upd=upd.flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    bpd=mdata.filter(lambda kv: kv["type"] == "business_profile")
    bpd=bpd.map(lambda kv: {kv['business_index']: kv["business_profile"]}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    finalres = sc.textFile(test_file_path).map(lambda row: json.loads(row)).map(lambda kv: (kv['user_id'], kv["business_id"]))
    temp=finalres
    count+=1
    finalres=finalres.map(lambda kv: (uidict.get(kv[0], -1), bidict.get(kv[1], -1))).filter(lambda uid_bid: uid_bid[0] != -1 and uid_bid[1] != -1) \
        .map(lambda kv: ((kv), calCS(upd.get(kv[0], set()),bpd.get(kv[1], set()))))
    count+=1
    finalres=finalres.filter(lambda kv: kv[1] > 0.01).map(lambda kv:{"user_id":ruidict[kv[0][0]],"business_id":rbidict[kv[0][1]],"sim":kv[1]})
    count+=1
    genModel(finalres.collect(), output_file_path)
    k=count
    # print(k)
    finaltime=(time.time() - start)
    print("Duration: %d s." % (time.time() - start))
