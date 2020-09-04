import collections
import json
import math
import re
import string
import sys
import time
from functools import reduce
from pyspark import SparkConf, SparkContext

def cleanData(texts, stop_words):
    word_list = []
    for text in texts:
        text = text.translate(str.maketrans('', '',"0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        # word_list.extend()
        pList=list(filter(lambda word: word not in stop_words and word != '' and word not in string.ascii_lowercase,
                        re.split(r"[~\s\r\n]+", text)))
        for p in pList:
            word_list.append(p)
    # print(word_list)
    return word_list



def countWords(wlist):
    mtimes,nk = 0,0
    cdict = collections.defaultdict(list)
    for word in wlist:
        if word not in cdict.keys():
            cdict[word]=[1]
        else:
            cdict[word].append(1)

        # if word in cdict.keys():
        #     cdict[word].append(1)
        # else:
        #     cdict[word] = [1]

    nk=len(reduce(lambda a, b:max(a,b),cdict.values()))
    if nk>mtimes:
        mtimes=nk
    # mtimes = max(nk, mtimes)
    cdict = dict(filter(lambda kv: len(kv[1]) > 3, cdict.items()))
    lis=sorted([(k, len(v), mtimes) for k,v in cdict.items()],key=lambda kv: kv[1], reverse=True)
    return lis


# def removeDuplicateIds(fList):
#     uidst,bidst = {},{}
#     for item in fList:
#         m,n=item[0],item[1]
#         uidst.add(m)
#         bidst.add(n)
#     p=sorted(uidst)
#     q=sorted(bidst)
#     return p,q


# def genModel(json_array, file_path):
#     with open(file_path, 'w+') as output_file:
#         for item in json_array:
#             output_file.writelines(json.dumps(item) + "\n")
#         output_file.close()

def flatIt(dict_list):
    result = collections.defaultdict(list)
    for item in dict_list:
        m=list(item.keys())
        n=list(item.values())
        result[m[0]] = n[0]
    return result

def genModel(json_array, file_path):
    print("Started Writing")
    f=open(file_path, 'w+')
    for item in json_array:
        f.writelines(json.dumps(item) + "\n")
    f.close()

def joinList(l1, l2):
    result = list(l1)
    # print(len(res))
    for i in l2:
        result.append(i)
    # result.extend(l2)
    # print(len(res))
    return result


# def genRes(data, type, keys):
#     result = []
#     if isinstance(data, dict):
#         for k, v in data.items():
#             result.append({"type": type,keys[0]: k,keys[1]: v})
#     elif isinstance(data, list):
#         for kv in data:
#             for k, v in kv.items():
#                 result.append({"type": type,keys[0]: k,keys[1]: v})
#     return result

def genRes(data, type, keys):
    result = []
    if isinstance(data, list):
        for kv in data:
            for k, v in kv.items():
                str1={"type": type,keys[0]: k,keys[1]: v}
                result.append(str1)
        
    elif isinstance(data, dict):
        for k, v in data.items():
            str1={"type": type,keys[0]: k,keys[1]: v}
            result.append(str1)
    return result


if __name__ == '__main__':

    export_model_file_path = sys.argv[2]
    train_file_path = sys.argv[1]
    stop_words_file_path = sys.argv[3]

    l=[]
    start = time.time()
    f=open(stop_words_file_path)
    for w in f:
         l.append(w.rstrip("\n"))
    stset=set(l)
    # print(stset)
    print(len(stset))

    # def starter():
    #   SparkContext.setSystemProperty('spark.executor.memory', '4g')
    #   SparkContext.setSystemProperty('spark.driver.memory', '4g')
    #   sc = SparkContext('local[*]', 'task1')
    #   start = time.time()

    # starter()

    SparkContext.setSystemProperty('spark.executor.memory', '4g')
    SparkContext.setSystemProperty('spark.driver.memory', '4g')
    sc = SparkContext('local[*]', 'task2')
    
    # USER_ID = 'user_id'
    # USER_PROFILE = 'user_profile'
    # USER_INDEX = "user_index"
    # BUSINESS_ID = 'business_id'
    # BUSINESS_PROFILE = 'business_profile'
    # BUSINESS_INDEX = "business_index"
    # REVIEW_TEXT = "text"
    # TOP_200 = 200
    # RARE_THRESHOLD = 3
    start = time.time()

    model_content = []
    count=0

    inp=sc.textFile(train_file_path)
    inp=inp.map(lambda row: json.loads(row))

    uirdd=inp.map(lambda kv: kv["user_id"]).distinct()  
    uirdd=uirdd.sortBy(lambda x: x).zipWithIndex().map(lambda kv: {kv[0]: kv[1]})
    uirdd=uirdd.flatMap(lambda x: x.items())

    bidict = inp.map(lambda kv: kv["business_id"]).distinct().sortBy(lambda item: item).zipWithIndex().map(lambda kv: {kv[0]: kv[1]})
    bidict=bidict.flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    uidict = uirdd.collectAsMap()
    model_content.extend(genRes(uidict, "user_index",keys=["user_id", "user_index"]))
    model_content.extend(genRes(bidict, "business_index",keys=["business_id", "business_index"]))

    btfrdd = inp.map(lambda kv: (bidict[kv["business_id"]],str(kv["text"].encode('utf-8')).lower())).groupByKey() \
        .mapValues(lambda texts: cleanData(list(texts), stset)).map(lambda x:(x[0], countWords(x[1])))
    btfrdd=btfrdd.flatMap(lambda bvs: [((bvs[0], words_vv[0]),words_vv[1] / words_vv[2]) for words_vv in bvs[1]]).persist()
    count+=1
    bidfrdd = btfrdd.map(lambda x: (x[0][1],x[0][0])).groupByKey().mapValues(lambda bids: list(set(bids)))
    bidfrdd=bidfrdd.flatMap(lambda wbd: [((bid, wbd[0]),math.log(len(bidict)/len(wbd[1]), 2)) for bid in wbd[1]])


    btfidfrdd = btfrdd.leftOuterJoin(bidfrdd)
    count+=1
    # print(count)
    btfidfrdd=btfidfrdd.mapValues(lambda tf_idf: tf_idf[0] * tf_idf[1]) \
        .map(lambda bid_word_val: (bid_word_val[0][0],(bid_word_val[0][1], bid_word_val[1])))
    btfidfrdd=btfidfrdd.groupByKey().mapValues(lambda val: sorted(list(val), reverse=True,key=lambda item: item[1])[:200]) \
        .mapValues(lambda word_vals: [item[0] for item in word_vals])

    wdict = btfidfrdd.flatMap(lambda kv: [(word, 1) for word in kv[1]]).groupByKey()\
    .map(lambda kv: kv[0]).zipWithIndex().map(lambda kv: {kv[0]: kv[1]}).flatMap(lambda kv_items: kv_items.items()).collectAsMap()

    bus_profile = btfidfrdd.mapValues(lambda words: [wdict[word] for word in words]).map(lambda x:{x[0]:x[1]}).persist()

    bfdata = bus_profile.collect()
    bus_profile_dict = flatIt(bfdata)
    
    user_profile = inp.map(lambda kv: (kv["user_id"], kv["business_id"])).groupByKey().map(lambda kv: (uidict[kv[0]], list(set(kv[1]))))
    k=len(model_content)
    user_profile=user_profile.mapValues(lambda bids: [bidict[bid] for bid in bids]).flatMapValues(lambda bids: [bus_profile_dict[bid] for bid in bids]).reduceByKey(joinList)\
        .filter(lambda p: len(p[1]) > 1).map(lambda x: {x[0]: list(set(x[1]))})

    model_content.extend(genRes(bfdata, "business_profile",keys=["business_index", "business_profile"]))

    model_content.extend(genRes(user_profile.collect(), "user_profile",keys=["user_index","user_profile"]))

    genModel(model_content, export_model_file_path)

    print("Duration: %d s." % (time.time() - start))
