from pymemcache.client.base import Client
import pymongo
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import math
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from helper.upload import convert_files_csv_tfidf, convert_files_csv_lda, convert_txt_to_csv, allowed_file, get_file, calculate_meta
from helper.general import add_code
from helper.visualization import generate_histogram, generate_boxplot, generate_atag, generate_correlation, save_plot, apply_pca
from helper.mining import generate_report, generate_rmse_table, save_model
from helper.scaler import apply_std_scaling
import uuid
import subprocess
import pandas as pd
import os
import ast
import copy
import requests
import json
import datetime
from io import StringIO
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora, matutils, models
import pickle
import scipy.sparse
from nltk.corpus import stopwords
import pyLDAvis.gensim
from gensim.models import Word2Vec
from helper.authentication import hash_password, verify_password

def json_serializer(key, value):
    if type(value) == str:
        return value, 1
    return json.dumps(value), 2


def json_deserializer(key, value, flags):
    if flags == 1:
        return value
    if flags == 2:
        return json.loads(value)
    raise Exception("Unknown serialization format")


app = Flask(__name__, static_folder='static')
CORS(app, support_credentials=True)
print('current path of application', os.getcwd())
app.config['UPLOAD_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\data-file'
app.config['VIZ_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static'
app.config['WE_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\word-embedding'



ALLOWED_EXTENSIONS_STRUCTURED_FILE = set(['csv', 'txt', 'xlsx'])
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["data-mining"]
file_collection = db['data_files']
memcache_client = Client(('localhost', 11211), serializer=json_serializer, deserializer=json_deserializer)

"""
Data Upload API: It is an API endpoint for uploading the data
provided by the user in CSV Format. It takes the filename as
input to the API and returns the raw input data in JSON format. 
"""


def find_stats(filename):
    """
        if stats are available in memacched, get them from there. Otherwise, check database
    :param filename: key of the file
    :return: stats as dictionary
    """
    # stats = memcache_client.get(filename)
    # if stats is not None:
    #     return stats
    # else:
    stats_collection = db['statistics']
    print('filename: ',filename)
    stats = list(stats_collection.find({'filename': filename}))[0]['stats']
    return stats


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return json.dumps({'hello':'world from Rahul'}), 200

@app.route('/register', methods=['POST'])
def register():
    print("called /register")
    username = request.form['username']
    password = request.form['password']
    hashed_password = hash_password(password)
    file_collection = db['user_accounts']
    ls = list(file_collection.find({'username': username})) #check if an existing account with the same username exists
    if (len(ls)>0):
        return json.dumps({'message': 'duplicate'}), 200

    record = {'username': username, 'password': hashed_password}
    file_collection.insert(record)
    return json.dumps({'message': 'success'}), 200

@app.route('/login', methods=['POST'])
def login():
    print("called /login")
    username = request.form['username']
    file_collection = db['user_accounts']
    ls = list(file_collection.find({'username': username}))  # check if the username exists

    # wrong username
    if (len(ls) == 0):
        return json.dumps({'message': 'wrong username'}), 200

    stored_password = ls[0]['password']
    provided_password = request.form['password']

    # wrong password
    if not verify_password(stored_password, provided_password):
        return json.dumps({'message': 'wrong password'}), 200

    return json.dumps({'message': 'success'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    print("called /upload")
    print("request:",request.form)
    if request.method == 'POST':
        # Structured data has been uploaded
        if 'file' in request.files:
            file = request.files['file']
            if 'description' in request.form:
                description = request.form['description']
            else:
                description = ''
            if 'miningType' in request.form:
                mining_type = request.form['miningType']
            else:
                mining_type = 'classification'
            print('file', file)
            print('miningType', mining_type)
            # print("fileType:",request.form['fileType'])
            print("isStructured:",request.form['isStructured'])
            filename = secure_filename(file.filename)
            type = filename.rsplit('.', 1)[1].lower()
            if type == 'txt':
                print("request.form is ",request.form)
                if 'delimiter' not in request.form:
                    delimiter = '\s+'
                else:
                    delimiter = request.form['delimiter']
                    if (delimiter == ''):
                        delimiter = '\s+'
                    print("delimeter:",delimiter)
            if not allowed_file(filename, ALLOWED_EXTENSIONS_STRUCTURED_FILE):
                return 'File not allowed', 400
            if file.filename == '':
                return 'No file selected', 400
            if file:
                # generating unique id every time so same filenames do not overlap
                unique_id = str(uuid.uuid4())
                filename = unique_id + '_' + filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # convert txt file to csv
                if type == "txt":
                    filename = convert_txt_to_csv(filename, app.config['UPLOAD_FOLDER'], delimiter)
                    print('file name after conversion to csv:', filename)
                file_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename)) / 1024**2  # size in MB
                print('file size in MB', file_size)
                file_collection = db['data_files']
                print("file_collection: ",file_collection)
                file_stats = db['statistics']
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                record = {'filename': filename, 'timestamp': timestamp, 'description': description, 'miningType': mining_type, 'code': []}
                file_collection.insert_one(record)

                data = get_file(filename, app.config['UPLOAD_FOLDER'])
                print("data is: ",data)
                data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], filename+'.json'), orient='records')
                add_code(filename, file_collection, "import pandas as pd")
                add_code(filename, file_collection, "import numpy as np")
                add_code(filename, file_collection, "from sklearn.preprocessing import StandardScaler")
                add_code(filename, file_collection, "from sklearn.ensemble import RandomForestClassifier")
                add_code(filename, file_collection, "import seaborn as sns")
                add_code(filename, file_collection, "import matplotlib.pyplot as plt")
                add_code(filename, file_collection, "data = pd.read_csv('"+filename+"')")
                metadata, code = calculate_meta(data)
                file_stats.insert({'filename': filename, 'stats': metadata}, check_keys=False)
                #memcache_client.set(filename, metadata)
                print('metadata', metadata)
                # after successful completion of function, add that code to database
                add_code(filename, file_collection, code)

                if 'workflow' in request.form:
                    # function has been called from the workflow screen
                    username = request.form['username']
                    workflow_name = request.form['workflow_name']
                    type = request.form['type']
                    workflow_file_db = db['workflow-file'] # db to map username and workflow name with uploaded file
                    # check if a file has been uploaded with the same username and same workflow name and same workflow type - structured or unstructured
                    ls = list(workflow_file_db.find({'username': username,'workflow_name': workflow_name,'type':type}))

                    if (len(ls)==0): # no file has been uploaded with the same username and same workflow name
                        workflow_file_db.insert({'version': 1,'username': username,'workflow_name': workflow_name, 'type':type, 'filename': filename})
                    else:
                        latest_version = int(ls[len(ls)-1]['version'])
                        workflow_file_db.insert({'version': latest_version+1,'username': username, 'workflow_name': workflow_name, 'type':type, 'filename': filename})


                return json.dumps({'message': 'file uploaded successfully',
                                   'filename': filename, 'metadata': metadata}), 200
        else:
            # Unstructured data has been uploaded
            listOfFiles = []
            key = "file0"
            i = 1
            while key in request.files:
                listOfFiles.append(request.files[key])
                key = key[0:4] + str(i)
                i += 1

            if 'description' in request.form:
                description = request.form['description']
            else:
                description = ''

            if 'onlyOneFile' in request.form:
                onlyOneFile = request.form['onlyOneFile']
            else:
                onlyOneFile = 'No'

            if 'stemming' in request.form:
                stemming = request.form['stemming']
            else:
                stemming = 'No'

            if 'lemmatization' in request.form:
                lemmatization = request.form['lemmatization']
            else:
                lemmatization = 'No'

            if 'parsingType' in request.form:
                parsingType = request.form['parsingType']
            else:
                parsingType = 'TF-IDF'


            filename = secure_filename(listOfFiles[0].filename)
            # generating unique id every time so same filenames do not overlap
            unique_id = str(uuid.uuid4())
            filename = unique_id + '_' + filename
            # create path of the newly created csv file containing all the tf-idf values
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            csvPath = filePath.split('.')[0] + '.csv'
            # create list of file paths
            listOfFilePaths = []
            for eachFile in listOfFiles:
                eachFileName = secure_filename(eachFile.filename)
                eachFile.save(os.path.join(app.config['UPLOAD_FOLDER'], eachFileName))
                listOfFilePaths.append(os.path.join(app.config['UPLOAD_FOLDER'], eachFileName))

            # call method to create to generate the dataframe and create the csv files
            if parsingType == "TF-IDF":
                filename = convert_files_csv_tfidf(onlyOneFile, stemming, lemmatization, listOfFiles, listOfFilePaths, csvPath)
            else:
                if 'topics' in request.form:
                    topics = request.form['topics']
                    topics = int(topics) # convert no. of topics to int from string
                else:
                    topics = 5

                filename = convert_files_csv_lda(onlyOneFile, stemming, lemmatization, listOfFiles, listOfFilePaths, csvPath, topics)

            # handle case when files uploaded have no data
            if filename == "fail":
                return json.dumps({'message': 'Error while parsing the unstructured file. Could be because the file has no data.'}), 200

            file_size = os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename)) / 1024 ** 2  # size in MB
            print('file size in MB', file_size)
            file_collection = db['data_files']
            print("file_collection: ", file_collection)
            file_stats = db['statistics']
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            record = {'filename': filename, 'timestamp': timestamp, 'description': description,
                      'stemming': stemming, 'lemmatization': lemmatization, 'code': []}
            file_collection.insert_one(record)

            data = get_file(filename, app.config['UPLOAD_FOLDER'])
            #print("data is: ", data)
            data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.json'), orient='records')
            add_code(filename, file_collection, "import pandas as pd")
            add_code(filename, file_collection, "import numpy as np")
            add_code(filename, file_collection, "from sklearn.preprocessing import StandardScaler")
            add_code(filename, file_collection, "from sklearn.ensemble import RandomForestClassifier")
            add_code(filename, file_collection, "import seaborn as sns")
            add_code(filename, file_collection, "import matplotlib.pyplot as plt")
            add_code(filename, file_collection, "data = pd.read_csv('" + filename + "')")
            metadata, code = calculate_meta(data)
            file_stats.insert({'filename': filename, 'stats': metadata}, check_keys=False)
            # memcache_client.set(filename, metadata)
            #print('metadata', metadata)
            # after successful completion of function, add that code to database
            add_code(filename, file_collection, code)

            if 'workflow' in request.form:
                # function has been called from the workflow screen
                username = request.form['username']
                workflow_name = request.form['workflow_name']
                type = request.form['type']
                workflow_file_db = db['workflow-file']  # db to map username and workflow name with uploaded file
                # check if a file has been uploaded with the same username and same workflow name and same workflow type - structured or unstructured
                ls = list(workflow_file_db.find({'username': username, 'workflow_name': workflow_name, 'type': type}))

                if (len(ls) == 0):  # no file has been uploaded with the same username and same workflow name
                    workflow_file_db.insert(
                        {'version': 1, 'username': username, 'workflow_name': workflow_name, 'type': type,
                         'filename': filename})
                else:
                    latest_version = int(ls[len(ls) - 1]['version'])
                    workflow_file_db.insert(
                        {'version': latest_version + 1, 'username': username, 'workflow_name': workflow_name,
                         'type': type, 'filename': filename})


            return json.dumps({'message': 'File uploaded successfully',
                               'filename': filename, 'metadata': metadata}), 200



@app.route('/stats', methods=['GET', 'POST'])
def get_stats():
    print("called /stats")
    if 'fileKey' not in request.form:
        return 'please send file-key', 500
    filename = request.form['fileKey']
    print('file key in stats', filename)
    stats = find_stats(filename)
    column_names = stats['columns']
    table_data = []
    print('stats', stats)
    # for mean, median and mode
    for stat_type in ['mean', 'median', 'mode']:
        stat_row = dict()
        stat_row['stat type'] = stat_type
        for column in column_names:
            if column not in stats[stat_type]:
                stat_row[column] = '-'
            else:
                stat_row[column] = stats[stat_type][column]
        table_data.append(stat_row)

    stats['table'] = table_data
    return json.dumps(stats), 200


@app.route('/replace-by-mean', methods=['GET', 'POST'])
def replace_by_mean():
    print("called /replace-by-mean")
    if 'fileKey' not in request.form or 'columns' not in request.form:
        return 'Please send file-key and column namess', 500
    filename = request.form['fileKey']
    columns = json.loads(request.form['columns'])
    print('replacing by mean')
    print('incoming file', filename)
    print('incoming columns', columns, 'type', type(columns))
    data = get_file(filename, app.config['UPLOAD_FOLDER'])
    stats = find_stats(filename)
    print('current file stats', stats)
    means = dict()
    missing_indexes = dict()
    for column in columns:
        missing_indexes[column] = list(np.where(data[column].isnull())[0])
        if column in stats['mean']:
            means[column] = stats['mean'][column]
    print('found means for', means)
    missing_val_set = set(['-', '?', 'na', 'n/a', 'NA', 'N/A'])
    for i in range(len(data)):
        for column in columns:
            if data[column][i] in missing_val_set:
                missing_indexes[column].append(i)
    print('missing indexes', missing_indexes)
    for column in columns:
        if column in missing_indexes:
            data.loc[missing_indexes[column], column] = means[column]
    # now write updated dictionary to csv and json again
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.json'), orient='records')
    code = '# code to replace missing values by mean\n'
    code += "missing_values = ['-', '?', 'na', 'n/a', 'NA', 'N/A']\n"
    code += "# columns you have selected\n"
    code += "columns = " + str(columns) + "\n"
    code += "for column in columns:\n"
    code += "\tmean=data[column].mean()\n"
    code += "\tdata[column].replace(missing_values, mean, inplace=True)\n"
    code += "\tdata[column].fillna(column, inplace=True)\n"
    add_code(filename, file_collection, code)
    return json.dumps({'message': 'file updated by mean successfully'}), 200


@app.route('/replace-by-median', methods=['GET', 'POST'])
def replace_by_median():
    print("called /replace-by-median")
    if 'fileKey' not in request.form or 'columns' not in request.form:
        return 'Please send fileKey and column names', 500
    filename = request.form['fileKey']
    columns = json.loads(request.form['columns'])
    print('incoming file', filename)
    print('incoming data', columns)
    data = get_file(filename, app.config['UPLOAD_FOLDER'])
    stats = find_stats(filename)
    print('current file stats', stats)
    medians = dict()
    missing_indexes = dict()
    for column in columns:
        missing_indexes[column] = list(np.where(data[column].isnull())[0])
        if column in stats['median']:
            medians[column] = stats['median'][column]
    print('found median for', medians)
    missing_val_set = set(['-', '?', 'na', 'n/a', 'NA', 'N/A'])
    for i in range(len(data)):
        for column in columns:
            if data[column][i] in missing_val_set:
                missing_indexes[column].append(i)
    print('missing indexes', missing_indexes)
    for column in columns:
        if column in missing_indexes:
            data.loc[missing_indexes[column], column] = medians[column]
    # now write updated dictionary to csv and json again
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], filename)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.json'), orient='records')
    code = '# code to replace missing values by mean\n'
    code += "missing_values = ['-', '?', 'na', 'n/a', 'NA', 'N/A']\n"
    code += "# columns you have selected\n"
    code += "columns = " + str(columns) + "\n"
    code += "for column in columns:\n"
    code += "\tmedian = data[column].median()\n"
    code += "\tdata[column].replace(missing_values, mean, inplace=True)\n"
    code += "\tdata[column].fillna(column, inplace=True)\n"
    add_code(filename, file_collection, code)
    return json.dumps({'message': 'file updated by median successfully'}), 200


@app.route('/file-data', methods=['GET', 'POST'])
def get_file_data():
    print("called /file-data")
    print('form', request.form)
    if 'fileKey' not in request.form:
        print('sending 500')
        return 'Please send file-key', 500
    file_key = request.form['fileKey']
    print('file-key', file_key)
    file_collection = db['data_files']

    ls = list(file_collection.find({'filename': file_key}))

    code = "# reading the head of the file\n"
    code += "print(data.head())"
    add_code(file_key, file_collection, code)

    if len(ls) == 1:
        return open(os.path.join(app.config['UPLOAD_FOLDER'], ls[0]['filename'])+'.json').read(), 200
    else:
        return 'Internal Server Error', 500


@app.route('/stats-by-column', methods=['POST'])
def stats_by_col():
    print("called /stats-by-column")
    if 'fileKey' not in request.form or 'column' not in request.form:
        print('sending 500')
        return 'Please send file-key and/or column', 500
    file_key = request.form['fileKey']
    column = request.form['column']
    code = "# getting statistics by columns\n"
    code += "column = '" + column + "'\n"
    print('column', column)
    print('file-key', file_key)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    stats = data[column].describe().to_dict()
    code += "stats = data[column].describe().to_dict()\n"
    print('our stats are', stats)
    modified_stats = dict()
    code += "modified_stats = dict()\n"

    if 'mean' in stats:     # if column is numeric
        modified_stats['Total Count'] = stats['count']
        code += "modified_stats['Total Count'] = stats['count']\n"
        modified_stats['Average/Mean'] = stats['mean']
        code += "modified_stats['Average/Mean'] = stats['mean']\n"
        modified_stats['Standard Deviation (std)'] = stats['std']
        code += "modified_stats['Standard Deviation (std)'] = stats['std']\n"
        modified_stats['Quartile 1(Q1)'] = stats['25%']
        code += "modified_stats['Quartile 1(Q1)'] = stats['25%']\n"
        modified_stats['Quartile 2 (Q2)/Median'] = stats['50%']
        code += "modified_stats['Quartile 2 (Q2)/Median'] = stats['50%']\n"
        modified_stats['Quartile 3(Q3)'] = stats['75%']
        code += "modified_stats['Quartile 3(Q3)'] = stats['75%']\n"
        modified_stats['Minimum Value'] = stats['min']
        code += "modified_stats['Minimum Value'] = stats['min']\n"
        modified_stats['Maximum Value'] = stats['max']
        code += "modified_stats['Maximum Value'] = stats['max']\n"
        modified_stats['Missing Count'] = int(data[column].isnull().sum())
        code += "modified_stats['Missing Count'] = int(data[column].isnull().sum())\n"
        modified_stats['Missing Count (%)'] = modified_stats['Missing Count'] / len(data) * 100
        code += "modified_stats['Missing Count (%)'] = modified_stats['Missing Count'] / len(data) * 100\n"

    else:
        modified_stats['Total Count'] = int(stats['count'])
        code += "modified_stats['Total Count'] = int(stats['count'])\n"
        modified_stats['Total Unique Values'] = int(stats['unique'])
        code += "modified_stats['Total Unique Values'] = int(stats['unique'])\n"
        modified_stats['Most Occurring value/mode'] = stats['top']
        code += "modified_stats['Most Occurring value/mode'] = stats['top']\n"
        modified_stats['Frequency of Most Occurring value'] = int(stats['freq'])
        code += "modified_stats['Frequency of Most Occurring value'] = int(stats['freq'])\n"
        modified_stats['Frequency of Most Occurring value (%)'] = stats['freq'] / len(data) * 100
        code += "modified_stats['Frequency of Most Occurring value (%)'] = stats['freq'] / len(data) * 100\n"
        modified_stats['Missing Count'] = int(data[column].isnull().sum())
        code += "modified_stats['Missing Count'] = int(data[column].isnull().sum())\n"
        modified_stats['Missing Count (%)'] = modified_stats['Missing Count'] / len(data) * 100
        code += "modified_stats['Missing Count (%)'] = modified_stats['Missing Count'] / len(data) * 100\n"
    add_code(file_key, file_collection, code)
    return json.dumps(modified_stats)


@app.route('/sample-files', methods=['GET'])
def get_sample_files():
    print("called /sample-files")
    public_tokens = ['b83aa9a7-d265-45df-b584-223b81a3b557_1000_Sales_Records.csv',
                     'ef805257-1f8d-4bbd-a31e-ab78c42e5890_FL_insurance_sample.csv',
                     'e76d9c91-05d4-4307-a9fa-d7ae79884931_Diabetess.csv']
    file_collection = db['data_files']
    file_collection.find()
    files = list(file_collection.find({'filename': {'$in': public_tokens}}))
    for file in files:
        del file['_id']
        del file['timestamp']
        file['token'] = file['filename']
        split_index = file['filename'].find('_')
        file['filename'] = file['filename'][split_index+1:]
    print('output', files)
    return json.dumps(files), 200


@app.route('/naive-bayes', methods=['POST'])
def naive_bayes_classifier():
    print("called /naive-bayes")
    if 'fileKey' not in request.form or 'columns' not in request.form:
        return 'Please send fileKey and column names', 500
    filename = request.form['fileKey']
    columns = json.loads(request.form['columns'])
    print('incoming file', filename)
    print('incoming data', columns)
    data = get_file(filename, app.config['UPLOAD_FOLDER'])


@app.route("/remove-missing", methods=['POST'])
def remove_missing():
    print("called /remove-missing")
    if 'fileKey' not in request.form or 'column' not in request.form:
        return 'Please send fileKey and column names', 500
    file_key = request.form['fileKey']
    column = request.form['column']
    print('column', column)
    print('file-key', file_key)
    print('incoming', request.form)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    data = data[pd.notnull(data[column])].reset_index(drop=True)
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')

    # now adding code
    '''code = '# code to remove missing values\n'
    code += "# columns you have selected\n"
    code += "column = '" + column + "'\n"
    code += "data = data[pd.notnull(data[column])].reset_index(drop=True)\n"
    add_code(file_key, file_collection, code)'''
    print('removed missing value')
    return json.dumps({"message" : "Removed missing values successfully"}), 200


@app.route("/replace-by-specific", methods=['POST'])
def replace_by_specific():
    print("called /replace-by-specific")
    if 'fileKey' not in request.form or 'column' not in request.form or 'value' not in request.form:
        return 'Please send fileKey and column names', 500
    file_key = request.form['fileKey']
    column = request.form['column']
    value = request.form['value']
    print('column', column)
    print('file-key', file_key)
    print('value', value)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    value = np.dtype(data[column].dtype.name).type(value)
    data[column].fillna(value, inplace=True).reset_index(inplace=True, drop=True)
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')

    # now adding code
    '''code = '# code to replace by specific value\n'
    code += "# columns you have selected\n"
    code += "column = '" + column + "'\n"
    code += "value = np.dtype(data[column].dtype.name).type(value)\n"
    code += "data[column].fillna(value, inplace=True)\n"
    code += "data = data[pd.notnull(data[column])].reset_index(drop=True)\n"
    add_code(file_key, file_collection, code)'''

    return json.dumps({"message" : "File Updated Successfully"}), 200


@app.route("/replace-value", methods=['POST'])
def replace_value():
    print("called /replace-value")
    if 'fileKey' not in request.form or 'column' not in request.form or 'oldValue' not in request.form and 'newValue' not in request.form:
        return 'Please send fileKey and column names', 500
    file_key = request.form['fileKey']
    column = request.form['column']
    old_value = request.form['oldValue']
    new_value = request.form['newValue']
    print('replace value')
    print('column', column)
    print('file-key', file_key)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    old_value = np.dtype(data[column].dtype.name).type(old_value)
    new_value = np.dtype(data[column].dtype.name).type(new_value)
    data[column].replace(old_value, new_value, inplace=True)
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')

    # now adding code
    '''code = '# code to replace by given value with specific value\n'
    code += "column = '" + column + "'\n"
    code += "old_value = '" + request.form['oldValue'] + '"\n'
    code += "old_value = np.dtype(data[column].dtype.name).type(old_value)\n"
    code += "new_value = '" + request.form['newValue'] + '"\n'
    code += "new_value = np.dtype(data[column].dtype.name).type(new_value)\n"
    code += "data[column].replace(old_value, new_value, inplace=True)\n"
    add_code(file_key, file_collection, code)'''
    return json.dumps({"message" : "Replaced values successfully"}), 200


@app.route("/remove-range", methods=['POST'])
def remove_range():
    print("called /remove-range")
    if 'fileKey' not in request.form or 'column' not in request.form or 'min' not in request.form and 'max' not in request.form:
        return 'Please send fileKey and column names', 500
    file_key = request.form['fileKey']
    column = request.form['column']
    min_value = str(request.form['min'])
    max_value = str(request.form['max'])
    print('column', column)
    print('file-key', file_key)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    print('min value', min_value, 'type', type(min_value))
    print('max value', max_value)
    print('pandas version', pd.__version__)
    min_value = np.dtype(data[column].dtype.name).type(min_value)
    max_value = np.dtype(data[column].dtype.name).type(max_value)
    indexes = data.loc[(data[column] > max_value) | (data[column] < min_value)].index
    print('indexes', indexes, 'pd', type(data))
    data.drop(labels=list(indexes), axis=0, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')

    # now adding code
    '''code = '# code to remove values outside range\n'
    code += "column = '" + column + "'\n"
    code += "min_value = '" + min_value + "'\n"
    code += "max_value = '" + max_value + "'\n"
    code += "min_value = np.dtype(data[column].dtype.name).type(min_value)\n"
    code += "max_value = np.dtype(data[column].dtype.name).type(max_value)\n"
    code += "indexes = data.loc[(data[column] > max_value) | (data[column] < min_value)].index\n"
    code += "data.drop(labels=list(indexes), axis=0, inplace=True)\n"
    code += "data.reset_index(inplace=True, drop=True)\n"
    add_code(file_key, file_collection, code)'''

    return json.dumps({"message": "Removed rows outside range Successfully"}), 200


@app.route('/get-help', methods=['GET', 'POST'])
def get_help():
    print("called /get-help")
    content = "Nothing to display"
    # jsonified_data = json.loads(req_data)
    topic = request.form['topic']
    print('topic', topic)
    data_collection = db['help-data']
    results = list(data_collection.find({'topic': topic}))
    print(results)
    print('results', results[0]['description'])
    if len(results) == 1:
        return json.dumps({'description': results[0]['description']})
    else:
        print('results', results)
    return json.dumps({'description': "something is wrong!"})


@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    print("called /visualization")
    if 'fileKey' not in request.form or 'columns' not in request.form or 'visType' not in request.form:
        return 'Please send fileKey and column names', 500
    print('columns', request.form['columns'], 'type', type(request.form['columns']))
    file_key= request.form['fileKey']
    columns = request.form['columns'].split(',')
    vis_type = request.form['visType']
    # code to generate visualization
    code = "# code to generate visualization\n"
    code += "columns = " + str(columns) + "\n"
    print('columns', columns)
    print('file-key', file_key)
    print('visType', vis_type)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    data = data[(data.values != '?').all(axis=1)]
    data.dropna(inplace=True)
    code += "# need to remove missing values\n"
    code += "data.dropna(inplace=True)"
    if vis_type == 'histogram':
        output = []
        for column in columns:
            print('generating histogram for column', column)
            hist_filename, hist_code = generate_histogram(data[column].copy(), column, app.config['VIZ_FOLDER'], data[column].dtype.name)
            output.append(hist_filename)
            code += hist_code
        print('output', output)
        add_code(file_key, file_collection, code)
        return json.dumps({'images': output})
    elif vis_type == 'Correlation':
        new_cols, rows, heatmap, corr_code = generate_correlation(data, columns, app.config['VIZ_FOLDER'])
        code += corr_code
        print('sending', json.dumps({'correlation': {'rows':rows, 'columns': new_cols}, 'images': [heatmap]}))
        add_code(file_key, file_collection, code)
        return json.dumps({'correlation': {'rows':rows, 'columns': new_cols}, 'images': [heatmap]})
    elif vis_type == 'Box Plot':
        print('will generate box plot')
        plots = []
        for column in columns:
            image, new_code = generate_boxplot(data[column], column, app.config['VIZ_FOLDER'])
            plots.append(image)
            code += new_code
        add_code(file_key, file_collection, code)
        print('box plots', plots)
        return json.dumps({'images': plots}), 200
    elif vis_type == 'Apply PCA':
        pca_plot = apply_pca(data, columns, app.config['VIZ_FOLDER'])
        return json.dumps({'images': [pca_plot]}), 200



@app.route('/standard-scale', methods=['GET', 'POST'])
def standard_scale():
    print("called /standard-scale")
    if 'fileKey' not in request.form or 'columns' not in request.form:
        return 'Please send fileKey and column names', 500
    print('columns', request.form['columns'], 'type', type(request.form['columns']))
    file_key= request.form['fileKey']
    columns = request.form['columns'].split(',')
    print('columns', columns)
    print('file-key', file_key)
    code = "# standard-scaling given columns\n"
    code += " columns = " + str(columns) + "\n"
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    data = data[(data.values != '?').all(axis=1)]
    data.dropna(inplace=True)
    std_code = apply_std_scaling(data, columns)
    code += std_code
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')
    add_code(file_key, file_collection, code)
    return json.dumps({"message": "Updated data with standard scaling Successfully"}), 200


@app.route('/random-forest', methods=['GET', 'POST'])
def random_forest():
    print("called /random-forest")
    feature_file = None
    roc_file = None
    conf_file = None
    accuracy = None
    report_dict = None
    imgErr = None
    model_file = None
    code = "# Code to generate random forest classifier and its report\n"
    if any([i not in request.form for i in ['fileKey', 'valuation', 'columns', 'n_estimator', 'max_depth', 'valuationType']]):
        print('value not found')
        return 'Please send missing values', 500
    print('valuationType', request.form['valuationType'])
    valuation = request.form['valuation']
    file_key = request.form['fileKey']
    columns = request.form['columns'].split(',')
    print('selected columns for random forest', columns)
    code += "columns = " + str(columns) + "\n"
    n_estimator = request.form['n_estimator']
    max_depth = request.form['max_depth']
    valuation_type = request.form['valuationType']
    print('valuation', valuation, type(valuation))

    if len(valuation) == 0:
        valuation = 0.3
    code += "valuation = " + str(valuation) + "\n"

    if valuation_type is None:
        valuation_type = 'TTS'

    print('n esti', n_estimator)
    print('depth', max_depth, type(max_depth), 'len', len(max_depth))
    if n_estimator == '':
        n_estimator = 100
    else:
        n_estimator = int(n_estimator)

    code += "n_estimator = " + str(n_estimator) + "\n"

    if max_depth == '' or max_depth is None:
        max_depth = 2
    else:
        max_depth = int(max_depth)

    code += "max_depth = " + str(max_depth)
    rmse_score_dict = dict()
    rmse_mean = None
    rmse_std = None
    imgErr = None
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    data = data[(data.values != '?').all(axis=1)]
    data.dropna(inplace=True)
    target = request.form['target_col']
    code += "target = " + str(target) + "\n"
    data = data[columns + [target]]     # only selecting necessary columns
    code += "data = data[columns + [target]]     # only selecting necessary columns\n"
    clf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=0)
    code += "clf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=0)\n"

    fig, ax = plt.subplots()
    if valuation_type == 'TTS':
        print('target value', target)
        valuation = float(valuation)    # converting incoming split ratio to float
        code += "valuation = float(valuation)"
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, 1), data[target], test_size=valuation, random_state=42)
        code += "x_train, x_test, y_train, y_test = train_test_split(data.drop(target, 1), data[target], test_size=valuation, random_state=42)\n"
        clf.fit(x_train, y_train)
        code += "clf.fit(x_train, y_train)\n"
        y_pred = clf.predict(x_test)
        code += "y_pred = clf.predict(x_test)\n"
        feat = clf.feature_importances_
        code += "feat = clf.feature_importances_\n"
        feat = pd.DataFrame(clf.feature_importances_, index=x_train.columns, columns=['importance']).sort_values(
            'importance', ascending=False)
        code += "feat = pd.DataFrame(clf.feature_importances_, index=x_train.columns, columns=['importance']).sort_values('importance', ascending=False)\n"
        feat.plot(kind='bar')
        code += "feat.plot(kind='bar')\n"
        plt.title("Feature importance ranging from highest to lowest")
        plt.grid(True, axis='y')
        plt.hlines(y=0, xmin=0, xmax=len(feat), linestyles='dashed')
        plt.tight_layout()
        feature_file= save_plot(plt, app.config['VIZ_FOLDER'], 'feature_importance')
        print('feature importance file', feature_file)
        # now generating and saving ROC plot
        preds = clf.predict_proba(x_test)[:, 1]
        code += "preds = clf.predict_proba(x_test)[:, 1]\n"
        y_test_temp = pd.Categorical(y_test).codes
        code += "y_test_temp = pd.Categorical(y_test).codes\n"
        fpr, tpr, thresholds = metrics.roc_curve(y_test_temp, preds)
        code += "fpr, tpr, thresholds = metrics.roc_curve(y_test_temp, preds)\n"
        roc_auc = metrics.auc(fpr, tpr)
        code += "roc_auc = metrics.auc(fpr, tpr)\n"
        plt.figure()
        code += "plt.figure()\n"
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        code += "plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)\n"
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        roc_file = save_plot(plt, app.config['VIZ_FOLDER'], 'roc_curve')
        print('roc file', roc_file)
        # now generating ans saving confusion matrix
        accuracy = metrics.accuracy_score(y_test, y_pred)
        code += "accuracy = metrics.accuracy_score(y_test, y_pred)\n"
        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        code += "report = metrics.classification_report(y_test, y_pred, output_dict=True)\n"
        code += "print('report dictionary', report)\n"
        print(report)
        print(type(report))
        report_dict = generate_report(report)
        # now generating confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        code += "cm = metrics.confusion_matrix(y_test, y_pred)\n"
        classes = data[target].unique()
        code += "classes = data[target].unique()\n"
        classes = list(classes)
        code += "classes = list(classes)\n"
        length = len(classes)
        code += "length = len(classes)\n"
        classes = []
        code += "classes = []\n"
        for i in range(length):
            classes.append(i)
        code += "for i in range(length):\n"
        code += "\tclasses.append(i)\n"
        fig, ax = plt.subplots()
        code += "fig, ax = plt.subplots()\n"
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        code += "im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)\n"
        ax.figure.colorbar(im, ax=ax)
        code += "ax.figure.colorbar(im, ax=ax)\n"
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')
        code += """
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')\n
        """
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        plt.tight_layout()
        conf_file = save_plot(plt, app.config['VIZ_FOLDER'], 'confusion_matrix')
        model_file, _ = save_model(clf, columns, 'random_forest', app.config['UPLOAD_FOLDER'])
    else:
        cv_fold = int(request.form['valuation'])
        code += "cv_fold = int(request.form['valuation'])\n"
        target_col_data = pd.Categorical(data[target]).codes
        code += "target_col_data = pd.Categorical(data[target]).codes\n"
        scores = cross_val_score(clf, data.drop([target], axis=1), target_col_data,
                                 scoring="neg_mean_squared_error", cv=cv_fold)
        code += """
        scores = cross_val_score(clf, data.drop([target], axis=1), target_col_data,
                                 scoring="neg_mean_squared_error", cv=cv_fold)\n"""
        print(scores)
        code += "print('scores', scores)\n"
        rmse_scores = np.sqrt(-scores).round(4)
        code += "rmse_scores = np.sqrt(-scores).round(4)\n"
        print(rmse_scores)
        code += "print('rmse score', rmse_score)\n"
        rmse_mean = rmse_scores.mean().round(4)
        code += "rmse_mean = rmse_scores.mean().round(4)\n"
        print("Mean:", rmse_mean)
        code += "print('Mean', rmse_mean)\n"
        rmse_std = rmse_scores.std().round(4)
        code += "rmse_std = rmse_scores.std().round(4)\n"
        print("Standard deviation:", rmse_std)
        code += "print('Standard deviation:', rmse_std)\n"
        key = 0
        for item in rmse_scores:
            rmse_score_dict[key] = item
            key += 1

        plt.figure()
        code += "plt.figure()\n"
        rdf = pd.DataFrame({'error':rmse_scores})
        code += "rdf = pd.DataFrame({'error':rmse_scores})\n"
        rdf.plot(kind='bar')
        code += "rdf.plot(kind='bar')\n"
        # plt.bar(np.arange(len(rmse_score_dict)), rmse_scores, align='center', alpha=0.5)
        # rdf.plot(kind='bar')
        plt.xlabel('folds')
        plt.ylabel('Error')
        # plt.show()
        plt.tight_layout()
        imgErr = save_plot(plt, app.config['VIZ_FOLDER'], 'random_error')

    output = dict()
    output['imgFeat'] = feature_file
    output['imgROC'] = roc_file
    output['imgConf'] = conf_file
    output['accuracy'] = accuracy
    output['report'] = report_dict
    output['rmse_scores'] = rmse_score_dict
    output['mean'] = rmse_mean
    output['std'] = rmse_std
    output['imgErr'] = imgErr
    output['model_file'] = model_file

    print('returning', output)
    # adding code
    add_code(file_key, file_collection, code)
    return json.dumps(output), 200


@app.route('/linear-regression', methods=['GET', 'POST'])
def linear_regression():
    print("called /linear-regression")
    if any([i not in request.form for i in ['fileKey', 'valuation', 'columns', 'valuationType', 'target_col']]):
        print('value not found')
        return 'Please send missing values', 500
    code = "# Code to generate random forest classifier and its report\n"
    rmse_score = None
    rmse_mean = None
    rmse_std = None
    feature_file = None
    linear_fit_file = None
    mse = None
    r2s = None
    model_file = None
    valuation = request.form['valuation']
    file_key = request.form['fileKey']
    columns = request.form['columns'].split(',')
    code += "columns = " + str(columns) + "\n"
    valuation_type = request.form['valuationType']
    print('valuation', valuation, type(valuation))
    target = request.form['target_col']
    print("columns: ",columns)
    print("target: ",target)
    if len(valuation) == 0:
        valuation = 0.3

    valuation = float(valuation)
    code += "valuation = " + str(valuation) + "\n"

    if valuation_type is None:
        valuation_type = 'TTS'

    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    print('data:', data)
    #data = data[(data.values != '?').all(axis=1)]
    print('data after conversion:', data)
    data.dropna(inplace=True)
    data = data[columns + [target]]     # only selecting necessary columns
    code += "data = data[columns + [target]]     # only selecting necessary columns\n"
    # before putting data into model, need to convert it into labels
    regression = LinearRegression()
    encoder_object = dict()     # this will store encoder for each column which is object
    for feature in columns:
        print('converting column', feature)
        if str(data[feature].dtype) == 'object':
            encoder = LabelEncoder()    # need encoder to convert object to numeric
            encoder.fit(data[feature])
            data[feature] = encoder.transform(data[feature])
            encoder_object[feature] = encoder
    if valuation_type == 'TTS':
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, 1), data[target], test_size=valuation,
                                                            random_state=42)
        regression.fit(x_train, y_train)
        predicted = regression.predict(x_test)
        mse = metrics.mean_squared_error(y_test, predicted).round(4)
        print('mse', mse)
        r2s = metrics.r2_score(y_test, predicted).round(4)
        print(r2s)
        coef = regression.coef_

        bias = regression.intercept_
        print('bias', bias)
        coefs_df = pd.DataFrame(coef, columns)
        coefs_df.columns = ["coef"]
        coefs_df["abs"] = coefs_df.coef.apply(np.abs)
        coefs_df = coefs_df.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

        plt.figure()
        coefs_df.coef.plot(kind='bar')
        plt.title("Feature importance ranging from highest to lowest")
        plt.grid(True, axis='y')
        plt.hlines(y=0, xmin=0, xmax=len(coefs_df), linestyles='dashed')
        plt.tight_layout()
        # saving the plot
        feature_file = save_plot(plt, app.config['VIZ_FOLDER'], '_feature_importance')
        print('feature importance file', feature_file)

        # another image
        attributes = x_test.columns
        r = math.ceil(len(attributes) / 3)
        print(attributes)
        for count, attribute in enumerate(attributes):
            print('attr', attribute)
            plt.subplot(r, 3, count+1).scatter(x_test[attribute], y_test)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_test[attribute], y_test)
            line = slope * x_test[attribute] + intercept

            plt.plot(x_test[attribute], y_test, 'o', x_test[attribute], line)
            plt.title(attribute)

        plt.tight_layout()
        linear_fit_file = save_plot(plt, app.config['VIZ_FOLDER'], '_linear_fit_')
        model_file, _ = save_model(regression, columns, 'linear_regression', app.config['UPLOAD_FOLDER'], encoder_object)
    else:
        valuation = int(valuation)
        scores = cross_val_score(regression, data.drop([target], axis=1), data[target], scoring="neg_mean_squared_error",
                                 cv=valuation)
        print(scores)
        rmse_scores = np.sqrt(-scores).round(4)
        print(rmse_scores)
        rmse_mean = rmse_scores.mean().round(4)
        print("Mean:", rmse_mean)
        rmse_std = rmse_scores.std().round(4)
        print("Standard deviation:", rmse_std)
        rmse_score_dict = dict()
        for index, item in enumerate(rmse_scores):
            rmse_score_dict[index] = item

        rmse_score = generate_rmse_table(rmse_score_dict)
    return json.dumps({
        'report': rmse_score,
        'mean': rmse_mean,
        'std': rmse_std,
        'imgFeat': feature_file,
        'linearFit': linear_fit_file,
        'r2s': r2s,
        'mse': mse,
        'model_file': model_file
    }), 200


@app.route('/hierarchical', methods=['GET', 'POST'])
def hierarchical():
    print("called /hierarchical")
    print("request.form: ",request.form)
    if request.method == 'POST':
        print('inside hierarchical clustering')
        features = request.form['columns'].split(',')
        print('incoming features', features, 'type', type(features))
        file_key = request.form['fileKey']
        data = get_file(file_key, app.config['UPLOAD_FOLDER'])
        data = data[(data.values != '?').all(axis=1)]
        data.dropna(inplace=True)
        data = data[features]
        if 'column1' not in request.form or 'column2' not in request.form:
            column1 = features[0]
            column2 = features[1]
        else:
            column1 = request.form['column1']
            column2 = request.form['column2']
        if 'n_clusters' not in request.form:    # did not came with request, set default to 2
            n_clusters = 2
        else:   # came in request, convert it into integer
            n_clusters = int(request.form['n_clusters'])
        # convert features if they are not already
        for feature in features:
            if str(data[feature].dtype) == 'object':
                data[feature] = pd.Categorical(data[feature]).codes

        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(data)
        plt.figure()
        plt.scatter(data[column1], data[column2], c=cluster.labels_)
        plt.tight_layout()
        plot_name = save_plot(plt, app.config['VIZ_FOLDER'], 'clustering')
        # below code for building dendogram

        plt.figure()
        dend = shc.dendrogram(shc.linkage(data, method='ward'))
        plt.tight_layout()
        dend_plot = save_plot(plt, app.config['VIZ_FOLDER'], 'dendrogram')
        return json.dumps({'cluster': plot_name, 'dendrogram': dend_plot}), 200


@app.route('/svm', methods=['POST'])
def svm():
    print("called /svm")
    feature_file = None
    roc_file = None
    conf_file = None
    accuracy = None
    report_dict = None
    imgErr = None
    code = "# Code to generate random forest classifier and its report\n"
    if any([i not in request.form for i in ['fileKey', 'valuation', 'columns', '', 'valuationType']]):
        print('value not found')
        return 'Please send missing values', 500
    code = "\n"
    print('valuationType', request.form['valuationType'])
    valuation = request.form['valuation']
    file_key = request.form['fileKey']
    columns = request.form['columns'].split(',')
    print('selected columns for random forest', columns)
    code += "columns = " + str(columns) + "\n"
    n_estimator = request.form['n_estimator']
    max_depth = request.form['max_depth']
    valuation_type = request.form['valuationType']
    print('valuation', valuation, type(valuation))
    rmse_score_dict = dict()
    rmse_mean = None
    rmse_std = None
    imgErr = None
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    data = data[(data.values != '?').all(axis=1)]
    data.dropna(inplace=True)
    target = request.form['target_col']
    code += "target = " + str(target) + "\n"
    data = data[columns + [target]]     # only selecting necessary columns

    if len(valuation) == 0:
        valuation = 0.3
    code += "valuation = " + str(valuation) + "\n"

    if valuation_type is None:
        valuation_type = 'TTS'
    if 'kernel-value' not in request.form:
        kernel_value = 'linear'
    else:
        kernel_value = request.form['kerna-value']
    clf = SVC(gamma='auto', kernel=kernel_value, probability=True)
    fig, ax = plt.subplots()
    if valuation_type == 'TTS':
        valuation = float('valuation')
        x_train, x_test, y_train, y_test = train_test_split(data.drop('Class', 1), data['Class'], random_state=98)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        cols = x_train.columns
        coef = clf.coef_
        col_names = list(features)
        importance = list(coef[0])
        coefs_df = pd.DataFrame(importance, col_names)
        coefs_df.columns = ["coef"]
        coefs_df["abs"] = coefs_df.coef.apply(np.abs)
        coefs_df = coefs_df.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

        plt.figure()
        coefs_df.coef.plot(kind='bar')
        plt.title("Feature importance ranging from highest to lowest")
        plt.grid(True, axis='y')
        plt.hlines(y=0, xmin=0, xmax=len(coefs_df), linestyles='dashed')
        plt.tight_layout()
        preds = clf.predict_proba(x_test)[:, 1]
        y_test_temp = pd.Categorical(y_test).codes
        fpr, tpr, thresholds = metrics.roc_curve(y_test_temp, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.tight_layout()

        accuracy = metrics.accuracy_score(y_test, y_pred)
        print('accuracy', accuracy)

        report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
        print(generate_report(report_dict))

        cm = metrics.confusion_matrix(y_test, y_pred)
        classes = data[label].unique()
        classes = list(classes)
        length = len(classes)
        classes = []
        for i in range(length):
            classes.append(i)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        plt.tight_layout()

@app.route('/codebox', methods=['GET', 'POST'])
def codebox():
    print("called /codebox")
    if 'code' not in request.form:
        return {'message': 'Please send the code'}

    code = request.form['code']
    print('incoming code', code)
    temp_filename = app.config['UPLOAD_FOLDER'] + "\\" + str(uuid.uuid1()) + '.py'
    print('temp filename', temp_filename)
    with open(temp_filename, 'w') as file:
        file.write(code)
    output = subprocess.check_output('python ' + temp_filename, shell=True, cwd=app.config['UPLOAD_FOLDER'])
    print('output', output)
    os.remove(temp_filename)
    return json.dumps({'output': output.decode('utf-8')}), 200


@app.route('/get-code', methods=['GET', 'POST'])
def get_code():
    print("called /get-code")
    if 'fileKey' not in request.form:
        filename = 'f722e823-df64-423e-ba58-4913a316ab57_Diabetess.csv'
        # return {'message': 'Please send the fileKey'}
    else:
        filename = request.form['fileKey']
    query = {'filename': filename}
    file_collection = db['data_files']
    new_file = file_collection.find_one(query)
    print('new file', new_file)
    # code = ''.join(new_file['code']).replace('\n', '<br/>').replace('\t', '&nbsp'*4)
    code = ''.join(new_file['code'])
    print('generated code', code)
    return json.dumps({'code': code}), 200


@app.route('/predict', methods=['POST'])
def predict():
    print("called /predict")
    if 'model_key' not in request.form:  # first, check we got the key or not
        return json.dumps({'message': 'please send model key'}), 500
    model_key = request.form['model_key']
    # load required features for model, it will be list, check filename's format in helper.mining.save_mode
    meta_object = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], model_key+'_meta.sav'))
    features = meta_object['features']
    if any([feature not in request.form for feature in features]):
        return json.dumps({'message': 'Please send all the feature values'}), 500
    # load model only after features are checked
    model = joblib.load(os.path.join(app.config['UPLOAD_FOLDER'], model_key))
    test_data = []
    print('meta', meta_object)
    for feature in features:
        """
        convert object value to numeric if encoder is available in meta_object.
        Because it must have been converted during prediction
        """
        if feature in meta_object['encoder_object']:
            test_data.append(meta_object['encoder_object'][feature].transform([request.form[feature]])[0])
        else:
            test_data.append(float(request.form[feature]))

    prediction = str(model.predict([test_data])[0])    # convert output to string, just to be safe
    return json.dumps({'output': prediction}), 200


@app.route('/posts', methods=['GET', 'POST'])
def posts():
    print("called /posts")
    print('incoming', request.get_json())
    print('form', request.form['body'])
    print('sending', json.dumps({'id': 101, 'title': request.form['title'], 'body': request.form['body']}))
    return json.dumps({'userId': 123, 'id': 101, 'title': request.form['title'], 'body': request.form['body']}), 200

@app.route('/topic-modeling', methods=['GET', 'POST'])
def topicModeling():
    print("called /topic-modeling")
    print(request.form)
    print(request.form['fileKey'])
    file_key = request.form['fileKey']
    selectAllCols = request.form['selectAllCols']

    # retrieve latest file key from db if not retrieved from front end
    if file_key == "testkey":
        file_collection = db['data_files']
        lastEntry = file_collection.find().sort([('timestamp', -1)]).limit(1)
        if lastEntry.count()==0:
            return json.dumps({'description': "No file has been uploaded. Please upload atleast 1 file in the upload file tab."}), 200
        else:
            for entry in lastEntry:
                file_key = entry['filename']

    print("file_key: ", file_key)
    num_topics = request.form['n_topics']
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])

    if selectAllCols == "Yes":
        listOfWords = list(data.columns.values)
        listOfWords = listOfWords[1:]
    else:
        listOfWords = request.form['columns'].split(',')

    text_data = []
    text_data.append(listOfWords)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    ldamodel.save('model.gensim')

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model.gensim')
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

    file_collection = db['help-data']
    ls = list(file_collection.find({'topic': "TM_output"}))

    if len(ls)>0:
        versionNos = []
        for i in range(len(ls)):
            versionNos.append(int(ls[i]['version']))

        versionNos.sort()
        version = str(versionNos[len(versionNos)-1]+1)
        html_filename = 'topics_' + version + '.html'

    else:
        html_filename = 'topics_0.html'
        version = '0'


    html_file_url = app.config['VIZ_FOLDER'] + "\\" + html_filename
    pyLDAvis.save_html(lda_display, html_file_url)



    record = {'topic': "TM_output", 'version':version, 'description': html_filename}
    file_collection.insert_one(record)

    return json.dumps({'description': html_filename}), 200

@app.route('/word-embedding', methods=['GET', 'POST'])
def wordEmbedding():
    print("called /word-embedding")
    print(request.form['fileKey'])
    file_key = request.form['fileKey']
    print("file_key: ", file_key)
    selectAllCols = request.form['selectAllCols']

    # retrieve latest file key from db if not retrieved from front end
    if file_key == "testkey":
        file_collection = db['data_files']
        lastEntry = file_collection.find().sort([('timestamp', -1)]).limit(1)
        if lastEntry.count() == 0:
            return json.dumps({'result': "No file has been uploaded. Please upload atleast 1 file in the upload file tab."}), 200
        else:
            for entry in lastEntry:
                file_key = entry['filename']

    data = get_file(file_key, app.config['UPLOAD_FOLDER'])

    if (selectAllCols == "Yes"):
        listOfWords = list(data.columns.values)
    else:
        listOfWords = request.form['columns'].split(',')
        data = data[listOfWords]

    print("data: ",data)
    dataAsList = data.values.tolist()
    print("dataAsList: ",dataAsList)


    allWordsInAllFiles = []
    '''for eachRow in dataAsList:
        wordsPerFile = []
        for i in range(1,len(eachRow)):
            if float(eachRow[i])>0:
                wordsPerFile.append(listOfWords[i])
        allWordsInAllFiles.append(wordsPerFile)'''
    allWordsInAllFiles.append(listOfWords)

    print("allWordsInAllFiles:",allWordsInAllFiles)
        # train model
    model = Word2Vec(allWordsInAllFiles, min_count=1)
    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    print("words:",words)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    word2vec_plot = save_plot(plt, app.config['WE_FOLDER'], 'word2vec')
    print("word2vec_plot: ",word2vec_plot)

    return json.dumps({'saved_file': word2vec_plot}), 200

@app.route('/remove-stop-words', methods=['GET', 'POST'])
def removeStopWords():
    print("called /remove-stop-words")
    file_key = request.form['fileKey']
    list_columns_to_remove = request.form['columns'].split(',')
    print("list_columns_to_remove: ",list_columns_to_remove)
    data = get_file(file_key, app.config['UPLOAD_FOLDER'])
    print("data: ",data)
    data = data.drop(columns=list_columns_to_remove)
    print("data after removing columns/chosen words to remove:", data)
    data.to_csv(os.path.join(os.path.join(app.config['UPLOAD_FOLDER'], file_key)), index=False)
    data.to_json(os.path.join(app.config['UPLOAD_FOLDER'], file_key + '.json'), orient='records')
    # Update modified column - schema in statistics table in db
    metadata, code = calculate_meta(data)
    file_stats = db['statistics']
    file_stats.update({'filename': file_key}, {'$set':{'stats': metadata}})

    return json.dumps({"message": "Removed stop words from the tf-idf table successfully"}), 200

@app.route('/getUsersWorkflows', methods=['GET', 'POST'])
def getUsersWorkflows():
    print("called /getUsersWorkflows")
    print("request.form:", request.form)
    username = request.form['username']
    type = request.form['type']
    file_collection = db['workflow-data']
    ls = list (
        file_collection.find({'username': username, 'type': type}))
    list_of_workflow_names = []
    for row in ls:
        list_of_workflow_names.append(row['workflow_name'])

    return json.dumps({'result': list_of_workflow_names}), 200

@app.route('/getWorkflow', methods=['GET', 'POST'])
def getWorkflow():
    print("called /getWorkflow")
    print("request.form:", request.form)
    username = request.form['username']
    workflow_name = request.form['workflow_name']
    type = request.form['type']
    file_collection = db['workflow-data']
    ls = list(
        file_collection.find({'username': username, 'workflow_name': workflow_name, 'type': type}))  # check if an existing account with the same username exists
    if (len(ls) == 0):
        return json.dumps({'message': 'no workflow'}), 200

    lastWorkflowEntry = ls[len(ls)-1] # get last entry since we need to retrieve recently created workflow
    print("lastWorkflowEntry:",lastWorkflowEntry)
    texts = lastWorkflowEntry['texts']
    colors = lastWorkflowEntry['colors']
    locs = lastWorkflowEntry['locs']
    filename = lastWorkflowEntry['filename']
    return json.dumps({'message': 'success', 'texts':texts, 'colors':colors, 'locs':locs, 'filename':filename}), 200

@app.route('/deleteWorkflow', methods=['GET', 'POST'])
def deleteWorkflow():
    print("called /deleteWorkflow")
    print("request.form:", request.form)
    username = request.form['username']
    workflow_name = request.form['workflow_name']
    type = request.form['type']
    file_collection = db['workflow-data']
    file_collection.delete_many({'username': username,'workflow_name': workflow_name, 'type':type})
    return json.dumps({'message': 'Successly deleted workflow: '+workflow_name}), 200

@app.route('/saveWorkflow', methods=['GET', 'POST'])
def saveWorkflow():
    print("called /saveWorkflow")
    print("request.form:",request.form)
    username = request.form['username']
    workflow_name = request.form['workflow_name']
    type = request.form['type']
    filename = request.form['filename']
    # retrieve all the texts, colors, locations of the node data array
    texts = []
    key = "text0"
    i = 1
    while key in request.form:
        texts.append(request.form[key])
        key = key[0:4] + str(i)
        i += 1

    print("texts: ",texts)

    colors = []
    key = "color0"
    i = 1
    while key in request.form:
        colors.append(request.form[key])
        key = key[0:5] + str(i)
        i += 1

    print("colors: ",colors)


    locs = []
    key = "loc0"
    i = 1
    while key in request.form:
        locs.append(request.form[key])
        key = key[0:3] + str(i)
        i += 1

    print("locs: ",locs)


    file_collection = db['workflow-data']
    ls = list(file_collection.find({'username': username, 'workflow_name': workflow_name, 'type':type}))
    if (len(ls) == 0):
        record = {'username': username, 'workflow_name':workflow_name, 'filename':filename, 'type':type, 'texts':texts, 'colors': colors, "locs":locs}
        file_collection.insert(record)
        return json.dumps({'message': 'Saved workflow ' + workflow_name + ' successfully'}), 200
    else:
        file_collection.update({'username': username, 'workflow_name':workflow_name}, {'$set': {'filename':filename, 'type':type, 'texts':texts, 'colors': colors, "locs":locs}})
    return json.dumps({'message': 'Overwritten workflow '+workflow_name+' successfully'}), 200

@app.route('/displayDataWorkflow', methods=['GET', 'POST'])
def displayDataWorkflow():
    print("called /displayDataWorkflow")
    print("request.form:", request.form)
    username = request.form['username']
    workflow_name = request.form['workflow_name']
    type = request.form['type']
    key = request.form['key']
    filename = request.form['filename']
    topic = username + '_' + workflow_name + '_' + type + '_' + key
    html_files_db = db['help-data']
    # check if a html file containing the data at this node has been created before
    ls = list(html_files_db.find({'topic': topic}))
    description = username + '_' + workflow_name + '_' + type + '_' + key + '_' + filename.split('.')[0] + '.html'
    if (len(ls) == 0):  # no file has been uploaded with the same username and same workflow name
        print("HTML File does not exist. Created first time.")
        data = get_file(filename, app.config['UPLOAD_FOLDER'])
        data_html = data.to_html()
        # write html to file
        html_file_url = app.config['UPLOAD_FOLDER'] + "\\" + description
        data_html_file = open(html_file_url, "w")
        data_html_file.write(data_html)
        data_html_file.close()
        html_files_db.insert({'topic': topic, 'description': description})
    else:
        print("HTML File exists.")
        workflow_db = db['workflow-data']
        workflow_ls = list(workflow_db.find({'username': username,'workflow_name':workflow_name}))
        if (len(workflow_ls)>0):
            saved_filename = workflow_ls[len(workflow_ls) - 1]['filename']
            print("filename: ",filename)
            print("saved_filename: ",saved_filename)

            # different file uploaded for same workflow or file has been modified
            if filename!=saved_filename:
                print("New File uploaded.")

                # update help data in this case
                data = get_file(filename, app.config['UPLOAD_FOLDER'])
                data_html = data.to_html()
                # write html to file
                html_file_url = app.config['UPLOAD_FOLDER'] + "\\" + description
                data_html_file = open(html_file_url, "w")
                data_html_file.write(data_html)
                data_html_file.close()
                html_files_db.update({'topic': topic}, {'$set': {'description': description}})

    return json.dumps({'description': description}), 200


if __name__ == '__main__':
    app.run(debug=True)
