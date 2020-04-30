import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, freeze_support
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pytesseract
import textract
from nltk.corpus import words, stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('words')

supportedImageExtension = ['bmp', 'gif', 'jfif', 'jpeg', 'jpg', 'png', 'pnm', 'tiff']

def performStemming(listOfFilesAsStrs):
    listOfFilesAsStrsStem = []
    for eachStr in listOfFilesAsStrs:
        eachStrNoPunc = re.sub(r'[^\w\s]', ' ', eachStr)  # replace all punctuations with spaces
        tokenizedWords = word_tokenize(eachStrNoPunc)
        # remove words that are not tokenized properly - meaningless words
        #tokenizedWords = list(filter(lambda a: a not in words.words(), tokenizedWords))
        rootWords = [] # new list which will contains all the stemmed version of all the words
        ps = PorterStemmer()
        for w in tokenizedWords:
            rootWord = ps.stem(w)
            # print(w ," stemmed as ", rootWord)
            rootWords.append(rootWord)
        listOfFilesAsStrsStem.append(' '.join([str(eachRootWord) for eachRootWord in rootWords]))
    return listOfFilesAsStrsStem

def performLemmatization(listOfFilesAsStrs):
    listOfFilesAsStrsLem = []
    for eachStr in listOfFilesAsStrs:
        eachStrNoPunc = re.sub(r'[^\w\s]',' ',eachStr) # replace all punctuations with spaces
        tokenizedWords = word_tokenize(eachStrNoPunc)
        # remove words that are not tokenized properly - meaningless words
        # tokenizedWords = list(filter(lambda a: a not in words.words(), tokenizedWords))
        rootWords = [] # new list which will contains all the lemmatized version of all the words
        lemmatizer = WordNetLemmatizer()
        for w in tokenizedWords:
            rootWord = lemmatizer.lemmatize(w) # convert nouns to root form
            rootWord = lemmatizer.lemmatize(rootWord, pos="v") # convert verbs to root form
            rootWord = lemmatizer.lemmatize(rootWord, pos="a") # convert adjectives to root form
            # print(w ," lemmatized as ", rootWord)
            rootWords.append(rootWord)
        listOfFilesAsStrsLem.append(' '.join([str(eachRootWord) for eachRootWord in rootWords]))
    return listOfFilesAsStrsLem

def removeStopWordsAndOtherUselessText(wordList):
    '''
    Remove stop words and other useless text like numbers from a given list of words
    :param wordList:
    :return:
    '''
    # get stop words from the stopwords library
    stopWordsInLibrary = set(stopwords.words('english'))
    updatedWordList = [word for word in wordList if word not in stopWordsInLibrary and not word.isdigit()]
    print("Update word list:",updatedWordList)
    return updatedWordList

def getFileExtension(filename):
    '''
    Return extensin of the filename - txt, csv, jpg, png, etc.
    :param filename: the full filename including the extension
    :return:
    '''
    extension = filename.rsplit('.', 1)[1].lower()
    return extension

def returnUnstructuredFileAsListOfParas(filePath):
    '''
    Return a list containing all paragraphs of the unstructured file
    :param filePath: the file path of the unstructured file
    :return:
    '''
    filename = getNameFromPath(filePath)
    extension = getFileExtension(filename)
    print("extension:"+extension)
    data = ''
    listOfParas = []
    if extension == 'txt':
        with open(filePath, encoding="utf8") as file:
            data = file.read()
            listOfParas = data.split('\n')

    elif extension == 'docx':
        data = textract.process(filePath)
        data = data.decode("utf-8")
        listOfParas = data.split('\n\n')

    print("listOfParas after conversion: ",listOfParas)
    return listOfParas

def returnUnstructuredFileAsSingleString(filePath):
    '''
    Return a single string containing all lines of the unstructured file
    :param filePath: the file path of the unstructured file
    :return:
    '''
    filename = getNameFromPath(filePath)
    extension = getFileExtension(filename)
    print("extension:"+extension)
    data = ''
    if extension == 'txt':
        with open(filePath, encoding="utf8") as file:
            data = file.read().replace('\n', '')
    elif extension == 'docx':
        data = textract.process(filePath)
        data = data.decode("utf-8")

    print("data after conversion: "+data)
    return data

def getNameFromPath(filePath):
    '''
    Get the exact filename from the file path
    :param filePath: the file path / file directory
    :return:
    '''
    folderList = filePath.split('\\')
    return folderList[len(folderList)-1]

def convert_files_csv_tfidf(onlyOneFile, stemming, lemmatization, listOfFiles, listOfFilePaths, csvPath):
    """
    Generate a csv file with all the td-idf values of all words in the passed list if files
    :param: onlyOneFile: flag to check if only 1 unstructured file has been uploaded so that the parsing happens differently
    :param: stemming: flag to to check if user wants stemming to be performed on the file
    :param: lemmatization: flag to to check if user wants lemmatization to be performed on the file
    :param listOfFiles: list of all text files
    :param listOfFilePaths: list of all respective paths of the text files
    :param csvPath: path where the created csv file is saved
    :return:
    """
    listOfFilesAsStrs = []

    if (onlyOneFile=="No"):
        for eachFilePath in listOfFilePaths:
            allContentsAsSingleString = returnUnstructuredFileAsSingleString(eachFilePath)
            if (allContentsAsSingleString!=""):
                listOfFilesAsStrs.append(allContentsAsSingleString)
    else:
        listOfFilesAsStrs = returnUnstructuredFileAsListOfParas(listOfFilePaths[0])

    if (stemming=="Yes"):
        listOfFilesAsStrs = performStemming(listOfFilesAsStrs)

    if (lemmatization=="Yes"):
       listOfFilesAsStrs = performLemmatization(listOfFilesAsStrs)

    if (len(listOfFilesAsStrs)==0):
        return "fail"

    # perform tf-idf and create result dataframe
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(listOfFilesAsStrs)
    feature_names = vectorizer.get_feature_names()
    print("feature_names:\n", feature_names)
    #feature_names = removeStopWordsAndOtherUselessText(feature_names)
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    # remove stop words and other useless words as columns from the dataframe df
    stopWordsInLibrary = set(stopwords.words('english'))
    listOfUselessWords = [word for word in feature_names if word in stopWordsInLibrary or word.isdigit()]
    print("list of useless words:\n", listOfUselessWords)
    df = df.drop(columns=listOfUselessWords)

    dfRowsAsCSV = df.to_csv(index=False).split('\r\n')

    count = 0
    with open(csvPath, 'w') as csvFile:
        for eachLine in dfRowsAsCSV:
            if (eachLine==''):
                break
            if (count==0):
                # first row - header
                if (onlyOneFile == "No"):
                    eachLine = "filename,"+eachLine
                else:
                    eachLine = "Paragraph Number," + eachLine
                count+=1
            else:
                # first column contains original file name
                if (onlyOneFile == "No"):
                    filename = listOfFiles[count-1].filename
                    print("filename:",filename)
                    eachLine = filename+","+eachLine
                else:
                    eachLine = str(count) + "," + eachLine

                count += 1

            eachLine = eachLine+'\n'
            csvFile.write(eachLine)
    csvFileName = getNameFromPath(csvPath)
    return csvFileName

def convert_files_csv_lda(onlyOneFile, stemming, lemmatization, listOfFiles, listOfFilePaths, csvPath, topics):
    """
    Generate a csv file with all the lda values of all words in the passed list if files
    :param: onlyOneFile: flag to check if only 1 unstructured file has been uploaded so that the parsing happens differently
    :param: stemming: flag to to check if user wants stemming to be performed on the file
    :param: lemmatization: flag to to check if user wants lemmatization to be performed on the file
    :param listOfFiles: list of all text files
    :param listOfFilePaths: list of all respective paths of the text files
    :param csvPath: path where the created csv file is saved
    :param topics: number of chosen topics
    :return:
    """
    listOfFilesAsStrs = []

    if (onlyOneFile=="No"):
        for eachFilePath in listOfFilePaths:
            allContentsAsSingleString = returnUnstructuredFileAsSingleString(eachFilePath)
            if (allContentsAsSingleString!=""):
                listOfFilesAsStrs.append(allContentsAsSingleString)
    else:
        listOfFilesAsStrs = returnUnstructuredFileAsListOfParas(listOfFilePaths[0])

    if (stemming=="Yes"):
        listOfFilesAsStrs = performStemming(listOfFilesAsStrs)

    if (lemmatization=="Yes"):
       listOfFilesAsStrs = performLemmatization(listOfFilesAsStrs);

    if (len(listOfFilesAsStrs)==0):
        return "fail"

    # perform lda and create result dataframe
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(listOfFilesAsStrs)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=topics, random_state=0).fit(tf)

    resultList = lda.transform(tf)
    print("resultList: ",resultList)

    list_of_columns = []
    for i in range(0, topics):
        col = 'Topic' + str(i+1)
        list_of_columns.append(col)

    df = pd.DataFrame(resultList, columns=list_of_columns)

    dfRowsAsCSV = df.to_csv(index=False).split('\r\n')

    count = 0
    with open(csvPath, 'w') as csvFile:
        for eachLine in dfRowsAsCSV:
            if (eachLine==''):
                break
            if (count==0):
                # first row - header
                if (onlyOneFile == "No"):
                    eachLine = "filename,"+eachLine
                else:
                    eachLine = "Paragraph Number," + eachLine
                count+=1
            else:
                # first column contains original file name
                if (onlyOneFile == "No"):
                    filename = listOfFiles[count-1].filename
                    print("filename:",filename)
                    eachLine = filename+","+eachLine
                else:
                    eachLine = str(count) + "," + eachLine

                count += 1

            eachLine = eachLine+'\n'
            csvFile.write(eachLine)
    csvFileName = getNameFromPath(csvPath)
    return csvFileName


def convert_txt_to_csv(filename, path, delimiter):
    """
    Convert txt file to csv format so that it can be parsed easily.
    Each line is assumed to be a row.
    The first line is assumed to be the header.
    :param filename: exact name of the txt file
    :param path: directory path where the txt file exists and where the converted csv should be saved as well
    :param: delimiter: column separator for each row - like space, comma, semicolon
    :return:
    """
    filePath = os.path.join(path, filename)
    csvPath = filePath.split('.')[0] + '.csv'
    with open(filePath) as txtFile:
        with open(csvPath, 'w') as csvFile:
            for lineInTXT in txtFile:
                    if (delimiter != ','):
                            lineInCSV = re.sub(delimiter, ",", lineInTXT.strip())+'\n'
                    else:
                        lineInCSV = lineInTXT.strip()+'\n'
                    #print("lineInCSV:",lineInCSV)
                    csvFile.write(lineInCSV)

    csvFileName = getNameFromPath(csvPath)
    return csvFileName


def allowed_file(filename, allowd_types):
    print('filename', filename.rsplit('.', 1)[1].lower())
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowd_types

def get_file(filename, path, delimiter='\t'):
    """
    Returns whole file
    :param filename: file-key
    :param path: path to the file
    :param delimiter: how you want to split the file
    :return: dataframe of the file
    """
    extension = getFileExtension(filename)
    if extension == 'csv':
        return pd.read_csv(os.path.join(path, filename))


def calculate_meta(data):
    code = '# generating information about CSV file\n'
    output = dict()
    output['columns'] = list(data.columns)
    data_types = dict()
    mean = dict()
    median = dict()
    mode = dict()
    for column in data.columns:
        data_type = str(data[column].dtype)
        if data_type == 'object':
            mode[column] = data[column].mode()[0]
            overall_type = 'categorical'
        else:
            mean[column] = data[column].mean()
            median[column] = data[column].median()
            overall_type = 'numerical'
        data_types[column] = data_type + '/' + overall_type

    output['dataTypes'] = data_types
    shape = data.shape
    output['shape'] = {'rows': shape[0], 'columns': data.shape[1]}
    output['missing'] = na_count(data)
    output['mean'] = mean
    output['median'] = median
    output['mode'] = mode
    print('missing', output['missing'])

    # now generating code for what we have done
    code = '# generating information about CSV file\n'
    code += "output = data.describe().to_dict()  # getting description of file and into Python dictionary\n"
    code += "for column in output.keys():\n"
    code += "\tdata_type = str(data[column].dtype)\n"
    code += "\tif data_type == 'object':\n"
    code += "\t\toverall_type = 'categorical'\n"
    code += "\telse:\n"
    code += "\t\toverall_type = 'numerical\n"
    code += "\toutput[column]['data_type'] = overall_type\n"
    code += "\toutput[column]['missing'] = data[column].isna().sum()\n"
    code += "print('overall stats', output)\n\n"
    return output, code


def na_count(data):
    print('type of data', type(data))
    missing_list = dict()
    cols = list(data.columns)
    for col in cols:
        #print('counting for col', col)
        count = 0
        temp = data[col]
        for i, v in temp.items():
            if v in ('-', '?', 'na', 'n/a', 'NA', 'N/A'):
                count += 1
        #print('column data', data[col].isnull())
        count += len(np.where(data[col].isnull())[0])
        missing_list[col] = count

    return missing_list

