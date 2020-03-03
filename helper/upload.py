import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, freeze_support
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def returnTXTasSingleString(filePath):
    '''
    Return a single string containing all lines of the txt file
    :param filePath: the file path of the txt file
    :return:
    '''
    with open(filePath, encoding="utf8") as file:
        data = file.read().replace('\n', '')
    return data

def getNameFromPath(filePath):
    '''
    Get the exact filename from the file path
    :param filePath: the file path / file directory
    :return:
    '''
    folderList = filePath.split('\\')
    return folderList[len(folderList)-1]

def convert_files_csv_tfidf(listOfFiles, listOfFilePaths, csvPath):
    """
    Generate a csv file with all the td-idf values of all words in the passed list if files
    :param listOfFiles: list of all text files
    :param listOfFilePaths: list of all respective paths of the text files
    :param csvPath: path where the created csv file is saved
    :return:
    """
    listOfFilesAsStrs = []

    for eachFilePath in listOfFilePaths:
        listOfFilesAsStrs.append(returnTXTasSingleString(eachFilePath))

    # perform tf-idf and create result dataframe
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(listOfFilesAsStrs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    #df = df.transpose()
    #print("transposed df:",df)
    # create the csv file using the dataframe
    dfRowsAsCSV = df.to_csv(index=False).split('\r\n')
    #dfRowsAsCSV = df.to_csv().split('\r\n')
    #print("transposed df as csv:",df.to_csv())
    count = 0
    with open(csvPath, 'w') as csvFile:
        for eachLine in dfRowsAsCSV:
            if (eachLine==''):
                break
            if (count==0):
                # first row - header
                eachLine = "filename,"+eachLine
                count+=1
            else:
                # first column contains original file name
                filename = listOfFiles[count-1].filename
                print("filename:",filename)
                eachLine = filename+","+eachLine
                count+=1

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
    extension = filename.rsplit('.', 1)[1].lower()
    if extension == 'csv':
        '''return pd.read_csv(os.path.join(path, filename))
    elif extension == 'txt':
        filename = convert_txt_to_csv(os.path.join(path, filename))'''
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

