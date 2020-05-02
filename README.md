# capstone-backend-master
Back-end code (Python, MongoDB) of a web-based data mining tool used for various data mining operations like data cleaning, transformation, modelling, visualization on structured/unstructured files along with developer tools.

To run the back-end code:
1) Go to app1.py

2) Change the following 3 lines to point to your specific folder: 
app.config['UPLOAD_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\data-file'
app.config['VIZ_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static'
app.config['WE_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\word-embedding'

3) Open a terminal in the main folder and run - python app1.py
