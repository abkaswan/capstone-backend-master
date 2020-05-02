# capstone-backend-master
Back-end code (Python, MongoDB) of a web-based data mining tool used for various data mining operations like data cleaning, transformation, modelling, visualization on structured/unstructured files along with drag-drop workflow creation.

Adding data to the database:
1) mongo/help-data.json
2) The file contains all rows that need to be added to the table help-data.
3) Create a table help-data and add all the rows (JSON objects).

To run the back-end code:
1) Make sure Python is installed in your local machine.

2) Go to app1.py

3) Change the following 3 lines to point to your specific folder: 
app.config['UPLOAD_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\data-file',
app.config['VIZ_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static',
app.config['WE_FOLDER'] = r'C:\Users\rchak\Desktop\Capstone\capstone-backend-master\static\word-embedding'

4) Open a terminal in the main folder and run - python app1.py

5) Certain python libraries may not work or be installed in your local machine. The program shall throw an error and indicate the specific library name. Just run:
pip install libraryname
