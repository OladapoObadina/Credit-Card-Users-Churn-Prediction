import pandas as pd
from flask import Flask, request,render_template
import pickle


app = Flask("__name__")

model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def loadPage():
	return render_template('homepage.html')

def get_data():
    inputQuery1 = int(request.form['query1'])
    inputQuery2 = request.form['query2']
    inputQuery3 = float(request.form['query3']) 
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = int(request.form['query8'])
    inputQuery9 = int(request.form['query9'])
    inputQuery10 = int(request.form['query10'])
    inputQuery11 = int(request.form['query11'])
    inputQuery12 = float(request.form['query12'])
    inputQuery13 = int(request.form['query13'])
    inputQuery14 = float(request.form['query14'])
    inputQuery15 = float(request.form['query15'])
    inputQuery16 = int(request.form['query16'])
    inputQuery17 = int(request.form['query17'])
    inputQuery18 = float(request.form['query18'])
    inputQuery19 = float(request.form['query19'])
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data, columns = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
                                           'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
                                           'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'])
     
     
    # Function to check if customer has revolving balance
    def rev_bal(x):
        if x['Total_Revolving_Bal'] == 0: 
            return False
        elif x['Total_Revolving_Bal'] != 0 : 
            return True
        else: 
            return 0
    
    new_df['Has_Revolving_Balance'] = new_df.apply(rev_bal, axis=1)
    
    
    #Convert object columns to category datatype
    categories = ['Gender','Education_Level', 'Marital_Status', 'Income_Category', 
              'Card_Category','Has_Revolving_Balance']
    new_df = new_df.apply(lambda col: col.astype('category') if col.name in categories else col)
    
    return new_df


@app.route('/send', methods=['POST'])
def show_data():
    user_data = get_data()
    prediction = model.predict(user_data.tail(1))
    probablity = model.predict_proba(user_data.tail(1))[:,1]
    
    if prediction==1:
        outcome1 = "Churn"
        outcome2 = "{}%".format(probablity*100)
    else:
        outcome1 = "Remain with the Bank"
        outcome2 = "{}%".format(probablity*100)


    return render_template('results.html', tables = [user_data.to_html(classes='data', header=True)], result1=outcome1, result2=outcome2)

if __name__ == "__main__":
    app.run(debug=True)