from flask import Flask, render_template,url_for,request , redirect
import pandas as pd
from sklearn.externals import joblib
import json
from flask import jsonify, make_response

app = Flask(__name__)
@app.route('/')
def home():
 return render_template('home.html')
 
#pour l'affichage des données en format json
@app.route('/data', methods=['GET'])
def view_data():
    """Page de rendu des données pour pouvoir les examiner."""
    with open('D:/flask-project/data.json') as f:
        data = json.load(f)
    return render_template('data.html',  data=json.dumps(data))



@app.route('/predict', methods = [ 'POST'])
def predict():
 modelfile = 'D:/flask-project/final_prediction.pickle'
 model_load= open(modelfile, 'rb')
 model = joblib.load(model_load)
 if request.method == 'POST':
  comment = request.form['comment']
  data = [comment]
  dataph = {'phrase': data }
  df_ph= pd.DataFrame(dataph)
  my_prediction = model.predict_proba(df_ph['phrase'])
  ##recupération de resultat en df et l'enregistré de format json
  dialect= ['ALE', 'ALG','ALX','AMM','ASW','BAG','BAS','BEI','BEN','CAI','DAM','DOH','FES','JED','JER','KHA','MOS','MSA','MUS',
          'RAB','RIY','SAL','SAN','SFX','TRI','TUN']
  jdict = {'Dialect': dialect, 'Probabilitydialect': my_prediction[0].tolist()   }
  dfjsn= pd.DataFrame(jdict,columns=['Dialect', 'Probabilitydialect'])
  resultJSON = dfjsn.sort_values(by=['Probabilitydialect'], ascending=False)[:5]
  #resultJSON = dfjsn.to_json(orient= "index")
  Probabilitydialect = list(resultJSON['Probabilitydialect'])
  Dialect = list(resultJSON['Dialect'])
  #with open('D:/flask-project/data.json', 'w') as f:
    #json.dump(resultJSON, f)
  return render_template('data.html', data = data[0] , dialects = Dialect, probs= Probabilitydialect)

  #return redirect('/data')
 else :
  return render_template('home.html')
if __name__ == '__main__':


  app.run(debug=True)
