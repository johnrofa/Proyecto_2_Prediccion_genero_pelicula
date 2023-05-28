from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import os
import pandas as pd
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


#Definición nombres columnas
cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

#Importar modelo y transformación
Model_clfRF21 = joblib.load('modelo_clfRF21.pkl')
Transformacion_X= joblib.load('transformacion_x.pkl')


# Definición aplicación Flask y habilitación del modelo para todas las rutas y orígenes
app = Flask(__name__); CORS(app)
api = Api( app, version='1.0', title='Prediccion género de películas',  description='Prediccion género de películas')
ns = api.namespace('predict', description='Prediccion género de películas')
parser = api.parser()

# Definición Parámetros de entrada
parser.add_argument('Descripcion', type=str, required=True,  help='Descripcion película', location='args')



resource_fields = api.model('Resource', { 'result': fields.String, })

@ns.route('/')
class PredictionApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        print(args)

        data = { "Descripcion": args['Descripcion']}
        ser = pd.Series(data)
        print('ser', ser)

        X_vec_0003 = Transformacion_X.fit_transform(ser)

        print('shape1',X_vec_0003.shape)

        X_vec_0003._shape = (1, 30000)
        print('shape2',X_vec_0003.shape)

        ypred_04 = Model_clfRF21.predict_proba(X_vec_0003)
        ypred_04 = np.array(ypred_04)
        ypred_04 = np.array(ypred_04[:, :, 1]).T
        print('ypred_04', ypred_04)

        result = pd.DataFrame(ypred_04[0], index=cols, columns=['Probabilidad'])
        result['Probabilidad'] = round(result['Probabilidad'] * 100, 0)
        result['Probabilidad'] = result['Probabilidad'].astype(str) + '%'

        print('result', result)

        #result_01 = result.to_json(orient = 'columns')
        result_01 = result.to_dict()

        print('result_01', result_01)

        return { "result": result_01}, 200





if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
