from flask import Flask,request,jsonify
from flask_cors import CORS
import startup_recommender_v3

app = Flask(__name__)
CORS(app) 
        
@app.route('/recc', methods=['GET'])
def recommend_movies():
        print("here")
        res = startup_recommender_v3.results(request.args.get('email'))
        return jsonify(res)

if __name__=='__main__':
        app.run(port = 8000, debug = True)