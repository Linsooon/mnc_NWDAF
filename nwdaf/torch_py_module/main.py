from flask import Flask, request
import json
from MTLF import *
from AnLF import *
#from Model import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def parser():
    data = {}
    if request.method == 'POST':
        data = request.json
        print(data)

        if str(data['nfService']) == 'training':
            MTLF()
            #data["data"] = "training finish"
        elif str(data['nfService']) == 'inference':
            #AnLF()
            inference_result = AnLF()
            inference_result = inference_result[0]

            #inference_result = AnLF(model,int(data['data']))
            #print(type(inference_result))
            #print(inference_result)
            return json.dumps({'data' : str(inference_result)})
            #data["data"] = print("inference_result : " + str(inference_result))
        else:
            data['data'] = "None (Wrong)"
        '''
        data['reqNFInstanceID'] = data['reqNFInstanceID'] + 'hi'
        data['nfService'] = data['nfService'] + '(reply)'
        data['reqTime'] = data['reqTime']
        '''
    return json.dumps("0")
    
    


if __name__ == '__main__':
    app.run(port=9537)
