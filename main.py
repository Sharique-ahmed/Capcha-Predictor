from captcha.image import ImageCaptcha
import random
from flask import Flask,request,jsonify,render_template
from prediction import predictImage
import os


app = Flask(__name__)

# Every possible character in a capcha
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # the len is 62


@app.route('/randomCapcha',methods=["GET"])
def generateRandomCapcha():
    # lets generate the capcha elements
    if request.method == "GET":
        try:
            n = 5 #  since I need just four letter capcha
            capcha_text = ""

            for i in range(n):
                capcha_text += characters[random.randint(0,61)]

            # This is the setting up the width and the height
            image = ImageCaptcha(width=250,height=90)

            # saving the capcha image
            image.write(capcha_text,f'static\capcha\{capcha_text}.png')

            return jsonify({'status':True,'path':f'static\capcha\{capcha_text}.png'}),200
        except Exception as e:
            return jsonify({'status':False,'error':f'There was error -->{e}',}),500
    else:
        return jsonify({'status':False,'error':'Use Get method'}),405


@app.route('/generateCapcha',methods=['POST'])
def generateCustomCapcha():
    if request.method == 'POST':
        try:
            capcha_text = request.json['capcha']
            if len(capcha_text) != 5:
                return jsonify({'status':False,'error':'The len of the capcha should be 4 letters'}),500
            
            # This is the setting up the width and the height
            image = ImageCaptcha(width=250,height=90)

            # saving the capcha image
            image.write(capcha_text,f'static\capcha\{capcha_text}.png')

            return jsonify({'status':True,'path':f'static\capcha\{capcha_text}.png'}),200
        except Exception as e:
            return jsonify({'status':False,'error':f'There was error -->{e}',}),500
    else:
        return jsonify({'status':False,'error':'Use POST method'}),405

@app.route('/uploadCapcha',methods=["POST"])
def UploadCapcha():
    if request.method == 'POST':
        try:
            file = request.files["file"]
            try:
                count = int(request.form.get("count"))  # Retrieve count as form data
            except:
                return jsonify({'status':False,'data':'No Count mentioned'})

            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            if file:
                # Save the file
                file_path = os.path.join("static","Uploaded Images", file.filename)
                file.save(file_path)

            try:
                result_status,result = predictImage(file_path,count)
            except Exception as e:
                print("There was a error while calling predictImage",e)

            # # remove the uploaded image to prevent storage
            # try:
            #     os.remove(file_path)
            # except:
            #     print('There was a errror while deleting the image')
            if  result_status:
                return jsonify({'status':True,'data':result,'imgpath':file_path}),200
            else:
                return jsonify({'status':False,'data':result}),500
        except Exception as e:
            return jsonify({'status':False,'error':f'There was a error --->{e}'}),500
    else:
        return jsonify({'status':False,'error':'Use POST Method'}),502



@app.route("/",methods=['GET'])
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
