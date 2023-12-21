from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import json
from scipy import stats
from scipy.signal import find_peaks
import pywt
import spkit
from spkit import cwt
import soundfile as sf
import tensorflow as tf
from transformers import TFViTModel

# Configuring env variables
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#load the model
custom_objects = {'TFViTMainLayer': TFViTModel}
model1 = tf.keras.models.load_model('./models/tf_model.h5')
model2 = tf.keras.models.load_model('./models/subject_pcg_model.h5')
#---------------------------------------------------------------------------------------------------------------------------

list_class = ['aorta_stenosis', 'mitral_regurgitation', 'mitral_stenosis', 'mitral_valve_prolapse', 'normal']

#defining all function
def load_signal(filename):
    y, sr = librosa.load(filename, sr = 2000)
    return y, sr
def threshold(coef) :
  result = []
  p75 = np.percentile(np.absolute(coef), 75)
  x_bar = np.mean(np.absolute(coef))
  var = np.var(np.absolute(coef))
  ## estimated Threshold
  T = p75*(1-(var-p75)) if p75 < var else p75 if p75 > var and p75 < x_bar else p75 +(p75-x_bar)
  ##rescaling threshold
  var_noise = np.mean(np.absolute(coef))/0.6745
  T = T * var_noise
  alpha = 1
  beta = 1.3 if p75 <= var else 1.4
  T1 = alpha * T
  T2 = beta * T
  for i in coef :
    if abs(i) < T1:
      result.append(0)
    elif abs(i) >= T1 and abs(i) <= T2:
      i = i**3/T2**2
      result.append(i)
    else :
      result.append(i)
  result = np.array(result)
  return result

def denoised(signal) :
    coeffs = pywt.wavedec(signal, 'coif5', level=5)
    coeffs[0] = np.zeros(len(coeffs[0]))
    # thresholding and change other freq to 0
    coeffs[1] = threshold(coeffs[1])
    coeffs[2] = threshold(coeffs[2])
    coeffs[3] = np.zeros(len(coeffs[3]))
    coeffs[4] = np.zeros(len(coeffs[4]))
    coeffs[5] = np.zeros(len(coeffs[5]))
    # Reconstruction
    denoised_signal = pywt.waverec(coeffs, 'coif5')
    return denoised_signal

def teo(signal):
    TEO = []
    for n in range(1,len(signal)-1):
        TEO.append(signal[n]**2 - signal[n-1]*signal[n+1])
    TEO[0] = TEO[1]
    TEO[len(TEO)-1] = TEO[len(TEO)-2]
    return TEO

def smoothing(filt) :
  w = 30
  ma = np.convolve(filt, np.ones(w), 'valid') / w
  ma[ma < 0] = 0
  square = np.sqrt(ma)
  return ma

def standarize(s_sign):
  standardized_data = stats.zscore(s_sign)
  standardized_data[standardized_data<0] = 0
  return standardized_data

def pacf(signal) :
  n = len(signal) - 1
  N = len(signal)
  pacf = []
  for i in range(0, n):
    R = 0
    for m in range(0, N-i-1) :
      R += signal[m] * signal[m + i]
    R = R/N
    pacf.append(R)
  pacf = np.array(pacf)
  return pacf

def segment(signal):
    segments = []
    # TKEO
    TKEO = teo(signal)

    # smoothing signal
    smooth_signal = smoothing(TKEO)
    final_signal = standarize(smooth_signal)

    # pacf
    par_acf = pacf(final_signal)

    # peaks detection
    peaks, _ = find_peaks(par_acf, distance=800, threshold=np.percentile(par_acf, 30))
    peaks = np.insert(peaks, 0, 0)

    # segment
    if len(peaks) > 2:
        first_segment = signal[:peaks[2]]
        segments.append(first_segment.tolist())
    else:
        par_acf = pacf(signal)
        # peaks detection
        peaks, _ = find_peaks(par_acf, distance=600)
        peaks = np.insert(peaks, 0, 0)
        if len(peaks) > 4:
            first_segment = signal[peaks[2]:peaks[4]]
            segments.append(first_segment.tolist())
        else:
            par_acf = pacf(signal)
            # peaks detection
            peaks, _ = find_peaks(par_acf, distance=600)
            peaks = np.insert(peaks, 0, 0)
            first_segment = signal[peaks[1]:peaks[2]]
            segments.append(first_segment.tolist())
    return segments

def transform_cwt(signal, sr, filename, i):
    t = np.linspace(0, len(signal) / 2000, num=len(signal))
    XW, S = cwt.ScalogramCWT(signal, t, fs=sr, wType='Morlet')
    plt.imsave('out.png', np.abs(XW), origin='lower', cmap='jet')
    image = Image.open('out.png')
    img_resized = image.resize((224, 224))
    base, ext = os.path.splitext(filename)
    transform_dir = './transform_image'
    filepath = f"{os.path.join(transform_dir, base)} - {i}.png"
    img_resized.save(filepath)
    os.remove('out.png')
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict the file ingredients
def pred_segment(picture):
    img = tf.keras.utils.load_img(picture, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # images = np.vstack([x])
    classes = model1(x)
    classes = classes.numpy().tolist()
    return classes

def pred_subject(picture):
    img = tf.keras.utils.load_img(picture, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # images = np.vstack([x])
    classes = model2(x)
    classes = classes.numpy().tolist()
    return classes
#----------------------------------------------------------------------------------------------------------------------------

#route
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello, world!!'

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data={}
        filePath = ''
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                return {
                    "status": "error",
                    "message": 'No file part'
                }
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                return {
                    "status": "error",
                    "message": 'No selected file'
                }
            if file and allowed_file(file.filename):
                # Saving file to local storage
                filename = secure_filename(file.filename)
                filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filePath)

                # load_signal
                signal, sr = load_signal(filePath)
                time = np.linspace(0, len(signal) / sr, num=len(signal))
                denoised_signal = denoised(signal=signal)
                segmenting = segment(denoised_signal)
                time_segment = np.linspace(0, len(segmenting[0]) / sr, num=len(segmenting[0]))
                for i in range(len(segmenting)):
                    transform_cwt(np.array(segmenting[i]), sr, filename, i)
                result_predict_segment =[]
                for f in os.listdir('./transform_image'):
                    confident_score = pred_segment(os.path.join('./transform_image', f))
                    result_predict_segment.append(confident_score[0])
                    os.remove(os.path.join('./transform_image', f))
                subject_level = denoised_signal[:3200]
                transform_cwt(subject_level, sr, filename, 'subject')
                confident_score_subject = []
                for f in os.listdir('./transform_image'):
                    predicted_subject = pred_subject(os.path.join('./transform_image', f))
                    confident_score_subject.append(predicted_subject)
                    os.remove(os.path.join('./transform_image', f))
                segment_result = list_class[np.argmax(np.array(result_predict_segment[0]))]
                subject_result = list_class[np.argmax(np.array(confident_score_subject[0]))]

                # Delete it as soon the picture is predicred
                os.remove(filePath)
                data = {
                    'signal_data' : {
                        'labels' : time.tolist(),
                        'datasets' :[{
                            'label' : 'PCG Signal',
                            'data' : signal.tolist()
                        }]
                    },
                    'denoised_signal_data' : {
                        'labels' : time.tolist(),
                        'datasets' :[{
                            'label' : 'Denoised PCG Signal',
                            'data' : denoised_signal.tolist()
                        }]
                    },
                    'segment_data': {
                        'labels' : time_segment.tolist(),
                        'datasets': [{
                            'label': 'Segment PCG',
                            'data' : segmenting
                        }]
                    },
                    'predict_segment': result_predict_segment,
                    'predict_subject': confident_score_subject,
                    'segment_result' : segment_result,
                    'subject_result' : subject_result
                }
            return json.dumps(data)
        # catch error if something unexpected happened
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        return 'Get error'


if __name__ == '__main__':
    app.run()
