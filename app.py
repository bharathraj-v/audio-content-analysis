from django.shortcuts import render
from flask_assets import Bundle, Environment
from flask import Flask, render_template,request, flash, redirect, url_for
import librosa
import torch
from textblob import Word
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import mimetypes
import os
import moviepy.editor as mp
import eng_to_ipa as ipa


# Using Wav2Vec2 model for speech recognition as training a good model for general ASR from scratch is not possible due to the time constraint!


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def correct(text):
    return " ".join([Word(word).spellcheck()[0][0] for word in text.split()])

def accuracy(text):
    return str((sum([int(Word(word).spellcheck()[0][1]==1) for word in text.split()])/len(text.split()))*100)+"%"


app = Flask(__name__)
app.secret_key = "super secret key"


@app.route('/')
def home():
    return render_template('index.html',translation="")

@app.route('/', methods=['GET','POST'])
def audio():
    if 'audio' not in request.files:
            flash('No file part')
            return redirect(request.url)
    file = request.files['audio']
    if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
    if file:
        file.save(file.filename)
        if mimetypes.guess_type(file.filename)[0].startswith('video'):
            clip = mp.VideoFileClip(file.filename)
            clip.audio.write_audiofile("audio.mp3")
            os.remove(file.filename)
            file.filename = "audio.mp3"        
        speech, rate = librosa.load(file.filename,sr=16000, duration=30)
        input_values = tokenizer(speech, return_tensors = 'pt').input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim =-1)
        transcriptions = tokenizer.decode(predicted_ids[0])
        os.remove(file.filename)
        return render_template('index.html',
        translation=transcriptions.lower(),
        accuracy=accuracy(transcriptions.lower()),
        incorrect_words = len([word for word in transcriptions.lower().split() if Word(word).spellcheck()[0][1]!=1]),
        phonetics=ipa.convert(transcriptions.lower()),
        stutters = sum([1 for word in [word for word in transcriptions.lower().split() if Word(word).spellcheck()[0][1]!=1] if len(word)<4])+ sum([1 for word in transcriptions.lower().split() if word in ['oh','uh','uhm','umm', 'ah','aa','aah']]))



if __name__ == "__main__":
    app.run(debug=True)