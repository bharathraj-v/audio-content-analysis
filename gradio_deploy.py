## Deploying in Heroku was not possible due to the memory usage. So,
## gradio was chosen


import nltk
import librosa
import torch
import gradio as gr
from textblob import Word
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
nltk.download("punkt")
import warnings
import eng_to_ipa as ipa
warnings.filterwarnings("ignore")
print(f"All libraries are installed!!")


def correct(text):
    return " ".join([Word(word).spellcheck()[0][0] for word in text.split()])

def accuracy(text):
    return str((sum([int(Word(word).spellcheck()[0][1]==1) for word in text.split()])/len(text.split()))*100)+"%"

def stutter(text):
  return sum([1 for word in [word for word in text.split() if Word(word).spellcheck()[0][1]!=1] if len(word)<4])+ sum([1 for word in text.split() if word in ['oh','uh','uhm','umm', 'ah','aa','aah']])


#Loading the pre-trained model and the tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def load_data(input_file):
  speech, sample_rate = librosa.load(input_file)
  print(speech)
  print()
  print(sample_rate)
  if len(speech.shape) > 1: 
        speech = speech[:,0] + speech[:,1]
  if sample_rate !=16000:
    speech = librosa.resample(speech, sample_rate,16000)
    return speech
    
def correct_casing(input_sentence):
    sentences = nltk.sent_tokenize(input_sentence)
    return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))
 
def asr_transcript(input_recording, input_file):
  if input_recording:
    speech = load_data(input_recording)
  else:
    speech = load_data(input_file)
  input_values = tokenizer(speech, return_tensors="pt").input_values
  logits = model(input_values).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = tokenizer.decode(predicted_ids[0])
  transcription = correct_casing(transcription.lower())
  return (transcription, accuracy(transcription.lower()), ipa.convert(transcription.lower()), stutter(transcription.lower()))

gr.Interface(asr_transcript,
             inputs = [gr.inputs.Audio(source="microphone", type="filepath", optional=True, label="Speaker"),
             gr.inputs.Audio(source="upload", type="filepath", optional=True, label="Speaker")],
             outputs = [gr.outputs.Textbox(label="Transcription"),
                        gr.outputs.Textbox(label="Pronounciation Accuracy"),
                        gr.outputs.Textbox(label="Phonetics"),
                        gr.outputs.Textbox(label="Approx. Stutter Count")],
             title="Audio Content Analysis",
             description = "Analyzing Grammatical Accuracy & Phonetics").launch()