from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import telebot
import torch
import subprocess
import os 
import librosa
import warnings
import re
import nemo.collections.asr as nemo_asr

#CONFIG_PATH = configs.spelling_correction.levenshtein_corrector_ru

#corector = build_model(CONFIG_PATH, download=True)


bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['variables']['BERT_PATH'] = 'sentence_ru_cased_L-12_H-768_A-12_pt'

m = build_model(bert_config)
print("BERT загружен")
model = torch.load("model6.pt")
model.eval()

encode = torch.load("encoder4.pt")
print("классификатор загружен")
asr_model = nemo_asr.models.EncDecCTCModel.restore_from("ZMv")
print("ASR модель загружена")
def pred(mes,m = m,model = model,encode = encode):#, cor = corector):
  model.eval()
  #mes = corector(mes)
  mes = mes[0][0].upper()+mes[0][1:]
  emb = m([mes])[4]
  probas = torch.nn.functional.softmax(model(torch.tensor(emb)).detach()).numpy()[0]
  top1 = np.argmax(probas)
  val1 = np.max(probas)
  probas[top1] = -1
  top2 = np.argmax(probas)
  val2 = np.max(probas)
  top1 = encode.categories_[0][top1]
  top2 = encode.categories_[0][top2]
  return "Ответ по запросу:"+mes+" Вероятные ответы: 1) "+str(top1)+": "+str(val1)+" 2) "+str(top2)+": "+str(val2)

token = '1731028966:AAH_TVC3hjxLAPoPq35TeFypmIaPYHFsY4o'
bot = telebot.TeleBot(token);

@bot.message_handler(content_types=['text'])
def mes(message):
    #bot.send_message(message.from_user.id,str(message))
    res = pred([message.text])
    bot.send_message(message.from_user.id, res);

@bot.message_handler(content_types=['document', 'voice'])
def speech(message):
    #print(message.voice)
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    src = 'Bot/'+str(message.from_user.id)+'.ogg'
    with open(src, 'wb') as new_file:
    # записываем данные в файл
        new_file.write(downloaded_file)

    src_filename = 'Bot/'+str(message.from_user.id)+'.ogg'
    dest_filename = 'Bot/wav/'+str(message.from_user.id)+'.wav'

    process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
    if process.returncode != 0:
        raise Exception("Something went wrong")
    else:
        os.remove('Bot/'+str(message.from_user.id)+'.ogg')
    transcripts = asr_model.transcribe(paths2audio_files=[dest_filename], batch_size=1)
    res = pred(transcripts)
    os.remove(dest_filename)
    bot.send_message(message.from_user.id, res);

bot.polling(none_stop=True, interval=0)
