from multiprocessing.sharedctypes import Value
import os

import openai
from flask import Flask, redirect, render_template, request, url_for
from lstm.src.model.model import CharRNN, sample, normalize_haiku_text, split_haiku_by_line
import torch

"""
    Renders index.html asking for a promt. Then eithers uses openai api
    or using model.py, depending on which preference the user has selected 
    to generate haikus 

    To access the openai api either follow the steps in the readme or paste the 
    api in openai.api_key = {api key}
"""

app = Flask(__name__)
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

strict = True


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        input_type = request.form["input-type"]
        cue = request.form["cue"]
        print("input type: ", input_type)
        if input_type == "lstm":
            haiku = generate_prompt(input_type, cue)
            return redirect(url_for("index", result=haiku))
        else: 
            while(True):
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=generate_prompt(input_type, cue),
                    temperature=0.6,
                )
                for i in range(len(response.choices)):
                    haiku = response.choices[i].text
                    if strict:
                        if len(haiku.split('\n')) == 3 and haiku[-1] == "." or haiku[-1] == "!":
                            return redirect(url_for("index", result=haiku))
                    else:
                        return redirect(url_for("index", result=haiku))
    result = request.args.get("result")
    return render_template("index.html", result=result)

def generate_prompt(input_type, cue):
    """
    Logic to decide what models to generate from.
    """
    
    if input_type == "subject":
        return generate_prompt_from_subject(cue)
    elif input_type == "symbol":
        return generate_prompt_from_symbol(cue)
    elif input_type == "lstm":
        return generate_prompt_from_seed(cue)
    else:
        raise ValueError(f'Invalid input type {input_type}')


def generate_prompt_from_seed(cue):
    """
    Sends the cue to the LSTM and returns the sample.
    """

    with open('./lstm/src/model/checkpoints/rnn (haikus + shakespeare).net', 'rb') as f:
        checkpoint = torch.load(f)
        
    loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])

    # Normalize text
    normalized_haiku = normalize_haiku_text(sample(loaded, 75, cuda=True, top_k=5, prime=cue))

    # Split into haiku format
    return split_haiku_by_line(normalized_haiku)


def generate_prompt_from_subject(cue):
    """ 
    Feeds the openai api both the training data and the cue word for subject hakius
    """

    return """Write three lines of a haiku from a cue word/phrase.
Cue: old pond
Haiku: An old silent pond\nA frog jumps into the pond—\nSplash! Silence again.
Cue: dew
Haiku: A world of dew,\nAnd within every dewdrop\nA world of struggle.
Cue: candle
Haiku: The light of a candle\nIs transferred to another candle—\nSpring twilight.
Cue: river
Haiku: love between us is\nspeech and breath. loving you is\na long river running.
Cue: summer grasses
Haiku: The summer grasses\nAll that remains\nOf warriors' dreams.
Cue: rotten log
Haiku: Like a half-exposed rotten log\nmy life, which never flowered,\nends barren.
Cue: cherry blossoms, full moon
Haiku: Let me die in spring\nbeneath the cherry blossoms\nwhile the moon is full.
Cue: river
Haiku: The calm,\nCool face of the river\nAsked me for a kiss.
Cue: butterfly
Haiku: Life: a solitary butterfly\nswaying unsteadily on a slender grass-stalk,\nnothing more. But ah! so exquisite!
Cue: road
Haiku: Along this road\nGoes no one,\nThis autumn eve.
Cue: Kyoto
Haiku: Even in Kyoto,\nHearing the cuckoo’s cry,\nI long for Kyoto.
Cue: {}
Haiku: """.format(cue.lower())


def generate_prompt_from_symbol(cue):
    """
    Feeds the openai api both the training data and the cue word for symbol hakius
    """

    return """Write three lines of a haiku from a cue word\nphrase.
Cue: silence
Haiku: An old silent pond\nA frog jumps into the pond—\nSplash! Silence again.
Cue: struggle
Haiku: A world of dew,\nAnd within every dewdrop\nA world of struggle.
Cue: light
Haiku: The light of a candle\nIs transferred to another candle—\nSpring twilight.
Cue: love
Haiku: love between us is\nspeech and breath. loving you is\na long river running.
Cue: dreams
Haiku: The summer grasses\nAll that remains\nOf warriors' dreams.
Cue: death
Haiku: Like a half-exposed rotten log\nmy life, which never flowered,\nends barren.
Cue: death
Haiku: Let me die in spring\nbeneath the cherry blossoms\nwhile the moon is full.
Cue: suicide
Haiku: The calm,\nCool face of the river\nAsked me for a kiss.
Cue: life
Haiku: Life: a solitary butterfly\nswaying unsteadily on a slender grass-stalk,\nnothing more. But ah! so exquisite!
Cue: loneliness
Haiku: Along this road\nGoes no one,\nThis autumn eve
Cue: {}
Haiku:""".format(cue.lower())