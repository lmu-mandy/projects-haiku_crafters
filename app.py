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
openai.api_key = os.getenv("OPENAI_API_KEY")
strict = False


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
    if input_type == "lstm":
        return generate_prompt_from_seed(cue)
    else:
        with open(f'./static/{input_type}_prompt.txt', "r") as f:
            prompt = f.read()
            return prompt.format(cue.lower())



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