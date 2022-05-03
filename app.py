from multiprocessing.sharedctypes import Value
import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
strict = False


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        input_type = request.form["input-type"]
        cue = request.form["cue"]
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
    with open(f'{input_type}_prompt.txt', "r") as f:
        prompt = f.read()
        return prompt.format(cue.lower())