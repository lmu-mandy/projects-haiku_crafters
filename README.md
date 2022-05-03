Weclome to the project: Haiku Crafters

## Project Description 
This a webapp that can generates peoms that closely follows the haiku format of 5-7-5 syllable and line count.

It generators haikus from either openai GPT-3 model or a LSTM model that we trained ourself.

The front end uses flask to connect from either the openai api or the LSTM model.

## Installation Instructions

1. Clone this repository

2. Navigate into the project directory

   ```bash
   $ cd openai-quickstart-python
   ```

3. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

4. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

5. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

6. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file

8. Run the app

   ```bash
   $ flask run
   ```

9. Copy the url link shown in the terminal and paste it into a browser

