# Chatbot Evaluation and Voice Processing Scripts

This repository contains scripts for evaluating the responses of a chatbot and processing voice inputs.

## Installation

To install the required dependencies for this project, run the following command:

```bash
pip install -r requirements.txt
```

You also need to set up your AssemblyAI API key by creating a .env file in the root directory of the project and adding your API key as follows:

```bash
ASSEMBLYAI_API_KEY="your_api_key_here"
OPENAI_API_KEY="your_api_key_here"
```

## Usage

To use the scripts in this project, you can run them with Python:

```bash
python eval.py
python voice.py
```

## Documentation

## eval.py

The eval.py script evaluates the responses of a chatbot. It calculates the cosine similarity between the chatbot's response and the expected output, judges the response based on certain criteria, and saves the evaluation results to a CSV file.

It contains the following functions:
calculate_cosine_similarity(vector1, vector2): Calculates the cosine similarity between two vectors.
similar(a, b): Calculates the similarity between two strings using OpenAI embeddings.
eval_agent(test_case_str, response_str, cosine_similarity, static_criteria, number_of_criteria=5, expected_similarity=0.95): Evaluates the chatbot's response based on the test case, response, cosine similarity, and criteria.
test_agent(): Tests the chatbot by providing a test case and expected output

## voice.py

The voice.py script processes voice inputs. It converts voice inputs to text, sends the text to the chatbot, receives the chatbot's response, and converts the response to voice.

### Usage

To use the script, follow these steps:

Ensure you have Python installed on your system.
Install the required dependencies as mentioned above.
Set up your AssemblyAI API key in the .env file.
Prepare your test cases in a CSV file named cases.csv following the format specified below.
Run the script using the following command:

```bash
python your_script_name.py
Replace your_script_name.py with the name of your Python script containing the provided code.
```

### Test Cases

The test cases are initialized from a CSV file named cases.csv. Each test case should be formatted as follows:

```bash
id,name,description,number_of_replies,replies,expected_output
id: Unique identifier for the test case.
name: Name of the test case.
description: Description of the test case.
number_of_replies: Number of replies in the test case.
replies: List of audio files containing replies.
expected_output: Expected transcriptions for the replies.
```

### Functionality

play_audio(filename): Plays the audio file specified by filename.
rec_audio(filename, length): Records audio and saves it to the file specified by filename for the given length of time.
transcribe_audio(filename): Transcribes the audio file specified by filename using the AssemblyAI API.
test_test_case(test_case_id): Tests the specified test case by playing audio, recording responses, transcribing them, and checking against expected output.
check_transcription(transcription_list, expected_output): Compares the transcribed text with the expected output and calculates the similarity.
calculate_cosine_similarity(vector1, vector2): Calculates the cosine similarity between two vectors.
similar(a, b): Calculates the similarity between two strings using embeddings and cosine similarity.

### Credits

Audio manipulation: Deepgram
Transcription: AssemblyAI
CSV handling: Python CSV Library
Evaluation Agent: OpenAI
