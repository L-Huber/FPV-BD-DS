# Chatbot Evaluation and Voice Processing Scripts

This repository contains scripts for evaluating the responses of a chatbot and processing voice inputs.

## Installation

To install the required dependencies for this project, run the following command:

```bash
pip install -r requirements.txt
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

It contains the following functions:
TODO: Add functions
voice_to_text(): Converts voice input to text.
text_to_voice(text): Converts text to voice.
process_voice_input(): Processes voice input by converting it to text, sending it to the chatbot, receiving the chatbot's response, and converting the response to voice.
