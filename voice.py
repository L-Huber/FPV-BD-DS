"""
This script contains functions for audio manipulation, transcription, and testing of test cases.
It imports necessary libraries and APIs, defines data classes, and initializes test cases from a CSV file.
The script also includes functions for playing and recording audio, transcribing audio using AssemblyAI API,
testing test cases, checking transcriptions, calculating cosine similarity, and embedding strings.
The main function initializes the test cases and can be used to run specific test cases.
"""

#TODO: document similar to eval.py

# scr: for audio: https://deepgram.com/learn/best-python-audio-manipulation-tools
# scr: for transcribing: https://www.assemblyai.com/app
# scr: for csv: https://docs.python.org/3/library/csv.html , https://www.geeksforgeeks.org/read-a-csv-into-list-of-lists-in-python/

from dataclasses import dataclass
import csv
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile
import time
import assemblyai as aai
import unittest
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# fetch env variables
load_dotenv(find_dotenv())


# insert assembly ai key
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
transcripton_list = []


# defining the test data class
# data class is a class that is used to store data, similar to a struct in C
@dataclass
class test_case:
    id: int
    name: str
    description: str
    number_of_replies: int
    replies: list[str]
    expected_output: list[str]


# defining the list of test cases
# this function initializes the test cases from the csv file
# it splits them into the different parts and stores them in the list
def init_test_cases():
    with open("cases.csv", "r") as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj, delimiter=",")

        # convert string to list
        list_of_csv = list(csv_reader)

        list_of_csv.pop(0)
        number_of_test_cases = len(list_of_csv)
        print(number_of_test_cases)

        list_of_csv = list_of_csv[0][0].split(",")
        list_of_csv[4] = list_of_csv[4].split("|")
        expected_output = [
            "Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?",
            "Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?",
            "Haben Sie noch ein weiteres Anliegen?",
        ]
        print(list_of_csv)
        test_case1 = test_case(
            list_of_csv[0],
            list_of_csv[1],
            list_of_csv[2],
            list_of_csv[3],
            list_of_csv[4],
            expected_output,
        )
        list_of_test_cases.append(test_case1)
        print(list_of_test_cases[0].id)
        print("testoutput: " + str(list_of_test_cases[0].expected_output))


# defining the functions for the audio to play
# @param filename: file that should be played
def play_audio(filename):
    data, fs = soundfile.read(filename, dtype="float32")
    sounddevice.play(data, fs)
    status = sounddevice.wait()


# defining the function for recording the audio
# @param filename: filename of the recorded file
# @param length: length of the recording in seconds
def rec_audio(filename, length):
    fs = 44100
    seconds = length

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)


# function that transcribes the audio with the assemblyai api
# @param filename: filename of the audio file to transcribe
def transcribe_audio(fileneame):
    config = aai.TranscriptionConfig(language_code="de")
    FILE_URL = fileneame
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(FILE_URL)

    # check if an error occured, if then print the error
    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        transcripton_list.append(transcript.text)
        print(transcript.text)


# function that tests the test case
# @param test_case_id: id of the test case that should be tested
def test_test_case(test_case_id):
    # sleep to wait until the audio is played
    sleep_list = [10, 72, 9]
    transcribe_list = []
    time.sleep(23)

    # steps are the different replies that are played and recorded
    for steps in range(0, int(list_of_test_cases[test_case_id - 1].number_of_replies)):
        play_audio("audio_input/" + list_of_test_cases[test_case_id - 1].replies[steps])
        print("sleep time: " + str(sleep_list[steps]))
        rec_audio(str(test_case_id) + "_" + str(steps) + ".wav", sleep_list[steps])
        transcribe_list.append(str(test_case_id) + "_" + str(steps) + ".wav")
        print("Trascribed finished")
    # the last reply is always this one
    play_audio("audio_input/" + "Outro.mp3")
    # the recorded audio gets transcribed
    for steps in transcribe_list:
        transcribe_audio(steps)
    print(transcripton_list)
    print(list_of_test_cases[test_case_id - 1].expected_output)
    # let's check the transcripton
    check_transcripton(
        transcripton_list, list_of_test_cases[test_case_id - 1].expected_output
    )


# function that checks the transcripton
# @param transcripton_list: list of the transcriptons
# @param expected_output: list of the expected output
def check_transcripton(transcripton_list, expected_output):
    # scr:https: // stackoverflow.com / questions / 17388213 / find - the - similarity - metric - between - two - strings, https://stackoverflow.com/questions/4481724/convert-a-list-of-characters-into-a-string
    # check if the transcripton and the output have the same length
    result_list = []
    if len(transcripton_list) != len(expected_output):
        return 0
    else:
        pos = 0
        for transcriptions in transcripton_list:
            print("Transcription: " + "".join(transcriptions))
            print("Expected: " + "".join(expected_output[pos]))
            result_list.append(
                similar("".join(transcriptions), "".join(expected_output[pos]))
            )
            pos = pos + 1
    # percentage_passed = 0
    print(result_list)

    return min(result_list)


# function that calculates the cosine similarity
# @param vector1: first vector
# @param vector2: second vector
def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]


# function that calculates the similarity between two strings
# @param a: first string
# @param b: second string
def similar(a, b):
    # old code
    # return SequenceMatcher(None, a, b).ratio()

    # embed the strings
    # embedding model for embeddings query
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    a_embedding = embeddings.embed_query(a)
    b_embedding = embeddings.embed_query(b)
    # print(a_embedding, b_embedding)

    # similarity query using cosine similarity
    return calculate_cosine_similarity(a_embedding, b_embedding)


# main function
def main():
    global list_of_test_cases
    list_of_test_cases = []
    # initialize the test cases
    init_test_cases()
    # to check test case 1: remove the comment
    # test_test_case(1)


# testing the functions
class TestStringMethods(unittest.TestCase):
    """
    A test case for testing string methods.
    """

    def test_upper(self):
        """
        Test the 'check_transcripton' function with a list of transcriptions and expected output.
        """
        transcripton_list = [
            ["test", "test2", "test3"],
            ["test4", "test5", "test6", "test7", "test8", "test9"],
            ["test10"],
            ["test11"],
        ]
        expected_output = [
            ["test", "test2", "test3"],
            ["test4", "test5", "test6", "test7", "test8", "test9"],
            ["test10"],
            ["test_wrong"],
        ]
        self.assertEqual(
            check_transcripton(transcripton_list, expected_output), 0.8470205155983312
        )

    def test_upper2(self):
        """
        Test the 'check_transcripton' function with a list of transcriptions and expected output.
        """
        transcripton_list = [
            "Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?",
            "Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migrosbank.ch-konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung der SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Armlegen zu lösen?",
            "Haben Sie noch ein weiteres Anliegen?",
        ]
        expected_output = [
            "Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?",
            "Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?",
            "Haben Sie noch ein weiteres Anliegen?",
        ]
        self.assertEqual(
            check_transcripton(transcripton_list, expected_output), 0.9765484386251064
        )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
    unittest.main()
    init_test_cases()
    test_test_case(1)
