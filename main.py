# scr: for audio: https://deepgram.com/learn/best-python-audio-manipulation-tools
#scr: for transcribing: https://www.assemblyai.com/app
#scr: for csv: https://docs.python.org/3/library/csv.html , https://www.geeksforgeeks.org/read-a-csv-into-list-of-lists-in-python/
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from dataclasses import dataclass
import csv
import sounddevice as sd
from scipy.io.wavfile import write
import sounddevice
import soundfile
import time
import assemblyai as aai
import unittest
from difflib import SequenceMatcher
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage, SystemMessage

#fetch env variables
load_dotenv(find_dotenv())


#insert assembly ai key
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
transcripton_list = []


@dataclass
class test_case:
    """Class defining the test cases."""
    id: int
    name: str
    description: str
    number_of_replies: int
    replies: list[str]
    expected_output: list[str]

def init_test_cases():
    with open('cases.csv', 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj, delimiter=',')

        # convert string to list
        list_of_csv = list(csv_reader)

        #print(list_of_csv)
        list_of_csv.pop(0)
        number_of_test_cases = len(list_of_csv)
        print(number_of_test_cases)
        #for number_of_test_cases in list_of_csv:
            #list_of_csv[number_of_test_cases-1]=list_of_csv[number_of_test_cases-1].split(',')
            #list_of_test_cases.append(test_case(rows[0], rows[1], rows[2], rows[3], rows[4].split('|'), rows[5].split('|')))
        #print(list_of_csv)
        list_of_csv =list_of_csv[0][0].split(',')
        list_of_csv[4] = list_of_csv[4].split('|')
        #list_of_csv[5] = list_of_csv[5].split('|')
        expected_output = ["Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?","Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?","Haben Sie noch ein weiteres Anliegen?"]
        print(list_of_csv)
        test_case1 = test_case(list_of_csv[0], list_of_csv[1], list_of_csv[2], list_of_csv[3], list_of_csv[4], expected_output)
        list_of_test_cases.append(test_case1)
        print(list_of_test_cases[0].id)
        print("testoutput: "+str(list_of_test_cases[0].expected_output))

            #print(rows[0])
            #print(list_to_create_test_case[0])
            #print(len(rows))
            #rows[0] = test_case(rows[0], rows[1], rows[2], rows[3], rows[4].split('|'), rows[5].split('|'))
            # rows[1], rows[2], rows[3], rows[4].split('|'), rows[5].split('|')
            #print(rows[0])

def play_audio(filename):
    data, fs = soundfile.read(filename, dtype='float32')
    sounddevice.play(data, fs)
    status = sounddevice.wait()

def rec_audio(filename, length):
    fs = 44100
    seconds = length

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)

def transcribe_audio(fileneame):
    config = aai.TranscriptionConfig(language_code="de")
    FILE_URL = fileneame
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(FILE_URL)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        transcripton_list.append(transcript.text)
        print(transcript.text)

def test_test_case(test_case_id):
    sleep_list = [10,72,9]
    transcribe_list = []
    time.sleep(23)
    #print(list_of_test_cases[test_case_id-1].number_of_replies)
    for steps in range(0, int(list_of_test_cases[test_case_id-1].number_of_replies)):
        play_audio("audio_input/"+list_of_test_cases[test_case_id-1].replies[steps])
        print("sleep time: "+str(sleep_list[steps]))
        rec_audio(str(test_case_id)+"_"+str(steps)+".wav", sleep_list[steps])
        transcribe_list.append(str(test_case_id)+"_"+str(steps)+".wav")
        print("Trascribed finished")
    #the last reply is alwways this one
    play_audio("audio_input/"+"Outro.mp3")
    for steps in transcribe_list:
        transcribe_audio(steps)
    print(transcripton_list)
    print(list_of_test_cases[test_case_id-1].expected_output)
    check_transcripton(transcripton_list, list_of_test_cases[test_case_id-1].expected_output)
def check_transcripton(transcripton_list, expected_output):
    # scr:https: // stackoverflow.com / questions / 17388213 / find - the - similarity - metric - between - two - strings, https://stackoverflow.com/questions/4481724/convert-a-list-of-characters-into-a-string
    #check if the transcripton and the output have the same length
    result_list = []
    if len(transcripton_list) != len(expected_output):
        return 0
    else:
        pos = 0
        for transcriptions in transcripton_list:
            print("Transcription: "+"".join(transcriptions))
            print("Expected: "+"".join(expected_output[pos]))
            result_list.append(similar("".join(transcriptions), "".join(expected_output[pos])))
            pos = pos + 1
    #percentage_passed = 0
    print(result_list)

    return (min(result_list))

def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def similar(a, b):
    #old code
    # return SequenceMatcher(None, a, b).ratio()

    # embed the strings
    # embedding model for embeddings query
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    a_embedding = embeddings.embed_query(a)
    b_embedding = embeddings.embed_query(b)
    #print(a_embedding, b_embedding)

    # similarity query using cosine similarity
    return calculate_cosine_similarity(a_embedding, b_embedding)

def eval_agent(test_case_str, response_str, cosine_similarity, static_criteria, number_of_criteria=5, expected_similarity=0.95):
    # Initialize variables to store pass/fail status and error type
    test_outcome = "Pass"
    error_types = []

    # create agent with prompt instructions based on GPT
    from langchain_openai import ChatOpenAI
    #chat = ChatOpenAI(model="gpt-4")
    chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

    if cosine_similarity < expected_similarity:
        test_outcome = "Fail"

    messages = [
    SystemMessage(content="""You're an evaluation agent. You're evaluating the response of a chatbot to a test case.
                  The test case is as follows: """ + test_case_str + """
                  First I want you to identify """+ str(number_of_criteria) + """ criteria to judge the response on.
                  Then I want you to evaluate the response based on the criteria.
                  The following criteria are mandatory: """ + static_criteria + """. 
                  If a criteria is not met by the response, you should classify the error based on the criteria."""
                  ),
    HumanMessage(content= response_str),
    SystemMessage(content="If you have identified a failed criteria please provide the error type at the very end of your response! Use this format: //ERROR START// <error type> //ERROR END//. Please avoid any comments or additional information at all cost! Only include a short description of the error type in the start and end flags. USE DIFFERENT FLAGS FOR MULTIPLE ERRORS!"),
        ]

    # Invoke the chat model
    chat.invoke(messages)

    response = []
    for chunk in chat.stream(messages):
        #print(chunk.content, end="", flush=True)
        #save answer to a file
        response.append(chunk.content)

    #save answer to a file
    with open("output.txt", "w") as text_file:
        for line in response:
            text_file.write(line)

    #if test_outcome == "Fail":
    # Read the content of the file
    with open("output.txt", "r") as file:
        content = file.read()
        # Flags to search for
        start_flag = "//ERROR START//"
        end_flag = "//ERROR END//"
        error_types = []
        # Loop to extract multiple error types
        while start_flag in content and end_flag in content:
            start_index = content.find(start_flag)
            end_index = content.find(end_flag)
            if start_index != -1 and end_index != -1:
                # Extract the error type between the flags
                error_type = content[start_index + len(start_flag):end_index].strip()
                error_types.append(error_type)
                # Remove the extracted error type from the message content
                content = content[end_index + len(end_flag):].strip()

    if error_types == []:
        error_types = ["No errors found in the response."]

    #structure return format
    return test_outcome, cosine_similarity, error_types

def main():
    global list_of_test_cases
    list_of_test_cases = []
    # Use a breakpoint in the code line below to debug your script.
    #rec_audio('output_sounddevice.wav', 3)
    #time.sleep(2)
    #play_audio('output_sounddevice.wav')
    #transcribe_audio('output_sounddevice.wav')
    init_test_cases()
    #test_test_case(1)

def test_agent(): 
    transcripton_list= ['Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?', 'Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migrosbank.ch-konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung der SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Armlegen zu lösen?', 'Haben Sie noch ein weiteres Anliegen?']
    expected_output_list = ['Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?', 'Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?', 'Haben Sie noch ein weiteres Anliegen?']

    test_case_str = "Sie können Ihr Konto einfach und schnell auf unserer Webseite eröffnen. Besuchen Sie www.beispielbank.de/konten und folgen Sie den Anweisungen für die Kontoeröffnung. Wir werden Ihnen dann eine E-Mail mit weiteren Details senden. Bitte überprüfen Sie auch Ihren Spam-Ordner, falls Sie die E-Mail nicht in Ihrem Posteingang finden. Haben Sie Fragen oder benötigen Sie weitere Unterstützung?"

    static_criteria = "1. Link sent per SMS, 2.Link contains correct URL = www.migrosbank.ch-konten, 3. Politness, 4. Clear instructions, 5. Follow-up question."
    expected_similarity = 0.95

    # list to string to get embeddings
    transcription = ", ".join(transcripton_list)
    expected_output = ", ".join(expected_output_list)


    print("\nEvaluate the agent with the transcription case, response, and cosine similarity\n")
    print(f' TEST CASE: {transcription} \n \n RESPONSE: {expected_output} \n')
    # Embed the strings and calculate cosine similarity
    cosine_sim = similar(transcription, expected_output)
    print(f'Cosine similarity: {cosine_sim}')

    outcome = eval_agent(expected_output, transcription, cosine_sim, static_criteria, expected_similarity=expected_similarity)
    print(f" \n TEST CASE PASSED.")
    print(f"Outcome: {outcome}")
    print(f"Test case outcome Reason: cosine_sim of case: {cosine_sim} vs. min cosine_sim: {expected_similarity}")


    print("\nEvaluate the agent with the test case, response, and cosine similarity\n")
    print(f' TEST CASE: {test_case_str} \n \n RESPONSE: {expected_output} \n')
    # Embed the strings and calculate cosine similarity
    cosine_sim = similar(test_case_str, expected_output)
    print(f'Cosine similarity: {cosine_sim}')

    outcome = eval_agent(expected_output, test_case_str, cosine_sim, static_criteria, expected_similarity=expected_similarity)
    print(f" \n TEST CASE COMPLETE.")
    print(f"Outcome: {outcome}")
    print(f"Test case outcome Reason: cosine_sim of case: {cosine_sim} vs. min cosine_sim: {expected_similarity}")
        


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        transcripton_list= [["test","test2","test3"],["test4","test5","test6","test7","test8","test9"],["test10"],["test11"]]
        expected_output = [["test","test2","test3"],["test4","test5","test6","test7","test8","test9"],["test10"],["test_wrong"]]
        self.assertEqual(check_transcripton(transcripton_list, expected_output), 0.5)
    def test_upper2(self):
        transcripton_list= ['Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?', 'Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migrosbank.ch-konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung der SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Armlegen zu lösen?', 'Haben Sie noch ein weiteres Anliegen?']
        expected_output = ['Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?', 'Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?', 'Haben Sie noch ein weiteres Anliegen?']
        self.assertEqual(check_transcripton(transcripton_list, expected_output), 0.9692946058091286)

test_agent()

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #main()
    #unittest.main()
    #init_test_cases()
    #test_test_case(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



"""
old version of check_transcripton
def check_transcripton(transcripton_list, expected_output):

    number_of_tests = len(transcripton_list)
    #print("Number of tests: " + str(number_of_tests))
    passed_tests = 0
    failed_tests = 0
    test_getting_checked = 0
    # check if at least 80% of the transcripton is correct
    # scr: https://stackoverflow.com/questions/40581010/how-to-calculate-the-similarity-between-two-strings
    for transcriptons in transcripton_list:
        correct_words = 0
        pos = 0
        for words in transcriptons:
            print("Words: "+words)
            print("Expected: "+expected_output[test_getting_checked][pos])
            if words == expected_output[test_getting_checked][pos]:
                correct_words += 1
            pos += 1
        if correct_words/len(expected_output[test_getting_checked]) >= 0.8:
            passed_tests += 1
        else:
            failed_tests += 1
        test_getting_checked += 1
    percentage_passed = passed_tests/number_of_tests
    print("Passed tests: "+str(percentage_passed))
    return (percentage_passed)

"""

"""
number_of_tests = len(transcripton_list)
    #print("Number of tests: " + str(number_of_tests))
    passed_tests = 0
    failed_tests = 0
    test_getting_checked = 0
    # check if at least 80% of the transcripton is correct
    # scr: https://stackoverflow.com/questions/40581010/how-to-calculate-the-similarity-between-two-strings
    for transcriptons in expected_output:
        transcriptons = [item.replace("'", "") for item in transcriptons]
        for items in expected_output:
            items = [item.replace(" ", ",") for item in items]
            print("expected output: " + str(items))

    for transcriptons in transcripton_list:
        transcriptons = [item.replace("'", "") for item in transcriptons]
        for items in transcripton_list:
            items = [item.replace(" ", ",") for item in items]
            print("items: "+str(items))
        correct_words = set(transcripton_list[test_getting_checked]) & set(expected_output[test_getting_checked])
        print("Correct words: "+str(correct_words))
        if len(correct_words)/len(expected_output[test_getting_checked]) >= 0.8:
            passed_tests += 1
        test_getting_checked += 1
    percentage_passed = passed_tests/number_of_tests
    print("Passed tests: "+str(percentage_passed))







"""