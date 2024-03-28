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

#insert assembly ai key
aai.settings.api_key = ""


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
        list_of_csv[5] = list_of_csv[5].split('|')
        print(list_of_csv)
        test_case1 = test_case(list_of_csv[0], list_of_csv[1], list_of_csv[2], list_of_csv[3], list_of_csv[4], list_of_csv[5])
        list_of_test_cases.append(test_case1)
        print(list_of_test_cases[0].id)

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
def main():
    global list_of_test_cases
    list_of_test_cases = []
    # Use a breakpoint in the code line below to debug your script.
    #rec_audio('output_sounddevice.wav', 3)
    #time.sleep(2)
    #play_audio('output_sounddevice.wav')
    #transcribe_audio('output_sounddevice.wav')
    init_test_cases()
    test_test_case(1)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
