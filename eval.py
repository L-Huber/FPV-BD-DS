"""
This script evaluates the response of a chatbot to a test case. It calculates the cosine similarity between the response and the expected output, judges the response based on criteria, and saves the evaluation results to a CSV file.

The script contains the following functions:
- calculate_cosine_similarity: Calculates the cosine similarity between two vectors.
- similar: Calculates the similarity between two strings using OpenAI embeddings.
- eval_agent: Evaluates the agent's response based on the test case, response, cosine similarity, and criteria.
- test_agent: Tests the agent by providing a test case and expected output.

The script also imports the following modules:
- csv: Provides functionality for reading and writing CSV files.
- os: Provides a way to interact with the operating system.
- langchain_openai: Provides access to OpenAI's language models and embeddings.
- dotenv: Loads environment variables from a .env file.
- sklearn.metrics.pairwise: Provides functions for calculating pairwise distances and similarities between vectors.
- langchain_core.messages: Defines message classes for the chatbot evaluation.
"""

import csv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv

# fetch env variables
load_dotenv(find_dotenv())


def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    @param vector1: The first vector.
    @param vector2: The second vector.
    @return: The cosine similarity between the two vectors.
    """
    return cosine_similarity([vector1], [vector2])[0][0]


def similar(a, b):
    """
    Calculate similarity between two strings.

    @param a: The first string.
    @param b: The second string.
    @return: The similarity between the two strings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    a_embedding = embeddings.embed_query(a)
    b_embedding = embeddings.embed_query(b)
    return calculate_cosine_similarity(a_embedding, b_embedding)


def eval_agent(
    test_case_str,
    response_str,
    cosine_similarity,
    static_criteria,
    number_of_criteria=5,
    expected_similarity=0.95,
):
    """
    Evaluate the agent's response.

    @param test_case_str: The test case string.
    @param response_str: The response string.
    @param cosine_similarity: The cosine similarity between the response and expected output.
    @param static_criteria: The static criteria to judge the response on.
    @param number_of_criteria: The number of criteria to judge the response on.
    @param expected_similarity: The expected cosine similarity.
    @return: The test outcome, cosine similarity, and error types.
    """
    test_outcome = "Pass"
    error_types = []

    chat = ChatOpenAI(model="gpt-4", temperature=0)

    if cosine_similarity < expected_similarity:
        test_outcome = "Fail"

    messages = [
        SystemMessage(
            content="""You're an evaluation agent. You're evaluating the response of a chatbot to a test case.
                  The test case is as follows: """
            + test_case_str
            + """
                  First I want you to identify """
            + str(number_of_criteria)
            + """ criteria to judge the response on.
                  Then I want you to evaluate the response based on the criteria.
                  The following criteria are mandatory: """
            + static_criteria
            + """. 
                  If a criteria is not met by the response, you should classify the error based on the criteria."""
        ),
        HumanMessage(content=response_str),
        SystemMessage(
            content="""If you have identified a failed criteria please provide the error type at the very end of your response! Use this format:
              //ERROR START// <error type> //ERROR END//. Please avoid any comments or additional information at all cost! Only include a short description
                of the error type in the start and end flags. USE DIFFERENT FLAGS FOR MULTIPLE ERRORS!"""
        ),
    ]

    # Invoke the chat model
    chat.invoke(messages)

    response = []
    for chunk in chat.stream(messages):
        # save answer to a file
        response.append(chunk.content)

    # Flags to search for
    content = "".join(response)
    eval_message = content.replace("\n", "\\n")  # Replace newlines with "\\n"
    start_flag = "//ERROR START//"
    end_flag = "//ERROR END//"
    error_types = []
    # Loop to extract multiple error types
    while start_flag in content and end_flag in content:
        start_index = content.find(start_flag)
        end_index = content.find(end_flag)
        if start_index != -1 and end_index != -1:
            # Extract the error type between the flags
            error_type = content[start_index + len(start_flag) : end_index].strip()
            error_types.append(error_type)
            # Remove the extracted error type from the message content
            content = content[end_index + len(end_flag) :].strip()

    if error_types == []:
        error_types = ["No errors found in the response."]

    # save response to csv
    with open("eval_history.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Check if the file is empty
        if os.stat("eval_history.csv").st_size == 0:
            # If it is, write the descriptors
            writer.writerow(
                [
                    "Test Case",
                    "Response",
                    "Cosine Similarity",
                    "Outcome",
                    "Error Types",
                    "Eval_Message",
                ]
            )
        # Write the data
        writer.writerow(
            [
                test_case_str,
                response_str,
                cosine_similarity,
                test_outcome,
                error_types,
                eval_message,
            ]
        )

    # structure return format
    return test_outcome, cosine_similarity, error_types


# Function to test the agent
def test_agent():
    # List of transcripton and expected output
    transcripton_list = [
        "Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?",
        "Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migrosbank.ch-konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung der SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Armlegen zu lösen?",
        "Haben Sie noch ein weiteres Anliegen?",
    ]
    expected_output_list = [
        "Habe ich Sie richtig verstanden, dass Sie gerne ein Konto eröffnen möchten?",
        "Ein Konto können Sie bequem über unsere Webseite beantragen. Unter dem Link www.migosbank.ch slash Konten begleiten wir Sie bei der Kontoeröffnung. Ich schicke Ihnen umgehend einen Link mit der Anleitung per SMS, um ein Konto digital zu beantragen. Bitte öffnen Sie die SMS, klicken Sie auf den Link und beantragen Sie die Kontoeröffnung online auf unserer Webseite. Danach warte ich kurz bis sie das SMS gelesen und die Anleitung unter dem angegebenen Link gelesen haben. In 20 Sekunden werde ich sie fragen, ob ihnen die Anleitung weiter geholfen hat. Hilft Ihnen meine Anleitung, Ihr Anliegen zu lösen?",
        "Haben Sie noch ein weiteres Anliegen?",
    ]

    test_case_str = "Sie können Ihr Konto einfach und schnell auf unserer Webseite eröffnen. Besuchen Sie www.beispielbank.de/konten und folgen Sie den Anweisungen für die Kontoeröffnung. Wir werden Ihnen dann eine E-Mail mit weiteren Details senden. Bitte überprüfen Sie auch Ihren Spam-Ordner, falls Sie die E-Mail nicht in Ihrem Posteingang finden. Haben Sie Fragen oder benötigen Sie weitere Unterstützung?"

    static_criteria = "1. Link sent per SMS, 2.Link contains correct URL = www.migrosbank.ch-konten, 3. Politness, 4. Clear instructions, 5. Follow-up question."
    expected_similarity = 0.95

    # list to string to get embeddings
    transcription = ", ".join(transcripton_list)
    expected_output = ", ".join(expected_output_list)

    # Print the test case and expected response
    print(
        "\nEvaluate the agent with the transcription case, response, and cosine similarity\n"
    )
    print(f" TEST CASE: {transcription} \n \n EXPECTED RESPONSE: {expected_output} \n")
    # Embed the strings and calculate cosine similarity
    cosine_sim = similar(transcription, expected_output)
    print(f"Cosine similarity: {cosine_sim}")

    # Evaluate the agent
    outcome = eval_agent(
        expected_output,
        transcription,
        cosine_sim,
        static_criteria,
        expected_similarity=expected_similarity,
    )
    print(f" \n TEST COMPLETE.")
    print(f"Outcome: {outcome}")
    print(
        f"Test case outcome Reason: cosine_sim of case: {cosine_sim} vs. min cosine_sim: {expected_similarity}"
    )

    # Print the test case and expected response
    print("\nEvaluate the agent with the test case, response, and cosine similarity\n")
    print(f" TEST CASE: {test_case_str} \n \n EXPECTED RESPONSE: {expected_output} \n")
    # Embed the strings and calculate cosine similarity
    cosine_sim = similar(test_case_str, expected_output)
    print(f"Cosine similarity: {cosine_sim}")

    # Evaluate the agent
    outcome = eval_agent(
        expected_output,
        test_case_str,
        cosine_sim,
        static_criteria,
        expected_similarity=expected_similarity,
    )
    print(f" \n TEST COMPLETE.")
    print(f"Outcome: {outcome}")
    print(
        f"Test case outcome Reason: cosine_sim of case: {cosine_sim} vs. min cosine_sim: {expected_similarity}"
    )


test_agent()
