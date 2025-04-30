import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import json
import re
import pandas as pd


client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-123")
MODEL_NAME = "/home/himanshu.dutta/hf_models/llama-3.1-8b-instruct/"

SYSTEM_PROMPT = """You are a highly knowledgeable language model specializing in classical Sanskrit poetics.You will be given a prose/poetry excerpt in Sanskrit (Romanized) which has presence of the figure of speech called Upamā alaṅkāra. Your task is to identify the essential elements of Upamā alaṅkāra: Upameya, Upamāna, Sādhāraṇadharma, and Upamādyotaka. Upamā alaṅkāra and its elements are described below.


**Explanation of Upamā alaṅkāra:**
Upamā alaṅkāra (simile) is a poetic device where a comparison is drawn between two entities. The essential elements of Upamā alaṅkāra are:
    - Upameya (Object of comparison): The entity being described.
    - Upamāna (Standard of comparison): The entity being compared to.
    - Sādhāraṇadharma (Common property/State/Event): The common quality between the two. This common quality can be an object or an action. A noun or verb can denote this event in a verse or sentence.
    - Upamādyotaka (Comparator): Words like iva, yathā, tulya that indicate the comparison are called comparators. For e.g. yathā, iva, vā, va, vat, sadṛśa, tulya, tulya, saṅkāśa, sannibha, upama, nīkāśa, sama, ābha, nibha, pratīkāśa, prakhya, pratinidhi, savarṇa.


**Classification Categories of Upamā alaṅkāra:**
    - Pūrṇopamā (Complete Simile): All four elements are present in the prose or poetry.
    - Luptopamā (Elided Simile): One or more elements, namely, the Upamāna, Upameya, Upamādyotaka or sādhāraṇadharma are missing, but the comparison is implied.
    - None: No elements of Upamā alaṅkāra are present.


**Input-Output Format:**

Input:
    - A Romanized Sanskrit prose or poetry excerpt.
Output:
    - Explanation: Reasoning based on which the four elements are identified.
    - The four elements: upameya, upamāna, sādhāraṇadharma, and upamādyotaka, in the specified format.
Output Format:
    {{"upameya": "<upameya>", "upamāna": "<upamāna>", "sādhāraṇadharma": "<sādhāraṇadharma>", "upamādyotaka": "<upamādyotaka>"}}


**Examples:**

Example 1:
Input: "rāmaḥ kālāgnisadṛśaḥ krodhe।"
Explanation: The comparison here is between the ‘rāmaḥ’ and ‘kālāgni’, where ‘rāma’ is the ‘upameya’ and ‘kālāgni’is the ‘upamāna’. The common property is anger, indicated by the word ‘krodhe’. The upamādyotaka used here is ‘sadṛśaḥ’. Since all four components namely, Upameya, Upamāna, sādhāraṇadharma and upamādyotaka are present, this is an example of Pūrṇopamā.
Output: {{"upameya": "rāma", "upamāna": "kālāgni", "sādhāraṇadharma": "krodhe", "upamādyotaka": "sadṛśaḥ"}}

Example 2:
Input: "sītā api anugatā rāmaṃ śaśinaṃ rohiṇī yathā ।"
Explanation: Here, a comparison is being made between ‘sītā’ and ‘rohiṇī’ which means Gods. The sādhāraṇadharma is indicated by the word ‘anugatā’ which means to follow. The Upamā alaṅkāra is indicated by the upamādyotaka/ comparator ‘yathā’. Since, all the four components namely, Upameya, Upamāna, sādhāraṇadharma and upamādyotaka are present, this is an example of Pūrṇopamā.
Output: {{"upameya": "sītā", "upamāna": "rohiṇī", "sādhāraṇadharma": "anugatā", "upamādyotaka": "yatha"}}

Example 3:
Input: "salabdhamānairvinayānvitairnṛpaiḥ purālayairjānapadaiśca mānavaiḥ । upopaviṣṭairnṛpatirvṛto babhau sahasracakṣurbhagavāniva amaraiḥ ।।"
Explanation: Here, a comparison is being made between ‘mānavaiḥ’ which means men and amaraiḥ which means Gods. The sādhāraṇadharma is ‘vṛtaḥ’ which means the common property is to encircle. The Upamā alaṅkāra is indicated by the comparator ‘iva’. Since, all the four components namely, Upameya, Upamāna, sādhāraṇadharma and upamādyotaka are present, this is Pūrṇopamā.    
Output: {{"upameya": "mānavaiḥ", "upamāna": "amaraiḥ", "sādhāraṇadharma": "vṛtaḥ", "upamādyotaka": "iva"}}

Example 4:
Input: "bhārgavēna piturniyōgāt mātari prahr̥tam dviṣadvat।"
Explanation: Here, a comparison is being made between ‘mātari’ which means ‘on mother’ and ‘dviṣad’ which means enemy. The sādhāraṇadharma is ‘prahr̥tam’ which means to execute. The Upamā alaṅkāra is indicated by the comparator ‘vat’. Since, all the four components namely, Upameya, Upamāna, sādhāraṇadharma and upamādyotaka are present, this is Pūrṇopamā.    
Output: {{"upameya": "mātari", "upamāna": "dviṣad", "sādhāraṇadharma": "prahr̥tam", "upamādyotaka": "vat"}}

Give only the output in the specified format and nothing else.
"""

USER_PROMPT_TEMPLATE = """
Input: {sentence} 
Output: 
"""


def main(args):
    with open(args.input_file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    outputs = list()
    for item in tqdm(data):
        label = item["label"]
        if label != "pūrṇopamā":
            continue
        sentence = item["sentence"]
        response = ""
        user_prompt = USER_PROMPT_TEMPLATE.format(sentence=sentence)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=1024,
            )
            response = response.choices[0].message.content.strip().lower()
            # response = response.replace("\n", " ")
            response = json.loads(response)

            item["components"] = response
            outputs.append(item)

        except Exception as e:
            print(f"Exception for sentence: {sentence}")
            print(f"Response: {response}")
    print("Number of successful sentences: ", len(outputs))
    with open(args.output_file_path, "w", encoding="utf-8") as fp:
        json.dump(outputs, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file-path", type=str, required=True)
    parser.add_argument("-o", "--output-file-path", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    args = parser.parse_args()

    main(args)
