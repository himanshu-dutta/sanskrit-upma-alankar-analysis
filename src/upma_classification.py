import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import json
import re
import pandas as pd


client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-123")
MODEL_NAME = "/home/himanshu.dutta/hf_models/llama-3.1-8b-instruct/"

SYSTEM_PROMPT = """You are a highly knowledgeable language model specializing in classical Sanskrit poetics. Your task is to classify a given prose passage in Sanskrit (Romanized) into one of four categories based on the presence of the figure of speech called Upamā alaṅkāra.


**Explanation of Upamā alaṅkāra:**
Upamā alaṅkāra (simile) is a poetic device where a comparison is drawn between two entities. The essential elements of Upamā alaṅkāra are:
   - Upameya (Object of comparison): The entity being described.
   - Upamāna (Standard of comparison): The entity being compared to.
   - Sādhāraṇadharma (Common property/State/Event): The common quality between the two. This common quality can be an object or an action. A noun or verb can denote this event in a verse or sentence.
   - Upamādyotaka (Comparator): Words like iva, yathā, tulya that indicate the comparison are called comparators. For e.g. yathā, iva, vā, va, vat, sadṛśa, tulya, tulya, saṅkāśa, sannibha, upama, nīkāśa, sama, ābha, nibha, pratīkāśa, prakhya, pratinidhi, savarṇa.


**Classification Categories of Upamā alaṅkāra:**
   - Pūrṇopamā (Complete Simile): All four elements are present in the prose or poetry.
   - Luptopamā (Ellided Simile): One or more elements, namely, the Upamāna, Upameya, Upamādyotaka or sādhāraṇadharma are missing, but the comparison is implied.
   - None: No elements of Upamā alaṅkāra are present.


**Input-Output Format:**


Input:
   - A Romanized Sanskrit prose or poetry excerpt.


Output:
   - reason: A text explanation of how the elements of Upamā alaṅkāra are identified or absent.
   - label: One of the categories: Pūrṇopamā, Luptopamā, None.
Output Format:
   {{"reason": "<reason>", "label": "<label>"}}


**Examples:**


Example 1:
Input: "Bhrātarau aśvinau iva rūpeṇa samupasthitayauvanau।।"
Output: {{"reason": "All elements are present: Upameya: bhrātarau, Upamāna: aśvinau, Sādhāraṇadharma: rūpeṇa, Upamādyotaka: iva.", "label": "Pūrṇopamā"}}


Example 2:
Input: "Kāminīgaṇdapaṇḍunā chandreṇa prācīdik alaṅkṛtā"
Output: {{"reason": "The comparative word (Upamādyotaka) is missing, but the comparison is implied.", "label": "Luptopamā"}}


Example 3:
Input: "Vṛkṣaḥ sthiraḥ tiṣṭhati."
Output: {{"reason": "No comparison elements are present.", "label": "None"}}


Give only the output in the specified format and nothing else.
"""

USER_PROMPT_TEMPLATE = """
Input: {sentence} 
Output: 
"""


def main(args):
    sentences = pd.read_csv(args.input_file_path, sep="\t")["sentence"].tolist()
    outputs = list()

    for sentence in tqdm(sentences):
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

            reasoning = response["reason"]
            label = response["label"]
            outputs.append(
                {
                    "sentence": sentence,
                    "reasoning": reasoning,
                    "label": label,
                    "human_label": "",
                    "is_reasoning_correct": True,
                }
            )

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
