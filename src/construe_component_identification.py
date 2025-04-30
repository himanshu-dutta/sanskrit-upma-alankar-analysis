import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import json
import re
import pandas as pd


client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-123")
MODEL_NAME = "/home/himanshu.dutta/hf_models/llama-3.1-8b-instruct/"

SYSTEM_PROMPT = """You are a highly knowledgeable language model specializing in classical Sanskrit poetics.You will be given a prose/poetry excerpt in Sanskrit (Romanized) which has presence of the figure of speech called Upamā alaṅkāra. Your task is to construct the construe and identify the essential elements of Upamā alaṅkāra: Upameya, Upamāna, Sādhāraṇadharma, and Upamādyotaka. Upamā alaṅkāra and its elements are described below.


**Explanation of Upamā alaṅkāra:**
Upamā alaṅkāra (simile) is a poetic device where a comparison is drawn between two entities. The essential elements of Upamā alaṅkāra are:
    - Upameya (Object of comparison): It is the object of comparison in a given instance. It is an object which being described by the speaker and more information is being provided. It is a noun or a pronoun. It is compared with other standard of comparison which is in the grammatical case. It is similar to the notion of ‘topic’ in the figure of speech called ‘Simile’.
    - Upamāna (Standard of comparison): It is the standard of comparison. It an entity with which the object of comparison is compared. It is an entity which is peculiarly known for a specific attribute or quality. It is known popularly for possessing a certain quality. It is similar to the notion of vehicle in the figure of speech called ‘Simile’.
    - Sādhāraṇadharma (Common property/State/Event): It is the common quality or property or attribute on the basis of the two entities are compared. The sādhāraṇadharma in a given instance can be an object or action that is expressed by noun or verb. This is known as ‘event’ or ‘state’ in Simile.
    - Upamādyotaka (Comparator): It is the comparator that indicates an object is being compared with a standard of comparison. There is a list of comparators that re employed in Sanskrit. The comparator can either be compounded with the standard of comparison or can be present in a sentence separately. Words like iva, yathā, tulya that indicate the comparison are called comparators. For e.g. yathā, iva, vā, va, vat, sadṛśa, tulya, tulya, saṅkāśa, sannibha, upama, nīkāśa, sama, ābha, nibha, pratīkāśa, prakhya, pratinidhi, savarṇa.

**Steps for Consture Creation:**
- The first step is to separate all words in the shloka by dissolving sandhi-s.
- The sentence starts with the subject first. The subject can either be in a nominative case or instrumental case or, in a few cases, a genetive case too. If the sentence is in active voice, the subject is a nominative case. If the sentence is in the passive case, the sentence is in the instrumental case. The subject can be a noun or a pronoun.
- If there are any adjectives for the subject, they are placed before it.
- Then the words in the locative case are placed. This is followed by one or more words in the ablative case, and then words in the instrumental case.
- The next step is to identify the object. It is a word that is in the accusative case.
- Adjectives of the object are placed before it.
- If there are certain gerunds ending in ‘’tvā’, they are to be placed now.
- Words in the dative case could be placed now.
- Then, a special type of adjective called ‘vidhi-viśeṣaṇa’ is to be placed. It is an adjective that comes in the construe after the noun that it qualifies. This adjective usually answers the question of what when asked to the noun that it qualifies. E.g., The king became what? The answer is ‘The king became rich’. Rich, despite being an adjective, follows the noun that it qualifies. Similarly, depending upon the context, the words that are ‘vidhi-viśeṣaṇa’ in a given instance are to be placed.
- Then, the adverbs are to be placed.
- Then, the verb is to be placed.
- If there is a negative particle ‘na’. It is to be placed before the verb.
- Upameya and upamāna are in the same grammatical case if the comparators are yathā, iva, vā and va. Of these two words, Upamāna is the one that is often near the upamādyotaka.  Some upamādyotaka-s like yathā are placed before the Upamāna in the construe. While other comparators such as iva, vā and va are placed after the Upamāna.
- In the case of other comparators, such as tulya, saṅkāśa, sadṛkṣa, sannibha, upama, etc., are placed after the Upamāna. If comparators are compounded with the standard of comparison, the grammatical case of this entire word will be the same as that of the object of comparison. Suppose the comparator is not compounded with the standard of comparison. In that case, the standard of comparison will be in the instrumental case, and the grammatical case of the comparator will be the same as that of the object of comparison. Such standards of comparison along with the comparator, are placed after the object of comparison.
- In case of comparator vat, it always follows the standard of comparison and is a suffix. 
- In the construe, out of Upameya and Upamāna, the Upameya is placed before the Upamāna in the sequence of words.


**Input-Output Format:**

Input:
    - A Romanized Sanskrit prose or poetry excerpt.
Output:
    - Construe
    - The four elements: upameya, upamāna, sādhāraṇadharma, and upamādyotaka, in the specified format.
Output Format:
    - Construe
    - Upameya
    - Upamāna
    - Sādhāraṇadharma
    - Upamādyotaka

**Examples:**

Example 1:
Input: "svapne'pi samareṣu tvāṃ vijayaśrīrna muñcati । prabhāvaprabhavaṃ kāntaṃ svādhīnapatikā yathā ।।"
Output:
Construe: vijayaśrīḥ svapne api samareṣu tvāṃ na muñcati, yathā svādhīnapatikā prabhāvaprabhavaṃ kāntaṃ (na muñcati).
Upameya: vijayaśrīḥ
Upamāna: svādhīnapatikā
Sādhāraṇadharma: na muñcati
Upamādyotaka: yathā

Example 2:
Input: "Rāmo asti dhanadena samastyāge"
Output:
Construe: rāmaḥ tyāge dhanadena samaḥ asti
Upameya: rāmaḥ
Upamāna: dhanadena
Sādhāraṇadharma: tyāge
Upamādyotaka: samaḥ

Example 3:
Input: "marutprayuktāśca marutsakhābhaṃ tamarcyamārādabhivartamānam। avākiranbālalatāḥ prasūnairācāralājairiva paurakanyāḥ॥"
Output:
Construe: marutprayuktāḥ bālalatāḥ ca ārād abhivartamānaṃ marutsakhābham arcyaṃ tam ācāralājaiḥ paurakanyāḥ iva prasūnaiḥ avākiran ।
Upameya: bālalatāḥ
Upamāna: paurakanyāḥ
Sādhāraṇadharma: avākiran
Upamādyotaka: iva

Example 4:
Input: "Nagarī sā nityamattaiḥ unnataiḥ sadā pūrṇā nāgaiacalasaṃnibhaiḥ"
Output:
Construe: sā nagarī nityamattaiḥ acalasaṃnibhaiḥ unnataiḥ nāgaiḥ sadā pūrṇā asti.
Upameya: nāgaiḥ
Upamāna: acala
Sādhāraṇadharma: unnataiḥ
Upamādyotaka: saṃnibhaiḥ

Example 5:
Input: "gāmbhīryagarimā tasya satyaṃ gaṅgābhujagaṅgavat ।"
Output:
Construe: tasya gāmbhīryagarimā satyaṃ gaṅgābhujagaṅgavat asti
Upameya: tasya
Upamāna: gaṅgābhujagaṅga
Sādhāraṇadharma: gāmbhīryagarimā
Upamādyotaka: vat

Example 6:
Input: "durālokaḥ sa samare nidaghāmbāraratnavat।"
Output:
Construe: saḥ samare nidaghāmbāraratnavat durālokaḥ asti ।
Upameya: saḥ
Upamāna: nidaghāmbāraratna
Sādhāraṇadharma: durālokaḥ
Upamādyotaka: vat

Give only the output in the specified format and nothing else.
"""

USER_PROMPT_TEMPLATE = """
Input: {sentence} 
Output: 
"""


def parse_string_to_dict(inp):
    # Split the input string into lines
    lines = inp.splitlines()

    # Create an empty dictionary to store the parsed data
    parsed_dict = {}

    # Define the expected keys
    keys = ["construe", "upameya", "upamāna", "sādhāraṇadharma", "upamādyotaka"]

    # Iterate over each line and extract key-value pairs
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()  # Normalize the key to lowercase
            value = value.strip()
            if key in keys:
                parsed_dict[key] = value

    return parsed_dict


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
                temperature=0.5,
            )
            response = response.choices[0].message.content.strip().lower()
            # response = response.replace("\n", " ")
            response = parse_string_to_dict(response)

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
