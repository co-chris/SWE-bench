import os
import json
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import cohere
import asyncio
import aiohttp
from typing import List, Dict
import requests

from inference.make_datasets.utils import extract_diff
from swebench.harness.colours import blue



def add_toks(text):
    return f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{text}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

def get_checkpoint(ip_address):
    url = f"http://{ip_address}/config"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps({}))
    response_json = json.loads(response.text)
    checkpoint_path = response_json['checkpoint_path']
    return checkpoint_path



class Fax_Sampler():

    def __init__(self, server_url): #, max_sequence_length):
        self.server_url = server_url
        # self.max_sequence_length = max_sequence_length
        # self.tokenizer = BPTokenizer("gs://cohere-dev/encoders/releases/0.3.1/75k_bos_eos_eop.json")

    async def raw_sample_request(self, session: aiohttp.ClientSession, queries) -> List[Dict]:
        async with session.post(f"http://{self.server_url}/batch_generate", json=queries) as resp:
            return await resp.json()

    async def run_generation(self, queries):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(None)) as sess:
            responses = await self.raw_sample_request(sess, queries)
            try:
                completions = [resp['completions'][0] for resp in responses]
            except Exception as e:
                print (responses)
                raise e
            return completions

    def sample(self, prompt_list, n_completions, temp): #, total_tokens=None):
        """
        Pass in a list of prompts and get back df where each completion is a row
        """
        # if total_tokens is None:
        #     total_tokens = self.max_sequence_length

        prompt_list = [add_toks(prompt) for prompt in prompt_list]
        prompt_lengths = [len(prompt)//3 for prompt in prompt_list]
        # print (prompt_lengths)

        # make max tokens be 2000
        n_tokens = [40000 - int(length) for length in prompt_lengths]
        n_tokens = [min(4000, n) for n in n_tokens]
        # print (n_tokens)
        for nt in n_tokens:
            if nt < 4000:
                print (f"Warning: n_tokens is less than 4000: {nt}")

        # Prepare the dataframe
        df = pd.DataFrame({"prompt": prompt_list})
        df["prompt"] = df.prompt
        df["temperature"] = temp
        df["n_completions"] = n_completions
        df["truncate"] = "START"# "NONE"
        df["p"] = .9

        df["n_tokens"] = n_tokens
        
        if temp == 0:
            df["k"] = 1
        else:
            df["k"] = 0
        
        # repeat the prompts n_completions times
        df = df.loc[df.index.repeat(n_completions)].reset_index(drop=True)

        # convert to list of dicts
        queries = df.to_dict('records')
        # for query in queries:
        #     print (query['prompt'])

        # generate completions
        samples = asyncio.run(self.run_generation(queries))
        # print (len(samples))
        # print (samples[0])

        df['completion'] = samples

        # for i, row in df.iterrows():
        #     # print (row.prompt)
        #     print (row.completion)
        #     print ('-----------------')
        # fadsf
        return df







def get_responses(client, model_name, datapoint):
    prompt = datapoint["prompt"]
    try:
        response = client.chat(
            message=prompt,
            temperature=0,
            model=model_name,
        )
    except Exception as e:
        print (f"Error: {e}")
        fadsfasd
    return response.__dict__







def cohere_inference(
    datapoints,
    model_path,
    output_file,
):



    if ':' in model_path:
        ckpt = get_checkpoint(model_path)
        print (f"Using checkpoint: {blue(ckpt)}")
        # Fax model
        sampler = Fax_Sampler(model_path)
        batch_size = 16
        n_datapoints = len(datapoints)
        batch = []
        for i in tqdm(range(n_datapoints)):
            # if i < 47:
            #     continue

            batch.append(datapoints[i])
            if len(batch) == batch_size or i == n_datapoints-1:
                prompt_list = [datum["prompt"] for datum in batch]
                df = sampler.sample(prompt_list, n_completions=1, temp=0)
                for i, row in df.iterrows():
                    output_dict = {
                        "instance_id": batch[i]["instance_id"], 
                        "model_name_or_path": ckpt,
                        "prompt": batch[i]["prompt"],
                        "full_output": row["completion"],
                        # "model_patch": extract_diff(row["completion"]),
                    }

                    with open(output_file, "a+") as f:
                        f.write(json.dumps(output_dict) + "\n")
                batch = []

    




    else:
        api_key = os.environ.get("COHERE_API_KEY", None)
        cohere_client = cohere.Client(api_key)

        # api_key = os.environ.get("COHERE_STG_API_KEY")
        # cohere_client = cohere.Client(base_url='https://stg.api.cohere.ai', api_key=api_key)



        # set the client to the cohere client
        get_responses_cohere = lambda x: get_responses(cohere_client, model_path, x)

        n_datapoints = len(datapoints)
        batch_size = 4
        # batch = []
        batch = []
        for i in tqdm(range(n_datapoints)):

            # batch.append(test_dataset[i])
            batch.append(datapoints[i])
            if len(batch) == batch_size or i == n_datapoints-1:
                # text_batch = [datum['text'] for datum in batch]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    responses = list(executor.map(get_responses_cohere, batch))

                for datum, response in zip(batch, responses):
                    completion = response['text']
                    # print (response.keys())
                    output_dict = {
                        "instance_id": datum["instance_id"], 
                        "model_name_or_path": model_path,
                        "prompt": datum["prompt"],
                        "full_output": completion,
                        # "model_patch": extract_diff(completion),
                    }

                    with open(output_file, "a+") as f:
                        f.write(json.dumps(output_dict) + "\n")
                batch = []






