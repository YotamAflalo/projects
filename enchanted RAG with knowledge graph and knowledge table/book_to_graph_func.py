import re
import pandas as pd
import json
import time
import anthropic

clode_key ='ENTER YOUR KEY HERE OR IN THE OS'

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=clode_key,
)

SYS_PROMPT = (
    "You are a network graph maker who extracts terms and their relations from a given context. "
    "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
    "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
    "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, \n"
        "\tcondition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible\n\n"
    "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms\n\n"
    "Thought 3: Find out the relation between each such related pair of terms. \n\n"
    "Format your output as a list of json. Each element of the list contains a pair of terms"
    "and the relation between them, like the follwing: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology",\n'
    '       "node_2": "A related concept from extracted ontology",\n'
    '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
    "   }, {...}\n"
    "]"

    "do not use \" in the edge text"
)
def remove_quotes(text):
    """
    This function removes uneeded quotes from a text to ajust formatting to json.
    Parameters:
        text (str): The input text that may contain quotes to be removed.
    Returns:
        str: The text with specific quotes removed or replaced.
    """
    # Define the regex pattern
    pattern = r'(?<![,:{])\"(?![,:{\n])'

    # Remove all matches using the pattern
    result = re.sub(pattern, '', text)
    result = result.replace('node_1": ','"node_1": "')
    result = result.replace('node_2": ','"node_2": "')
    result = result.replace('edge": ','"edge": "')
    #pattern2 = r'",(?!\n)'
    # Replace `",` not followed by `\n` with `,`
    #fixed_text = re.sub(pattern2, ',', text)
    return result


def ask_haike(user_prompt,system_prompt=SYS_PROMPT,max_tokens=4000,model ="claude-3-haiku-20240307",only_text = False):
    '''
    This function interacts with an AI model to get a response based on the user's prompt.
    Parameters:
        user_prompt (str): The input prompt from the user + added data.
        system_prompt (str): The system's predefined prompt. Default is SYS_PROMPT.
        max_tokens (int): The maximum number of tokens to generate in the response. Default is 4000.
        model (str): The model to use for generating the response. Default is "claude-3-haiku-20240307".
        only_text (bool): Whether to return only the text content of the response. Default is False.
    Returns:
        If only_text is True, returns the text content of the response.
        Otherwise, returns the full message response.'''
    message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    temperature=0.0,
    system=system_prompt,
    messages=[
        {"role": "user", "content": user_prompt}])
    if only_text:
        return message.dict()['content'][0]['text']
    return message

def chanks_to_df_for_graph(chank_dict,prompt=SYS_PROMPT,max_token_per_chank=4000,verbos=False,test = False):
    '''
    This function processes chunks of text, interacts with the AI model to get structured data,
    and compiles the data into a DataFrame suitable for graphing.
    Parameters:
        chank_dict (dict): A dictionary where keys are chunk names and values are the corresponding text chunks.
        prompt (str): The system's predefined prompt. Default is SYS_PROMPT.
        max_token_per_chank (int): The maximum number of tokens to generate per chunk response. Default is 4000.
        verbos (bool): Whether to print verbose output during processing. Default is False.
        test (bool): Whether to run in test mode, which limits the number of chunks processed. Default is False.
    Returns:
        tuple: A tuple containing:
            - full_df (pd.DataFrame): The compiled DataFrame with all processed chunks.
            - chanks_not_added (list): A list of chunks that were not successfully added to the DataFrame.'''
    chanks_names = list(chank_dict.keys())
    message = ask_haike(f"context: ```{chank_dict[chanks_names[0]]}``` \n\n output: ",max_tokens=max_token_per_chank,only_text = False)
    txt = message.dict()['content'][0]['text']
    data = json.loads(txt[txt.find('['):txt.find(']')+1])
    full_df = pd.DataFrame(data)
    full_df['Chunk Name'] = chanks_names[0]
    chanks_not_added = []
    num =1
    api_eror = 0
    for chank in chanks_names[1:]:
        time.sleep(0.5)
        num +=1
        if (not num % 10) and verbos : 
            print(f'pass {num} chanks')
            if test: 
                full_df['count'] = 4
                return full_df,chanks_not_added
        try:
            message = ask_haike(f"context: ```{chank_dict[chank]}``` \n\n output: ",max_tokens=max_token_per_chank,only_text = False)
            txt = message.dict()['content'][0]['text']
        except:
            try:
                time.sleep(30)
                message = ask_haike(f"context: ```{chank_dict[chank]}``` \n\n output: ",max_tokens=max_token_per_chank,only_text = False)
                txt = message.dict()['content'][0]['text']
            except:
                print('eror in the api')
                try:
                    chanks_not_added.append([chank, message.dict()])
                except:
                    chanks_not_added.append(chank)
                print(chank) 
                api_eror+=1
                if api_eror >10: return full_df,chanks_not_added         
        try:
            data = json.loads(txt[txt.find('['):txt.find(']')+1])
        except:
            try:
                data = json.loads(remove_quotes(txt[txt.find('['):txt.find(']')+1]))
            except:
                print('eror in the JSON.LOADS')
                chanks_not_added.append([chank,txt])
                if verbos:
                    print(chank)
                continue
        try:   
            new_df = pd.DataFrame(data)
            new_df['Chunk Name'] = chank
            # Append the new data to the existing DataFrame
            full_df = pd.concat([full_df, new_df], ignore_index=True)
        except:
            print('eror in the pd add')
            chanks_not_added.append([chank,txt])
            if verbos:
                print(chank)
    full_df['count'] = 4
    return full_df,chanks_not_added