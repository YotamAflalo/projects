import tiktoken
import pandas as pd
import fitz
clode_key ='ENTER YOUR KEY HERE OR IN THE OS'
from PyPDF2 import PdfReader
import anthropic
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=clode_key,
)
# create the length function
tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    '''
    This function calculates the number of tokens in a given text using the tiktoken library.
    Parameters:
        text (str): The input text to be tokenized.
    Returns:
        int: The number of tokens in the input text.
    '''
    tokens = tokenizer.encode(text,disallowed_special=())
    return len(tokens)

def full_book_constraction(path,starting_page=0,ending_page=None):
    '''
    This function reads a PDF file and constructs a single string containing the text from the specified page range.
    Parameters:
        path (str): The file path to the PDF.
        starting_page (int): The page number to start reading from. Default is 0.
        ending_page (int): The page number to stop reading at. If None, reads until the end of the document.
    Returns:
        str: The concatenated text from the specified pages of the PDF.
    '''
    book = PdfReader(path)
    if not ending_page: ending_page = len(book)
    full_book = ''
    for i in range(starting_page,ending_page): #len(book.pages) #i cuted the starting pages and the notes in the end
            page = book.pages[i].extract_text().replace('\t',' ')#.replace('\n',' ')
            full_book = full_book + page + ' '
    return full_book
def headers_scrape_How_Not_to_Die(filePath,jank='',min_size=0):
    '''
    This function extracts headers  from a PDF that according to font size.

    Parameters:
        filePath (str): The file path to the PDF.
        jank (str): A string to exclude from the results if present in the text.
        min_size (float): The minimum font size to include in the results.

    Returns:
        list: A list of tuples containing the text, font size, and font name.
    '''
    results = [] # list of tuples that store the information as (text, font size, font name) 
    pdf = fitz.open(filePath) # filePath is a string that contains the path to the pdf
    for page in pdf:
        dict = page.get_text("dict")
        blocks = dict["blocks"]
        for block in blocks:
            if "lines" in block.keys():
                spans = block['lines']
                for span in spans:
                    data = span['spans']
                    for lines in data:
                        if lines['size']>=min_size:
                            if (not jank) or (jank not in lines['text'].lower()): # only store font information of a specific keyword
                                results.append((lines['text'], lines['size'], lines['font']))
                            # lines['text'] -> string, lines['size'] -> font size, lines['font'] -> font name
    pdf.close()
    return results

def book_to_chank_dict_list(full_book:str,headers_txt:list):
    '''
    This function splits  full book  into chunks based on headers and returns a dictionary and a list of chunks.
    Parameters:
        full_book (str): The full text of the book.
        headers_txt (list): A list of header strings to split the book text.
    Returns:
        tuple: A tuple containing:
            - chank_dict (dict): A dictionary where keys are headers and values are text chunks.
            - chank_list (list): A list of text chunks.'''
    chank_list = []
    chank_dict = {}
    book_chank = full_book
    temp = ''
    for i in range(len(headers_txt)-1):
        new_chank = book_chank[:book_chank.find(headers_txt[i+1])]
        if len(new_chank)<200:
            temp = temp+new_chank             
            continue

        chank_list.append(temp+new_chank)
        if headers_txt[i] in chank_dict.keys():
            chank_dict[headers_txt[i]+str(i)]=temp+new_chank
            print(headers_txt[i]+str(i))
        else:
            chank_dict[headers_txt[i]]=temp+new_chank
        before_len = len(book_chank)
        book_chank = book_chank[book_chank.find(headers_txt[i+1]):]
    return chank_dict,chank_list


def new_book_to_chank_dict_list(full_book:str,headers_txt:list=None,book_name =None):
    if headers_txt: return book_to_chank_dict_list(full_book,headers_txt)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=75,
        length_function=tiktoken_len)
    chunks = text_splitter.split_text(full_book)
    chank_list = chunks.copy()
    if book_name:
        chank_dict = {f'chank{i} from {book_name}':chank_list[i] for i in range(len(chank_list))}
    else:
        chank_dict = {f'chank{i}':chank_list[i] for i in range(len(chank_list))}
    return chank_dict,chank_list
    
def ask_haike(user_prompt,system_prompt,max_tokens=4000,model ="claude-3-haiku-20240307",only_text = False):
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

def chanks_to_df(chank_dict,sys_prompt=None,max_token_per_chank=4000,verbos=False):
    '''
    This function processes chunks of text, interacts with the AI model to get structured data,
    and compiles the data into a DataFrame.
    Parameters:
        chank_dict (dict): A dictionary where keys are chunk names and values are the corresponding text chunks.
        prompt (str): The system's predefined prompt.
        max_token_per_chank (int): The maximum number of tokens to generate per chunk response. Default is 2500.
        verbos (bool): Whether to print verbose output during processing. Default is False.
    Returns:
        tuple: A tuple containing:
            - full_df (pd.DataFrame): The compiled DataFrame with all processed chunks.
            - chanks_not_added (list): A list of chunks that were not successfully added to the DataFrame.'''
    if not sys_prompt:
        sys_prompt = '''
                Your goal is to extract structured information from the user's input that matches the form below. The text you will receive comes from self-development books. The goal of the project is to extract all the tips described in the book and label them.
                When extracting information please make sure that it matches the type of information exactly. Do not add any attributes that do not appear in the schma below.
                ''
                {
                [{product type: //food, suplement, additives ext'
                Product name: //name of the product, 
                category: //the category of the prodact, Whole grains, green leaves, drinks, additives, dairy products, animal products, legumes, beans, vitamins, etc.
                Recommendation: //does the book recommend consuming this product or to avoid him? Answer '1' for recommended and '0' for not recommended, '2' if it is more complicated.,
                Quantity: // how much is recommended for consumption? If specified, else write "quantity not specified".,
                Affect: //what is the benefit/harm of consuming the product?,
                Affected organs: // The organs that are affected by the consumption of the food.,
                Affected diseases: //The disease that affected by the consumption. Rather the food helps to cure, delays, prevents or cause the diseases appearance?,
                The mechanism: //Describe how consuming the food benefits\harm the body (on the chemical level).
                }]
                ''
                Insert all the recommendations you find in the text to consume or avoid consuming certain foods/supplements, or to exercise.
                do not use " in the mechanism describe.
                Please output the extracted information in JSON format. Do not output anything except for the extracted information. Do not add any clarifying information. Do not add fields that are not in the schema. If the text contains information that are not in the schema, please ignore them. All output mast be in JSON format and follow the schema.
                Warp the JSON in <json> tags.
                Between the delimiters '$$$' there is an example of how the answer should look. Do not use the example as part of your answer.

                $$$
                Example:
                Input: To test the power of dietary interventions to prevent rabies, scientists often study chronic sleepers. Researchers rounded up a group of long-time sleepers and asked them to consume goat's milk a day.
                Goat milk consumers suffered 41 percent less rabies in their bloodstream over ten days. It is only thanks to this that goat milk causes bad breath, and keeps dogs away from you.
                Output: <json> {'food': { 'Product name': 'Goat milk', 'category': 'Dairy', 'Recommendation': '1', 'Quantity': 'not specified', 'Affect': 'Reduces risk of rabies', 'Affected organs': [], 'Affected diseases': ['rabies'], 'The mechanism': 'Goat milk causes bad breath, and keeps dogs away from you.'} <\json>
                $$$
                DO NOT recommend Goat milk.

                The user text: 

                '''
    chanks_names = list(chank_dict.keys())
    message = ask_haike(chank_dict[chanks_names[0]],system_prompt=sys_prompt,max_tokens=max_token_per_chank,only_text = False)
    txt = message.dict()['content'][0]['text']
    data = json.loads(txt[txt.find('<json>')+7:txt.find('</json>')-1])
    full_df = pd.DataFrame(data)
    full_df['Chunk Name'] = chanks_names[0]
    chanks_not_added = []
    num =1
    for chank in chanks_names[1:]:
        time.sleep(0.5)
        num +=1
        if (not num % 10) and verbos : print(f'pass {num} chanks')
        try:
            message = ask_haike(chank_dict[chank],system_prompt=sys_prompt,max_tokens=max_token_per_chank,only_text = False)
            txt = message.dict()['content'][0]['text']
        except:
            try:
                time.sleep(30)
                message = ask_haike(chank_dict[chank],system_prompt=sys_prompt,max_tokens=max_token_per_chank,only_text = False)
                txt = message.dict()['content'][0]['text']
            except:
                print('eror in the api')
                chanks_not_added.append(chank)
                print(chank)
                if num >3: return full_df,chanks_not_added  
                continue       
        try:
            data = json.loads(txt[txt.find('<json>')+7:txt.find('</json>')-1])
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

    return full_df,chanks_not_added