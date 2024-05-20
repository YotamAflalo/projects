import pandas as pd
import numpy as np
from rag_graph import query_df_contaxt_retriver,query_node_contaxt_retriver
from context_retriver import query_with_meta
import anthropic
clode_key ='ENTER YOUR KEY HERE OR IN THE OS'
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=clode_key,
)
df = pd.read_csv(r'C:\Users\yotam\Desktop\how_not_to_project\books_to_table_llm\new_prompt_df.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
df_node = pd.read_csv(r'C:\Users\yotam\Desktop\how_not_to_project\books_to_table_llm\df_saved\How_Not_to_Die_df_graph_v2.csv')

#legacy:
#sys_prompt_answer = '''
#    You are a chat designed to give advice for healthy living.
#    The excerpts below are taken from instructional books on how to live a healthy life
#    write a long answer.
#    Answer the question based on the context below.
#    the context contains excerpts from the books themselves marked <context>,
#    as well as information extracted from a knowledge table marked <table_knowledge>
#    spesify where you got each pise of information before you give the information.
#    Try to reduce the length of the answer, and answer the question accurately without much background
#    If there is a list of answers, you will display them in the following format:
#    "
#    1. base on chank [chank number] of the book [book name], answer1
#    2. base on chank [chank number] of the book [book name], answer2
#    4. base on the table knowledge, answer3
#    "
#    Context:\n    '''

sys_prompt_answer = '''
    You are a chat designed to give advice for healthy living.
    The knowledge below are taken from instructional books on how to live a healthy life
    write a long answer.
    Answer the question based on the context below.
    the context contains excerpts from the books themselves marked <context>,
    the context may contain information extracted from a knowledge table marked <table_knowledge>,
    the context may contain information extracted from a node graph marked <node>,
    use the node data for understending the context.
    spesify where you got each pise of information before you give the information.
    Answer the question accurately without much background
    If there is a list of answers, you will display them in the following format:
    "
    1. base on chank [chank number] of the book [book name], answer1
    2. base on chank [chank number] of the book [book name], answer2
    3. base on the table knowledge, answer3
    
    "
    Context:\n
    
    '''

def user_prompt_esmbly(query:str,df:pd.DataFrame=None,node_df:pd.DataFrame=None,books = None,inclusive = True,context = True,table = False,node = False):
    '''
    This function assembles the user prompt based on the provided query, data frame, and node data frame.
    Parameters:
        query (str): The query text.
        df (pd.DataFrame, optional): Data frame containing knowledge table information. Default is None.
        node_df (pd.DataFrame, optional): Data frame containing node information. Default is None.
        books (list, optional): List of book names to filter results. Default is None.
        inclusive (bool, optional): Flag to include all results. Default is True.
        context (bool, optional): Flag to include context knowledge. Default is True.
        table (bool, optional): Flag to include table knowledge. Default is False.
        node (bool, optional): Flag to include node knowledge. Default is False.
    Returns:
        str: The assembled user prompt.'''
    user_prompt = ''
    if node: user_prompt+='<node>' + query_node_contaxt_retriver(query=query,df=node_df,inclusive=inclusive) +'<node> \n'
    if table: user_prompt+="<table_knowledge>" + query_df_contaxt_retriver(query=query,df=df,inclusive=inclusive) + " <table_knowledge> \n"
    if context: user_prompt+='<context>' + "\n\n---\n\n".join(query_with_meta(query,book_name=books)) + '<context>'
    user_prompt +="\n query: "+ query+"\nAnswer:"
    return user_prompt

def ask_claude(user_prompt,system_prompt=sys_prompt_answer,max_tokens=4000,model ="claude-3-sonnet-20240229",only_text = False):
    '''
    This function sends a query to the Claude model with the specified user and system prompts.

    Parameters:
        user_prompt (str): The user prompt text.
        system_prompt (str, optional): The system prompt text. Default is sys_prompt_answer_new.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 4000.
        model (str, optional): The model to use for generating the response. Default is "claude-3-sonnet-20240229".
        only_text (bool, optional): Flag to return only the text part of the response. Default is False.

    Returns:
        dict or str: The response message from the Claude model, either as a full dictionary or only the text part.'''
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

def rag_answer(query,system_prompt=sys_prompt_answer,model='claude-3-sonnet-20240229',df = None,node_df = None,books = None,inclusive = True,context = True,table = False,node = False):
    '''
    This function generates a response to the query using the RAG (Retrieval-Augmented Generation) approach.
    Parameters:
        query (str): The query text.
        system_prompt (str, optional): The system prompt text. Default is sys_prompt_answer.
        model (str, optional): The model to use for generating the response. Default is 'claude-3-sonnet-20240229'.
        df (pd.DataFrame, optional): Data frame containing knowledge table information. Default is None.
        node_df (pd.DataFrame, optional): Data frame containing node information. Default is None.
        books (list, optional): List of book names to filter results. Default is None.
        inclusive (bool, optional): Flag to include all results. Default is True.
        context (bool, optional): Flag to include context knowledge. Default is True.
        table (bool, optional): Flag to include table knowledge. Default is False.
        node (bool, optional): Flag to include node knowledge. Default is False.
    Returns:
        str: The response text from the Claude model.'''
    prompt = user_prompt_esmbly(query=query,df=df,node_df=df_node,books = books,inclusive = inclusive,context = context,table = table,node = node)

    message = ask_claude(user_prompt=prompt,system_prompt=system_prompt,model=model)
    return message.dict()['content'][0]['text']

