from pinecone import Pinecone
import pandas as pd

from sentence_transformers import SentenceTransformer

model_emmbed = SentenceTransformer('BAAI/bge-large-en-v1.5')
def get_embeddings_vector(txt):
    '''
    This function generates an embedding vector for a given text using the SentenceTransformer model.
    Parameters:
        txt (str): The input text to be converted into an embedding vector.
    Returns:
        numpy.array: The embedding vector of the input text.'''
    return model_emmbed.encode(txt)
pinecone = Pinecone(api_key='ENTER YOUR KEY HERE OR IN THE OS')
index = pinecone.Index('health-bge-large')
#books_names = ['Outlive','The_4_Hour_Body','How_Not_to_Die']

def query_with_meta(query, book_name:list[str]=None):
    '''
    This function queries the Pinecone index with an embedding vector generated from the query text,
    and optionally filters the results by book names.
    Parameters:
        query (str): The query text to search for in the index.
        book_name (list[str], optional): A list of book names to filter the results by. Default is None.
    Returns:
        list: A list of text metadata from the top matching results in the index.'''
    embed = get_embeddings_vector(query)
    books_names = ['Outlive','The_4_Hour_Body','How_Not_to_Die']
    if book_name:
        clean_list = []
        for book in book_name:
            if book in books_names: 
                clean_list.append(book)
        if clean_list:
            res = index.query(vector=embed.tolist(), top_k=4,
                            filter={"book_name": {"$in": clean_list}},
                            include_metadata=True)
        else:
            res = index.query(vector=embed.tolist(), top_k=4, include_metadata=True)  
    else: 
        res = index.query(vector=embed.tolist(), top_k=4, include_metadata=True)
    return [x['metadata']['text'] for x in res['matches']]
def test():
    '''
    This function is a test case for the query_with_meta function. It queries the index with a sample query
    and returns the contexts of the top matching results.

    Returns:
        list: A list of text metadata from the top matching results for the sample query.
'''
    query = "How not to die from cancer ?"
    contexts = query_with_meta(query=query)
    return contexts

