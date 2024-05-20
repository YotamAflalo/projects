import pandas as pd
import numpy as np

# fluten lists
def flatten_list(lst):
    '''
    Flattens a list of lists into a single list.
    Parameters:
        lst (list): The list to flatten.  
    Returns:
        list: A flattened list.
    '''
    flattened_list = []
    for item in lst:
        if isinstance(item, str):
            item = item.replace('[','')
            item = item.replace(']','')
            item = item.replace("'",'')
            if ', ' in item:
                item = item.split(', ')
                flattened_list.extend(item)
            else:
                flattened_list.append(item)
    return flattened_list

def remove_empty(lst:list):
    ''' remove empty strings from list'''
    if '' in lst:
        lst.remove('')
    return lst

# df ->dict
def df_to_entities_dict(df:pd.DataFrame):
    '''
    Converts a DataFrame into a dictionary of entities.
    Parameters: df (pd.DataFrame): The DataFrame containing entity information.
    Returns: dict: A dictionary containing entity types as keys and lists of unique entities as values.
    '''
    entities_dict = {}
    for var in ['Product name','Affected organs','Affected diseases']:
        entities_dict[var] = remove_empty(list(set(list_standertizer(flatten_list(list(df[var].unique()))))))
    return entities_dict

# node -> list
def node_to_noeds_entities(df:pd.DataFrame):
    '''
    Extracts node entities from a DataFrame containing node information.
    Parameters: df (pd.DataFrame): The DataFrame containing node information.
    Returns: list: A list of standardized node entities.
    '''
    entities_nodes = list_standertizer(list(set(df.node_2.unique()).union((set(df.node_1.unique())))))
    return entities_nodes

# clean strings 
def string_standertizer(s:str,sings:dict=None):
    '''
    Standardizes a string by replacing certain characters and converting it to lowercase.
    Parameters:
        s (str): The input string.
        sings (dict): A dictionary mapping characters to be replaced to their replacements. 
    Returns:
        str: The standardized string.
    '''
    if not sings:
        sings = {'-':' ',' - ':' ',"'":'','?':'','!':''}
    for sing,new in sings.items():
        s = s.replace(sing,new)
    s = s.lower()
    return s

# clean list
def list_standertizer(entitie_list:list,sings:list=None):
    '''
    Standardizes a list of strings using `string_standertizer`.
    Parameters:
        entitie_list (list): The list of strings to be standardized.
        sings (list): A list of characters to be replaced. 
    Returns:
        list: The standardized list of strings.
    '''
    entitie_list = [string_standertizer(name,sings) for name in entitie_list]
    return entitie_list

# q ->  entities - one list
def query_entitie_list_extract(q:str,entitie_list:list = None):
    '''
    Extracts entities from a query based on a given list of entities.
    Parameters:
        q (str): The query string.
        entitie_list (list): The list of entities to search for in the query. 
    Returns:
        list: A list of entities found in the query.
    '''
    entities = []
    q = string_standertizer(q)
    if entitie_list:
        for name in entitie_list:
            if name in q:
                entities.append(name)
    else:
        pass
    return entities

# q - > entities dict
def query_entitie_dict_extract(q:str,entitie_dict:dict[str:list]):
    '''
    Extracts entities from a query based on a dictionary of entity types and lists of entities.
    Parameters:
        q (str): The query string.
        entitie_dict (dict): The dictionary of entity types and lists of entities.
    Returns:
        dict: A dictionary containing entity types as keys and lists of entities found in the query as values.
    '''
    query_entitie_dict = {}
    for ent_type,ent_lst in entitie_dict.items():
        query_entitie_dict[ent_type] = query_entitie_list_extract(q,entitie_list=ent_lst)
    return query_entitie_dict

def non_zero_entitie_checker(entitie_dict:dict[str:list]):
    '''
    Checks if any entity list in the dictionary is non-empty.
    Parameters:entitie_dict (dict): The dictionary of entity types and lists of entities.
        
    Returns: bool: True if any entity list is non-empty, False otherwise.
    '''
    for ent_type,ent_lst in entitie_dict.items():
        if len(ent_lst)>0: return True
    return False

#df sclicer by dict entitie list and type - inclusive and exclusive
def df_entitie_sclicer(df:pd.DataFrame,query_entitie_dict:dict[str:list],inclusive = True):
    '''
    Slices a DataFrame based on the entities extracted from a query.
    Parameters:
        df (pd.DataFrame): The DataFrame to slice.
        query_entitie_dict (dict): The dictionary of entity types and lists of entities extracted from the query.
        inclusive (bool): Flag to determine inclusive or exclusive slicing.
    Returns:
        pd.DataFrame: The sliced DataFrame.
    '''
    if not non_zero_entitie_checker(query_entitie_dict): return None
    df_scliced = df.copy()
    if inclusive:
        df_scliced['keep'] = 0
        if len(query_entitie_dict['Product name'])>0:df_scliced['keep'] = np.where(df_scliced['Product name'].isin(query_entitie_dict['Product name']),1,df_scliced['keep'] ) 
        for ent_type in ['Affected organs','Affected diseases']:
             for i in range(len(df_scliced)):
                if isinstance(df_scliced[ent_type].iloc[i], str):
                    temp_entities = df_scliced[ent_type].iloc[i].replace('[','')
                    temp_entities = temp_entities.replace(']','')
                    temp_entities = temp_entities.replace("'",'')
                    if ', ' in temp_entities:
                        if any(item in query_entitie_dict[ent_type] for item in temp_entities): df_scliced.loc[i, 'keep'] = 1
                    else:
                        if temp_entities in query_entitie_dict[ent_type]: df_scliced.loc[i, 'keep'] = 1
                
        df_scliced = df_scliced[df_scliced['keep']==1]
        df_scliced.drop(columns=['keep'],inplace=True)
    else:
        for ent_type,ent_lst in query_entitie_dict.items():
            if len(ent_lst)>0: df_scliced = df_scliced[df_scliced[ent_type].isin(ent_lst)] #need more work
    return df_scliced

#for node - inclosive and exclusive
def node_entitie_sclicer(df:pd.DataFrame,query_entitie_list:list,inclusive = True):
    '''
    Slices a DataFrame based on the node entities extracted from a query.
    Parameters:
        df (pd.DataFrame): The DataFrame to slice.
        query_entitie_list (list): The list of node entities extracted from the query.
        inclusive (bool): Flag to determine inclusive or exclusive slicing.
    Returns:
        pd.DataFrame: The sliced DataFrame.
    '''
    df_scliced = df.copy()
    if len(query_entitie_list)==0:return
    if inclusive:
        df_scliced['keep'] = 0
        df_scliced['keep'] = np.where(df_scliced['node_1'].isin(query_entitie_list),1,df_scliced['keep'] )
        df_scliced['keep'] = np.where(df_scliced['node_2'].isin(query_entitie_list),1,df_scliced['keep'] )
        df_scliced = df_scliced[df_scliced['keep']==1]
        df_scliced.drop(columns=['keep'],inplace=True)       
    else:
        df_scliced = df_scliced[df_scliced['node_1'].isin(query_entitie_list)]
        df_scliced = df_scliced[df_scliced['node_2'].isin(query_entitie_list)]
    return df_scliced

#add the sliced data to the prompt
def df_slice_to_string(df: pd.DataFrame):
    '''
    Converts a sliced DataFrame to a string.
    Parameters: df (pd.DataFrame): The DataFrame to convert.
    Returns:  str: The DataFrame converted to a JSON string.
    '''
    if isinstance(df, pd.DataFrame): return df['Affect'].to_json(orient='records')

def generate_string_from_table(df: pd.DataFrame):
    '''
    Generates a string from a DataFrame containing table data.
    Parameters: df (pd.DataFrame): The DataFrame containing table data.
    Returns: str: The generated string.
    '''
    strings = []
    for index, row in df.iterrows():
        product_name = row['Product name']
        affected_organs = row['Affected organs']
        affected_diseases = row['Affected diseases']
        affect = row['Affect']
        #string = f"The {product_name} affects {affected_organs}, {affected_diseases} by {affect}"
        string = f"The {product_name} {affect}"
        strings.append(string)
    return "\n".join(strings)

def generate_string_from_node_table(df: pd.DataFrame):
    '''
    Generates a string from a DataFrame containing node data.
    Parameters: df (pd.DataFrame): The DataFrame containing node data.
    Returns: str: The generated string.
    '''
    strings = []
    for index, row in df.iterrows():
        node_1 = row['node_1']
        node_2 = row['node_2']
        edge = row['edge']
        string = f"The connection between {node_1} and {node_2} is {edge}"
        strings.append(string)
    return "\n".join(strings)

def query_df_contaxt_retriver(query:str,df:pd.DataFrame,df_entities_dict:pd.DataFrame=None,inclusive = True,verb=False):
    '''
    Retrieves context from a DataFrame based on a query.
    Parameters:
        query (str): The query string.
        df (pd.DataFrame): The DataFrame containing the context.
        df_entities_dict (pd.DataFrame): The dictionary of entities extracted from the DataFrame.
        inclusive (bool): Flag to determine inclusive or exclusive slicing.
        verb (bool): Flag to print verbose output.
    Returns:
        str: The context retrieved from the DataFrame.
    '''
    if not df_entities_dict : df_entities_dict= df_to_entities_dict(df)
    query_entitie_dict =query_entitie_dict_extract(query,df_entities_dict)
    slice_df = df_entitie_sclicer(df = df,query_entitie_dict = query_entitie_dict,inclusive = inclusive)
    if verb: print(query_entitie_dict, '\nnumber of rows: ',len(slice_df))
    return generate_string_from_table(slice_df)

def query_node_contaxt_retriver(query:str,df:pd.DataFrame,df_entities_list:pd.DataFrame=None,inclusive = True):
    '''
    Retrieves context from a DataFrame containing node data based on a query.
    Parameters:
        query (str): The query string.
        df (pd.DataFrame): The DataFrame containing the context.
        df_entities_list (pd.DataFrame): The list of entities extracted from the DataFrame.
        inclusive (bool): Flag to determine inclusive or exclusive slicing.
    Returns:
        str: The context retrieved from the DataFrame.
    '''
    if not df_entities_list : df_entities_list= node_to_noeds_entities(df)
    query_entitie_list =query_entitie_list_extract(query,df_entities_list)
    slice_df = node_entitie_sclicer(df = df,query_entitie_list = query_entitie_list,inclusive = inclusive)
    return generate_string_from_node_table(slice_df)
