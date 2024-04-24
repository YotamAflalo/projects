import math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def category_cleaner(df):
    df['Category'] = df['Category'].str.strip(r'[]\'.,\"')
    df['Category'] = df['Category'].str.lower()
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:11] != 'Young Adult' else 'Young Adult')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:11] != 'Young adult' else 'Young Adult')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:3] != 'Zoo' else 'Zoo')
    df['Category'] = df['Category'].apply(lambda txt: txt if txt[0:7] != 'Cookery' else 'Cookery')
    df['Category'] = df['Category'].apply(lambda txt: 'literary' if 'literary' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'biography & autobiography' if 'biography' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'biography & autobiography' if 'autobiography' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'history' if 'history' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'business & economics' if 'business' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'business & economics' if 'economics' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'psychology' if 'psychology' in txt else txt)
    df['Category'] = df['Category'].apply(
        lambda txt: 'fiction' if ('fiction' in txt) and ('juvenile' not in txt) and ('nonfiction' not in txt) else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'health & fitness' if 'health' in txt else txt)
    df['Category'] = df['Category'].apply(lambda txt: 'philosophy' if 'philosophy' in txt else txt)
    big_cat = list(
        df.groupby(['Category'])['Category'].count()[df.groupby(['Category'])['Category'].count() > 90].sort_values(
            ascending=False).index)
    big_cat = big_cat[3:]
    for cat in big_cat:
        df['Category'] = df['Category'].apply(lambda txt: cat if cat in txt else txt)
        cat_lst = cat.split()
        df['Category'] = df['Category'].apply(lambda txt: cat if cat_lst[0] in txt else txt)
    return df


def df_cleaner(df, min_book_ratings: int = 5, min_user_ratings: int = 2):
    '''
    drop books with les then 'min_book_rating' raters, and users that rate less then 'min_user_ratings' books
      '''
    filter_books = df['isbn'].value_counts() > min_book_ratings
    filter_books = filter_books[filter_books].index.tolist()

    filter_users = df['user_id'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()
    print('The original data frame shape:\t{}'.format(df.shape))

    df_new = df[(df['isbn'].isin(filter_books))]  # &
    print('The data frame shape after bookk filtering:\t{}'.format(df_new.shape))
    df_new = df_new[df_new['user_id'].isin(filter_users)]
    print('The new data frame shape:\t{}'.format(df_new.shape))
    return df_new


def category_compliter(df):
    df_books = df.groupby('isbn').agg(
        {'book_title': 'first', 'book_author': 'first', 'year_of_publication': 'first', 'user_id': 'count',
         'age': 'mean', 'rating': 'mean', 'publisher': 'first', 'Category': 'first', 'img_s': 'first', 'img_m': 'first',
         'img_l': 'first', 'Summary': 'first',
         'Language': 'first', 'city': pd.Series.mode, 'state': pd.Series.mode, 'country': pd.Series.mode})

    df_author = df_books.groupby('book_author').agg(
        {'Category': pd.Series.mode, 'user_id': 'sum', 'publisher': 'count'})
    df_author['freq_Category'] = df_books.groupby('book_author')['Category'].agg(
        lambda x: x.mode()[0] if (x.mode()[0] != '9') or (len(x.mode()) == 1) else x.mode()[1])
    df_author['num_topic'] = df_author['Category'].apply(lambda lst: 1 if type(lst) == str else len(lst))
    df_author['topic9'] = df_author['Category'].apply(lambda lst: 1 if '9' in lst else 0)

    df_author = df_author[df_author['topic9'] == 1]
    df_author_relevant = df_author[df_author['num_topic'] > 1]
    df_author9_relevant_two_options = df_author_relevant[df_author_relevant['num_topic'] == 2]
    df_author9_relevant_two_options['pred_Category'] = df_author9_relevant_two_options['Category'].apply(
        lambda lst: lst[-1])

    df_add_cat = df_author9_relevant_two_options['freq_Category']
    add_lst = list(df_add_cat.index)
    df = df.reset_index()
    for i in range(len(df)):
        if df.loc[i, 'book_author'] in add_lst:
            df.loc[i, 'Category'] = df_add_cat[df.loc[i, 'book_author']]
    df = df.drop(['index', 'Unnamed: 0'], axis=1)
    return df

def zero_droper(df):
    '''drop rating of 0, they look odd'''
    df = df[df['rating']>0]
    df.reset_index(inplace = True, drop = True)
    return df

def precentage_null(df):
    '''return % of null in df'''
    null_count = df.isnull().sum()
    total_count = len(df)
    precentage_nulls = (null_count / total_count) * 100
    print(precentage_nulls)

def category_embedding(df,trans_model=None,col ='Category'):
    '''doing emmbeding to the collume 'col' and returning df with the emmbeding data inside'''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if trans_model:
        model = SentenceTransformer(trans_model)
    df_category = pd.DataFrame(df[col].unique())
    # Encode category names into embeddings
    category_embeddings = model.encode(df_category[0], show_progress_bar=True)
    df_category['Embedding_cat'] = category_embeddings.tolist()
    for i in range(len(category_embeddings[0])):
        df_category[f'Embedding_{col}{i}'] = df_category['Embedding_cat'].apply(lambda arr: arr[i])
    df_category=df_category.rename(columns = {0:col})
    df = pd.merge(df, df_category, on = col, how = "left")
    return df
from sklearn.decomposition import PCA
def pca_embedding(df,comp=230):
    '''
    using pca to reduce the number of dimensions of the problem with minimal loss of information'''
    df = df.drop(['Embedding_cat_x','Embedding_cat_y'],axis=1)
    embeded_cols = [col for col in df.columns if ('Embedding' in col)]
    not_embeded_cols = [col for col in df.columns if ('Embedding' not in col)]
    df_embeding = df.drop(not_embeded_cols,axis=1)
    df = df.drop(embeded_cols,axis=1)
    pca = PCA()
    df_embeding = pca.fit_transform(df_embeding)
    df_embeding_df = pd.DataFrame(df_embeding)
    df_embeding_df = df_embeding_df.drop([i for i in range(230,768)],axis=1)
    df = pd.merge(df.reset_index(), df_embeding_df.reset_index(), on = 'index', how = "left")
    df = df.drop('index',axis =1)
    return df
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

def reader_encode(df,y=None):
    '''one hot encoding the 'Language'and 'country' categories'''

    cat_var = ['Language','country']
    one_hot = OneHotEncoder(sparse=False)  # , drop = 'first')
    encoder_var_array = one_hot.fit_transform(df[cat_var])
    encoder_name = one_hot.get_feature_names_out(cat_var)
    encoder_vars_df = pd.DataFrame(encoder_var_array, columns=encoder_name)
    df = pd.concat([df, encoder_vars_df], axis=1)
    return df
from collections import defaultdict

def precision_recall_at_k(predictions, k=10, threshold=8):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        #precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1 #שיניתי ל1
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else None
        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else None

        #recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


# recomending func:
import warnings

warnings.filterwarnings('ignore')
import requests
from PIL import Image
import matplotlib.pyplot as plt


def rated_books(df, reader_id):
    '''showing the books the user rated'''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    df_reader = df.loc[df['user_id'] == reader_id]
    fig, axs = plt.subplots(1, min(5, len(df_reader)), figsize=(18, 5))
    fig.suptitle('Yours previous ratings:', size=22)
    print(len(df_reader))
    for i in range(min(5, len(df_reader))):
        img_url = df_reader.iloc[i]['img_l']

        try:
            response = requests.get(img_url, headers=headers, stream=True)

            response.raise_for_status()  # Check if the request was successful

            # Open the image from the response content (bytes) and convert to RGB mode
            raw_image = Image.open(response.raw).convert('RGB')

            # You can perform further processing with the `raw_image` object here

        except requests.exceptions.RequestException as e:
            raw_image = 'eror'  # לשים פה תמונה של מסך שחור
            # print(f"Error fetching the image from the URL: {e}")
            continue
        except Image.UnidentifiedImageError:
            raw_image = 'eror'  # לשים פה תמונה של מסך שחור
            # print("Unable to identify the image file. Please check the URL or image format.")
            continue

        axs[i].imshow(raw_image)
        axs[i].axis("off")
        axs[i].set_title('your rating: ' + str(df_reader.iloc[i]['rating'])
                         , y=-0.18, color="blue", fontsize=12)
        fig.show()
    return


def deep_recommender(df, reader_id,model,x_train, rec=5):
    '''predict and sho the prediction visualy'''
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    df_books = df.drop_duplicates(subset='isbn_num', keep="first")
    df_reader = df.loc[df['user_id'] == reader_id]
    read_already = list(df_reader['isbn_num'].drop_duplicates())
    reader_cols = ['user_id'] + list(x_train.columns)[251:]
    book_cols = set(df_books.columns) - set(reader_cols)
    df_reader = df_reader.drop(list(book_cols), axis=1)
    df_books = df_books[~df_books['isbn_num'].isin(read_already)]
    df_reader = df_reader.iloc[0:1]

    df_books = df_books.drop(list(reader_cols), axis=1)
    for col in list(df_reader.columns):
        df_books[col] = df_reader.iloc[0][col]

    pred = model.predict({
        "user": df_books["user_id"],
        "book": df_books["isbn_num"],
        "author": df_books['author_num'],
        "book_data": df_books[[str(i) for i in range(0, 230)]],
        'user_data': df_books[list(x_train.columns)[251:]]})

    df_books['pred_score'] = np.mean(pred, axis=1)
    df_books = df_books.sort_values(by='pred_score', ascending=False)

    # proses to get 5 recommended books
    fig, axs = plt.subplots(1, rec, figsize=(18, 5))
    fig.suptitle('You may also like these books', size=22)
    for i in range(rec):
        img_url = df_books.iloc[i]['img_l']

        try:
            response = requests.get(img_url, headers=headers, stream=True)
            response.raise_for_status()  # Check if the request was successful
            # Open the image from the response content (bytes) and convert to RGB mode
            raw_image = Image.open(response.raw).convert('RGB')
            # You can perform further processing with the `raw_image` object here

        except requests.exceptions.RequestException as e:
            raw_image = 'eror'  # לשים פה תמונה של מסך שחור
            # print(f"Error fetching the image from the URL: {e}")
            continue
        except Image.UnidentifiedImageError:
            raw_image = 'eror'  # לשים פה תמונה של מסך שחור
            # print("Unable to identify the image file. Please check the URL or image format.")
            continue

        axs[i].imshow(raw_image)
        axs[i].axis("off")
        axs[i].set_title('predicted joy: ' + str(df_books.iloc[i]['pred_score'])
                         , y=-0.18, color="blue", fontsize=12)
        fig.show()
