import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
import nltk
# from cleaning_func import replace_starting_characters,extract_text_without_pattern,check_pattern,extract_matching_patterns,process_icd_codes_O,combo_code
# nltk.download('punkt')
# nltk.download('punkt_tab')
import warnings
import os
from scipy.spatial.distance import cosine
import time
warnings.filterwarnings('ignore')
from jproperties import Properties
from tqdm import tqdm
import re
import pickle
from gensim.models import Word2Vec
import h5py
import pyarrow as pa
import pyarrow.parquet as pq

def load_data(save_dir):
    """
    Load the Word2Vec models, matrices, and dataframe from the save directory.
    """
    
    # Load Word2Vec models more efficiently
    model_path = os.path.join(save_dir, 'word2vec_model.model')
    model = Word2Vec.load(model_path)

    df_path = os.path.join(save_dir, 'df_with_embeddings.parquet')
    # Read the Parquet file into a PyArrow Table
    table = pq.read_table(df_path)
    # Convert the Table back to a pandas DataFrame
    df = table.to_pandas()
    # print(f"DataFrame loaded from {df_path}")

    combined_matrix_path = os.path.join(save_dir, 'combined_matrix.h5')
    with h5py.File(combined_matrix_path, 'r') as h5f:
        combined_matrix = h5f['matrix'][:]
    
    return df, model, combined_matrix


df, model, embedding_matrix = load_data(r"C:/Users/vc/project/icd_automation/model")
print("df :",df.shape)

configs=Properties()

# Function to compute the average Word2Vec embedding for a list of words
def get_avg_word2vec(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:  # If no tokens in vocabulary, return a zero vector
        return np.zeros(vector_size)
    return np.mean([model.wv[token] for token in valid_tokens], axis=0)


# Function to find the most similar diagnosis code
def single_description(input_description, df, model, embedding_matrix):

    ele_dia_desc_S = input_description
    ele_dia_desc_S=ele_dia_desc_S.split("~")
    ele_code_list=[]
    ele_sim_list=[]

    for dia_desc_S in ele_dia_desc_S:
   
        filtered_text = dia_desc_S.lower()
        input_tokens = word_tokenize(filtered_text)
        # input_embedding = get_avg_word2vec(input_tokens, model, 150).reshape(1, -1)
        input_embedding = get_avg_word2vec(input_tokens, model, 150)
        # Compute cosine similarity between input description and all embeddings in the dataset
        # similarities = cosine_similarity(input_embedding, embedding_matrix)[0]
        similarities = np.dot(embedding_matrix, input_embedding) / (
                np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(input_embedding) + 1e-8
        )
        # Add the similarity scores to the DataFrame
        df['similarity'] = similarities

        # Sort the results by similarity (descending) and get top 5 matches
        most_similar_df = df[['Description', 'ICD Code', 'similarity']].sort_values(by='similarity', ascending=False).head(5)
        # print(most_similar_df)
        sorted_df1=most_similar_df[most_similar_df['Description']==filtered_text]
        # Check if filtered DataFrame is empty
        if sorted_df1.empty:
            # Return rows where column does not match the value
            sorted_df1 = most_similar_df[most_similar_df["Description"] != filtered_text]
            # print("No matches found. Returning other rows:")
        else:
            # print("Matches found:")
            # print(sorted_df1.head(1))
            pass
        # print(sorted_df1.head(5))
        # Filtering rows where the 'Score' is greater than 30
        sorted_df1 = sorted_df1[sorted_df1['similarity'] >= 0.8]
        if(len(sorted_df1)>0):
            ele_description=dia_desc_S
            ele_diagnosis_code=sorted_df1['ICD Code'].iloc[0]
            ele_similarity=sorted_df1['similarity'].iloc[0]
        else:
            ele_description=dia_desc_S
            ele_diagnosis_code="0000"
            ele_similarity=0
            
        if((len(ele_diagnosis_code.split(",")))>1):
            ele_code_list.append(ele_diagnosis_code)
            ele_sim_list.append(','.join(map(str,np.full(len(ele_diagnosis_code.split(",")),round(ele_similarity,3)).tolist())))

        else:
            # print(out,":",m_cos)
            ele_code_list.append(ele_diagnosis_code)
            ele_sim_list.append(round(ele_similarity,3))
            # ele_sim_list1=ele_code_list.copy()
        

    ele_code_list=",".join(ele_code_list)  
    ele_sim_list=",".join(map(str, ele_sim_list))  
    response1 = {
                "ICD Code":ele_code_list,
                "Similarity":ele_sim_list
            }
    response_1 = {
            "ICD Code":ele_code_list.split(","),
            "Similarity":ele_sim_list.split(",")
        }
    
    response=pd.DataFrame(response_1)
    response['ICD Code'] = response['ICD Code'].str.strip()
    response.drop_duplicates(subset=['ICD Code'],inplace=True)
    # print(response)
    response=response.to_dict(orient='list')
    
    ele_code_list1=",".join(map(str,response['ICD Code']))
    ele_sim_list1=",".join(map(str,response['Similarity']))
    response_act = {
            "ICD Code":ele_code_list1,
            "Similarity":ele_sim_list1
        }
    print(response)

    return response