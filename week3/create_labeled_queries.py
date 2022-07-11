import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'
train_output_file_name = r'/workspace/datasets/train'
test_output_file_name = r'/workspace/datasets/test'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]
# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
stemmer = nltk.stem.snowball.SnowballStemmer("english")
regexes =   [(r"[\+]", " plus"),
            (r"\$(\d+)", r"\1 dollars"),
            (r"\&", " and "),
            (r"\s*[\./_-]\s*", " "),
            (r"[^\w\s]", "")]

df["query"] =  df["query"].str.lower()
for regex, replace in regexes:
    df["query"] = df["query"].str.replace(regex, replace, regex = True)
df["query"] = df["query"].apply(lambda x: stemmer.stem(x))

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
parents_dict = parents_df.set_index("category")["parent"]
parents_dict[root_category_id] = root_category_id

category_cutoff_count = 10000

def categories_below_cutoff():
    return (df["category"].value_counts()<category_cutoff_count).sum()

num_of_categories_below_cutoff = categories_below_cutoff()
while num_of_categories_below_cutoff > 0:
    categories_below = df["category"].value_counts().where(lambda x: x<category_cutoff_count).dropna()
    df["category"] = df["category"].apply(lambda x: parents_dict[x] if x in categories_below else x)
    num_of_categories_below_cutoff = categories_below_cutoff()

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
train = df[['output']].sample(frac = 0.8)
test = df[["output"]].drop(train.index)
train.to_csv(train_output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
test.to_csv(test_output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)