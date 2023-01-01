import json
import os
import zipfile
import urllib.request
import dask.bag as db
import pandas as pd
from datasets import Dataset

def retrieve_url(url, filename):
    if not os.path.exists(filename) and not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f'{filename} exists! Nothing to download')


def unzip_file(zip_filename, file_path, filename):
    if not os.path.exists(os.path.join(file_path,filename)) and not os.path.isfile(os.path.join(file_path,filename)):
        with zipfile.ZipFile(zip_filename, 'r') as zObject:
            zObject.extract(filename, path=file_path)
            zObject.close()
    else:
        print(f'{filename} exists! Nothing to unzip')


def get_json_from_file(filename):
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def print_dict(json_dict, items=5):
    print({x: json_dict[x] for (i, x) in enumerate(json_dict) if i < items})

def get_df_from_file(json_filename, filename):

    docs = db.read_text(json_filename).map(json.loads)
    # print(docs.count().compute())
    # print(docs.take(1))

    category_map = {'cs.AI': 'Artificial Intelligence',
                    'cs.LG': 'Machine Learning',
                    'cs.SE': 'Software Engineering',
                    'econ.EM': 'Econometrics',
                    'math.PR': 'Probability',
                    'math.ST': 'Statistics Theory',
                    'q-bio.QM': 'Quantitative Methods',
                    'q-fin.CP': 'Computational Finance',
                    'q-fin.EC': 'Economics',
                    'q-fin.GN': 'General Finance',
                    'q-fin.MF': 'Mathematical Finance',
                    'q-fin.PM': 'Portfolio Management',
                    'q-fin.PR': 'Pricing of Securities',
                    'q-fin.RM': 'Risk Management',
                    'q-fin.ST': 'Statistical Finance',
                    'stat.ML': 'Machine Learning',
                    'stat.OT': 'Other Statistics',
                    'stat.TH': 'Statistics Theory'}

    # print(category_map.keys())

    get_latest_version = lambda x: x['versions'][-1]['created']

    trim = lambda x: {'id': x['id'],
                      'date': int(x['versions'][-1]['created'][11:16]),
                      'authors': x['authors'],
                      'title': x['title'],
                      'doi': x['doi'],
                      'category': x['categories'].split(' '),
                      'abstract': x['abstract'], }

    docs_df = (docs.filter(lambda x: int(get_latest_version(x).split(' ')[3]) >= 2018)
                   .filter(lambda x: x['categories'] in category_map.keys())
                   .map(trim)
                   .compute())

    # convert to pandas
    docs_df = pd.DataFrame(docs_df)
    docs_df.to_csv(filename, index=False)

   # docs_df = pd.read_csv(filename, low_memory=False)
    docs_df['abstract_word_count'] = docs_df['abstract'].apply(lambda x: len(x.strip().split()))
    docs_df.drop_duplicates(['abstract', ], inplace=True)

    return docs_df


if __name__ == '__main__':
    url = 'https://storage.googleapis.com/kaggle-data-sets/612177/4767544/compressed/arxiv-metadata-oai-snapshot.json.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com/20221231/auto/storage/goog4_request&X-Goog-Date=20221231T181626Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=668a1069656b6dea3ef6ff6bec5126b5b9af787a4f1e1b077efb1532f88f82b3eee543fa1b84f5032fd0350183c18e59a5b05aed5a5bcb77c5b72c1c22020897e944fe41c876c16e04de7e4e078859652d18f89e650d390dffbfa3e82b5618f52c82aaa83bbb962839de6fe51ce3a4f9e28139259b85e748d58a1628982593f9f0ccef88c9548830260b9ec96ca0675e9e656990b746507335c2307e568cff795e9b4d07c6a384cb4d4254c82440c2abda542a888ab5c3f94d281dd0ec5e956950066fbd666ea63d722e3a1a6e960dbcff57ec0b3ef50b22eabf33370f42a6083f7c22526d870a0b013de1e8ece8cc1f7c47646bf0926ec07480506ebf773463'
    data_path = './datasets/Cornell-University/arxiv/'
    zip_filename = f'{data_path}/arxiv-metadata-oai-snapshot.json.zip'
    json_filename = 'arxiv-metadata-oai-snapshot.json'
    json_filename_with_path = os.path.join(data_path, json_filename)
    filename = 'trimmed_arxiv_docs.csv'
    filename_with_path = os.path.join(data_path, filename)
    df_path = './datasets/Cornell-University/arxiv/trimmed_arxiv'

    retrieve_url(url, zip_filename)
    unzip_file(zip_filename, data_path, json_filename)
    docs_df = get_df_from_file(json_filename_with_path, filename_with_path)


    print("Shape: " , docs_df.shape)
    print("describe: " , docs_df.describe())
    print("docs_df['abstract'].describe(include='all'): ", docs_df['abstract'].describe(include='all'))
    print("head(1).T ", docs_df.head(1).T)

    # convert pandas dataframes to hugginface datasets
    dataset = Dataset.from_pandas(docs_df).train_test_split(test_size=0.1)

    dataset.save_to_disk(df_path)


    print(dataset)