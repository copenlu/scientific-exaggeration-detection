import pandas as pd


if __name__ == '__main__':
    pubmed = pd.read_csv('./annotated_pubmed.csv')
    eureka = pd.read_csv('./annotated_eureka.csv')

    eureka = eureka[['sentence', 'label']]

    all_data = pd.concat([pubmed, eureka])
    all_data.columns = ['text', 'label']
    all_data.to_csv('./combined_strength_data.csv', index=None)