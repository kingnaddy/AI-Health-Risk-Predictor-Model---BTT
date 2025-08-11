# generate_plots.py
# Usage: python generate_plots.py
# Requires: train.csv in the same folder

import os
import pandas as pd
import matplotlib.pyplot as plt

project_path = os.path.dirname(__file__)
images_dir = os.path.join(project_path, 'images')
os.makedirs(images_dir, exist_ok=True)

train_path = os.path.join(project_path, 'train.csv')

def main():
    if not os.path.exists(train_path):
        print('train.csv not found. Please place train.csv in the project root.')
        return

    df = pd.read_csv(train_path)

    # 1) Smoking distribution
    plt.figure()
    df['smoking'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Smoking Status (0 = Non-smoker, 1 = Smoker)')
    plt.xlabel('smoking')
    plt.ylabel('count')
    plt.savefig(os.path.join(images_dir, 'smoking_distribution.png'), bbox_inches='tight')
    plt.close()

    # 2) Boxplot for common numerical features
    num_cols = [c for c in ['Gtp','hemoglobin','serum creatinine','weight(kg)','triglyceride','height(cm)','waist(cm)','age'] if c in df.columns]
    if len(num_cols) >= 1:
        plt.figure()
        df[num_cols].boxplot(rot=45)
        plt.title('Boxplot of Numerical Features')
        plt.ylabel('value')
        plt.savefig(os.path.join(images_dir, 'numerical_features_boxplot.png'), bbox_inches='tight')
        plt.close()
    else:
        print('None of the expected numerical columns were found; skipping boxplot.')

if __name__ == '__main__':
    main()
