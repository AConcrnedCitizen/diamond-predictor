import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import data

sns.set_theme()

data_frame = data.load_data()

os.makedirs('figures', exist_ok=True)

# Create a graph for each continuous column
plt.figure(figsize=(10,6))
sns.histplot(data_frame['carat'], bins=50)
plt.title('Carat Distribution')
plt.savefig('figures/carat_dist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
sns.countplot(data=data_frame, x='cut')
plt.title('Cut Distribution')
plt.xticks(rotation=45)
plt.savefig('figures/cut_dist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
sns.countplot(data=data_frame, x='color')
plt.title('Color Distribution')
plt.savefig('figures/color_dist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
sns.countplot(data=data_frame, x='clarity')
plt.title('Clarity Distribution')
plt.xticks(rotation=45)
plt.savefig('figures/clarity_dist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
sns.histplot(data_frame['price'], bins=50)
plt.axvline(data_frame['price'].mean(), color='red', linestyle='--')
plt.text(data_frame['price'].mean()*1.1, plt.ylim()[1]*0.9, 
         f'Mean: ${data_frame["price"].mean():,.2f}')
plt.title('Distribution of Diamond Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('figures/price_dist.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,8))
numerical_cols = ['price', 'carat', 'x', 'y', 'z']
correlation_matrix = data_frame[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been generated in the 'figures' directory.")
