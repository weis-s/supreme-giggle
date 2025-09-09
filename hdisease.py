import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df = pd.read_csv('C:\Users\Ziak\Desktop\weis\archive')
df.head()
df.info()
cols = ['trestbps', 'chol']
new_df = df[cols].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(data=new_df, y='trestbps', ax=axes[0,0], color='#0180CC')
sns.histplot(data=new_df, x='trestbps', kde=True, ax=axes[0,1], color='#0180CC')
sns.boxplot(data=new_df, y='chol', ax=axes[1,0], color='#0180CC')
sns.histplot(data=new_df, x='chol', kde=True, ax=axes[1,1], color='#0180CC')
plt.tight_layout()
plt.savefig('outlier_plot')
plt.show()
column1, column2 = 'trestbps', 'chol'

# Function to find outliers using a chosen method
def find_outliers(method, col):
    lower, upper = method(col)
    outliers = new_df[(new_df[col] < lower) | (new_df[col] > upper)][col]
    return outliers
    
# Function to print an outlier detection report
def print_outlier_report(outliers):
    print("Detected outliers : ", outliers.values)
    print("No. of outliers detected:", len(outliers))
    print("Removed data : {:.2f}%".format(len(outliers)/len(new_df) * 100))

# trimming function
def trimming(outliers, col):
    outlierIndex = outliers.index
    removed = new_df.drop(outlierIndex, axis=0)
    print(f"Data Shape ->\nBefore trimming : {new_df.shape}\nAfter trimming : {removed.shape}")
    
# Capping function
def capping(method, col, method_name):
    lower_threshold, upper_threshold = method(col)
    new_df[col + '_capped' + method_name] = np.where(
        new_df[col] > upper_threshold, upper_threshold,
        np.where(new_df[col] < lower_threshold, lower_threshold, new_df[col])
    )

# Plot KDE distribution to observe the overall shape and skewness of the column's data  
def kdeplot(col):
    if col == 'chol':
        col_name='Cholesterol'  
    else:
        col_name='Blood Pressure'    
        
    plt.figure(figsize=(12,6))
    
    sns.kdeplot(new_df[col], label="Original", color="#FF6F61", linewidth=2)
    sns.kdeplot(new_df[col + "_capped_zscore"], label="Z-score Capped", color="#6A5ACD", linewidth=2)
    sns.kdeplot(new_df[col + "_capped_percentile"], label="Percentile Capped", color="#20B2AA", linewidth=2)
    sns.kdeplot(new_df[col + "_capped_iqr"], label="IQR Capped", color="#FFD700", linewidth=2)

    plt.title(f"{col_name} Distribution Before and After Capping", fontsize=14)
    plt.legend()
    plt.show()

# Plot boxplot to visualize the spread and detect potential outliers in the column  
def boxplot(col):
    if col == 'chol':
        col_name='Cholesterol'  
    else:
        col_name='Blood Pressure'    
        
    melted = new_df.melt(value_vars=[col, col + "_capped_zscore", col + "_capped_percentile", col + "_capped_iqr"],
        var_name="Method", value_name=col_name
    )

    plt.figure(figsize=(14,6))
    sns.boxplot(data=melted, x="Method", y=col_name, palette="Set2")
    plt.title(f"{col_name} Distribution Across Methods")
    plt.show()
def z_score_val(col):
    Mean = new_df[col].mean()
    Std = new_df[col].std()
    return (Mean - 3*Std), (Mean + 3*Std)
# For column1 ['trestbps']
outliers = find_outliers(z_score_val, column1)
print_outlier_report(outliers)
trimming(outliers, column1)
capping(z_score_val, column1, '_zscore')
# For column2 ['chol']
outliers = find_outliers(z_score_val, column2)
print_outlier_report(outliers)
trimming(outliers, column2)
capping(z_score_val, column2, '_zscore')
def percentile_val(col):
    lower = new_df[col].quantile(0.01)
    upper = new_df[col].quantile(0.99)
    return lower, upper
# For column1 ['trestbps']
outliers = find_outliers(percentile_val, column1)
print_outlier_report(outliers)
trimming(outliers, column1)
capping(percentile_val, column1, '_percentile')
# For column2 ['chol']
outliers = find_outliers(percentile_val, column2)
print_outlier_report(outliers)
trimming(outliers, column2)
capping(percentile_val, column2, '_percentile')
def iqr_val(col):
    Q1 = new_df[col].quantile(0.25)
    Q3 = new_df[col].quantile(0.75)
    IQR = Q3 - Q1
    return (Q1 - 1.5*IQR), (Q3 + 1.5*IQR)
# For column1 ['trestbps']
outliers = find_outliers(iqr_val, column1)
print_outlier_report(outliers)
trimming(outliers, column1)
capping(iqr_val, column1, '_iqr')
# For column2 ['chol']
outliers = find_outliers(iqr_val, column2)
print_outlier_report(outliers)
trimming(outliers, column2)
capping(iqr_val, column2, '_iqr')
kdeplot(column1)
boxplot(column1)
kdeplot(column2)
boxplot(column2)
new_df.describe()
skew = [new_df[c].skew() for c in new_df.columns]

p = pd.DataFrame(
    {"Method": ["Original", "Z-score", "Percentile", "IQR"],
    "Column1 ['trestbps']": skew[0::2],
    "Column2 ['chol']": skew[1::2]}
)

print(f"Skewness:\n{p}")
