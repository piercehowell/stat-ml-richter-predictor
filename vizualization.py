# %%
from sklearn.manifold import TSNE
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import comp_dataset, study_dataset

n_datapoints = 10000
# Get Dataset
X_train, y_train, X_val, y_val = comp_dataset.train_data()
target = y_train[:n_datapoints]
data = pd.DataFrame(X_train[:n_datapoints])


def show_plot(x, y, title):
    df = pd.DataFrame(data)
    df['damage grade']=target
    df['feature 1']=x
    df['feature 2']=y
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='feature 1',y='feature 2',hue='damage grade',data=df,
                legend="full")
    plt.title(title)
    plt.show()

# %% [markdown]
# ## TSNE

# %%
data = pd.DataFrame(X_train[:n_datapoints])
tSNE=TSNE(n_components=2, verbose=1, n_jobs=-1)
tSNE_result=tSNE.fit_transform(data)
x=tSNE_result[:,0]
y=tSNE_result[:,1]
show_plot(x,y,"t-SNE")

# %% [markdown]
# ## MDS

# %%
data = pd.DataFrame(X_train[:n_datapoints])
mds=MDS(n_components=2, verbose=1, n_jobs=-1)
mds_result=mds.fit_transform(data)
x=mds_result[:,0]
y=mds_result[:,1]
show_plot(x,y,"Multidimensional Scaling")

# %% [markdown]
# ## IsoMap

# %%
data = pd.DataFrame(X_train[:n_datapoints])
iso=Isomap(n_components=2, n_neighbors=50, n_jobs=-1)
iso_result=iso.fit_transform(data)
x=iso_result[:,0]
y=iso_result[:,1]
show_plot(x,y, "IsoMap")

# %% [markdown]
# ## Local Linear Embedding

# %%
data = pd.DataFrame(X_train[:n_datapoints])
LLE=LocallyLinearEmbedding(n_components=2, n_jobs=-1)
LLE_result=LLE.fit_transform(data)
x=LLE_result[:,0]
y=LLE_result[:,1]
show_plot(x,y,"Local Linear Embedding")

# %% [markdown]
# ## PCA

# %%
data = pd.DataFrame(X_train[:n_datapoints])
LLE=PCA(n_components=2)
LLE_result=LLE.fit_transform(data)
x=LLE_result[:,0]
y=LLE_result[:,1]
show_plot(x,y, "PCA")

# %%
data = pd.DataFrame(X_train[:n_datapoints])
KPCA=KernelPCA(n_components=2, kernel='poly', degree=1)
KPCA_result=LLE.fit_transform(data)
x=KPCA_result[:,0]
y=KPCA_result[:,1]
show_plot(x,y,"Kernel PCA, Degree=1")

data = pd.DataFrame(X_train[:n_datapoints])
KPCA=KernelPCA(n_components=2, kernel='poly', degree=2)
KPCA_result=LLE.fit_transform(data)
x=KPCA_result[:,0]
y=KPCA_result[:,1]
show_plot(x,y,"Kernel PCA, Degree=2")

# %% [markdown]
# ## Pairplot

# %%
import os
features_df = pd.read_csv("data/TRAIN.csv")
# X = features_df[ int_columns + categ_columns + binary_columns]
sns.pairplot(features_df[:n_datapoints], hue="damage_grade")

# %%
X_train, y_train, X_val, y_val = study_dataset.train_data()
target = y_train[:n_datapoints]
data = pd.DataFrame(X_train[:n_datapoints])
print(X_train.shape)

# %% [markdown]
# ## TSNE

# %%
data = pd.DataFrame(X_train[:n_datapoints])
tSNE=TSNE(n_components=2, verbose=1, n_jobs=-1)
tSNE_result=tSNE.fit_transform(data)
x=tSNE_result[:,0]
y=tSNE_result[:,1]
show_plot(x,y,"t-SNE")

# %% [markdown]
# ## MDS

# %%
data = pd.DataFrame(X_train[:n_datapoints])
mds=MDS(n_components=2, verbose=1, n_jobs=-1)
mds_result=mds.fit_transform(data)
x=mds_result[:,0]
y=mds_result[:,1]
show_plot(x,y,"Multidimensional Scaling")

# %% [markdown]
# ## IsoMap

# %%
data = pd.DataFrame(X_train[:n_datapoints])
iso=Isomap(n_components=2, n_neighbors=50, n_jobs=-1)
iso_result=iso.fit_transform(data)
x=iso_result[:,0]
y=iso_result[:,1]
show_plot(x,y,"IsoMap")

# %% [markdown]
# ## Local Linear Embedding

# %%
data = pd.DataFrame(X_train[:n_datapoints])
LLE=LocallyLinearEmbedding(n_components=2, n_jobs=-1)
LLE_result=LLE.fit_transform(data)
x=LLE_result[:,0]
y=LLE_result[:,1]
show_plot(x,y,"Local Linear Embedding")

# %% [markdown]
# ## Pairplot

# %%
import os
features_df = pd.read_csv("data/TRAIN_STUDY.csv")
# X = features_df[ int_columns + categ_columns + binary_columns]
sns.pairplot(features_df[:n_datapoints], hue="damage_grade")


