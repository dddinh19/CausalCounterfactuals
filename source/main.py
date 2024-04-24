import pandas as pd
import numpy as np
import lingam

import dice_ml
from dice_ml.utils.neuralnetworks import train_and_save_model
from sklearn.model_selection import train_test_split
from dice_ml.utils.neuralnetworks import AdvancedNet

import torch
from torch.utils.data import Dataset

from causal_data_augmentation.api_support.typing import GraphType


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def load_data(path: str) -> pd.DataFrame:
    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "matiral-status", "occupation",
                    "relationship", "race", "sex", "captital-gain", "captital-loss", "hours-per-week", "native-country", "class"]
    data = pd.read_csv(path, delimiter=', ', names=column_names)
    data['class'] = data['class'].replace({'<=50K': 0, '>50K': 1})
    data = data.drop(["captital-loss", "captital-gain", "education", "fnlwgt"], axis=1)
    # data = data.select_dtypes('number')
    data = data.replace('?', np.nan)
    data = data.dropna()
    return data

def load_model(model_path, x_train):
    model = AdvancedNet(x_train)  # MyModel() should be the architecture of your model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def normalize_df(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def estimate_lingam(X: pd.DataFrame, prior_knowledge_matrix=None) -> GraphType:
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge_matrix)
    model.fit(X)
    vertices = X.columns
    directed_edges = []
    bi_edges = []
    for i, j in zip(*np.nonzero(model.adjacency_matrix_)):
        directed_edges.append((vertices[j], vertices[i]))
    return vertices, directed_edges, bi_edges

def get_top_casual(X: pd.DataFrame, num_top_relationships, prior_knowledge_matrix=None):
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge_matrix)
    model.fit(X)
    column_name = X.columns
    casual_direction = []

    # Get the top causal relationships
    top_adjacency_matrix = model.get_top_causal_relationships(num_top_relationships)

    for i, j in zip(*np.nonzero(top_adjacency_matrix)):
        casual_direction.append((column_name[j], column_name[i]))
    return casual_direction

def get_adjacency_matrix(X: pd.DataFrame, prior_knowledge_matrix=None):
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge_matrix)
    model.fit(X)
    return model.adjacency_matrix_

def create_prior_knowledge_matrix(df, prior_causal_features):
    """
    Creates a prior knowledge matrix from a list of feature pairs with causal relationships.
    
    :param df: DataFrame containing the data
    :type df: pd.DataFrame
    :param prior_causal_features: List of tuples containing pairs of features with causal relationships (cause, effect)
    :type prior_causal_features: List[Tuple[str, str]]
    :return: Prior knowledge matrix
    :rtype: np.ndarray
    """
    n_features = df.shape[1]
    prior_knowledge_matrix = np.full((n_features, n_features), -1)
    
    def get_index(prefix):
        return [idx for idx, col in enumerate(df.columns) if col.startswith(prefix)]
    
    for prior_causal_feature in prior_causal_features:
        effect, cause, element = prior_causal_feature
    
        if cause and effect:  # cả cause và effect đều không rỗng
            cause_indices = get_index(cause)
            effect_indices = get_index(effect)
        
            for cause_idx in cause_indices:
                for effect_idx in effect_indices:
                    prior_knowledge_matrix[cause_idx, effect_idx] = int(element)
    
        elif cause:  # chỉ cause không rỗng
            indices = get_index(cause)
            if indices:
                prior_knowledge_matrix[indices, :] = int(element)
    
        elif effect:  # chỉ effect không rỗng
            indices = get_index(effect)
            if indices:
                prior_knowledge_matrix[:, indices] = int(element)
    
        else:
            print('At least one of the cause and effect must exist')
                
    return prior_knowledge_matrix


# Processing data, creating train_dataset and test_dataset
data = load_data("/Users/dinhdinh/Documents/SC_19125039_INTEGRATING CAUSAL CONSTRAINTS INTO COUNTERFACTUAL EXPLANATIONS/datasets/adult.csv")
dataset = data.drop(["class"], axis = 1)
target = data["class"]  # outcome variable
train_dataset, test_dataset, _, _ = train_test_split(data, target, test_size=0.2, random_state = 1, stratify=target)

d = dice_ml.Data(dataframe=train_dataset, data_name='adult',
                 continuous_features=['age', 'education-num', 'hours-per-week'],
                 outcome_name='class')

# Target encoding train_dataset, preparing for input into DirectLiNGAM
process_data = train_dataset.copy()
mapping_tables = {}

for categorical_feature_name in d.categorical_feature_names:
    mapping_dict = train_dataset.groupby(categorical_feature_name)['class'].mean().to_dict()
    process_data[categorical_feature_name] = train_dataset[categorical_feature_name].map(mapping_dict)
    mapping_tables[categorical_feature_name] = mapping_dict

process_dataset = process_data.drop(['class'], axis = 1)
# Using prior knowledge and finding causal relationships
prior_causal_features = [('','age','0'),('','race','0'),('','sex','0'),('','native-country','0')]
prior_knowledge_matrix = create_prior_knowledge_matrix(process_dataset, prior_causal_features)

adjacency_matrix = get_adjacency_matrix(process_dataset, prior_knowledge_matrix)

# Create new adjacency matrix
process_data = train_dataset.copy()
process_data = d.one_hot_encode_data(process_data)
process_dataset = process_data.drop(["class"], axis = 1)

# Initialize the weight matrix with zeros
weights = np.zeros(len(process_dataset.columns))

# Loop through all columns in process_dataset
for i, column in enumerate(process_dataset.columns):
    # If the column is a continuous feature, the weight will be 1
    if column in d.continuous_feature_names:
        weights[i] = 1
    else:
        # For categorical features, find the original feature and the category value
        # For example, 'gender_Male' will revert to 'gender' and 'Male'
        original_feature = column.split("_")[0]
        category = column.split("_")[-1]

        # Apply the corresponding weight from mapping_tables
        if original_feature in mapping_tables:
            # Check if the category exists in the mapping_dict of the original feature
            if category in mapping_tables[original_feature]:
                weights[i] = mapping_tables[original_feature][category]
                
new_matrix_size = len(process_dataset.columns)
new_matrix = np.zeros((new_matrix_size, new_matrix_size))

new_order_cols = d.continuous_feature_names + d.categorical_feature_names
data_order = data[new_order_cols]

# Create a new DAG matrix with the new column order
dag_matrix_size = len(dataset.columns)
dag_matrix = np.zeros((dag_matrix_size, dag_matrix_size))

# Create a mapping from old column names to new column indices
column_index_mapping = {col: index for index, col in enumerate(new_order_cols)}

# Map the values from the old DAG matrix to the new matrix based on the new column order
for old_index, col in enumerate(dataset.columns):
    new_index = column_index_mapping.get(col)
    if new_index is not None:
        dag_matrix[new_index, new_index] = adjacency_matrix[old_index, old_index]

        for inner_old_index, inner_col in enumerate(dataset.columns):
            inner_new_index = column_index_mapping.get(inner_col)
            if inner_new_index is not None:
                dag_matrix[new_index, inner_new_index] = adjacency_matrix[old_index, inner_old_index]

# We need a way to map old indices to new
index_mapping = {}
new_index = 0

# Iterate through all columns in the original data
for col in data_order.columns:
    if col in d.categorical_feature_names:
        # Create new columns from one-hot encoding
        new_features = [col_name for col_name in process_dataset.columns if col_name.startswith(col + "_")]

        # Create new columns from one-hot encoding
        for i in range(len(new_features)):
            index_mapping[new_index] = data_order.columns.get_loc(col)  # ánh xạ từ chỉ mục mới sang cũ
            new_index += 1
    else:
        # For non-categorical columns, we keep the mapping the same
        index_mapping[new_index] = data_order.columns.get_loc(col)
        new_index += 1

# Create a new matrix with updated weights
for i in range(new_matrix_size):
    for j in range(new_matrix_size):
        old_i = index_mapping.get(i)  
        old_j = index_mapping.get(j)  

        if old_i is not None and old_j is not None:  # Only update when both indices exist
            new_matrix[i, j] = dag_matrix[old_i, old_j]*weights[j] 

# Prepare data for training the model
input_data = normalize_df(process_dataset).values
output_data = process_data['class'].values

x_train = input_data
y_train = output_data

train_data = MyDataset(x_train,y_train)

train_and_save_model(train_data, x_train, 'model_weights_adult.pth')
model = load_model('model_weights_adult.pth', x_train)



# Prepare data fpr DiCE
process_data = train_dataset

d = dice_ml.Data(dataframe=process_data, data_name='adult',
                 continuous_features=['age', 'hours-per-week', 'education-num'],
                 outcome_name='class')

# Pre-trained ML model
backend = {'model': 'pytorch_model.PyTorchModel',
           'explainer': 'feasible_model_approx.FeasibleModelApprox'}

# Training DiCE
m = dice_ml.Model(model=model, backend=backend)

adj_matrix = torch.tensor(new_matrix, dtype=torch.float32)

exp = dice_ml.Dice(d, m, method=None, encoded_size=30, adj_matrix = adj_matrix, lr=1e-2,
                   batch_size=4096, validity_reg=76.0, margin=0.344, epochs=25,
                   wm1=1e-2, wm2=1e-2, wm3=1e-2)

data_training = d.one_hot_encode_data(process_data)
data_training = data_training.drop(['class'], axis = 1)

exp.train(None, 87, data_training, pre_trained=0)

# Select a random set from test data
query_instance = test_dataset.drop(columns="class")[0:1]

# generate counterfactuals
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_class="opposite")
# visualize the results
dice_exp.visualize_as_dataframe(show_only_changes=False)


           







