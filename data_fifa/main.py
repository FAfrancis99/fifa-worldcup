import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
#from google.colab import drive
import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')
matches = pd.read_csv('matches.csv')
teams = pd.read_csv('teams.csv')




# Feature Engineering and Preprocessing without computing outcome
def preprocess_data(matches):
    # Label encode the team names (home and away)
    all_teams = pd.concat([matches['home_team_name'], matches['away_team_name']]).unique()
    le = LabelEncoder()
    le.fit(all_teams)

    # Encode home and away team names
    matches['home_team_encoded'] = le.transform(matches['home_team_name'])
    matches['away_team_encoded'] = le.transform(matches['away_team_name'])

    # Calculate recent form based on past match results
    matches['home_team_form'] = matches['home_team_score_margin']  # Margin of victory in previous matches
    matches['away_team_form'] = matches['away_team_score_margin']  # Margin of victory in previous matches

    # Goals scored and conceded from the dataset
    matches['home_team_goals'] = matches['home_team_score']  # Known before the match
    matches['away_team_goals'] = matches['away_team_score']  # Known before the match

    return matches

# Apply the preprocessing function
matches = preprocess_data(matches)


def derive_match_outcome(row):
    if row['home_team_win'] == 1:
        return 1  # Home win
    elif row['away_team_win'] == 1:
        return 2  # Away win
    else:
        return 0  # Draw

matches['outcome'] = matches.apply(derive_match_outcome, axis=1)


numeric_cols = matches[['home_team_encoded', 'away_team_encoded', 'home_team_form', 'away_team_form',
                        'home_team_goals', 'away_team_goals']]

plt.figure(figsize=(6, 4))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.show()




# Select features for model training
X = matches[['home_team_encoded', 'away_team_encoded', 'home_team_form', 'away_team_form',
             'home_team_goals', 'away_team_goals']]
y = matches['outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Hyperparameter Tuning using RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15],  # Limit depth to prevent overfitting
    'min_samples_split': [5, 10, 20],  # Increase minimum samples required to split a node
    'min_samples_leaf': [4, 6, 8],  # Increase minimum samples required at leaf nodes
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                   n_iter=20, cv=3, random_state=42, n_jobs=-1, verbose=2)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters from random search
print(f"Best Parameters: {random_search.best_params_}")

# Use the best model from random search
best_rf = random_search.best_estimator_




# Predictions and evaluation
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Random Forest Classifier Metrics after Tuning:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}')



# Feature Importance Visualization
importances = best_rf.feature_importances_
feature_names = ['home_team_encoded', 'away_team_encoded', 'home_team_form', 'away_team_form',
                 'home_team_goals', 'away_team_goals']

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance from Random Forest after Tuning')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()




from sklearn.preprocessing import StandardScaler

# Scaling the performance-related features
X_clustering = matches[['home_team_form', 'away_team_form', 'home_team_goals', 'away_team_goals']]

# Initialize the scaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X_clustering)


# Apply PCA to reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Applying K-Means Clustering to the reduced PCA components
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_pca)

# Visualizing the clusters with PCA-reduced features
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_clusters, palette='viridis')
plt.title('K-Means Clustering with PCA-Reduced Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# Add description for each K-Means cluster
matches['cluster'] = kmeans_clusters
# Exclude non-numeric columns from mean calculation
numeric_columns = matches.select_dtypes(include=[np.number]).columns

# Now, perform the groupby and mean operation only on numeric columns
cluster_summary = matches.groupby('cluster')[numeric_columns].mean()[['home_team_goals', 'away_team_goals', 'home_team_form', 'away_team_form']]
print("Cluster Summary (Average Performance Metrics):")
print(cluster_summary)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=matches['home_team_goals'], y=matches['away_team_goals'], hue=matches['cluster'])
plt.title('K-Means Clustering of Teams Based on Goals Scored/Conceded')
plt.xlabel('Home Team Goals')
plt.ylabel('Away Team Goals')
plt.show()
# Create a directed graph based on match outcomes
G = nx.DiGraph()

# Add matches as directed edges in the graph with weights based on match outcome
for index, row in matches.iterrows():
    G.add_edge(row['home_team_name'], row['away_team_name'], weight=row['outcome'])

# Apply PageRank algorithm to rank teams
pagerank = nx.pagerank(G)
pagerank_df = pd.DataFrame(list(pagerank.items()), columns=['Team', 'PageRank']).sort_values(by='PageRank', ascending=False)

# Visualizing the PageRank of teams
plt.figure(figsize=(10,18))
sns.barplot(x='PageRank', y='Team', data=pagerank_df)
plt.title('Team Rankings by PageRank')
plt.xlabel('PageRank')
plt.ylabel('Teams')
plt.show()



# Analyze and explain the PageRank results
print("Top 10 Teams by PageRank:")
print(pagerank_df.head(10))  # Show top 10 teams by PageRank

plt.figure(figsize=(10, 6))
sns.barplot(x='PageRank', y='Team', data=pagerank_df.head(10))
plt.title('Top 10 Teams by PageRank')
plt.xlabel('PageRank')
plt.ylabel('Teams')
plt.show()


# Apply community detection
communities = nx.community.greedy_modularity_communities(G)

# Visualizing the team communities
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)

# Assign a color to each community
color_map = []
for node in G:
    for i, community in enumerate(communities):
        if node in community:
            color_map.append(i)

nx.draw_networkx_nodes(G, pos, node_size=50, node_color=color_map, cmap=plt.get_cmap('rainbow'))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Team Communities Based on Match Outcomes")
plt.show()

# Assign teams to communities
community_dict = {}
for i, community in enumerate(communities):
    for team in community:
        community_dict[team] = i
matches['community'] = matches['home_team_name'].map(community_dict)
# Group by community and summarize performance
community_summary = matches.groupby('community')[numeric_columns].mean()[['home_team_goals', 'away_team_goals', 'home_team_form', 'away_team_form']]
print("Community Summary (Average Performance Metrics):")
print(community_summary)

# Visualize community sizes or characteristics
community_sizes = pd.Series(community_dict.values()).value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=community_sizes.index, y=community_sizes.values)
plt.title('Community Sizes')
plt.xlabel('Community')
plt.ylabel('Number of Teams')
plt.show()