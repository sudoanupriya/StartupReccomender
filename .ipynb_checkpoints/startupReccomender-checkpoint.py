import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data into dataframes
root_dir = '/Users/rentala/Desktop/295/reccModel/data'
investor_data = pd.read_csv(root_dir+'/investors.csv')
org_data = pd.read_csv(root_dir+'/organizations.csv')
#org_data = pd.read_csv('/kaggle/input/traindata/organizations.csv').sample(frac=0.2)
investment_data = pd.read_csv(root_dir+'/investments.csv')


#DATA PREPROCESSING
def clean_text(text):
    return text.strip().lower()

# Drop irrelevant columns and rename columns for consistency
investor_data = investor_data[['uuid']]
investor_data = investor_data.rename(columns={'uuid': 'investor_uuid'})

#print(investor_data.columns.tolist())

org_data = org_data[['uuid', 'name', 'short_description']] #, 'category_list'
org_data = org_data.dropna(subset=['short_description']) #category_list
org_data['short_description'] = org_data['short_description'].apply(clean_text)
org_data = org_data.rename(columns={'uuid': 'org_uuid'})
org_data = org_data.rename(columns={'name': 'org_name'})

#print(org_data.columns.tolist())


investment_data = investment_data[['uuid', 'name', 'funding_round_uuid', 'funding_round_name', 'investor_uuid', 'investor_name']] #removing this 'org_uuid', 'organisation_name'
investment_data = investment_data.rename(columns={'uuid': 'investment_uuid'})
investment_data['org_name'] = investment_data['name'].apply(lambda x: x.split('-')[-1].strip())
investment_data = investment_data.drop('name', axis=1)

print(investment_data.columns.tolist())

#DATA MERGE AND DUPLICATE REMOVAL??

# Merge dataframes to create investor-organization relationship matrix
matrix = pd.merge(investment_data, investor_data, on='investor_uuid', how='left')
matrix = pd.merge(matrix, org_data, on='org_name', how='left')


# Drop duplicates and null values
matrix = matrix.drop_duplicates(subset=['investor_uuid', 'org_uuid'])
matrix = matrix.dropna(subset=['short_description']) #category_list
#print(matrix.shape)
#print(matrix.columns.tolist())
#print(matrix[['investor_name', 'org_name']])

def recommend_orgs(investor_id):
    # get all organizations not invested by the investor
    
    #investor_org_uuids
    invested_orgs = matrix[matrix['investor_uuid'] == investor_id]['org_uuid'].unique()
    all_orgs = org_data['org_uuid'].unique()
    print("invested_orgs: ", str(invested_orgs))
    non_invested_orgs_uuids = set(all_orgs) - set(invested_orgs)
    non_invested_orgs_matrix = org_data[org_data['org_uuid'].isin(non_invested_orgs_uuids)]

    #-------
    # vectorize data
    vectorizer = TfidfVectorizer(stop_words='english')
    org_features = vectorizer.fit_transform(non_invested_orgs_matrix['short_description'])
    #print(org_features)
    
    # calculate similarity scores
    cosine_sim = cosine_similarity(org_features)
    #print(cosine_sim)
    
    #------
    # get indices of investor's invested organizations in matrix
    #invested_indices = matrix[matrix['permalink'].isin(invested_orgs)].index

    # calculate similarity scores between investor's investments and all non invested organizations
    org_indices = [matrix[matrix['org_uuid'] == org_id].index[0] for org_id in invested_orgs]
    org_scores = cosine_sim[org_indices].sum(axis=0)
    #print("org_scores: "+str(org_scores))
    
    top_n_recc = min(20, len(org_scores))
    top_org_indices = org_scores.argsort()[::-1][:top_n_recc]
    top_orgs = non_invested_orgs_matrix.iloc[top_org_indices][['org_uuid', 'org_name']].drop_duplicates()
    
    return top_orgs

investor_uuids = ['5ff6c126-4aad-4d3e-89d3-6a14c442d588', '466da2c7-65a9-3144-ae57-00362b05f3b1', '2a936a56-b1fc-d684-81c8-cb8100c67f39', '68be9953-7439-1a89-36bc-0e98bf217ee2', '19802464-9761-158b-36ec-7ef0f0bf1a8a']
for investor_id in investor_uuids:
    top_orgs = recommend_orgs(investor_id)
    print(f"\ntop reccomendations for {investor_id} are \n {top_orgs}",end="\n****************************\n\n")
