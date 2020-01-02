import time
t0 = time.time()

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

print(f"Imports took {time.time() - t0}sec")

class JobRec():
    # Initialize the model (TF-IDF)
    def __init__(self):
        df = pd.read_csv('jobs_data.csv')

        # Preprocess dataframe
        df_dropped = self.preprocess_df(df)
        
        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')

        # Re-read the dropped dataframe, because... reasons
        df = df_dropped.fillna('')

        # Replace NaN with an empty string
        df['title'] = df['title'].fillna('')

        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix_title = tfidf.fit_transform(df['title'])

        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix_title, tfidf_matrix_title)

        # Reset index
        df = df.reset_index()

        # Construct a reverse map of indices and movie titles
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        self.df_dropped = df_dropped
        self.indices = indices
        self.cosine_sim = cosine_sim
        self.titles = df.title.drop_duplicates().dropna()
        self.df = df


    # Preprocess the jobFunction into 3 separate columns 
    def preprocess_jf(self, df):
        vals_jf = []

        # Convert each value in the jobFunction to a 'literal string'
        # and append it to the vals_jf array
        for val in df["jobFunction"]:
            x = ast.literal_eval(val)
            vals_jf.append(x)

        # Convert the vals array into a dataframe
        jf_df = pd.DataFrame(data=vals_jf)
        jf_df.columns = ['jf_1', 'jf_2', 'jf_3']

        # Drop the 'jobFunction' column
        df_no_jf = df.drop(columns='jobFunction') 

        # concatinate the two DFs
        df_new = pd.concat([df_no_jf, jf_df], axis=1)
        df_new.sort_values(by='title')
        
        return df_new


    # Preprocess the industry into 3 separate columns 
    def preprocess_ind(self, df):
        vals_ind = []

        for val in df["industry"]:
            x = ast.literal_eval(val)
            vals_ind.append(x)

        ind_df = pd.DataFrame(data=vals_ind)
        
        ind_df.columns = ['ind_1', 'ind_2', 'ind_3']
        df_no_ind = df.drop(columns='industry') 
        df_final = pd.concat([df_no_ind, ind_df], axis=1)
        
        return df_final


    # Preprocess the dataframe
    def preprocess_df(self, df):
        # Invoke the preprocess_ind and preprocess_ind methods
        df_pp = self.preprocess_jf(df)
        df_final = self.preprocess_ind(df_pp)

        # Drop the duplicates and fill the NaN and None values with empty strings
        df_final_drop = df_final.drop_duplicates()
        df2 = df_final_drop.fillna('')
        
        return df2


    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(self, df, title):
        # # Get indices and cosine similarity
        # df_dropped, indices, cosine_sim = self.init_title(df)
        
        # Get the index of the job that matches the title
        idx = self.indices[title]

        # Get the pairwsie similarity scores of all jobs with that job
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the jobs based on the similarity scores
        sim_scores = sorted(sim_scores, reverse=True)

        # Get the scores of the 10 most similar jobs
        sim_scores = sim_scores[0:10]

        # Get the jobs indices
        rec_jobs_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar job functions and industries
        recommendations = self.df_dropped.iloc[rec_jobs_indices, :]
        
        return recommendations


    # Get recommendations in form of a dictionary
    def get_recs(self, df, title):
        # Initialize variables
        rec_dict = {}   # recommendations dictionary
        
        # Get recommendations and drop duplicates
        recommendations = self.get_recommendations(df, title)
        recommendations = recommendations.drop_duplicates()
        
        # Append all job titles to the 'titles' section in the recommendations dictionary    
        recommended_job_titles = [recommendations['title'].drop_duplicates().tolist()]

        # Append the input title to the recommended_job_titles and convert them to an array
        title_list = sum(recommended_job_titles, [title])

        # Get rid of empty values
        title_list = list(filter(None, title_list))

        # Append the title_list to the titles section in the dictionary
        rec_dict['titles'] = title_list
        
        # Append all job functions to the 'functions' section in the recommendations dictionary
        recommended_job_functions = [recommendations[fun].drop_duplicates().tolist() for fun in ['jf_1', 'jf_2', 'jf_3']] 

        fun_list = sum(recommended_job_functions, [])
        fun_list = list(set(filter(None, fun_list)))    # filter out empty and duplicated enteries 
        rec_dict['functions'] = fun_list
        
        
        # Append all job industries to the 'industries' section in the recommendations dictionary
        recommended_job_industries = [recommendations[ind].drop_duplicates().tolist() for ind in ['ind_1', 'ind_2', 'ind_3']] 
        ind_list = sum(recommended_job_industries, [])
        ind_list = list(set(filter(None, ind_list)))
        rec_dict['industries'] = ind_list
        
        # Return the recommendations dictionary
        return rec_dict


    # Main function
    def recommend(self, substring):
        recommendations_list = []

        # Autocomplete - if the user enters a string, the system predicts the rest of the title automatically
        # I chose the first three titles only because othewise there will
        suggested_titles = [title for title in self.titles if title.lower().find(substring.lower()) != -1][:5]
        print(f"SUGGESTED TITLES: {suggested_titles}")

        # Get recommendations for each of the suggested, auto-completed titles
        for suggested_title in suggested_titles:
            recommendation_dict = self.get_recs(self.df, suggested_title)
            recommendations_list.append(recommendation_dict)
            print(f"RECOMMENDATIONS FOR {suggested_title}:\n {recommendation_dict}")
            print("\n--------------------------------\n")
        
        return suggested_titles, recommendations_list


if __name__ == '__main__':
    t0 = time.time()
    JR = JobRec()

    JR.recommend(substring='data')

    task_time = time.time() - t0
    print(f"This task took {task_time:.3f}sec")