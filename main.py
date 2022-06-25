import streamlit as st
import streamlit.components.v1 as stc
from streamlit_player import st_player
import json

#load EDA
import pandas as pd
import numpy as np


#function to load our dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def loading(data):
    df=pd.read_csv(data)
    return df

#Vectorize
def vectorize_text_to_cosine_mat(data):
    count_vect= CountVectorizer()

#Cosine similarity Matrix
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat=cosine_similarity(cv_mat)
    return cosine_sim_mat

#COSINE SIMILARITY
@st.cache
def get_recommendation(title, cosine_sim_mat, df,num_of_rec=10):

    #indices of the course
    course_indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    #index of the course
    idx = course_indices[title]

    #looking into cosine matrix for that index
    sim_score= list(enumerate(cosine_sim_mat[idx]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    selected_course_indices=[i[0] for i in sim_score[1:]]
    selected_course_scores = [i[0] for i in sim_score[1:]]
    result_df =df.iloc[selected_course_indices]
    result_df['similarity_score']=selected_course_scores
    final_rec_course= result_df[['Title','similarity_score','Link', 'Stars', 'Enrollment','Website']]

    return final_rec_course.head(num_of_rec)

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗: </span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🎓Stars: </span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students: </span>{}</p>
<p style="color:blue;"><span style="color:black;">Website: </span>{}</p>
</div>
"""



@st.cache
def search_term_if_not_found(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    rec_course=result_df[['Title','Link', 'Stars', 'Enrollment','Website']]
    return rec_course.head(num_of_rec)

RESULT_TEMP1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🎓Stars:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑Students:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Website:</span>{}</p>
</div>
"""






#PROJECTS
@st.cache
def get_recommendation_projects(title, cosine_sim_mat, df,num_of_rec=10):

    #indices of the course
    course_indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

    #index of the course
    idx = course_indices[title]

    #looking into cosine matrix for that index
    sim_score= list(enumerate(cosine_sim_mat[idx]))
    sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
    selected_course_indices=[i[0] for i in sim_score[1:]]
    selected_course_scores = [i[0] for i in sim_score[1:]]
    result_df =df.iloc[selected_course_indices]
    result_df['similarity_score']=selected_course_scores
    final_rec_course= result_df[['Title','similarity_score','Link','Website']]

    return final_rec_course.head(num_of_rec)


@st.cache
def search_term_if_not_found_project(term,df,num_of_rec=10):
    result_df=df[df['Title'].str.contains(term)]
    rec_course=result_df[['Title', 'Links','Website']]
    return rec_course.head(num_of_rec)



RESULT_TEMP_project1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">Website:</span>{}</p>
</div>
"""

RESULT_TEMP_project2 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #DDDDDD;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">Website:</span>{}</p>
</div>
"""


def get_btn_text(start_ms):
    seconds = int((start_ms / 1000) % 60)
    minutes = int((start_ms / (1000 * 60)) % 60)
    hours = int((start_ms / (1000 * 60 * 60)) % 24)
    btn_txt = ''
    if hours > 0:
        btn_txt += f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        btn_txt += f'{minutes:02d}:{seconds:02d}'
    return btn_txt


def add_btn(start_ms, key):
    start_s = start_ms / 1000
    if st.button(get_btn_text(start_ms), key):
        url_time = url + '&t=' + str(start_s) + 's'
        with placeholder.container():
            st_player(url_time, playing=True, muted=False)


if __name__ == '__main__':

    st.title("LearningBazaar: The best guide to start with")
    about=["Recommender","Simplifier"]
    choice=st.sidebar.selectbox("Cool Functionalities of this app",about)

    if choice=="Simplifier":
        st.subheader("About")
        st.text("Hey There, this is a fast tracker for youtube courses/videos""\n"
                "- It automatically split the course in chapters""\n"
                "- It generates small summary's of it""\n"
                "- It also list outs the timestamps so that anyone can fast track to that point""\n")
        demo=st.text_input("Enter your lecture url")


        url = 'https://www.youtube.com/watch?v=-DP1i2ZU9gk'

        file_highlights = 'obljugk7on-6ebd-4499-9832-fe5920ac3a89_highlights.json'
        file_chapters = 'obljugk7on-6ebd-4499-9832-fe5920ac3a89_chapters.json'


        placeholder = st.empty()
        with placeholder.container():
            st_player(url, playing=False, muted=True)

        mode = st.sidebar.selectbox("Summary Mode", ("Highlights", "Chapters"))




        if mode == "Highlights":
            with open(file_highlights, 'r') as f:
                data = json.load(f)
            results = data['results']

            cols = st.columns(3)
            n_buttons = 0
            for res_idx, res in enumerate(results):
                text = res['text']
                timestamps = res['timestamps']
                col_idx = res_idx % 3
                with cols[col_idx]:
                    st.write(text)
                    for t in timestamps:
                        start_ms = t['start']
                        add_btn(start_ms, n_buttons)
                        n_buttons += 1
        else:
            with open(file_chapters, 'r') as f:
                chapters = json.load(f)
            for chapter in chapters:
                start_ms = chapter['start']
                add_btn(start_ms, None)
                txt = chapter['summary']
                st.write(txt)


    if choice=="Recommender":
        st.subheader("About")
        st.text("Hey There, this is a mini version of policy bazaar but for:""\n"
                "- The best Educational Courses available out there""\n"
                "- The Projects to step up your learning game""\n"
                "- The Test series to sharpen up your fundamental curve""\n")
        st.text(
            "Note: These are all recommendations of the best possible website \nand has been trained on a very small dataset")

        menu = ["Courses", "Projects", "Test Series"]
        choice = st.sidebar.selectbox("What do you need us to recommend to?", menu)

        if choice == "Courses":
            st.subheader("Course Recommendation")
            st.text("Type any of your linked courses from the internet and we'll try to recommend you similar courses"
                    "\n or \n"
                    "just type the domain name and we'll try to recommend the best possible resource")
            userinput = st.text_input("Search")

            # ALL DATA
            st.subheader("All Courses")
            df = loading("Data/All2.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'].values.astype('U'))
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if userinput is not None:
                    try:
                        results = get_recommendation(userinput, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_star = row[1][3]
                            rec_rating = row[1][4]
                            rec_website = row[1][5]

                            stc.html(
                                RESULT_TEMP.format(rec_title, rec_score, rec_link, rec_star, rec_rating, rec_website),
                                height=250)

                    except:
                        results = "Hmm seems like you are searching through domains"
                        st.warning(results)
                        st.info("Here's our recommendation for the same :)")

                        result_df = search_term_if_not_found(userinput, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            rec_star = row[1][2]
                            rec_rating = row[1][3]
                            rec_website = row[1][4]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP1.format(rec_title, rec_link, rec_star, rec_rating, rec_website),
                                     height=250)
        elif choice == "Projects":
            st.subheader("Project Recommendations")
            st.text("Which domain you wanna work on?")
            search_term = st.text_input("Search")

            df = loading("Data/Projectsdata.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_website = row[1][3]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_website),
                                     height=250)
                    except:
                        results = "Yaay!, you finally decided to level up your game. Here are the best project recommendations for the same"
                        st.warning(results)

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            rec_website = row[1][2]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link, rec_website),
                                     height=150)


        elif choice == "Test Series":
            st.subheader("Tests Recommendations")
            st.text("Which topic you wanna test yourself on?")
            search_term = st.text_input("Search")

            df = loading("Data/TestData.csv")
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
            num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
            if st.button("Recommend"):
                if search_term is not None:
                    try:
                        results = get_recommendation_projects(search_term, cosine_sim_mat, df, num_of_rec)
                        for row in results.iterrows():
                            rec_title = row[1][0]
                            rec_score = row[1][1]
                            rec_link = row[1][2]
                            rec_website = row[1][3]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project2.format(rec_title, rec_score, rec_link, rec_website),
                                     height=250)
                    except:
                        results = "Yaay!, Here are the best test recommendations for the same"
                        st.warning(results)

                        result_df = search_term_if_not_found_project(search_term, df, num_of_rec)
                        # st.dataframe(result_df)
                        for row in result_df.iterrows():
                            rec_title = row[1][0]
                            rec_link = row[1][1]
                            rec_website = row[1][2]
                            # st.write("Title",rec_title)
                            stc.html(RESULT_TEMP_project1.format(rec_title, rec_link, rec_website),
                                     height=150)





else:
        st.text("Hello, idk why this page is made :/")



st.text("Developed by Siddhivinayak Dubey under the mentorship of Proff. Arjun Arora. ")