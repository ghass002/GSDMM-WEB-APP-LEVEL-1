import pandas as pd
import streamlit as st 
import SessionState
import input_output as io
import preprocessor as pp
import model_evaluator as mv
import gsdm
import base64


def get_download_link(df):
    """
    function to generate download link to a dataframe

    Parameters
    ----------
    df: DataFrame 
    """
    csv = df.to_csv(index=False,sep=';')
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="Report.csv"> Download the report file </a>' 
    return href


def display_app_header(main_txt,is_sidebar = False):
    """
    function to display major headers at user interface

    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#054029  ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
   
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def display_side_panel_header(txt):
    """
    function to display minor headers at side panel

    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')

def display_header(header):
    """
    function to display minor headers at user interface main pannel 

    Parameters
    ----------
    header: str -> the major text to be displayed
    """
     
    #view clean data
    html_temp = f"""
    <div style = "background.color:#054029; padding:10px">
    <h4 style = "color:white;text_align:center;"> {header} </h5>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

def space_header():
    """
    function to create space using html 

    Parameters
    ----------
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Design of the side panel
display_app_header(main_txt='GSDMM Topic Modelling',is_sidebar=True)
display_side_panel_header(txt='Steps:')
display_side_panel_header(txt='choose the parameters:')


# GSDMM Parameters Initialization
n_of_topics = st.sidebar.number_input('Expected Number of Topics',min_value=1,max_value=2000,value=5,step =1)
n_of_words = st.sidebar.number_input('Number of words per topic',min_value=1 ,max_value=30,value=5,step =1)
alpha= st.sidebar.slider('Alpha',min_value=0.01,max_value=0.5,value=0.05,step=0.01)
beta= st.sidebar.slider('Beta',min_value=0.01,max_value=0.5,value=0.15,step=0.01)
iterations=st.sidebar.number_input('Number of Iteration',min_value=30,max_value=5000,value=60,step =1)

#Session State 
ss = SessionState.get(output_df = pd.DataFrame(), 
_model=None,
text_col='Text',
is_file_uploaded=False,
to_clean_data = False,
to_encode = False,
to_train = False,
to_evaluate = False,
to_visualize = False,
to_download = False,
df = pd.DataFrame(),
clean_text = None,
topics_df = None)
st.cache()


def check_input():
    """
    function to generate a single dataframe from all csv files uploaded and check if the dataframe is not empty
    Returns:
    df: DataFrame
    """

    df= io.get_input()
    if df.shape[0]>0:
        ss.is_file_uploaded = True
    
    return df
    

############ APP Logical Flow ###############

space_header()
# Upload Step
ss.df = check_input()
if ss.is_file_uploaded:
    ss.df = io.extract_information(ss.df)
    ss.df, ss.text_col, export_df = io.select_text_feature(ss.df)       
    if ss.df[ss.text_col].dtype =='O':        
        ss.to_clean_data = True                      
    else:
        st.warning('plese select the column that contains the text data')
        ss.to_clean_data = False
        

# Cleaning data step  
if  ss.to_clean_data:
    display_header(header = 'Cleaning Data')  
    space_header()    
    st.success('Data cleaning successfuly done')
    ss.to_encode = True

#Encoding Step 
if ss.to_encode:
    display_header(header = 'Features Extraction') 
    space_header()    
    ss.clean_text, deleted_index  =pp.extract_features_de(ss.df,feature=ss.text_col)                
    st.success('Features Extraction Successfully done')
    ss.to_train = True
     
     
################### Training STEP ###########################

if ss.to_train:
    display_header(header = 'Model Training Section')
    space_header() 
    button_train = st.button('Train Model')
    if button_train:
        ss._model = gsdm.gsdmm_train(ss.clean_text,alpha,beta,iterations,number_of_topics=n_of_topics)       
        st.success('Training completed!!!')
        ss.to_evaluate = True
    

################### Model Evaluation  STEP ###########################
if ss.to_evaluate:
    display_header(header = 'Model Evaluation Section')  
    space_header()
    button_eva = st.button('Evaluate Model')

    if button_eva:               
        #Extracting the doc frequency dictionary
        ss.topics_df, dictionary = mv.get_model_results_gsdmm(export_df, deleted_index, n_of_words,texts = ss.clean_text,mgp=ss._model)
        #Generate Labels
        labels_count, topics_document_list = mv.generate_labels_de(dictionary,deleted_index,ss.topics_df, n_of_topics)
        #Generate Summaries
        ss.topics_df = mv.generate_summary_de(ss.topics_df, deleted_index)        
        # drop unnecessary column from final dataframe
        ss.topics_df.drop('Text',axis=1, inplace=True)
        ss.topics_df.drop('Topic number',axis=1, inplace=True)        
        # Delete column that has document what response less than 5 characters or document_title response less than 3
        for i in range(len(ss.topics_df)):
            
            if len(ss.topics_df['document_what_response'][i]) < 5 or len(ss.topics_df['document_title_response'][i]) < 3:
              ss.topics_df.drop(i,axis=0, inplace=True)                   
       
        ss.to_download = True

# Download Step          
if ss.to_download:
    display_header(header = 'Download Section')
    space_header()
    download_button= st.button('Generate download link')
    if download_button:
      st.success('please click the link below to download the report file')
      st.markdown(get_download_link(ss.topics_df), unsafe_allow_html=True)

       
      
