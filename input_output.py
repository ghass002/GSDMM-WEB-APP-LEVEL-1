#IMPORT LIBRARIES
import streamlit as st 
import pandas as pd  
import nltk
from HanTa import HanoverTagger as ht

#INTILIZE TAGGER
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

def get_input():
    """
    function get input from user by uploading multiple csv files

    Returns:
    df3: DataFrame Composed of all dataframes from the csv files uploaded 
    """

    multiple_files = st.file_uploader("Choose multiple CSV files to upload", type="csv", accept_multiple_files=True)


    if len(multiple_files) > 0:
        st.success('Files successfuly uploaded!!!')
        df_list = []

        for file in multiple_files:
            dataframe = pd.read_csv(file, sep=';', error_bad_lines=False)
            dataframe['MyIdx'] = dataframe.index
            dataframe.sort_values(by=['Conversation Id','MyIdx'], inplace=True)
            file.seek(0)
            df_list.append(dataframe)
        
        if df_list != []:
            df3 = pd.concat(df_list, ignore_index=True)            
            return df3
    
    else:
        st.info('Kindly upload a csv file')
        return pd.DataFrame()




def extract_information(df):
    """
    function to extract information from the chatbot dataframe and generate the report dataframe 
    
    Returns:
    export_df: DataFrame: Unfinished Report Dataframe
    """

    AI_stopwords = ['KI','Künstlicher','künsticher','Künstliche','künstiche','Intelligenz','AI','ki','kI','Ai','aI','Artifizielle','artifizielle','intelligenz']
    
    #DataFrame columns Initialization
    conv_id = []
    time = []
    document_type = []
    vision_keywords_questions = []
    vision_attitude_questions = []
    vision_keywords = []
    document_what_question = []
    document_why_question = []
    document_whom_question = []
    document_effect_question = []
    document_title_question = []
    vision_attitude = []
    document_what = []
    document_why = []
    document_whom = []
    document_effect = []
    document_title = []
    email = []
    document_title_keywords = []    

  

    #Extracting the information from chatbot dataframe and store it in the corresponding list
    for i in range(len(df)):        

        #Case 1: PROBLEM
        if df['Message'][i] in ["'@problem",'Problem','@problem','problem']:                        
            
            #Extracting document_what response
            document_what_question.append("Welches Problem sehen Sie, das bei der Nutzung Künstlicher Intelligenz auftreten könnte? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+1] == df['Conversation Id'][i]:
                  document_what.append(df['Message'][i+1])
                else:
                  document_what.append('')
            except:
                document_what.append('')

            #Extracting document_why response
            document_why_question.append("Wieso ist das Problem aus Ihrer Sicht wichtig?  Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+2] == df['Conversation Id'][i]:
                  document_why.append(df['Message'][i+2])
                else:
                  document_why.append('')
            except:
                document_why.append('')

            #Extracting document_whom response
            document_whom_question.append("Wen könnte das Problem besonders betreffen (z.B. bestimmte Berufs- oder Gesellschaftsgruppen, etc.)? Erklären Sie bitte in 1-2 Sätzen.")
            try:            
                if df['Conversation Id'][i+3] == df['Conversation Id'][i]:
                  document_whom.append(df['Message'][i+3])
                else:
                  document_whom.append('')
            except:
                document_whom.append('')

            #Extracting document_effect response
            document_effect_question.append("Welche gesellschaftliche Folgen kann es haben, wenn dieses Problem nicht gelöst wird? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+4] == df['Conversation Id'][i]:
                  document_effect.append(df['Message'][i+4])
                else:
                  document_effect.append('')
            except:
                document_effect.append('')

            #Extracting document_title response  
            document_title_question.append("Geben Sie Ihrem Problem eine kurze Überschrift (z.B. ein Stichwort oder das Hauptthema des Problems). Nutzen Sie max. 30 Zeichen.")
            try:
                if df['Conversation Id'][i+5] == df['Conversation Id'][i]:
                  document_title.append(df['Message'][i+5])
                else:
                  document_title.append('')
            except:
                document_title.append('')

            #Extracting Emails
            try:
                if df['Conversation Id'][i+6] == df['Conversation Id'][i]:                  
                  message = str(df['Message'][i+6])
                  mess = str(df['Message'][i+7])
                  if message.lower() == 'ja' and mess.lower() == 'ja':
                    if df['Conversation Id'][i+8] == df['Conversation Id'][i]:
                        email.append(df['Message'][i+8])
                    else:
                        email.append('')
                  else:
                    email.append('')
                else:
                  email.append('')
            except:
                email.append('')
            
            #Extracting Conversation IDs, time and filling document_type and non relevant lists such us vision keywords 
            conv_id.append(df['Conversation Id'][i])
            time.append(df['Received At'][i])                        
            document_type.append(2)
            vision_keywords_questions.append('')
            vision_attitude_questions.append('')
            vision_keywords.append('')
            vision_attitude.append('')
            

            
        #Case 2: Frage
        if df['Message'][i] in ['Frage','@frage','frage',"'@frage"]:
            
            #Extracting document_what response
            document_what_question.append("Was möchten Sie über künstliche Intelligenz von den WissenschaftlerIinnen wissen? Erklären Sie bitte in 1-2 Sätzen.")
            try:            
                if df['Conversation Id'][i+1] == df['Conversation Id'][i]:
                  document_what.append(df['Message'][i+1])
                else:
                  document_what.append('')
            except:
                document_what.append('')

            #Extracting document_why response
            document_why_question.append("Wieso ist diese Frage wichtig für Sie? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+2] == df['Conversation Id'][i]:
                  document_why.append(df['Message'][i+2])
                else:
                  document_why.append('') 
            except:
                document_why.append('')

            #Extracting document_whom response
            document_whom_question.append("Für wen könnte die Beantwortung dieser Frage wichtig sein (z.B. bestimmte Berufsgruppen, Gesellschaftsgruppen etc.)? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+3] == df['Conversation Id'][i]:
                  document_whom.append(df['Message'][i+3])
                else:
                  document_whom.append('')
            except:
                document_whom.append('')

            #Extracting document_effect response
            document_effect_question.append("Könnte die Beantwortung dieser Frage gesellschaftliche Folgen haben?  Wenn ja - welche? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+4] == df['Conversation Id'][i]:
                  document_effect.append(df['Message'][i+4])
                else:
                  document_effect.append('')
            except:
                document_effect.append('')

            #Extracting document_title response 
            document_title_question.append("Geben Sie Ihrer Frage eine kurze Überschrift (z.B. ein Stichwort oder das Hauptthema der Frage).")
            try:
                if df['Conversation Id'][i+5] == df['Conversation Id'][i]:
                  document_title.append(df['Message'][i+5])
                else:
                  document_title.append('')
            except:
                document_title.append('')

            #Extracting Emails
            try:
                if df['Conversation Id'][i+6] == df['Conversation Id'][i]:
                  message = str(df['Message'][i+6])
                  mess = str(df['Message'][i+7])
                  if message.lower() == 'ja' and mess.lower() == 'ja':
                    if df['Conversation Id'][i+8] == df['Conversation Id'][i]:
                        email.append(df['Message'][i+8])
                    else:
                        email.append('')
                  else:
                    email.append('')
                else:
                  email.append('')            
            except:
                email.append('')

            #Extracting Conversation IDs, time and filling document_type and non relevant lists such us vision keywords 
            conv_id.append(df['Conversation Id'][i])
            time.append(df['Received At'][i])                      
            document_type.append(1)
            vision_keywords_questions.append('')
            vision_attitude_questions.append('')
            vision_keywords.append('')
            vision_attitude.append('')
            
            
            
        #Case 3: VISION 
        if df['Message'][i] in ['Vision','@vision','vision',"'@vision"]:

            #Exracting vision keywords, attitude
            vision_keywords_questions.append("Welche 3 Schlagworte fallen Ihnen zum Thema Künstliche Intelligenz ein?")
            try:
                if df['Conversation Id'][i+1] == df['Conversation Id'][i]:
                  vision_keywords.append(df['Message'][i+1])
                else:
                  vision_keywords.append('')
            except:
                vision_keywords.append('')
            vision_attitude_questions.append("Glauben Sie, dass die zukünftigen gesellschaftlichen Auswirkungen eher positiv oder eher negativ wären?")
            try:
                if df['Conversation Id'][i+2] == df['Conversation Id'][i]:
                  vision_attitude.append(df['Message'][i+2])
                else:
                  vision_attitude.append('')
            except:
                vision_attitude.append('')
            
            #Extracting document_what response
            document_what_question.append("Wie stellen Sie sich konkret eine Nutzung von Künstlicher Intelligenz in der Zukunft vor? Erklären Sie in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+3] == df['Conversation Id'][i]:
                  document_what.append(df['Message'][i+3])
                else:
                  document_what.append('')
            except:
                document_what.append('')

            #Extracting document_why response
            document_why_question.append("Warum wäre das eine wichtige Entwicklung aus Ihrer Sicht? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+4] == df['Conversation Id'][i]:
                  document_why.append(df['Message'][i+4])
                else:
                  document_why.append('')
            except:
                document_why.append('')

            #Extracting document_whom response
            document_whom_question.append("Wer wird von dieser Entwicklung der Nutzung Künstlicher Intelligenz betroffen sein (z.B. bestimmte Berufs- oder Gesellschaftsgruppen, etc.)? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+5] == df['Conversation Id'][i]:
                    document_whom.append(df['Message'][i+5])
                else:
                    document_whom.append('')
            except:
                document_whom.append('')

            #Extracting document_effect response
            document_effect_question.append("Welche gesellschaftliche Folgen dieser künftigen Nutzung Künstlicher Intelligenz kann es geben? Erklären Sie bitte in 1-2 Sätzen.")
            try:
                if df['Conversation Id'][i+6] == df['Conversation Id'][i]:
                    document_effect.append(df['Message'][i+6])
                else:
                    document_effect.append('')
            except:
                document_effect.append('')

            #Extracting document_title response
            document_title_question.append("Geben Sie Ihren Beitrag eine Überschrift (z.B. ein Stichwort oder das Hauptthema des Problems). Nutzen Sie max. 30 Zeichen.")
            try:
                if df['Conversation Id'][i+7] == df['Conversation Id'][i]:
                    document_title.append(df['Message'][i+7])
                else:
                    document_title.append('')
            except:
                document_title.append('')

            #Extracting Email
            try:
                if df['Conversation Id'][i+8] == df['Conversation Id'][i]:
                    message = str(df['Message'][i+8])
                    mess = str(df['Message'][i+9])
                    if message.lower() == 'ja' and mess.lower() == 'ja':
                        if df['Conversation Id'][i+10] == df['Conversation Id'][i]:
                            email.append(df['Message'][i+10])
                        else:
                            email.append('')
                    else:
                        email.append('')
                else:
                    email.append('') 
            except:
                email.append('')

            #Extracting Conversation IDs, time and filling document_type
            conv_id.append(df['Conversation Id'][i])
            time.append(df['Received At'][i]) 
            document_type.append(3)

        
    # Counting the conversation IDS and creating the format conv ID - 1,2,3.. 
    conv_dict = {i:conv_id.count(i) for i in conv_id}
    conv_id = []
    for ids in conv_dict:
        for i in range(1,conv_dict[ids]+1):
            conversation_id = str(ids) + '-' + str(i)
            conv_id.append(conversation_id)

    # Generating the document title keywords (ADJ or nouns)
    for title in document_title:
      doc = nltk.tokenize.word_tokenize(title,language='german')
      tags = tagger.tag_sent(doc)
      token_list = []
      title_list = []
      for (word,lemma,pos) in tags:
        title_list.append(lemma)       
        if pos in ['ADJA','NN','NA','NE'] and word not in AI_stopwords:     
          token_list.append(lemma)
      if token_list != []:
        document_title_keywords.append(token_list)
      else:
        document_title_keywords.append(title_list)


    # Creating the dictionary to convert it then to a dataframe with all the extracted information
    export_dict = {'Received At': time,
                'Conversation Id': conv_id,
                'document number': [i for i in range(1,len(conv_id)+1)],
                'document_type: 1 = question, 2 = problem, 3 = vision': document_type,
                'vision_keywords_question' : vision_keywords_questions,
                'vision_keywords_response' : vision_keywords,
                'vision_attitude_question' : vision_attitude_questions,
                'vision_attitude_response': vision_attitude,
                'document_what_question': document_what_question,
                'document_what_response': document_what,
                'document_why_question': document_why_question,
                'document_why_response': document_why,
                'document_for_whom_question': document_whom_question,
                'document_for_whom_response': document_whom,
                'document_effect_question': document_effect_question,
                'document_effect_response': document_effect,
                'document_title_question': document_title_question,
                'document_title_response': document_title,
                'dcoument_title_keywords': document_title_keywords,
                'submitter_email': email 
                }    
    
    export_df = pd.DataFrame.from_dict(export_dict)

    export_df['Text'] = export_df['document_title_response'] + '. ' + export_df['document_what_response'] + '. ' + export_df['document_why_response'] + '. ' + export_df['document_for_whom_response'] + '. ' + export_df['document_effect_response']   


    return export_df


def select_text_feature(df) -> pd.DataFrame :
    """
    this function selects the text feature from the uploaded csv file
    Parameters
    ----------
    df: A pandas Dataframe 

    Returns:
    df: Dataframe containing only the text columns
    text_col: the name of text column
    export_df: a copy of the dataframe extracted from the csv file
    """
    export_df = df.copy()
    text_col = st.selectbox('plese select the column that contains the text data named Text',(list(df.columns)))
    df = pd.DataFrame(df[text_col])
    
    return df, text_col, export_df


    
