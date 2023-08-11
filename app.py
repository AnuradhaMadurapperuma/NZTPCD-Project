import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from lifelines import KaplanMeierFitter
import plotly.graph_objs as go
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import PyPDF2
from PyPDF2 import PdfMerger
from lxml import etree
from lifelines.plotting import plot_lifetimes
from streamlit.components.v1 import components
from matplotlib.ticker import LinearLocator
from pathlib import Path


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, SimpleDocTemplate, Table, TableStyle
import matplotlib.pyplot as plt
import numpy as np
import io
import subprocess
import os

import streamlit as st
from reportlab.lib.pagesizes import letter, inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, PageTemplate, Frame
from reportlab.lib.styles import ParagraphStyle


from reportlab.platypus import SimpleDocTemplate, Paragraph, KeepTogether, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet


from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


import base64
from io import BytesIO

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "styles.css"
#demo_report = current_dir / "assets" / "report_demographic_details.pdf"
#lab_report = current_dir / "assets" / "report_lab_details.pdf"
#demo_all_comp_report = current_dir / "assets" / "report_demographic_details_all_comp.pdf"
#lab_all_comp_report = current_dir / "assets" / "report_lab_details_all_comp.pdf"
#cover_image = current_dir / "assets" / "Diabetes_cover1.png"
cover_image1 = current_dir / "assets" / "Diabetes_Cover_new.png"
#cover_image1 = current_dir / "assets" / "Diabetes_cover3.png"
cover_image = current_dir / "assets" / "Diabetes_cover4.png"
demo_report_fig = current_dir / "assets" / "demo_report_fig.png"
lab_report_fig = current_dir / "assets" /"lab_report_fig.png"


image_path_E1122 = current_dir / "assets"/"demo_report_fig_E1122.png"
image_path_E1129 = current_dir / "assets"/"demo_report_fig_E1129.png"
image_path_E1131 = current_dir / "assets"/"demo_report_fig_E1131.png"
image_path_E1139 = current_dir / "assets"/"demo_report_fig_E1139.png"
image_path_E1142 = current_dir / "assets"/"demo_report_fig_E1142.png"
image_path_E1151 = current_dir / "assets"/"demo_report_fig_E1151.png"
image_path_E1164 = current_dir / "assets"/"demo_report_fig_E1164.png"
image_path_E1165 = current_dir / "assets"/"demo_report_fig_E1165.png"
image_path_E1171 = current_dir / "assets"/"demo_report_fig_E1171.png"
image_path_E1172 = current_dir / "assets"/"demo_report_fig_E1172.png"


image_path_E1122_lab = current_dir / "assets"/"lab_report_fig_E1122.png"
image_path_E1129_lab = current_dir / "assets"/"lab_report_fig_E1129.png"
image_path_E1131_lab = current_dir / "assets"/"lab_report_fig_E1131.png"
image_path_E1139_lab = current_dir / "assets"/"lab_report_fig_E1139.png"
image_path_E1142_lab = current_dir / "assets"/"lab_report_fig_E1142.png"
image_path_E1151_lab = current_dir / "assets"/"lab_report_fig_E1151.png"
image_path_E1164_lab = current_dir / "assets"/"lab_report_fig_E1164.png"
image_path_E1165_lab = current_dir / "assets"/"lab_report_fig_E1165.png"
image_path_E1171_lab = current_dir / "assets"/"lab_report_fig_E1171.png"
image_path_E1172_lab = current_dir / "assets"/"lab_report_fig_E1172.png"


t2dm_path = current_dir / "assets" /"t2dm_final.pkl"
#report_create = current_dir /"assets"/"report_create.py"

PAGE_ICON = ":computer:"
t2dm_final = pd.read_pickle(t2dm_path)


demo_Cox = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"
lab_Cox = current_dir/ "assets"/"Trained_Models"/"CPH_All"

E1122_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1122.sav"
E1129_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1129.sav"
E1131_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1131.sav"
E1139_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1139.sav"
E1142_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1142.sav"
E1151_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1151.sav"
E1164_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1164.sav"
E1165_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1165.sav"
E1171_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1171.sav"
E1172_demo_model = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"cph_E1172.sav"


E1122_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1122.pkl" 
E1129_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1129.pkl" 
E1131_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1131.pkl" 
E1139_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1139.pkl" 
E1142_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1142.pkl" 
E1151_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1151.pkl" 
E1164_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1164.pkl" 
E1165_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1165.pkl" 
E1171_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1171.pkl" 
E1172_demo_data = current_dir/ "assets"/"Trained_Models"/"CPH_Basic"/"Cox_E1172.pkl" 





E1122_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1122_All.sav"
E1129_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1129_All.sav"
E1131_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1131_All.sav"
E1139_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1139_All.sav"
E1142_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1142_All.sav"
E1151_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1151_All.sav"
E1164_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1164_All.sav"
E1165_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1165_All.sav"
E1171_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1171_All_orginal.sav"
E1172_lab_model = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"cph_E1172_All.sav"

E1122_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1122_All.pkl" 
E1129_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1129_All.pkl" 
E1131_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1131_All.pkl" 
E1139_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1139_All.pkl" 
E1142_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1142_All.pkl" 
E1151_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1151_All.pkl" 
E1164_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1164_All.pkl" 
E1165_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1165_All.pkl" 
E1171_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1171_All.pkl" 
E1172_lab_data = current_dir/ "assets"/"Trained_Models"/"CPH_All"/"Cox_E1172_All.pkl" 



st.set_page_config(page_title="NZTPCD - App", layout="wide", initial_sidebar_state="expanded", page_icon = PAGE_ICON)




# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
#with open(resume_file, "rb") as pdf_file:
 #   PDFbyte = pdf_file.read()
#profile_pic = Image.open(profile_pic)

 
#with open(demo_report, "rb") as pdf_file:
#    PDF_demo_report = pdf_file.read()
    
    
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage
    

# Display cover image
st.image(str(cover_image1), use_column_width=True)

st.markdown("""<link rel="stylesheet" type="text/css" href="styles.css">""", unsafe_allow_html=True)



# Define the options for the dropdown
options = ['E119 - Diabetes','E1122 - Diabetic Nephropathy', 'E1129 - Kidney Complications AKI', 'E1131 - Background Retinopathy' , 'E1139 - Other Opthalmic Complications','E1142 - Diabetic Polyneuropathy','E1151 -PVD','E1164 - Hypoglycaemia','E1165 - Poor Control - Hyperglycaemia','E1171 - Microvascular and other specified nonvascular complications','E1172 - Fatty Liver','All Complications']

# Add a dropdown widget to the sidebar
selected_option = st.sidebar.selectbox(
    'Select a Complication',
    options
)



options2 = ["None","Demographic Details", "Lab Vales"]
selected_option2 = st.sidebar.selectbox("Prediction", options2)

if selected_option2 == "Demographic Details":
    st.sidebar.write("Want to create a report?")
    demo_report_clicked = st.sidebar.button("Create Report with demographic data",on_click=set_stage, args=(1,) )

if selected_option2 == "Lab Vales":
    st.sidebar.write("Want to create a report?")
    lab_report_clicked = st.sidebar.button("Create Report with lab data",on_click=set_stage, args=(2,) )

def load_data_model_none(comp):
    t2dm_final = pd.read_pickle(t2dm_path)
    comp_df = t2dm_final[t2dm_final['Diagnosis ICD10 Code_comp']== comp]
    return comp_df
    
    

#def report_create_hyperlinked_page():
    # Replace "hyperlinked_page.py" with the name of your Python file
 #   os.system( "/Users/anuradhamadurapperuma/Documents/report_create.py") 
    
def load_data_model_demo(location,df_name,model_name,name):
    file_path = f"{location}/{df_name}"
    comp_cph_model_data = pd.read_pickle(file_path)
    time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    file_path1 = f"{location}/{model_name}"
    with open(file_path1,'rb') as file1:
        cph_basic_model = pickle.load(file1)
        num_rows = comp_cph_model_data.shape[0]        
    t = range(0, 3652)  
    survival_func = cph_basic_model.predict_survival_function(comp_cph_model_data.iloc[0:num_rows], times=t)
    survival_means = survival_func.mean(axis=1)
    gender_value, Maori_value, ethnicity_value, number_input,number_input_age, submit_button_demo  = demographic_form()
    data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
    data1 =pd.DataFrame(data1)
    
    
    

    if submit_button_demo:
        if number_input_age < number_input:
            st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
        else:  
            st.markdown('<p class="my-text">---Prediction results----</p>', unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)
            
            st.markdown('<p class="h1">Your prediction results for next 10 years</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.plot(t,survival_means, label='Survival curve of the cohort')
            b = cph_basic_model.predict_survival_function(data1, times=t)
            ax.plot(t,b, label='your survival curve')
           # plt.legend()
           # st.pyplot(plt)
        
        #rounded_time_points = min(time_points, key=lambda x: abs(x - 730))
        #surv_prob_1year = c[rounded_time_points].iloc[0]
        #st.write(surv_prob_1year)
            c = cph_basic_model.predict_survival_function(data1,times=time_points)

            age_diff = (number_input_age - number_input)*365
      
            partial_hazard = cph_basic_model.predict_partial_hazard(data1) 
   
            survival_prob_day = cph_basic_model.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_day
       
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today = cph_basic_model.predict_survival_function(data1,times=time_today)
            survival_from_age_today = (1 - survival_from_age_today)*100
            survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
            survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name = survival_from_age_today.columns[0]

        # rename the first column
            survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
            survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
            survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)

         
            
           
            ax.legend()
            ax.set_title('Survival Probability of {}'.format(name))
            ax.set_xlabel('Days')
            ax.set_ylabel('Survival Probability')

            
            
            
            # create two columns
        
            st.write("")
            st.write("")
            st.write("")
            left_column, right_column = st.columns(2)
            
        # add content to the left column
            with left_column:
                st.write('Your hazard of %s' % (name))
                st.table(survival_from_age_today) 
                

        # add content to the right column
            with right_column:
                st.write('Visualisation of your hazard of %s' % (name))
                x = [0,1,2,3,4,5,6,7,8,9,10]
                y = survival_from_age_today['Hazard probability in %'].values
                y = y.astype('float64')
                df1 = pd.DataFrame(y,x)
                #st.write(y)
                fig3, ax3 = plt.subplots()
                #plt.plot(x, y)
                ax3.plot(df1)
                
                ax3.set_xlabel('Time (Years)')
                ax3.set_ylabel('Hazard probability (%)')

                # Set the title
                ax3.set_title('Risk of experiencing %s' %(name))
                st.pyplot(fig3)
            
            
                        
            st.write("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="h1">Survival curve </p>', unsafe_allow_html=True)
            
            
            st.pyplot(fig) 
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
            #st.table(survival_from_age_today)
            
              


def load_data_model_lab(location,df_name,model_name,name):
    file_path = f"{location}/{df_name}"
    comp_cph_all_data = pd.read_pickle(file_path)
    file_path1 = f"{location}/{model_name}"
    with open(file_path1,'rb') as file1:
        cph_all_model = pickle.load(file1)
        num_rows = comp_cph_all_data.shape[0]
    t = range(0, 3500)
    survival_func = cph_all_model.predict_survival_function(comp_cph_all_data.iloc[0:num_rows], times=t)
    survival_means = survival_func.mean(axis=1)
    gender_value, Maori_value, ethnicity_value, number_input,number_input_age, hba1c, cholesterol, triglycerides, hdl,ldl,egfr,submit_button_lab = lab_form()
    data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value], 'HbA1c_ResultValue': hba1c,'Cholesterol_ResultValue': cholesterol,'Triglyceride_ResultValue': triglycerides ,'HDL_ResultValue': hdl,'LDL_ResultValue':ldl,'EGFR_ResultValue':egfr}
    data1 =pd.DataFrame(data1)
    if submit_button_lab:
        if number_input_age < number_input:
            st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="my-text">---Prediction results----</p>', unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)
            
            st.markdown('<p class="h1">Your prediction results for next 10 years</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.plot(t,survival_means, label='Survival curve of the cohort')
            
            b = cph_all_model.predict_survival_function(data1, times=t)
            ax.plot(t,b, label='your survival curve')
           # plt.legend()
           # st.pyplot(plt)
        
        
            age_diff = (number_input_age - number_input)*365
      
            partial_hazard = cph_all_model.predict_partial_hazard(data1) 
        
       
            survival_prob_day = cph_all_model.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_day
       
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today = cph_all_model.predict_survival_function(data1,times=time_today)
            survival_from_age_today = (1 - survival_from_age_today)*100
        
        
            survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
            survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name = survival_from_age_today.columns[0]

        # rename the first column
            survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
            survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
            survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)

            
           
            ax.legend()
            ax.set_title('Survival Probability of {}'.format(name))
            ax.set_xlabel('Days')
            ax.set_ylabel('Survival Probability')
            
            # create two columns
        
            st.write("")
            st.write("")
            st.write("")
            left_column, right_column = st.columns(2)
            
        # add content to the left column
            with left_column:
                st.write('Your hazard of %s' % (name))
                st.table(survival_from_age_today) 
                

        # add content to the right column
            with right_column:
                st.write('Visualisation of your hazard of %s' % (name))
                x = [0,1,2,3,4,5,6,7,8,9,10]
                y = survival_from_age_today['Hazard probability in %'].values
                y = y.astype('float64')
                df1 = pd.DataFrame(y,x)
                #st.write(y)
                fig3, ax3 = plt.subplots()
                #plt.plot(x, y)
                ax3.plot(df1)
                
                ax3.set_xlabel('Time (Years)')
                ax3.set_ylabel('Hazard probability (%)')

                # Set the title
                ax3.set_title('Risk of experiencing %s' %(name))
                st.pyplot(fig3)
            
            
                        
            st.write("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="h1">Survival curve </p>', unsafe_allow_html=True)
            
            
            st.pyplot(fig) 

    

def demographic_report_form():
    with st.form("demo_report_form"):
        st.write('Enter your details to create a report')
        name = st.text_input("Enter your name")
        address = st.text_input("Enter your address")
        gender = ['Male', 'Female']
        # Add a selectbox widget to the sidebar
        #gender_select = st.sidebar.selectbox('Gender', gender)
        #st.write("Selected option:", gender_select)
        # define a dictionary to map options to values
        gender_mapper = {'Male': 0,'Female': 1}
        # create select box for user to choose an option
        gender_select = st.selectbox('Gender', list(gender_mapper.keys()))
        gender_value = gender_mapper[gender_select]
        
        
        Maori = ['Maaori', 'Non-Maaori']
        # Add a selectbox widget to the sidebar
        #Maori_select = st.sidebar.selectbox('Maori/Non-Maori', Maori)
        Maori_mapper = {'Maaori': 1,'Non-Maaori': 0}
        # create select box for user to choose an option
        Maori_select = st.selectbox('Maaori', list(Maori_mapper.keys()))
        Maori_value = Maori_mapper[Maori_select]
    
        
        ethnicity = ['European', 'Maaori', 'Other Ethnicity', 'Asian', 'Pacific','Middle Eastern/Latin American/ African']
        # Add a selectbox widget to the sidebar
        #ethnicity_select = st.sidebar.selectbox('Ethnicity', ethnicity)
        ethnicity_mapper = {'European': 0,'Maaori': 1,'Other Ethnicity': 2 ,'Asian': 3,'Pacific': 4,'Middle Eastern/Latin American/ African': 5}
        # create select box for user to choose an option
        ethnicity_select = st.selectbox('Ethnicity', list(ethnicity_mapper.keys()))
        ethnicity_value = ethnicity_mapper[ethnicity_select]    
        number_input = st.number_input(label='Enter your age at diagnosis of Diabetes', value=0,step=1,min_value=0,max_value=100)
        number_input_age = st.number_input(label='Enter your current age', value=0,step=1,min_value=0,max_value=100)
        #st.sidebar.button('Submit')
        submit_button_demo = st.form_submit_button('Submit',on_click=set_stage, args=(2,))
        #if st.sidebar.button('Submit'):
         #   st.write(f"Hello!")
        return name, address, gender_value, Maori_value, ethnicity_value, number_input,number_input_age, submit_button_demo
        #data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
        #data1 =pd.DataFrame(data1)    
    

def create_report_all_comp_demo(cph_basic_model,cph_model_data,survival_means,data1,age_diff, comp):
    st.write("read")
    #file_path = f"{location}/{df_name}"
    #comp_cph_model_data = pd.read_pickle(file_path)
    time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    #file_path1 = f"{location}/{model_name}"
    #with open(file_path1,'rb') as file1:
     #   cph_basic_model = pickle.load(file1)
    num_rows = cph_model_data.shape[0]        
    t = range(0, 3652)  
    survival_func = cph_basic_model.predict_survival_function(cph_model_data.iloc[0:num_rows], times=t)
    survival_means = survival_func.mean(axis=1)

    plt.plot(t,survival_means, label='Survival curve of the cohort')
    b = cph_basic_model.predict_survival_function(data1, times=t)
    plt.plot(t,b, label='your survival curve')

    c = cph_basic_model.predict_survival_function(data1,times=time_points)
      
 
    #age_diff = (number_input_age - number_input)*365
      
    partial_hazard = cph_basic_model.predict_partial_hazard(data1) 
  
    survival_prob_day = cph_basic_model.predict_survival_function(data1,times=age_diff)
    hazard = 1 - survival_prob_day
       
    time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
    survival_from_age_today = cph_basic_model.predict_survival_function(data1,times=time_today)
    survival_from_age_today = (1 - survival_from_age_today)*100
        
    survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
    survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
    old_name = survival_from_age_today.columns[0]

        # rename the first column
    survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
    survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
    survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
    st.table(survival_from_age_today)

            
    plt.legend()
    st.pyplot(plt)   

        
        # Save the plot as an image
    plt.savefig('demo_report_fig_{}.png'.format (comp))

    style_body = styles['Normal']
    # Create a new canvas and add some content to it
    c = canvas.Canvas("temp_canvas.pdf")
    c.drawString(100, 750, "This is some text added to the canvas")

    
        #dynamic_name = table_data_ + str(comp)
   
    table_data = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
    table = Table(table_data)
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
    table.hAlign = 'LEFT'   

         # Add an image
    image_path = "demo_report_fig.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
    image1 = Image(image_path, width=inch*3.5, height=inch*3.5)
        
    header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
    header_table = Paragraph(header_table, style_sheet["Heading2"])
        
    header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
    header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
    chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
    elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
    chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

    elements.append(Table([[table, image1]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

    spacer = Spacer(1, 0.25*inch)
       


def create_report_demo(location,df_name,model_name,comp):
    file_path = f"{location}/{df_name}"
    comp_cph_model_data = pd.read_pickle(file_path)
    time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    file_path1 = f"{location}/{model_name}"
    with open(file_path1,'rb') as file1:
        cph_basic_model = pickle.load(file1)
        num_rows = comp_cph_model_data.shape[0]        
    t = range(0, 3652)  
    survival_func = cph_basic_model.predict_survival_function(comp_cph_model_data.iloc[0:num_rows], times=t)
    survival_means = survival_func.mean(axis=1)
    name, address, gender_value, Maori_value, ethnicity_value, number_input,number_input_age, submit_button_demo  = demographic_report_form()
    data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
    data1 =pd.DataFrame(data1)

    if submit_button_demo:
        if number_input_age < number_input:
            st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
        else:
            plt.plot(t,survival_means, label='Survival curve of the cohort')
            b = cph_basic_model.predict_survival_function(data1, times=t)
            plt.plot(t,b, label='your survival curve')

            c = cph_basic_model.predict_survival_function(data1,times=time_points)
      
 
            age_diff = (number_input_age - number_input)*365
      
            partial_hazard = cph_basic_model.predict_partial_hazard(data1) 
  
            survival_prob_day = cph_basic_model.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_day
       
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today = cph_basic_model.predict_survival_function(data1,times=time_today)
            survival_from_age_today = (1 - survival_from_age_today)*100
        
            survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
            survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name = survival_from_age_today.columns[0]

        # rename the first column
            survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
            survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
            survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
            st.table(survival_from_age_today)

            
            plt.legend()
            plt.title('Survival Probability of {}'.format(comp))
            plt.xlabel('Days')
            plt.ylabel('Survival Probability')
            st.pyplot(plt)   

        
        # Save the plot as an image
            plt.savefig('assets/demo_report_fig.png')
        
         # Create a file-like buffer to receive PDF data.
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=inch/2, rightMargin=inch/2,
                            topMargin=inch/2, bottomMargin=inch/2)
        #doc = SimpleDocTemplate(report_filename, pagesize=letter,
                        #leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        
        
        
        # Define the page template with one frame
            page_template = PageTemplate(id='OneCol', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='Frame1')])
            doc.addPageTemplates([page_template])
            style_sheet = getSampleStyleSheet()
        # Define styles for the document
            styles = getSampleStyleSheet()
        
        
        # Create a list of flowable elements to be added to the document
            elements = []  
        # Add an image
        #image_path = "Diabetes_cover1.png"
            image = Image(cover_image1, width = inch*8)
        
            elements.append(image)
            style_title = styles['Title']
            style_body = styles['Normal']

            title_style = styles["Title"]
    
        # Define the current date and time
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # Create a ParagraphStyle for the date/time text
            style_datetime = ParagraphStyle(name='Date/time', fontSize=12, textColor=colors.black)
        # Add the title to the document
            header_text = "Report - {}".format(comp)
         # Create first large horizontal division
            header_paragraph = Paragraph(header_text, style_sheet["Heading1"])
            elements.append(header_paragraph)
            elements.append(Spacer(1, 20)) 
        # Create a Paragraph for the date/time text
            datetime_text = "Generated on: " + current_date
            datetime_paragraph = Paragraph(datetime_text, style_datetime)
        # Add the date/time text to the document elements
            elements.append(datetime_paragraph)
            elements.append(Spacer(1, 0.25*inch)) 
        

            name_text = "Name: {}".format(name)
            age_text = "Age: {}".format(number_input_age)
            address_text = "Address: {}".format(address)
        
        # Add the user input to the document
            elements.append(Paragraph(name_text, style_body))
            elements.append(Paragraph(age_text, style_body))
            elements.append(Paragraph(address_text, style_body))
            elements.append(Spacer(1, 0.25*inch))
        
        
            table_data = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
            table = Table(table_data)
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
            table.hAlign = 'LEFT'   
       # elements.append(table)

        #spacer = Spacer(1, inch * 0.5)
        #elements.append(spacer)


         # Add an image
        #image_path = "assets/demo_report_fig.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
            image1 = Image(demo_report_fig, width=inch*3.5, height=inch*3.5)
        
            header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
            header_table = Paragraph(header_table, style_sheet["Heading2"])
        
            header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
            header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
            chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
            elements.append(Table([[header_table, header_graph]],
                         colWidths=[4 * inch, 4 * inch,],
                         rowHeights=[1.5 * inch], style=chart_style))
        
            chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

            elements.append(Table([[table, image1]],
                     colWidths=[4 * inch, 4 * inch,],
                     rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

            spacer = Spacer(1, 0.25*inch)
        
        
        # Build the document by adding the elements
            doc.build(elements)
        
            now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            report_filename = f"assets/report_demographic_details_{now}.pdf"
        
         # File name of the report
        #report_name = "example_report.pdf"
        
        # Move the buffer's pointer to the beginning of the buffer
            buffer.seek(0)
        # generate the download link
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download Report</a>'

        # display the link using st.markdown
            st.markdown(href, unsafe_allow_html=True)
        # Set up the download link
            #href = f'<a href="data:application/pdf;base64,{base64.b64encode(buffer.read()).decode()}">Download Report</a>'
            #st.markdown(href, unsafe_allow_html=True)
       
    
       # doc.build([header_paragraph,datetime_paragraph,name_paragraph,age_paragraph,address_paragraph,[column1, column2]])
        # Display a link to the generated PDF
        #st.success("PDF generated! Download the PDF here")
        
       # with open(report_path, "rb") as pdf_file:
        #    PDF_demo_report = pdf_file.read()
        #st.download_button(
        #label=" 📄 Download Report",
        #data=PDF_demo_report,
        #file_name=report_filename.name,
        #mime="application/octet-stream",)


def lab_report_form():
    with st.form("demo_report_form"):
        st.write('Enter your details to create a report')
        name = st.text_input("Enter your name")
        address = st.text_input("Enter your address")
        gender = ['Male', 'Female']
    # Add a selectbox widget to the sidebar
        gender_mapper = {'Male': 0,'Female': 1}
        # create select box for user to choose an option
        gender_select = st.selectbox('Gender', list(gender_mapper.keys()))
        gender_value = gender_mapper[gender_select]
        
        Maori = ['Maaori', 'Non-Maaori']
    # Add a selectbox widget to the sidebar
        Maori_mapper = {'Maaori': 1,'Non-Maaori': 0}
        # create select box for user to choose an option
        Maori_select = st.selectbox('Maaori', list(Maori_mapper.keys()))
        Maori_value = Maori_mapper[Maori_select]
        
        ethnicity = ['European', 'Maaori', 'Other Ethnicity', 'Asian', 'Pacific','Middle Eastern/Latin American/ African']
    # Add a selectbox widget to the sideba
        #ethnicity_select = st.sidebar.selectbox('Ethnicity', ethnicity)
        ethnicity_mapper = {'European': 0,'Maaori': 1,'Other Ethnicity': 2 ,'Asian': 3,'Pacific': 4,'Middle Eastern/Latin American/ African': 5}
        # create select box for user to choose an option
        ethnicity_select = st.selectbox('Ethnicity', list(ethnicity_mapper.keys()))
        ethnicity_value = ethnicity_mapper[ethnicity_select]
        
        number_input = st.number_input(label='Enter your age at diagnosis of Diabetes', value=0,step=1,min_value=0,max_value=100)
        

        number_input_age = st.number_input(label='Enter your current age', value=0,step=1,min_value=0,max_value=100)
    # Add age and phone number input with two columns
        col1, col2 = st.columns(2)
        hba1c = col1.number_input('HbA1c', min_value=0.0, max_value=120.0, step=0.1, format="%.2f",key="hba1c", help="Enter HbA1c value as mmol/mol (Ex:47)",)

        hba1c = float(hba1c)
        
        cholesterol = col2.number_input('Cholesterol', min_value=0.0, max_value=120.0, step=0.1, format="%.2f",key="cholesterol", help="Enter Cholesterol value as mmol/l (Ex:5.2)",)
        cholesterol = float(cholesterol)
        
        col1, col2 = st.columns(2)
        triglycerides = col1.number_input('Triglyceride', min_value=0.0, max_value=20.0, step=0.1, format="%.2f",key="triglyceride", help="Enter Triglyceride value as mmol/l (Ex:1.5)",)
        triglycerides = float(triglycerides)
        
        hdl = col2.number_input('HDL', value=0.0, step=0.1, format="%.2f",key="hdl", help="Enter HDL value as mmol/l (Ex:1.55)",)
        hdl = float(hdl)
        
        col1, col2 = st.columns(2)
        ldl = col1.number_input('LDL', min_value=0.0, max_value=20.0, step=0.1, format="%.2f",key="ldl", help="Enter LDL value as mmol/l (Ex:2.5)",)
        ldl = float(ldl)
  
        egfr = col2.number_input('eGFR', value=0.0, step=0.1, format="%.2f",key="egfr", help="Enter eGFR value as mL/min/1.73m2 (Ex:47)",)
        egfr = float(egfr)
        
    # Add address input with full width
        #address = st.text_input('Address', max_chars=200)
        submit_button_lab = st.form_submit_button('Submit')
        # Submit button
       # if st.button('Submit'):
        # Process form data here
       # st.write('Name:', name)
       # st.write('Age:', age)
        #st.write('Phone Number:', phone_number)
        #st.write('Address:', address)  
        #if submit_button:
        #    st.write(f"Hello!")
        return name, address, gender_value, Maori_value, ethnicity_value, number_input,number_input_age,hba1c,cholesterol,triglycerides,hdl,ldl,egfr,submit_button_lab
        
        
def create_report_lab(location,df_name,model_name,comp):
    
    file_path = f"{location}/{df_name}"
    comp_cph_all_data = pd.read_pickle(file_path)
    file_path1 = f"{location}/{model_name}"
    with open(file_path1,'rb') as file1:
        cph_all_model = pickle.load(file1)
        num_rows = comp_cph_all_data.shape[0]
    t = range(0, 3500)
    survival_func = cph_all_model.predict_survival_function(comp_cph_all_data.iloc[0:num_rows], times=t)
    survival_means = survival_func.mean(axis=1)
    name, address, gender_value, Maori_value, ethnicity_value, number_input,number_input_age,hba1c,cholesterol,triglycerides,hdl,ldl,egfr,submit_button_lab  = lab_report_form()
    data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value], 'HbA1c_ResultValue': hba1c,'Cholesterol_ResultValue': cholesterol,'Triglyceride_ResultValue': triglycerides ,'HDL_ResultValue': hdl,'LDL_ResultValue':ldl,'EGFR_ResultValue':egfr}
    data1 =pd.DataFrame(data1)
    #st.write(submit_button_lab)
    if submit_button_lab:
        if number_input_age < number_input:
            st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
        else:
            b = cph_all_model.predict_survival_function(data1, times=t)
            plt.plot(t,b, label='your survival curve')
           # plt.legend()
           # st.pyplot(plt)
        
        
            age_diff = (number_input_age - number_input)*365
      
            partial_hazard = cph_all_model.predict_partial_hazard(data1) 
        
      

        
            survival_prob_day = cph_all_model.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_day
       
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today = cph_all_model.predict_survival_function(data1,times=time_today)
            survival_from_age_today = (1 - survival_from_age_today)*100
        
            survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
            survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name = survival_from_age_today.columns[0]

        # rename the first column
            survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
            survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
            survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
            st.table(survival_from_age_today)
         
   
            plt.plot(t,survival_means, label='Survival curve of the cohort')
      
            plt.legend()
            plt.title('Survival Probability of {}'.format(comp))
            plt.xlabel('Days')
            plt.ylabel('Survival Probability')
            st.pyplot(plt)    

        
        # Save the plot as an image
            plt.savefig('assets/lab_report_fig.png')
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                        leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        
        
        # Define the page template with one frame
            page_template = PageTemplate(id='OneCol', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='Frame1')])
            doc.addPageTemplates([page_template])
            style_sheet = getSampleStyleSheet()
        # Define styles for the document
            styles = getSampleStyleSheet()
        
        
        # Create a list of flowable elements to be added to the document
            elements = []  
        # Add an image
        #image_path = "Diabetes_cover1.png"
            image = Image(cover_image1, width = inch*8)
            
            elements.append(image)
            style_title = styles['Title']
            style_body = styles['Normal']

            title_style = styles["Title"]
    
        # Define the current date and time
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # Create a ParagraphStyle for the date/time text
            style_datetime = ParagraphStyle(name='Date/time', fontSize=12, textColor=colors.black)
        # Add the title to the document
            header_text = "Report - {}".format(comp)
         # Create first large horizontal division
            header_paragraph = Paragraph(header_text, style_sheet["Heading1"])
            elements.append(header_paragraph)
            elements.append(Spacer(1, 20)) 
        # Create a Paragraph for the date/time text
            datetime_text = "Generated on: " + current_date
            datetime_paragraph = Paragraph(datetime_text, style_datetime)
        # Add the date/time text to the document elements
            elements.append(datetime_paragraph)
            elements.append(Spacer(1, 0.25*inch)) 
        

            name_text = "Name: {}".format(name)
            age_text = "Age: {}".format(number_input_age)
            address_text = "Address: {}".format(address)
        
        # Add the user input to the document
            elements.append(Paragraph(name_text, style_body))
            elements.append(Paragraph(age_text, style_body))
            elements.append(Paragraph(address_text, style_body))
            elements.append(Spacer(1, 0.25*inch))
        
        
            table_data = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
            table = Table(table_data)
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
            table.hAlign = 'LEFT'   
       # elements.append(table)

            spacer = Spacer(1, inch * 0.5)
            elements.append(spacer)
            image1 = Image(lab_report_fig, width=inch*3.5, height=inch*3.5)
        
            header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
            header_table = Paragraph(header_table, style_sheet["Heading2"])
        
            header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
            header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
            chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
            elements.append(Table([[header_table, header_graph]],
                     colWidths=[4 * inch, 4 * inch,],
                     rowHeights=[1.5 * inch], style=chart_style))
        
            chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

            elements.append(Table([[table, image1]],
                     colWidths=[4 * inch, 4 * inch,],
                     rowHeights=[2.5 * inch], style=chart_style1))
            spacer = Spacer(1, 0.25*inch)
       
        # Build the document by adding the elements
            doc.build(elements)
        
     
            now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            report_filename = f"report_lab_details_{now}.pdf"

        
        # Move the buffer's pointer to the beginning of the buffer
            buffer.seek(0)
        # generate the download link
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download Report</a>'

        # display the link using st.markdown
            st.markdown(href, unsafe_allow_html=True)
        # Set up the download link
         #   href = f'<a href="data:application/pdf;base64,{base64.b64encode(buffer.read()).decode()}">Download Report</a>'
         #   st.markdown(href, unsafe_allow_html=True)

def graph_draw(df,name,des):
    st.write("")
    st.write("")
    st.write("")
    left_column, right_column = st.columns(2)

# add content to the left column
    with left_column:
        fig_Gender = go.Figure(data=[go.Histogram(x=df['Patient_Gender'],histnorm='percent')])
        fig_Gender.update_traces(marker=dict(color='#FFB900'))
        fig_Gender.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=50, b=50, t=100, pad=4),
            width=700,  # set the plot width to 100%
            height=500,  # set the plot height to 100%
            autosize=True, 
            yaxis=(dict(showgrid=False)),
            xaxis_title="Gender", yaxis_title="Percentage of Patients", title_text="Patients with {} - {} by Gender ".format(name, des)
                )

        st.plotly_chart(fig_Gender,use_container_width=True)

        fig_Maori = go.Figure(data=[go.Histogram(x=df['Patient_Maori_NonMaori'],histnorm='percent')])
        fig_Maori.update_traces(marker=dict(color='#008B8B'))
        fig_Maori.update_layout(
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=50, r=50, b=50, t=100, pad=4),
            width=700,  # set the plot width to 100%
            height=500,  # set the plot height to 100%
            autosize=True, 
            yaxis=(dict(showgrid=False)),xaxis_title="Maaori/Non-Maaori", yaxis_title="Percentage of Patients", title_text="Patients with {} - {} by Maaori/Non-Maaori ".format(name, des)
            )
        st.plotly_chart(fig_Maori,use_container_width=True)

# add content to the right column
    with right_column:
         vals=df['Patient_Ethnicity'].value_counts()
         keys=df['Patient_Ethnicity'].unique()
         color = ['#E65100']
         fig_pie_Ethnicity = go.Figure(data=[go.Pie(labels=keys, values=vals, marker=dict(colors=color))])
         fig_pie_Ethnicity.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#444444',margin=dict(l=50, r=50, b=50, t=100, pad=4),
         width=700,  # set the plot width to 100%
         height=500,  # set the plot height to 100%, 
         autosize=True, title_text="Patients with {} - {} by Ethnicity ".format(name, des))
                 # wrap the plot in a div
         st.plotly_chart(fig_pie_Ethnicity,use_container_width=True)

         df['Patient_DOB'] = pd.to_datetime(df['Patient_DOB'])
         df['Birth_year'] = df['Patient_DOB'].dt.year
         Age_comp = df['Year_Comp'] - df['Birth_year']

         layout = go.Layout(plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Age on diagosis of {} - {}".format(name,des),margin=dict(l=50, r=50, b=50, t=100, pad=4), width=700,  # set the plot width to 100%
            height=500,  # set the plot height to 100%
            autosize=True, yaxis_title="Number of Patients",title="Patients with {} - {} by Age on diagnosis ".format(name, des))   
         fig_Age = go.Figure(data=[go.Histogram(x=Age_comp, nbinsx=30,marker=dict(color='#E65100'))],layout=layout)
            # Define the histogram trace
         trace = go.Histogram(x=Age_comp,xbins=dict(start=10,end=100,size=1))
            #fig = go.Figure(data=[trace], layout=layout)

         st.plotly_chart(fig_Age,use_container_width=True)

    
   
    
    
        
    
        
    kmf = KaplanMeierFitter()
    kmf.fit(df['days'],event_observed=df['event'])
        
        
        #st.write(E1139_model_data.head())
        #time_E1122 = E1122_model_data['days']
        #model= pickle.load(f'/Users/anuradhamadurapperuma/Documents/thesis code/Thesis_codes/Trained_Models/Kaplan','rb')
    result = kmf.survival_function_at_times(df['days'])
    fig_kmf_comp = kmf.plot(color=['#FFA07A', '#2E8B57'])
    fig_kmf_comp.set_title('Survival Probability of {}'.format(name))
    fig_kmf_comp.set_xlabel('Days')
    fig_kmf_comp.set_ylabel('Survival Probability')

        # Display the chart in Streamlit
    st.pyplot(fig_kmf_comp.figure)
    st.write('The survival curve illustrates the changes in the survival rate within ten years. The time frame is in the format of days. The graph shows that the survival at the initial time is nearly 1, which reduces over time to 0. The survival curve can be used to interpret the survival at each time point. The hazard of complications can also be measured using the graph where hazard = 1 - survival.')
        



def demographic_form():
    with st.form("lab_form"):
        st.write('Enter your demographic details')
        gender = ['Male', 'Female']
        # Add a selectbox widget to the sidebar
        #gender_select = st.sidebar.selectbox('Gender', gender)
        #st.write("Selected option:", gender_select)
        # define a dictionary to map options to values
        gender_mapper = {'Male': 0,'Female': 1}
        # create select box for user to choose an option
        gender_select = st.selectbox('Gender', list(gender_mapper.keys()))
        gender_value = gender_mapper[gender_select]
        
        
        Maori = ['Maaori', 'Non-Maaori']
        # Add a selectbox widget to the sidebar
        #Maori_select = st.sidebar.selectbox('Maori/Non-Maori', Maori)
        Maori_mapper = {'Maaori': 1,'Non-Maaori': 0}
        # create select box for user to choose an option
        Maori_select = st.selectbox('Maaori', list(Maori_mapper.keys()))
        Maori_value = Maori_mapper[Maori_select]
    
        
        ethnicity = ['European', 'Maaori', 'Other Ethnicity', 'Asian', 'Pacific','Middle Eastern/Latin American/ African']
        # Add a selectbox widget to the sidebar
        #ethnicity_select = st.sidebar.selectbox('Ethnicity', ethnicity)
        ethnicity_mapper = {'European': 0,'Maaori': 1,'Other Ethnicity': 2 ,'Asian': 3,'Pacific': 4,'Middle Eastern/Latin American/ African': 5}
        # create select box for user to choose an option
        ethnicity_select = st.selectbox('Ethnicity', list(ethnicity_mapper.keys()))
        ethnicity_value = ethnicity_mapper[ethnicity_select]    
        number_input = st.number_input(label='Enter your age at diagnosis of Diabetes', value=0,step=1,min_value=0,max_value=100)
        number_input_age = st.number_input(label='Enter your current age', value=0,step=1,min_value=0,max_value=100)
        #st.sidebar.button('Submit')
        submit_button_demo = st.form_submit_button('Submit')
        #if st.sidebar.button('Submit'):
         #   st.write(f"Hello!")
        return gender_value, Maori_value, ethnicity_value, number_input, number_input_age, submit_button_demo
        #data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
        #data1 =pd.DataFrame(data1)    
    
       



def lab_form():
    st.write("Enter your details here")
    with st.form("lab_form"):
    #name = st.text_input('Name', max_chars=50)
        gender = ['Male', 'Female']
    # Add a selectbox widget to the sidebar
        gender_mapper = {'Male': 0,'Female': 1}
        # create select box for user to choose an option
        gender_select = st.selectbox('Gender', list(gender_mapper.keys()))
        gender_value = gender_mapper[gender_select]
        
        Maori = ['Maaori', 'Non-Maaori']
    # Add a selectbox widget to the sidebar
        Maori_mapper = {'Maaori': 1,'Non-Maaori': 0}
        # create select box for user to choose an option
        Maori_select = st.selectbox('Maaori', list(Maori_mapper.keys()))
        Maori_value = Maori_mapper[Maori_select]
        
        
        
        ethnicity = ['European', 'Maaori', 'Other Ethnicity', 'Asian', 'Pacific','Middle Eastern/Latin American/ African']
    # Add a selectbox widget to the sideba
        #ethnicity_select = st.sidebar.selectbox('Ethnicity', ethnicity)
        ethnicity_mapper = {'European': 0,'Maaori': 1,'Other Ethnicity': 2 ,'Asian': 3,'Pacific': 4,'Middle Eastern/Latin American/ African': 5}
        # create select box for user to choose an option
        ethnicity_select = st.selectbox('Ethnicity', list(ethnicity_mapper.keys()))
        ethnicity_value = ethnicity_mapper[ethnicity_select]
        
        number_input = st.number_input(label='Enter your age at diagnosis of Diabetes', value=0,step=1,min_value=0,max_value=100)
        number_input_age = st.number_input(label='Enter your current age', value=0,step=1,min_value=0,max_value=100)
    # Add age and phone number input with two columns
        col1, col2 = st.columns(2)
        hba1c = col1.number_input('HbA1c', min_value=0.0, max_value=120.0, step=0.1, format="%.2f",key="hba1c", help="Enter HbA1c value as mmol/mol (Ex:47)",)
        hba1c = float(hba1c)
        
        cholesterol = col2.number_input('Cholesterol', min_value=0.0, max_value=120.0, step=0.1, format="%.2f",key="cholesterol", help="Enter Cholesterol value as mmol/l (Ex:5.2)",)
        cholesterol = float(cholesterol)
        
        col1, col2 = st.columns(2)
        triglycerides = col1.number_input('Triglyceride', min_value=0.0, max_value=20.0, step=0.1, format="%.2f",key="triglyceride", help="Enter Triglyceride value as mmol/l (Ex:1.5)",)
        triglycerides = float(triglycerides)
        
        hdl = col2.number_input('HDL', value=0.0, step=0.1, format="%.2f",key="hdl", help="Enter HDL value as mmol/l (Ex:1.55)",)
        hdl = float(hdl)
        
        col1, col2 = st.columns(2)
        ldl = col1.number_input('LDL', min_value=0.0, max_value=20.0, step=0.1, format="%.2f",key="ldl", help="Enter LDL value as mmol/l (Ex:2.5)",)
        ldl = float(ldl)
  
        egfr = col2.number_input('eGFR', value=0.0, step=0.1, format="%.2f",key="egfr", help="Enter eGFR value as mL/min/1.73m2 (Ex:47)",)
        egfr = float(egfr)
        
    # Add address input with full width
        #address = st.text_input('Address', max_chars=200)
        submit_button_lab = st.form_submit_button('Submit')
        # Submit button
       # if st.button('Submit'):
        # Process form data here
       # st.write('Name:', name)
       # st.write('Age:', age)
        #st.write('Phone Number:', phone_number)
        #st.write('Address:', address)  
        #if submit_button:
        #    st.write(f"Hello!")
        return gender_value, Maori_value, ethnicity_value, number_input,number_input_age,hba1c,cholesterol,triglycerides,hdl,ldl,egfr,submit_button_lab
        


            
        
def all_comp_demo(model_name,data_comp,survival_means_name,data1,age_diff,name):  
            st.markdown("<hr>", unsafe_allow_html=True)
            st.write(f'<div style="font-family: Arial; color: #d33f4c;font-size: 20px;text-align: center;">{name}</div>', unsafe_allow_html=True)
            t = range(0, 3652)
            patient_comp = model_name.predict_survival_function(data1, times=t)
            fig2, ax2 = plt.subplots()
            ax2.plot(t,survival_means_name,label=' %s - Survival curve of the cohort' % (name))
            ax2.plot(t,patient_comp,label = 'Your survival curve')
            ax2.legend()
            st.pyplot(fig2)
               # plt.plot(t,b, label='your survival curve')
                #ax1.plot()
                #ax1.legend()
                #st.pyplot(fig1)
                #ax2.plot()
                
            
            partial_hazard = model_name.predict_partial_hazard(data1)
            survival_prob_comp_day = model_name.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_comp_day
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today_comp = model_name.predict_survival_function(data1,times=time_today)
            survival_from_age_today_comp = (1 - survival_from_age_today_comp)*100
            survival_from_age_today_comp = survival_from_age_today_comp.rename(columns={survival_from_age_today_comp.index.name: 'Time'})
            survival_from_age_today_comp.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name_comp = survival_from_age_today_comp.columns[0]
        # rename the first column
            survival_from_age_today_comp = survival_from_age_today_comp.rename(columns={old_name_comp: 'Hazard probability in %'})
            survival_from_age_today_comp['Hazard probability in %'] = round(survival_from_age_today_comp['Hazard probability in %'],2)
            survival_from_age_today_comp['Hazard probability in %'] = survival_from_age_today_comp['Hazard probability in %'].apply('{:.2f}'.format)
             
        # create two columns
        
            st.write("")
            st.write("")
            st.write("")
            left_column, right_column = st.columns(2)
            
        # add content to the left column
            with left_column:
                st.write('Your hazard of %s' % (name))
                st.table(survival_from_age_today_comp) 
                

        # add content to the right column
            with right_column:
                st.write('Visualisation of your hazard of %s' % (name))
                x = [0,1,2,3,4,5,6,7,8,9,10]
                y = survival_from_age_today_comp['Hazard probability in %'].values
                y = y.astype('float64')
                df1 = pd.DataFrame(y,x)
                #st.write(y)
                fig3, ax3 = plt.subplots()
                #plt.plot(x, y)
                ax3.plot(df1)
                
                plt.xlabel('Time (Years)')
                plt.ylabel('Hazard probability (%)')

                # Set the title
                plt.title('Risk of experiencing %s' %(name))
                st.pyplot(fig3)
            return survival_from_age_today_comp     

                
                
def all_comp_lab(model_name,data_comp,survival_means_name,data1,age_diff,name):  
            st.markdown("<hr>", unsafe_allow_html=True)
            st.write(name)
            t = range(0, 3652)
            patient_comp = model_name.predict_survival_function(data1, times=t)
            fig2, ax2 = plt.subplots()
            ax2.plot(t,survival_means_name,label=' %s - Survival curve of the cohort' % (name))
            ax2.plot(t,patient_comp,label = 'Your survival curve')
            ax2.legend()
            st.pyplot(fig2)
               # plt.plot(t,b, label='your survival curve')
                #ax1.plot()
                #ax1.legend()
                #st.pyplot(fig1)
                #ax2.plot()
                
            
            partial_hazard = model_name.predict_partial_hazard(data1)
            survival_prob_comp_day = model_name.predict_survival_function(data1,times=age_diff)
            hazard = 1 - survival_prob_comp_day
            time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
            survival_from_age_today_comp = model_name.predict_survival_function(data1,times=time_today)
            survival_from_age_today_comp = (1 - survival_from_age_today_comp)*100
            survival_from_age_today_comp = survival_from_age_today_comp.rename(columns={survival_from_age_today_comp.index.name: 'Time'})
            survival_from_age_today_comp.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
            old_name_comp = survival_from_age_today_comp.columns[0]
        # rename the first column
            survival_from_age_today_comp = survival_from_age_today_comp.rename(columns={old_name_comp: 'Hazard probability in %'})
            survival_from_age_today_comp['Hazard probability in %'] = round(survival_from_age_today_comp['Hazard probability in %'],2)
            survival_from_age_today_comp['Hazard probability in %'] = survival_from_age_today_comp['Hazard probability in %'].apply('{:.2f}'.format)
        # create two columns
        
            st.write("")
            st.write("")
            st.write("")
            left_column, right_column = st.columns(2)
            
        # add content to the left column
            with left_column:
                st.write('Your hazard of %s' % (name))
                st.table(survival_from_age_today_comp) 

        # add content to the right column
            with right_column:
                st.write('Visualisation of your hazard of %s' % (name))
                x = [0,1,2,3,4,5,6,7,8,9,10]
                y = survival_from_age_today_comp['Hazard probability in %'].values
                y = y.astype('float64')
                df1 = pd.DataFrame(y,x)
                #st.write(y)
                fig3, ax3 = plt.subplots()
                #plt.plot(x, y)
                ax3.plot(df1)
                
                plt.xlabel('Time (Years)')
                plt.ylabel('Hazard probability (%)')

                # Set the title
                plt.title('Risk of experiencing %s' %(name))
                st.pyplot(fig3)
                






def E119_page():
    st.markdown('<p class="my-text">E119 - Diabetes Mellitus</p>', unsafe_allow_html=True)
    text = "Diabetes is a chronic condition characterized by high levels of glucose (sugar) in the blood. Occurring when the body is unable to produce or effectively use insulin, a hormone that regulates blood sugar levels. There are two main types of diabetes: Type 1 diabetes, which is usually diagnosed in children and young adults, and Type 2 diabetes, which is typically diagnosed in adults and is often associated with lifestyle factors. Diabetes can lead to a number of complications, both chronic and acute. Acute complications can including hypoglycaemia (low blood sugar) or hyperglycaemia (high blood sugar), can cause symptoms such as shakiness, confusion, sweating, and fatigue. Long-term complications can include damage to the eyes, kidneys, nerves, and blood vessels, which can increase the risk of heart disease, stroke, blindness, kidney failure, and amputations. People with diabetes may also be at higher risk for infections and slow wound healing.Management of diabetes typically involves lifestyle changes such as diet and exercise, and in some cases, medication or insulin therapy. Regular monitoring of blood sugar levels is also important to help prevent and manage complications."
    st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
    statlist=["463 million adults with diabetes worldwide in 2019, and this number is expected to rise to 578 million by 2030.","In the United States, over 34 million people have diabetes, which is approximately 10.5% of the population. Of these, approximately 95% have type 2 diabetes.","Diabetes is a leading cause of kidney failure, and in the United States, diabetes accounts for approximately 44% of all new cases of kidney failure each year.Diabetes is a leading cause of blindness, and people with diabetes are two to five times more likely to develop cataracts and glaucoma than people without diabetes.","People with diabetes are at higher risk for heart disease, and are approximately twice as likely to die from heart disease than people without diabetes.","In addition to physical health complications, diabetes can also have a significant impact on mental health. People with diabetes are at higher risk for depression and anxiety, and may also experience diabetes distress, which is a form of emotional distress related specifically to the challenges of managing diabetes."]
    
    st.write("Global impact of diabetes:")
    bulleted_statlist = "<ul>" + "".join(f"<li>{item}</li>" for item in statlist) + "</ul>"
    justified_bulleted_statlist = f"<p style='text-align:justify'>{bulleted_statlist}</p>"

    st.write(justified_bulleted_statlist, unsafe_allow_html=True)
    st.write("Waikato District breakdown of Diabetes Mellitus")
    graph_draw(t2dm_final,'E119','Diabetes Mellitus')
    
    
def E1122_page():
    st.markdown('<p class="my-text">E1122 - Diabetic Nephropathy</p>', unsafe_allow_html=True)
   
    t = range(0, 3500)
    if selected_option2 == "None":
        text = "Diabetic nephropathy is a complication of diabetes that affects the kidneys, and it is one of the leading causes of end-stage renal disease (ESRD) worldwide.In diabetic nephropathy, high levels of blood sugar damage the small blood vessels in the kidneys, leading to progressive kidney damage and impaired kidney function. Over time, this can lead to proteinuria (excretion of protein in the urine), hypertension, and ultimately, kidney failure.Management of diabetic nephropathy involves controlling blood sugar levels, blood pressure, and other risk factors that can worsen kidney function. Treatment options may include lifestyle modifications, medications, and in severe cases, kidney transplant or dialysis. Early diagnosis and treatment can help slow the progression of the disease and reduce the risk of complications."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1122')
        comp_df = load_data_model_none('E1122')
        graph_draw(comp_df,'E1122','Diabetic Nephropathy')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1122.pkl','cph_E1122.sav','Diabetic Nephropathy')     
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1122.pkl','cph_E1122.sav','E1122')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1122_All.pkl','cph_E1122_All.sav','Diabetic Nephropathy')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1122_All.pkl','cph_E1122_All.sav','E1122')
      
   
def E1129_page():
    st.markdown('<p class="my-text">E1129 - Kidney Complications AKI</p>', unsafe_allow_html=True)
    
    t = range(0, 3500)
    if selected_option2 == "None":
        text = "E1129 is a diagnostic code used in the International Classification of Diseases, Tenth Revision, Clinical Modification (ICD-10-CM) to identify cases of acute kidney injury (AKI) that are not specified as any of the other types of AKI. AKI is a sudden and often reversible loss of kidney function that can occur as a complication of various medical conditions, including dehydration, infections, medications, and chronic diseases such as diabetes or hypertension. It is characterized by a rapid increase in serum creatinine levels and a decrease in urine output, and can lead to various complications such as electrolyte imbalances, acidosis, and fluid overload. The management of AKI typically involves addressing the underlying cause and providing supportive care to maintain fluid and electrolyte balance and prevent further damage to the kidneys."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1129')
        comp_df = load_data_model_none('E1129')
        graph_draw(comp_df,'E1129','Kidney Complications AKI')
        
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1129.pkl','cph_E1129.sav','Kidney Complications AKI')  
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1129.pkl','cph_E1129.sav','E1129') 
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1129_All.pkl','cph_E1129_All.sav','Kidney Complications AKI')
        
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1129_All.pkl','cph_E1129_All.sav','E1129')
               
       
def E1131_page():
    st.markdown('<p class="my-text">E1131 - Background Retinopathy</p>', unsafe_allow_html=True)
    
    t = range(0, 3500)
    if selected_option2 == "None":
        text = "E1131 - background retinopathy, which is a common complication of diabetes mellitus. Background retinopathy refers to the early stages of diabetic retinopathy, a progressive disease that affects the blood vessels in the retina and can lead to vision loss if left untreated. In background retinopathy, there are mild changes in the blood vessels, such as small bulges called microaneurysms, and some leakage of blood or fluid into the retina. These changes may not cause noticeable symptoms at first, but can progress to more severe forms of diabetic retinopathy if blood sugar levels are not well-controlled or if other risk factors such as high blood pressure or smoking are present. The management of background retinopathy involves close monitoring of blood sugar and blood pressure levels, regular eye exams, and, in some cases, laser treatment or other interventions to prevent or slow the progression of the disease."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1131')
        comp_df = load_data_model_none('E1131')
        graph_draw(comp_df,'E1131','Background Retinopathy')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1131.pkl','cph_E1131.sav','Background Retinopathy') 
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1131.pkl','cph_E1131.sav','E1131')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1131_All.pkl','cph_E1131_All.sav','Background Retinopathy')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:    
        create_report_lab(lab_Cox,'Cox_E1131_All.pkl','cph_E1131_All.sav','E1131')  
      
    
    
def E1139_page():
    st.markdown('<p class="my-text">E1139 - Other Opthalmic Complications</p>', unsafe_allow_html=True)
    
    t = range(0, 3500)
    if selected_option2 == "None":
        text = "These complications may include a wide range of conditions affecting different structures of the eye, such as the lens, cornea, retina, and optic nerve. Some examples of ophthalmic complications that may be classified under E1139 include cataracts, glaucoma, macular edema, vitreous hemorrhage, and tractional retinal detachment. These conditions may occur as a result of various underlying mechanisms, such as damage to the blood vessels, inflammation, or changes in the metabolism of the eye tissues. The management of ophthalmic complications in diabetes may involve a combination of lifestyle changes, medication, and surgery, depending on the severity and type of the condition. Regular eye exams and close monitoring of blood sugar and blood pressure levels are important to detect and manage these complications early and prevent or delay vision loss."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1139')
        comp_df = load_data_model_none('E1139')
        graph_draw(comp_df,'E1139','Other Opthalmic Complications')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1139.pkl','cph_E1139.sav','Other Opthalmic Complications')  
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1139.pkl','cph_E1139.sav','E1139') 
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1139_All.pkl','cph_E1139_All.sav','Other Opthalmic Complications')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1139_All.pkl','cph_E1139_All.sav','E1139')     
          
    
def E1142_page():
    st.markdown('<p class="my-text">E1142 - Diabetic Polyneuropathy</p>', unsafe_allow_html=True)
    
    t = range(0, 3500)
    if selected_option2 == "None":
        text = "Diabetic polyneuropathy is a common complication of diabetes mellitus that affects the peripheral nerves, which are responsible for transmitting signals between the central nervous system and the rest of the body. It is characterized by a gradual and progressive damage to the nerve fibers, which can lead to various symptoms such as numbness, tingling, burning, or shooting pain in the extremities, especially in the feet and hands. Other symptoms may include muscle weakness, loss of coordination, and difficulty in sensing changes in temperature or texture. The exact mechanisms underlying diabetic polyneuropathy are not fully understood, but may involve multiple factors such as hyperglycemia, inflammation, oxidative stress, and microvascular dysfunction. The management of diabetic polyneuropathy may involve a combination of lifestyle changes, medication, and physical therapy to alleviate the symptoms, improve nerve function, and prevent or delay the progression of the disease. Tight control of blood sugar levels is also crucial in reducing the risk of developing diabetic polyneuropathy and other diabetes-related complications."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1142')
        comp_df = load_data_model_none('E1142')
        graph_draw(comp_df,'E1142','Diabetic Polyneuropathy')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1142.pkl','cph_E1142.sav','Diabetic Polyneuropathy')  
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1142.pkl','cph_E1142.sav','E1142') 
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1142_All.pkl','cph_E1142_All.sav','Diabetic Polyneuropathy')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1142_All.pkl','cph_E1142_All.sav','E1142') 

             
    
    
def E1151_page():
    st.markdown('<p class="my-text">E1151 -PVD</p>', unsafe_allow_html=True)
    
    if selected_option2 == "None":
        text = "PVD is a condition that affects the blood vessels outside of the heart and brain, such as those in the legs, arms, or abdomen. In diabetes, PVD may occur as a result of various factors, including damage to the blood vessels caused by high blood sugar levels, inflammation, or changes in the lipid metabolism. The symptoms of PVD may vary depending on the location and severity of the disease, but may include pain, cramping, numbness, or tingling in the affected area, as well as changes in skin color or temperature. The management of PVD in diabetes may involve a combination of lifestyle changes, medication, and surgical interventions to improve blood flow, relieve symptoms, and prevent or delay the progression of the disease. Regular exercise, smoking cessation, and tight control of blood sugar levels and blood pressure are important in reducing the risk of developing PVD and other diabetes-related complications."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1151')
        comp_df = load_data_model_none('E1151')
        graph_draw(comp_df,'E1151','PVD')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1151.pkl','cph_E1151.sav','PVD')
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1151.pkl','cph_E1151.sav','E1151') 
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1151_All.pkl','cph_E1151_All.sav','PVD')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1151_All.pkl','cph_E1151_All.sav','E1151')     
             
       
    
# 'E119 - Diabetes','E1122 - Diabetic Nephropathy', 'E1129 - Kidney Complications AKI', 'E1131 - Background Retinopathy' , 'E1139 - Other Opthalmic Complications','E1142 - Diabetic Polyneuropathy','E1151 -PVD','E1164 - Hypoglycaemia','E1165 - Poor Control - Hyperglycaemia','E1171 - Microvascular and other specified nonvascular complications','E1172 - Fatty Liver'   

def E1164_page():
    st.markdown('<p class="my-text">E1164 - Hypoglycaemia</p>', unsafe_allow_html=True)
    
    if selected_option2 == "None":
        text = "Hypoglycemia is a condition characterized by a low blood sugar level, usually below 70 mg/dL. It can occur in people with diabetes who are taking medications that lower blood sugar, such as insulin or sulfonylureas, or as a result of other factors such as delayed or missed meals, excessive exercise, or alcohol consumption. The symptoms of hypoglycemia may vary depending on the severity of the condition, but may include sweating, trembling, dizziness, confusion, weakness, headache, or seizures. Severe hypoglycemia can lead to loss of consciousness, coma, or even death. The management of hypoglycemia in diabetes involves taking prompt action to raise the blood sugar level, such as consuming glucose-containing foods or beverages, adjusting the diabetes medications, or seeking medical attention if necessary. Prevention of hypoglycemia may involve a combination of strategies, such as monitoring blood sugar levels regularly, adjusting medications under the guidance of a healthcare provider, and maintaining a balanced diet and regular physical activity."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1164')
        comp_df = load_data_model_none('E1164')
        graph_draw(comp_df,'E1164','Hypoglycaemia')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1164.pkl','cph_E1164.sav','Hypoglycaemia')
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1164.pkl','cph_E1164.sav','E1164')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1164_All.pkl','cph_E1164_All.sav','Hypoglycaemia')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1164_All.pkl','cph_E1164_All.sav','E1164')        
            

        
def E1165_page():
    st.markdown('<p class="my-text">E1165 - Poor Control - Hyperglycaemia</p>', unsafe_allow_html=True)
    
    if selected_option2 == "None":
        text = "Hyperglycemia is a condition characterized by high blood sugar levels, which can occur when the body doesn't produce enough insulin or when the insulin doesn't work effectively. Poor glycemic control refers to a situation where blood sugar levels are consistently above the recommended target range despite the use of medications, diet, and lifestyle interventions to manage diabetes. Over time, high blood sugar levels can lead to a wide range of complications, such as damage to the blood vessels, nerves, and organs, which can increase the risk of heart disease, stroke, kidney disease, and other health problems. The management of poor glycemic control in diabetes involves a comprehensive approach that includes regular monitoring of blood sugar levels, adjustment of diabetes medications, lifestyle changes, and regular follow-up with a healthcare provider. In some cases, more intensive treatments such as insulin therapy or other medications may be necessary to achieve adequate glycemic control and prevent or delay the onset of diabetes-related complications."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1165')
        comp_df = load_data_model_none('E1165')
        graph_draw(comp_df,'E1165','Poor Control - Hyperglycaemia')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1165.pkl','cph_E1165.sav','Poor Control - Hyperglycaemia')     
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1165.pkl','cph_E1165.sav','E1165')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1165_All.pkl','cph_E1165_All.sav','Poor Control - Hyperglycaemia')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1165_All.pkl','cph_E1165_All.sav','E1165')        
                
    
    
def E1171_page():
    st.markdown('<p class="my-text">E1171 - Microvascular and other specified nonvascular complications</p>', unsafe_allow_html=True)
    
    if selected_option2 == "None":
        text = "Microvascular complications refer to damage to the small blood vessels in the body, which can occur as a result of high blood sugar levels over time. Examples of microvascular complications include diabetic retinopathy, diabetic nephropathy, and diabetic neuropathy. Other specified nonvascular complications refer to a range of other health problems that can be caused by diabetes or occur more frequently in people with diabetes. Examples of nonvascular complications include skin conditions, digestive problems, oral health issues, and hearing impairment. The management of microvascular and other specified nonvascular complications in diabetes involves a combination of strategies to control blood sugar levels, manage symptoms, and prevent or delay the progression of the disease. This may involve medications, lifestyle modifications such as diet and exercise, regular monitoring of blood sugar levels, and close follow-up with a healthcare provider."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1171')
        comp_df = load_data_model_none('E1171')
        graph_draw(comp_df,'E1171','Microvascular and other specified nonvascular complications')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1171.pkl','cph_E1171.sav','Microvascular and other specified nonvascular complications')     
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1171.pkl','cph_E1171.sav','E1171')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1171_All.pkl','cph_E1171_All_orginal.sav','Microvascular and other specified nonvascular complications')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1171_All.pkl','cph_E1171_All_orginal.sav','E1171')   
    
def E1172_page(): 
    st.markdown('<p class="my-text">E1172 - Fatty Liver</p>', unsafe_allow_html=True)
    
    if selected_option2 == "None":
        text = "Fatty liver, also known as hepatic steatosis, is a condition characterized by the accumulation of fat in the liver cells. In some cases, fatty liver may not cause any symptoms or complications, but in others, it can progress to a more serious condition called non-alcoholic steatohepatitis (NASH), which can cause inflammation and damage to the liver tissue. Fatty liver can occur in people with diabetes as a result of insulin resistance, which can cause the liver to produce and store more fat than normal. Other risk factors for fatty liver include obesity, high blood pressure, and high cholesterol levels. The management of fatty liver in diabetes involves a combination of strategies to control blood sugar levels, manage risk factors, and prevent or delay the progression of the disease. This may involve lifestyle modifications such as weight loss, regular physical activity, and a balanced diet, as well as medications to control blood sugar, cholesterol, and blood pressure levels. In some cases, more advanced treatments such as liver transplantation may be necessary for people with advanced liver disease."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        load_data_model_none('E1172')
        comp_df = load_data_model_none('E1172')
        graph_draw(comp_df,'E1172','Fatty Liver')
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        load_data_model_demo(demo_Cox,'Cox_E1172.pkl','cph_E1172.sav','Fatty Liver')     
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        create_report_demo(demo_Cox,'Cox_E1172.pkl','cph_E1172.sav','E1172')
        st.session_state.stage = False
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        load_data_model_lab(lab_Cox,'Cox_E1172_All.pkl','cph_E1172_All.sav','Fatty Liver')
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:   
        create_report_lab(lab_Cox,'Cox_E1172_All.pkl','cph_E1172_All.sav','E1172')     
    

def All_Comp_page(): 
    t = range(0, 3652) 
    plots = []
    if selected_option2 == "None":
        st.markdown('<p class="my-text">All Complications of Diabetes Mellitus</p>', unsafe_allow_html=True)
        text = "Diabetes is a chronic condition characterized by high levels of glucose (sugar) in the blood. Occurring when the body is unable to produce or effectively use insulin, a hormone that regulates blood sugar levels. There are two main types of diabetes: Type 1 diabetes, which is usually diagnosed in children and young adults, and Type 2 diabetes, which is typically diagnosed in adults and is often associated with lifestyle factors. Diabetes can lead to a number of complications, both chronic and acute. Acute complications can including hypoglycaemia (low blood sugar) or hyperglycaemia (high blood sugar), can cause symptoms such as shakiness, confusion, sweating, and fatigue. Long-term complications can include damage to the eyes, kidneys, nerves, and blood vessels, which can increase the risk of heart disease, stroke, blindness, kidney failure, and amputations. People with diabetes may also be at higher risk for infections and slow wound healing.Management of diabetes typically involves lifestyle changes such as diet and exercise, and in some cases, medication or insulin therapy. Regular monitoring of blood sugar levels is also important to help prevent and manage complications."
        text = "NZTPCD can predict 10 complications of diabetes including; "
        
        my_list = ["E1122 - Diabetic Nephropathy", "E1129 - Kidney Complications AKI", "E1131 - Background Retinopathy", "E1139 - Other Ophthalmic Complications", "E1142 - Diabetic Polyneuropathy", "E1151 -PVD, E1164 – Hypoglycemia", "E1165 - Poor Control – Hyperglycemia", "E1171 - Microvascular and other specified nonvascular complications", "E1172 - Fatty Liver"
]
        #E1122 - Diabetic Nephropathy, E1129 - Kidney Complications AKI, E1131 - Background Retinopathy, E1139 - Other Ophthalmic Complications, E1142 - Diabetic Polyneuropathy, E1151 -PVD, E1164 – Hypoglycemia, E1165 - Poor Control – Hyperglycemia, E1171 - Microvascular and other specified nonvascular complications, E1172 - Fatty Liver."
        st.write(f"<p style='text-align:justify'>{text}</p>", unsafe_allow_html=True)
        for item in my_list:
            st.write(item)
        
      
    elif selected_option2 == "Demographic Details" and not st.session_state.stage > 0:
        gender_value, Maori_value, ethnicity_value, number_input,number_input_age, submit_button_demo  = demographic_form()
        
        data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
        data1 =pd.DataFrame(data1)
        age_diff = (number_input_age - number_input)*365
            
         # Load the Cox models with basic features
        #E1122_demo_model
        with open(E1122_demo_model,'rb') as file1:
        #with open('/Users/anuradhamadurapperuma/Documents/thesis code/Thesis_codes/Trained_Models/CPH_Basic/cph_E1122.sav','rb') as file1:
            cph_basic_E1122_model = pickle.load(file1)
            
        with open(E1129_demo_model,'rb') as file1:
            cph_basic_E1129_model = pickle.load(file1)
        with open(E1131_demo_model,'rb') as file1:
            cph_basic_E1131_model = pickle.load(file1)
        with open(E1139_demo_model,'rb') as file1:
            cph_basic_E1139_model = pickle.load(file1)
        with open(E1142_demo_model,'rb') as file1:
            cph_basic_E1142_model = pickle.load(file1)
        with open(E1151_demo_model,'rb') as file1:
            cph_basic_E1151_model = pickle.load(file1)
        with open(E1164_demo_model,'rb') as file1:
            cph_basic_E1164_model = pickle.load(file1)
        with open(E1165_demo_model,'rb') as file1:
            cph_basic_E1165_model = pickle.load(file1)
        with open(E1171_demo_model,'rb') as file1:
            cph_basic_E1171_model = pickle.load(file1)
        with open(E1172_demo_model,'rb') as file1:
            cph_basic_E1172_model = pickle.load(file1)
        
        # Load the Data frames with basic features
        cph_basic_E1122_data = pd.read_pickle(E1122_demo_data)
        num_rows_E1122 = cph_basic_E1122_data.shape[0] 
        cph_basic_E1129_data = pd.read_pickle(E1129_demo_data)
        num_rows_E1129 = cph_basic_E1129_data.shape[0]
        cph_basic_E1131_data = pd.read_pickle(E1131_demo_data)
        num_rows_E1131 = cph_basic_E1131_data.shape[0]
        cph_basic_E1139_data = pd.read_pickle(E1139_demo_data)
        num_rows_E1139 = cph_basic_E1139_data.shape[0]
        cph_basic_E1142_data = pd.read_pickle(E1142_demo_data)
        num_rows_E1142 = cph_basic_E1142_data.shape[0]
        cph_basic_E1151_data = pd.read_pickle(E1151_demo_data)
        num_rows_E1151 = cph_basic_E1151_data.shape[0]
        cph_basic_E1164_data = pd.read_pickle(E1164_demo_data)
        num_rows_E1164 = cph_basic_E1164_data.shape[0]
        cph_basic_E1165_data = pd.read_pickle(E1165_demo_data)
        num_rows_E1165 = cph_basic_E1165_data.shape[0]
        cph_basic_E1171_data = pd.read_pickle(E1171_demo_data)
        num_rows_E1171 = cph_basic_E1171_data.shape[0]
        cph_basic_E1172_data = pd.read_pickle(E1172_demo_data)
        num_rows_E1172 = cph_basic_E1172_data.shape[0]
        # Survival curve of E1122   
        survival_func_E1122 = cph_basic_E1122_model.predict_survival_function(cph_basic_E1122_data.iloc[0:num_rows_E1122], times=t)
        survival_means_E1122 = survival_func_E1122.mean(axis=1)
        fig1, ax1 = plt.subplots()

        ax1.plot(t,survival_means_E1122,label='Survival curve of E1122')
        ax1.set_title('Survival curves of all complications')
   
        survival_func_E1129 = cph_basic_E1129_model.predict_survival_function(cph_basic_E1129_data.iloc[0:num_rows_E1129], times=t)
        survival_means_E1129 = survival_func_E1129.mean(axis=1)

        ax1.plot(t,survival_means_E1129,label='Survival curve of E1129') 

        # Survival curve of E1131   
        survival_func_E1131 = cph_basic_E1131_model.predict_survival_function(cph_basic_E1131_data.iloc[0:num_rows_E1131], times=t)
        survival_means_E1131 = survival_func_E1131.mean(axis=1)

        ax1.plot(t,survival_means_E1131,label='Survival curve of E1131') 

        # Survival curve of E1139   
        survival_func_E1139 = cph_basic_E1139_model.predict_survival_function(cph_basic_E1139_data.iloc[0:num_rows_E1139], times=t)
        survival_means_E1139 = survival_func_E1139.mean(axis=1)

        ax1.plot(t,survival_means_E1139,label='Survival curve of E1139') 

        # Survival curve of E1142   
        survival_func_E1142 = cph_basic_E1142_model.predict_survival_function(cph_basic_E1142_data.iloc[0:num_rows_E1142], times=t)
        survival_means_E1142 = survival_func_E1142.mean(axis=1)

        ax1.plot(t,survival_means_E1142,label='Survival curve of E1142') 

        # Survival curve of E1151   
        survival_func_E1151 = cph_basic_E1151_model.predict_survival_function(cph_basic_E1151_data.iloc[0:num_rows_E1151], times=t)
        survival_means_E1151 = survival_func_E1151.mean(axis=1)

        ax1.plot(t,survival_means_E1151,label='Survival curve of E1151') 

        # Survival curve of E1164   
        survival_func_E1164 = cph_basic_E1164_model.predict_survival_function(cph_basic_E1164_data.iloc[0:num_rows_E1164], times=t)
        survival_means_E1164 = survival_func_E1164.mean(axis=1)

        ax1.plot(t,survival_means_E1164,label='Survival curve of E1164') 

         # Survival curve of E1165   
        survival_func_E1165 = cph_basic_E1165_model.predict_survival_function(cph_basic_E1165_data.iloc[0:num_rows_E1165], times=t)
        survival_means_E1165 = survival_func_E1165.mean(axis=1)

        ax1.plot(t,survival_means_E1165,label='Survival curve of E1165') 

         # Survival curve of E1171   
        survival_func_E1171 = cph_basic_E1171_model.predict_survival_function(cph_basic_E1171_data.iloc[0:num_rows_E1171], times=t)
        survival_means_E1171 = survival_func_E1171.mean(axis=1)

        ax1.plot(t,survival_means_E1171,label='Survival curve of E1171') 

          # Survival curve of E1172   
        survival_func_E1172 = cph_basic_E1172_model.predict_survival_function(cph_basic_E1172_data.iloc[0:num_rows_E1172], times=t)
        survival_means_E1172 = survival_func_E1172.mean(axis=1)

        ax1.plot(t,survival_means_E1172,label='Survival curve of E1172') 
        ax1.legend()
        st.pyplot(fig1)
        st.write('The survival curve illustrates the changes in the survival rate within ten years. The time frame is in the format of days. The graph shows that the survival at the initial time is nearly 1, which reduces over time to 0. The survival curve can be used to interpret the survival at each time point. The hazard of complications can also be measured using the graph where hazard = 1 - survival.')

        
        if submit_button_demo:
            if number_input_age < number_input:
                st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="my-text">---Prediction results----</p>', unsafe_allow_html=True)
                st.write("<hr>", unsafe_allow_html=True)
            
                st.markdown('<p class="h1">Your prediction results for next 10 years</p>', unsafe_allow_html=True)
                all_comp_demo(cph_basic_E1122_model,cph_basic_E1122_data,survival_means_E1122,data1,age_diff,'E1122')   
                all_comp_demo(cph_basic_E1129_model,cph_basic_E1129_data,survival_means_E1129,data1,age_diff,'E1129')
                all_comp_demo(cph_basic_E1131_model,cph_basic_E1131_data,survival_means_E1131,data1,age_diff,'E1131')
                all_comp_demo(cph_basic_E1139_model,cph_basic_E1139_data,survival_means_E1139,data1,age_diff,'E1139')
                all_comp_demo(cph_basic_E1142_model,cph_basic_E1142_data,survival_means_E1142,data1,age_diff,'E1142')
                all_comp_demo(cph_basic_E1151_model,cph_basic_E1151_data,survival_means_E1151,data1,age_diff,'E1151')
                all_comp_demo(cph_basic_E1164_model,cph_basic_E1164_data,survival_means_E1164,data1,age_diff,'E1164')
                all_comp_demo(cph_basic_E1165_model,cph_basic_E1165_data,survival_means_E1165,data1,age_diff,'E1165')
                all_comp_demo(cph_basic_E1171_model,cph_basic_E1171_data,survival_means_E1171,data1,age_diff,'E1171')
                all_comp_demo(cph_basic_E1172_model,cph_basic_E1172_data,survival_means_E1172,data1,age_diff,'E1172')
         
        
        
    elif selected_option2 == "Demographic Details" and st.session_state.stage > 0:
        name, address, gender_value, Maori_value, ethnicity_value, number_input, number_input_age, submit_button_demo  = demographic_report_form()
        
        data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value]}
        data1 =pd.DataFrame(data1)
        age_diff = (number_input_age - number_input)*365
            
         # Load the Cox models with basic features
        with open(E1122_demo_model,'rb') as file1:
            cph_basic_E1122_model = pickle.load(file1)
            
        with open(E1129_demo_model,'rb') as file1:
            cph_basic_E1129_model = pickle.load(file1)
        with open(E1131_demo_model,'rb') as file1:
            cph_basic_E1131_model = pickle.load(file1)
        with open(E1139_demo_model,'rb') as file1:
            cph_basic_E1139_model = pickle.load(file1)
        with open(E1142_demo_model,'rb') as file1:
            cph_basic_E1142_model = pickle.load(file1)
        with open(E1151_demo_model,'rb') as file1:
            cph_basic_E1151_model = pickle.load(file1)
        with open(E1164_demo_model,'rb') as file1:
            cph_basic_E1164_model = pickle.load(file1)
        with open(E1165_demo_model,'rb') as file1:
            cph_basic_E1165_model = pickle.load(file1)
        with open(E1171_demo_model,'rb') as file1:
            cph_basic_E1171_model = pickle.load(file1)
        with open(E1172_demo_model,'rb') as file1:
            cph_basic_E1172_model = pickle.load(file1)
        
        # Load the Data frames with basic features
        cph_basic_E1122_data = pd.read_pickle(E1122_demo_data)
        num_rows_E1122 = cph_basic_E1122_data.shape[0] 
        cph_basic_E1129_data = pd.read_pickle(E1129_demo_data)
        num_rows_E1129 = cph_basic_E1129_data.shape[0]
        cph_basic_E1131_data = pd.read_pickle(E1131_demo_data)
        num_rows_E1131 = cph_basic_E1131_data.shape[0]
        cph_basic_E1139_data = pd.read_pickle(E1139_demo_data)
        num_rows_E1139 = cph_basic_E1139_data.shape[0]
        cph_basic_E1142_data = pd.read_pickle(E1142_demo_data)
        num_rows_E1142 = cph_basic_E1142_data.shape[0]
        cph_basic_E1151_data = pd.read_pickle(E1151_demo_data)
        num_rows_E1151 = cph_basic_E1151_data.shape[0]
        cph_basic_E1164_data = pd.read_pickle(E1164_demo_data)
        num_rows_E1164 = cph_basic_E1164_data.shape[0]
        cph_basic_E1165_data = pd.read_pickle(E1165_demo_data)
        num_rows_E1165 = cph_basic_E1165_data.shape[0]
        cph_basic_E1171_data = pd.read_pickle(E1171_demo_data)
        num_rows_E1171 = cph_basic_E1171_data.shape[0]
        cph_basic_E1172_data = pd.read_pickle(E1172_demo_data)
        num_rows_E1172 = cph_basic_E1172_data.shape[0]
        
        
        # Survival curve of E1122   
        survival_func_E1122 = cph_basic_E1122_model.predict_survival_function(cph_basic_E1122_data.iloc[0:num_rows_E1122], times=t)
        survival_means_E1122 = survival_func_E1122.mean(axis=1)
        
        survival_func_E1129 = cph_basic_E1129_model.predict_survival_function(cph_basic_E1129_data.iloc[0:num_rows_E1129], times=t)
        survival_means_E1129 = survival_func_E1129.mean(axis=1)
         
        survival_func_E1131 = cph_basic_E1131_model.predict_survival_function(cph_basic_E1131_data.iloc[0:num_rows_E1131], times=t)
        survival_means_E1131 = survival_func_E1131.mean(axis=1)
          
        survival_func_E1139 = cph_basic_E1139_model.predict_survival_function(cph_basic_E1139_data.iloc[0:num_rows_E1139], times=t)
        survival_means_E1139 = survival_func_E1139.mean(axis=1)
         
        survival_func_E1142 = cph_basic_E1142_model.predict_survival_function(cph_basic_E1142_data.iloc[0:num_rows_E1142], times=t)
        survival_means_E1142 = survival_func_E1142.mean(axis=1)
       
        survival_func_E1151 = cph_basic_E1151_model.predict_survival_function(cph_basic_E1151_data.iloc[0:num_rows_E1151], times=t)
        survival_means_E1151 = survival_func_E1151.mean(axis=1)
         
        survival_func_E1164 = cph_basic_E1164_model.predict_survival_function(cph_basic_E1164_data.iloc[0:num_rows_E1164], times=t)
        survival_means_E1164 = survival_func_E1164.mean(axis=1)
         
        survival_func_E1165 = cph_basic_E1165_model.predict_survival_function(cph_basic_E1165_data.iloc[0:num_rows_E1165], times=t)
        survival_means_E1165 = survival_func_E1165.mean(axis=1)
       
        survival_func_E1171 = cph_basic_E1171_model.predict_survival_function(cph_basic_E1171_data.iloc[0:num_rows_E1171], times=t)
        survival_means_E1171 = survival_func_E1171.mean(axis=1)
         
        survival_func_E1172 = cph_basic_E1172_model.predict_survival_function(cph_basic_E1172_data.iloc[0:num_rows_E1172], times=t)
        survival_means_E1172 = survival_func_E1172.mean(axis=1)
        #st.write('The survival curve illustrates the changes in the survival rate within ten years. The time frame is in the format of days. The graph shows that the survival at the initial time is nearly 1, which reduces over time to 0. The survival curve can be used to interpret the survival at each time point. The hazard of complications can also be measured using the graph where hazard = 1 - survival.')

        
        
        if submit_button_demo:
            if number_input_age < number_input:
                st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
            else:

                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter,
                        leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        
        
        # Define the page template with one frame
                #page_template = PageTemplate(id='OneCol', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='Frame1')])
                #doc.addPageTemplates([page_template])
                style_sheet = getSampleStyleSheet()
        # Define styles for the document
                styles = getSampleStyleSheet()
        
        
        # Create a list of flowable elements to be added to the document
                elements = []  
        # Add an image
               # image_path = "Diabetes_cover1.png"
                image = Image(cover_image1, width = inch*8)
        
                elements.append(image)
                style_title = styles['Title']
                style_body = styles['Normal']

                title_style = styles["Title"]
    
        # Define the current date and time
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # Create a ParagraphStyle for the date/time text
                style_datetime = ParagraphStyle(name='Date/time', fontSize=12, textColor=colors.black)
        # Add the title to the document
                header_text = "Report of all complications "
         # Create first large horizontal division
                header_paragraph = Paragraph(header_text, style_sheet["Heading1"])
                elements.append(header_paragraph)
                elements.append(Spacer(1, 20)) 
        # Create a Paragraph for the date/time text
                datetime_text = "Generated on: " + current_date
                datetime_paragraph = Paragraph(datetime_text, style_datetime)
        # Add the date/time text to the document elements
                elements.append(datetime_paragraph)
                elements.append(Spacer(1, 0.25*inch)) 
        

                name_text = "Name: {}".format(name)
                age_text = "Age: {}".format(number_input_age)
                address_text = "Address: {}".format(address)
        
        # Add the user input to the document
                elements.append(Paragraph(name_text, style_body))
                elements.append(Paragraph(age_text, style_body))
                elements.append(Paragraph(address_text, style_body))
                elements.append(Spacer(1, 0.25*inch))
        
                header_text_E1122 = "Results of E1122 "
         # Create first large horizontal division
                header_paragraph_E1122 = Paragraph(header_text_E1122, style_sheet["Heading1"])
                elements.append(header_paragraph_E1122)
                
                
                time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    #file_path1 = f"{location}/{model_name}"
    #with open(file_path1,'rb') as file1:
     #   cph_basic_model = pickle.load(file1)
        
       # all_comp_demo(cph_basic_E1122_model,cph_basic_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
                num_rows = cph_basic_E1122_data.shape[0]        
                t = range(0, 3652)  
                survival_func = cph_basic_E1122_model.predict_survival_function(cph_basic_E1122_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1122_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')

                c = cph_basic_E1122_model.predict_survival_function(data1,times=time_points)
      
 
    #age_diff = (number_input_age - number_input)*365
      
                partial_hazard = cph_basic_E1122_model.predict_partial_hazard(data1) 
  
                survival_prob_day = cph_basic_E1122_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1122_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
        
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
                #st.table(survival_from_age_today)

                
                plt.legend()  
                plt.plot()
        
        # Save the plot as an image
                plt.savefig('assets/demo_report_fig_E1122.png')

                style_body = styles['Normal']
     
    
                table_data_E1122 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1122 = Table(table_data_E1122)
                table_E1122.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1122.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1122 = "demo_report_fig_E1122.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1122 = Image(image_path_E1122, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1122, image_E1122]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                elements.append(spacer)
                elements.append(spacer)
                
                header_text_E1129 = "Results of E1129 "
         # Create first large horizontal division
                header_paragraph_E1129 = Paragraph(header_text_E1129, style_sheet["Heading1"])
                elements.append(header_paragraph_E1129)
                
                
                time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    #file_path1 = f"{location}/{model_name}"
    #with open(file_path1,'rb') as file1:
     #   cph_basic_model = pickle.load(file1)
        
       # all_comp_demo(cph_basic_E1122_model,cph_basic_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
                num_rows = cph_basic_E1129_data.shape[0]        
                t = range(0, 3652)  
                survival_func = cph_basic_E1129_model.predict_survival_function(cph_basic_E1129_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1129_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')

                c = cph_basic_E1129_model.predict_survival_function(data1,times=time_points)
      
 
    #age_diff = (number_input_age - number_input)*365
      
                partial_hazard = cph_basic_E1129_model.predict_partial_hazard(data1) 
  
                survival_prob_day = cph_basic_E1129_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1129_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
        
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
       # survival_from_age_today['survival_from_age_today.iloc[0, 0]']= 'Now'
                #st.table(survival_from_age_today)

                
                plt.legend()
                #pyplot(plt)
                plt.plot()

        
        # Save the plot as an image
                plt.savefig('assets/demo_report_fig_E1129.png')

                style_body = styles['Normal']
     
    
                table_data_E1129 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1129 = Table(table_data_E1129)
                table_E1129.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1129.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1129 = "demo_report_fig_E1129.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1129 = Image(image_path_E1129, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1129, image_E1129]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1131
        
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                elements.append(spacer)
                elements.append(spacer)
                
                header_text_E1131 = "Results of E1131 "
         # Create first large horizontal division
                header_paragraph_E1131 = Paragraph(header_text_E1131, style_sheet["Heading1"])
                elements.append(header_paragraph_E1131)
                num_rows = cph_basic_E1131_data.shape[0]          
                survival_func = cph_basic_E1131_model.predict_survival_function(cph_basic_E1131_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1131_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1131_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1131_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1131_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1131_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1131.png')
                style_body = styles['Normal']
     
                table_data_E1131 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1131 = Table(table_data_E1131)
                table_E1131.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1131.hAlign = 'LEFT'   

         # Add an image
                #image_path_E1131 = "demo_report_fig_E1131.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1131 = Image(image_path_E1131, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1131, image_E1131]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1139
        
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)

                
                header_text_E1139 = "Results of E1139 "
         # Create first large horizontal division
                header_paragraph_E1139 = Paragraph(header_text_E1139, style_sheet["Heading1"])
                elements.append(header_paragraph_E1139)
                num_rows = cph_basic_E1139_data.shape[0]          
                survival_func = cph_basic_E1139_model.predict_survival_function(cph_basic_E1139_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1139_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1139_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1139_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1139_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1139_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1139.png')
                style_body = styles['Normal']
     
                table_data_E1139 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1139 = Table(table_data_E1139)
                table_E1139.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1139.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1139 = "demo_report_fig_E1139.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1139 = Image(image_path_E1139, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1139, image_E1139]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)
        # E1142
        
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                elements.append(spacer)

                
                header_text_E1142 = "Results of E1142 "
         # Create first large horizontal division
                header_paragraph_E1142 = Paragraph(header_text_E1142, style_sheet["Heading1"])
                elements.append(header_paragraph_E1142)
                num_rows = cph_basic_E1142_data.shape[0]          
                survival_func = cph_basic_E1142_model.predict_survival_function(cph_basic_E1142_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1142_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1142_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1142_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1142_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1142_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1142.png')
                style_body = styles['Normal']
     
                table_data_E1142 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1142 = Table(table_data_E1142)
                table_E1142.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1142.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1142 = "demo_report_fig_E1142.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1142 = Image(image_path_E1142, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1142, image_E1142]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)
        # E1151
                
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
             
                elements.append(spacer)
                
                header_text_E1151 = "Results of E1151 "
         # Create first large horizontal division
                header_paragraph_E1151 = Paragraph(header_text_E1151, style_sheet["Heading1"])
                elements.append(header_paragraph_E1151)
                num_rows = cph_basic_E1151_data.shape[0]          
                survival_func = cph_basic_E1151_model.predict_survival_function(cph_basic_E1151_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1151_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1151_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1151_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1151_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1151_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1151.png')
                style_body = styles['Normal']
     
                table_data_E1151 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1151 = Table(table_data_E1151)
                table_E1151.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1151.hAlign = 'LEFT'   

         # Add an image
                #image_path_E1151 = "demo_report_fig_E1151.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1151 = Image(image_path_E1151, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1151, image_E1151]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1164
                
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                
                elements.append(spacer)
                
                header_text_E1164 = "Results of E1164 "
         # Create first large horizontal division
                header_paragraph_E1164 = Paragraph(header_text_E1164, style_sheet["Heading1"])
                elements.append(header_paragraph_E1164)
                num_rows = cph_basic_E1164_data.shape[0]          
                survival_func = cph_basic_E1164_model.predict_survival_function(cph_basic_E1164_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1164_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1164_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1164_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1164_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1164_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1164.png')
                style_body = styles['Normal']
     
                table_data_E1164 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1164 = Table(table_data_E1164)
                table_E1164.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1164.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1164 = "demo_report_fig_E1164.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1164 = Image(image_path_E1164, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1164, image_E1164]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1165
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                
                elements.append(spacer)
                
                header_text_E1165 = "Results of E1165 "
         # Create first large horizontal division
                header_paragraph_E1165 = Paragraph(header_text_E1165, style_sheet["Heading1"])
                elements.append(header_paragraph_E1165)
                num_rows = cph_basic_E1165_data.shape[0]          
                survival_func = cph_basic_E1165_model.predict_survival_function(cph_basic_E1165_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1165_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1165_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1165_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1165_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1165_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1165.png')
                style_body = styles['Normal']
     
                table_data_E1165 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1165 = Table(table_data_E1165)
                table_E1165.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1165.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1165 = "demo_report_fig_E1165.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1165 = Image(image_path_E1165, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1165, image_E1165]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1171
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                
                elements.append(spacer)
                
                header_text_E1171 = "Results of E1171 "
         # Create first large horizontal division
                header_paragraph_E1171 = Paragraph(header_text_E1171, style_sheet["Heading1"])
                elements.append(header_paragraph_E1171)
                num_rows = cph_basic_E1171_data.shape[0]          
                survival_func = cph_basic_E1171_model.predict_survival_function(cph_basic_E1171_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1171_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1171_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1171_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1171_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1171_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1171.png')
                style_body = styles['Normal']
     
                table_data_E1171 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1171 = Table(table_data_E1171)
                table_E1171.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1171.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1171 = "demo_report_fig_E1171.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1171 = Image(image_path_E1171, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1171, image_E1171]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)

        # E1172
                spacer = Spacer(1, 0.25*inch)
                elements.append(spacer)
                elements.append(spacer)
                elements.append(spacer)
                
                header_text_E1172 = "Results of E1172 "
         # Create first large horizontal division
                header_paragraph_E1172 = Paragraph(header_text_E1172, style_sheet["Heading1"])
                elements.append(header_paragraph_E1172)
                num_rows = cph_basic_E1172_data.shape[0]          
                survival_func = cph_basic_E1172_model.predict_survival_function(cph_basic_E1172_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)
                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')
                b = cph_basic_E1172_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                c = cph_basic_E1172_model.predict_survival_function(data1,times=time_points)
                partial_hazard = cph_basic_E1172_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_basic_E1172_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_basic_E1172_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
                old_name = survival_from_age_today.columns[0]
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
                plt.legend()
                plt.plot()
                plt.savefig('assets/demo_report_fig_E1172.png')
                style_body = styles['Normal']
     
                table_data_E1172 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1172 = Table(table_data_E1172)
                table_E1172.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1172.hAlign = 'LEFT'   

         # Add an image
               # image_path_E1172 = "demo_report_fig_E1172.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1172 = Image(image_path_E1172, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
      
                elements.append(Table([[header_table, header_graph]], colWidths=[4 * inch, 4 * inch,], rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1172, image_E1172]], colWidths=[4 * inch, 4 * inch,], rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)

                spacer = Spacer(1, 0.25*inch)



       
        # Build the document by adding the elements
                doc.build(elements)
                
                now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                report_filename = f"report_lab_details_{now}.pdf"
        
         # File name of the report
        #report_name = "example_report.pdf"
        
        # Move the buffer's pointer to the beginning of the buffer
                buffer.seek(0)
        # generate the download link
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download Report</a>'

        # display the link using st.markdown
                st.markdown(href, unsafe_allow_html=True)
        # Set up the download link
                #href = f'<a href="data:application/pdf;base64,{base64.b64encode(buffer.read()).decode()}">Download Report</a>'
                #st.markdown(href, unsafe_allow_html=True)
     
       # doc.build([header_paragraph,datetime_paragraph,name_paragraph,age_paragraph,address_paragraph,[column1, column2]])
        # Display a link to the generated PDF
               # st.success("PDF generated! Download the PDF [here](/Users/anuradhamadurapperuma/Documents/report_demo1.pdf).")

        st.session_state.stage = False
            
            
        ########### Lab values      
                
    elif selected_option2 == "Lab Vales" and not st.session_state.stage > 0:
        gender_value, Maori_value, ethnicity_value, number_input,number_input_age, hba1c, cholesterol, triglycerides, hdl,ldl,egfr,submit_button_lab = lab_form()
        data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value], 'HbA1c_ResultValue': hba1c,'Cholesterol_ResultValue': cholesterol,'Triglyceride_ResultValue': triglycerides ,'HDL_ResultValue': hdl,'LDL_ResultValue':ldl,'EGFR_ResultValue':egfr}
        data1 =pd.DataFrame(data1)
        age_diff = (number_input_age - number_input)*365
            
         # Load the Cox models with All features
        with open(E1122_lab_model,'rb') as file1:
            cph_all_E1122_model = pickle.load(file1)
            
        with open(E1129_lab_model,'rb') as file1:
            cph_all_E1129_model = pickle.load(file1)
        with open(E1131_lab_model,'rb') as file1:
            cph_all_E1131_model = pickle.load(file1)
        with open(E1139_lab_model,'rb') as file1:
            cph_all_E1139_model = pickle.load(file1)
        with open(E1142_lab_model,'rb') as file1:
            cph_all_E1142_model = pickle.load(file1)
        with open(E1151_lab_model,'rb') as file1:
            cph_all_E1151_model = pickle.load(file1)
        with open(E1164_lab_model,'rb') as file1:
            cph_all_E1164_model = pickle.load(file1)
        with open(E1165_lab_model,'rb') as file1:
            cph_all_E1165_model = pickle.load(file1)
        with open(E1171_lab_model,'rb') as file1:
            cph_all_E1171_model = pickle.load(file1)
        with open(E1172_lab_model,'rb') as file1:
            cph_all_E1172_model = pickle.load(file1)
        
        # Load the Data frames with basic features
        cph_all_E1122_data = pd.read_pickle(E1122_lab_data)
        num_rows_E1122 = cph_all_E1122_data.shape[0] 
        cph_all_E1129_data = pd.read_pickle(E1129_lab_data)
        num_rows_E1129 = cph_all_E1129_data.shape[0]
        cph_all_E1131_data = pd.read_pickle(E1131_lab_data)
        num_rows_E1131 = cph_all_E1131_data.shape[0]
        cph_all_E1139_data = pd.read_pickle(E1139_lab_data)
        num_rows_E1139 = cph_all_E1139_data.shape[0]
        cph_all_E1142_data = pd.read_pickle(E1142_lab_data)
        num_rows_E1142 = cph_all_E1142_data.shape[0]
        cph_all_E1151_data = pd.read_pickle(E1151_lab_data)
        num_rows_E1151 = cph_all_E1151_data.shape[0]
        cph_all_E1164_data = pd.read_pickle(E1164_lab_data)
        num_rows_E1164 = cph_all_E1164_data.shape[0]
        cph_all_E1165_data = pd.read_pickle(E1165_lab_data)
        num_rows_E1165 = cph_all_E1165_data.shape[0]
        cph_all_E1171_data = pd.read_pickle(E1171_lab_data)
        num_rows_E1171 = cph_all_E1171_data.shape[0]
        cph_all_E1172_data = pd.read_pickle(E1172_lab_data)
        num_rows_E1172 = cph_all_E1172_data.shape[0]
        
        # Survival curve of E1122   
        
        survival_func_E1122 = cph_all_E1122_model.predict_survival_function(cph_all_E1122_data.iloc[0:num_rows_E1122], times=t)
        survival_means_E1122 = survival_func_E1122.mean(axis=1)
        fig1, ax1 = plt.subplots()
        ax1.plot(t,survival_means_E1122,label='Survival curve of E1122')
        ax1.set_title('Survival curves of all complications')
         # Survival curve of E1129   
        survival_func_E1129 = cph_all_E1129_model.predict_survival_function(cph_all_E1129_data.iloc[0:num_rows_E1129], times=t)
        survival_means_E1129 = survival_func_E1129.mean(axis=1)
        ax1.plot(t,survival_means_E1129,label='Survival curve of E1129') 
        # Survival curve of E1131   
        survival_func_E1131 = cph_all_E1131_model.predict_survival_function(cph_all_E1131_data.iloc[0:num_rows_E1131], times=t)
        survival_means_E1131 = survival_func_E1131.mean(axis=1)
        ax1.plot(t,survival_means_E1131,label='Survival curve of E1131') 
        # Survival curve of E1139   
        survival_func_E1139 = cph_all_E1139_model.predict_survival_function(cph_all_E1139_data.iloc[0:num_rows_E1139], times=t)
        survival_means_E1139 = survival_func_E1139.mean(axis=1)
        ax1.plot(t,survival_means_E1139,label='Survival curve of E1139') 
        # Survival curve of E1142   
        survival_func_E1142 = cph_all_E1142_model.predict_survival_function(cph_all_E1142_data.iloc[0:num_rows_E1142], times=t)
        survival_means_E1142 = survival_func_E1142.mean(axis=1)
        ax1.plot(t,survival_means_E1142,label='Survival curve of E1142') 
        # Survival curve of E1151   
        survival_func_E1151 = cph_all_E1151_model.predict_survival_function(cph_all_E1151_data.iloc[0:num_rows_E1151], times=t)
        survival_means_E1151 = survival_func_E1151.mean(axis=1)
        ax1.plot(t,survival_means_E1151,label='Survival curve of E1151') 
        # Survival curve of E1164   
        survival_func_E1164 = cph_all_E1164_model.predict_survival_function(cph_all_E1164_data.iloc[0:num_rows_E1164], times=t)
        survival_means_E1164 = survival_func_E1164.mean(axis=1)
        ax1.plot(t,survival_means_E1164,label='Survival curve of E1164') 
         # Survival curve of E1165   
        survival_func_E1165 = cph_all_E1165_model.predict_survival_function(cph_all_E1165_data.iloc[0:num_rows_E1165], times=t)
        survival_means_E1165 = survival_func_E1165.mean(axis=1)
        ax1.plot(t,survival_means_E1165,label='Survival curve of E1165') 
        # Survival curve of E1171   
        survival_func_E1171 = cph_all_E1171_model.predict_survival_function(cph_all_E1171_data.iloc[0:num_rows_E1171], times=t)
        survival_means_E1171 = survival_func_E1171.mean(axis=1)
        ax1.plot(t,survival_means_E1171,label='Survival curve of E1171') 
        # Survival curve of E1172   
        survival_func_E1172 = cph_all_E1172_model.predict_survival_function(cph_all_E1172_data.iloc[0:num_rows_E1172], times=t)
        survival_means_E1172 = survival_func_E1172.mean(axis=1)
        ax1.plot(t,survival_means_E1172,label='Survival curve of E1172') 
        ax1.legend()
        st.pyplot(fig1)
        st.write('The survival curve illustrates the changes in the survival rate within ten years. The time frame is in the format of days. The graph shows that the survival at the initial time is nearly 1, which reduces over time to 0. The survival curve can be used to interpret the survival at each time point. The hazard of complications can also be measured using the graph where hazard = 1 - survival.')

        if submit_button_lab:
            if number_input_age < number_input:
                st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="my-text">---Prediction results----</p>', unsafe_allow_html=True)
                st.write("<hr>", unsafe_allow_html=True)
            
                st.markdown('<p class="h1">Your prediction results for next 10 years</p>', unsafe_allow_html=True)
                all_comp_demo(cph_all_E1122_model,cph_all_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
                all_comp_demo(cph_all_E1129_model,cph_all_E1129_data,survival_means_E1129,data1,age_diff,'E1129')
                all_comp_demo(cph_all_E1131_model,cph_all_E1131_data,survival_means_E1131,data1,age_diff,'E1131')
                all_comp_demo(cph_all_E1139_model,cph_all_E1139_data,survival_means_E1139,data1,age_diff,'E1139')
                all_comp_demo(cph_all_E1142_model,cph_all_E1142_data,survival_means_E1142,data1,age_diff,'E1142')
                all_comp_demo(cph_all_E1151_model,cph_all_E1151_data,survival_means_E1151,data1,age_diff,'E1151')
                all_comp_demo(cph_all_E1164_model,cph_all_E1164_data,survival_means_E1164,data1,age_diff,'E1164')
                all_comp_demo(cph_all_E1165_model,cph_all_E1165_data,survival_means_E1165,data1,age_diff,'E1165')
                all_comp_demo(cph_all_E1171_model,cph_all_E1171_data,survival_means_E1171,data1,age_diff,'E1171')
                all_comp_demo(cph_all_E1172_model,cph_all_E1172_data,survival_means_E1172,data1,age_diff,'E1172')
                
                

        
    elif selected_option2 == "Lab Vales" and st.session_state.stage > 0:
        name, address, gender_value, Maori_value, ethnicity_value, number_input,number_input_age, hba1c, cholesterol, triglycerides, hdl,ldl,egfr,submit_button_lab = lab_report_form()
        data1 = {'Patient_Age_OnDiagnosis': [number_input],'Patient_Ethnicity': [ethnicity_value],'Patient_Maori_NonMaori': [Maori_value],'Patient_Gender': [gender_value], 'HbA1c_ResultValue': hba1c,'Cholesterol_ResultValue': cholesterol,'Triglyceride_ResultValue': triglycerides ,'HDL_ResultValue': hdl,'LDL_ResultValue':ldl,'EGFR_ResultValue':egfr}
        data1 =pd.DataFrame(data1)
        age_diff = (number_input_age - number_input)*365
            
         # Load the Cox models with All features
        with open(E1122_lab_model,'rb') as file1:
            cph_all_E1122_model = pickle.load(file1)
            
        with open(E1129_lab_model,'rb') as file1:
            cph_all_E1129_model = pickle.load(file1)
        with open(E1131_lab_model,'rb') as file1:
            cph_all_E1131_model = pickle.load(file1)
        with open(E1139_lab_model,'rb') as file1:
            cph_all_E1139_model = pickle.load(file1)
        with open(E1142_lab_model,'rb') as file1:
            cph_all_E1142_model = pickle.load(file1)
        with open(E1151_lab_model,'rb') as file1:
            cph_all_E1151_model = pickle.load(file1)
        with open(E1164_lab_model,'rb') as file1:
            cph_all_E1164_model = pickle.load(file1)
        with open(E1165_lab_model,'rb') as file1:
            cph_all_E1165_model = pickle.load(file1)
        with open(E1171_lab_model,'rb') as file1:
            cph_all_E1171_model = pickle.load(file1)
        with open(E1172_lab_model,'rb') as file1:
            cph_all_E1172_model = pickle.load(file1)
        
        # Load the Data frames with basic features
        cph_all_E1122_data = pd.read_pickle(E1122_lab_data)
        num_rows_E1122 = cph_all_E1122_data.shape[0] 
        cph_all_E1129_data = pd.read_pickle(E1129_lab_data)
        num_rows_E1129 = cph_all_E1129_data.shape[0]
        cph_all_E1131_data = pd.read_pickle(E1131_lab_data)
        num_rows_E1131 = cph_all_E1131_data.shape[0]
        cph_all_E1139_data = pd.read_pickle(E1139_lab_data)
        num_rows_E1139 = cph_all_E1139_data.shape[0]
        cph_all_E1142_data = pd.read_pickle(E1142_lab_data)
        num_rows_E1142 = cph_all_E1142_data.shape[0]
        cph_all_E1151_data = pd.read_pickle(E1151_lab_data)
        num_rows_E1151 = cph_all_E1151_data.shape[0]
        cph_all_E1164_data = pd.read_pickle(E1164_lab_data)
        num_rows_E1164 = cph_all_E1164_data.shape[0]
        cph_all_E1165_data = pd.read_pickle(E1165_lab_data)
        num_rows_E1165 = cph_all_E1165_data.shape[0]
        cph_all_E1171_data = pd.read_pickle(E1171_lab_data)
        num_rows_E1171 = cph_all_E1171_data.shape[0]
        cph_all_E1172_data = pd.read_pickle(E1172_lab_data)
        num_rows_E1172 = cph_all_E1172_data.shape[0]
        
        # Survival curve of E1122   
        
        survival_func_E1122 = cph_all_E1122_model.predict_survival_function(cph_all_E1122_data.iloc[0:num_rows_E1122], times=t)
        survival_means_E1122 = survival_func_E1122.mean(axis=1)
        #fig1, ax1 = plt.subplots()
        #ax1.plot(t,survival_means_E1122,label='Survival curve of E1122')
        #ax1.set_title('Survival curves of all complications')
         # Survival curve of E1129   
        survival_func_E1129 = cph_all_E1129_model.predict_survival_function(cph_all_E1129_data.iloc[0:num_rows_E1129], times=t)
        survival_means_E1129 = survival_func_E1129.mean(axis=1)
        #ax1.plot(t,survival_means_E1129,label='Survival curve of E1129') 
        # Survival curve of E1131   
        survival_func_E1131 = cph_all_E1131_model.predict_survival_function(cph_all_E1131_data.iloc[0:num_rows_E1131], times=t)
        survival_means_E1131 = survival_func_E1131.mean(axis=1)
        #ax1.plot(t,survival_means_E1131,label='Survival curve of E1131') 
        # Survival curve of E1139   
        survival_func_E1139 = cph_all_E1139_model.predict_survival_function(cph_all_E1139_data.iloc[0:num_rows_E1139], times=t)
        survival_means_E1139 = survival_func_E1139.mean(axis=1)
        #ax1.plot(t,survival_means_E1139,label='Survival curve of E1139') 
        # Survival curve of E1142   
        survival_func_E1142 = cph_all_E1142_model.predict_survival_function(cph_all_E1142_data.iloc[0:num_rows_E1142], times=t)
        survival_means_E1142 = survival_func_E1142.mean(axis=1)
        #ax1.plot(t,survival_means_E1142,label='Survival curve of E1142') 
        # Survival curve of E1151   
        survival_func_E1151 = cph_all_E1151_model.predict_survival_function(cph_all_E1151_data.iloc[0:num_rows_E1151], times=t)
        survival_means_E1151 = survival_func_E1151.mean(axis=1)
        #ax1.plot(t,survival_means_E1151,label='Survival curve of E1151') 
        # Survival curve of E1164   
        survival_func_E1164 = cph_all_E1164_model.predict_survival_function(cph_all_E1164_data.iloc[0:num_rows_E1164], times=t)
        survival_means_E1164 = survival_func_E1164.mean(axis=1)
        #ax1.plot(t,survival_means_E1164,label='Survival curve of E1164') 
         # Survival curve of E1165   
        survival_func_E1165 = cph_all_E1165_model.predict_survival_function(cph_all_E1165_data.iloc[0:num_rows_E1165], times=t)
        survival_means_E1165 = survival_func_E1165.mean(axis=1)
        #ax1.plot(t,survival_means_E1165,label='Survival curve of E1165') 
        # Survival curve of E1171   
        survival_func_E1171 = cph_all_E1171_model.predict_survival_function(cph_all_E1171_data.iloc[0:num_rows_E1171], times=t)
        survival_means_E1171 = survival_func_E1171.mean(axis=1)
        #ax1.plot(t,survival_means_E1171,label='Survival curve of E1171') 
        # Survival curve of E1172   
        survival_func_E1172 = cph_all_E1172_model.predict_survival_function(cph_all_E1172_data.iloc[0:num_rows_E1172], times=t)
        survival_means_E1172 = survival_func_E1172.mean(axis=1)
        #ax1.plot(t,survival_means_E1172,label='Survival curve of E1172') 
        #ax1.legend()
        #st.pyplot(fig1)

        if submit_button_lab:
            if number_input_age < number_input:
                st.markdown('<p class="warning-text">Your current age should be higher than the age at diagnosis of diabetes.Enter valid age values...</p>', unsafe_allow_html=True)
            else:

                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter,
                        leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        
        
        # Define the page template with one frame
                page_template = PageTemplate(id='OneCol', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='Frame1')])
                doc.addPageTemplates([page_template])
                style_sheet = getSampleStyleSheet()
        # Define styles for the document
                styles = getSampleStyleSheet()
        
        
        # Create a list of flowable elements to be added to the document
                elements = []  
        # Add an image
                #image_path = "Diabetes_cover1.png"
                image = Image(cover_image1, width = inch*8)
        
                elements.append(image)
                style_title = styles['Title']
                style_body = styles['Normal']

                title_style = styles["Title"]
    
        # Define the current date and time
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d %H:%M:%S")

        # Create a ParagraphStyle for the date/time text
                style_datetime = ParagraphStyle(name='Date/time', fontSize=12, textColor=colors.black)
        # Add the title to the document
                header_text = "Report of all complications with laboratory values "
         # Create first large horizontal division
                header_paragraph = Paragraph(header_text, style_sheet["Heading1"])
                elements.append(header_paragraph)
                elements.append(Spacer(1, 20)) 
        # Create a Paragraph for the date/time text
                datetime_text = "Generated on: " + current_date
                datetime_paragraph = Paragraph(datetime_text, style_datetime)
        # Add the date/time text to the document elements
                elements.append(datetime_paragraph)
                elements.append(Spacer(1, 0.25*inch)) 
        

                name_text = "Name: {}".format(name)
                age_text = "Age: {}".format(number_input_age)
                address_text = "Address: {}".format(address)
        
        # Add the user input to the document
                elements.append(Paragraph(name_text, style_body))
                elements.append(Paragraph(age_text, style_body))
                elements.append(Paragraph(address_text, style_body))
                elements.append(Spacer(1, 0.25*inch))
        
        # E1122
                header_text_E1122 = "Results of E1122 "
         # Create first large horizontal division
                header_paragraph_E1122 = Paragraph(header_text_E1122, style_sheet["Heading1"])
                elements.append(header_paragraph_E1122)
                
                
                time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    #file_path1 = f"{location}/{model_name}"
    #with open(file_path1,'rb') as file1:
     #   cph_basic_model = pickle.load(file1)
        
       # all_comp_demo(cph_basic_E1122_model,cph_basic_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
   # all_comp_demo(cph_all_E1122_model,cph_all_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
                num_rows = cph_all_E1122_data.shape[0]        
                t = range(0, 3652)  
                survival_func = cph_all_E1122_model.predict_survival_function(cph_all_E1122_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1122_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
           # plt.legend()
           # st.pyplot(plt)
        
        
                age_diff = (number_input_age - number_input)*365
      
                partial_hazard = cph_all_E1122_model.predict_partial_hazard(data1) 
        
       # hazard = partial_hazard * cph_basic_model.baseline_hazard_(time_point)
        
        #baseline_hazard = cph_basic_model.baseline_hazard_
        #st.write(baseline_hazard)
        #hazard = partial_hazard * baseline_hazard[time_point]

        # Print the hazard
        #print("Hazard at time {}:".format(time_point))
        #print(hazard)
        
        #baseline_hazard = cph_basic_model.baseline_hazard_
        #hazard = partial_hazard * baseline_hazard[rounded_time_points]

        
                survival_prob_day = cph_all_E1122_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1122_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
        #survival_from_age_today['']  
        #survival_from_age_today = survival_from_age_today.rename_axis('Time')
        
        #survival_from_age_today = survival_from_age_today.rename_axis('Time').reset_index()
        
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1122.png')
        
        
        
                table_data_E1122 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1122 = Table(table_data_E1122)
                table_E1122.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1122.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.5)
                elements.append(spacer)


         # Add an image
               # image_path_E1122 = "lab_report_fig_E1122.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1122 = Image(image_path_E1122_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.2 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1122, image_E1122]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
                elements.append(spacer)
            
                
                 # E1129
                header_text_E1129 = "Results of E1129 "
         # Create first large horizontal division
                header_paragraph_E1129 = Paragraph(header_text_E1129, style_sheet["Heading1"])
                elements.append(header_paragraph_E1129)
                
                
                time_points = [365, 730, 1095, 1461, 1825, 2191, 2556, 2922, 3287, 3652]
    #file_path1 = f"{location}/{model_name}"
    #with open(file_path1,'rb') as file1:
     #   cph_basic_model = pickle.load(file1)
        
       # all_comp_demo(cph_basic_E1122_model,cph_basic_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
   # all_comp_demo(cph_all_E1122_model,cph_all_E1122_data,survival_means_E1122,data1,age_diff,'E1122')
                num_rows = cph_all_E1122_data.shape[0]        
                t = range(0, 3652)  
                survival_func = cph_all_E1129_model.predict_survival_function(cph_all_E1129_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1129_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
           # plt.legend()
           # st.pyplot(plt)
        
        
                age_diff = (number_input_age - number_input)*365
      
                partial_hazard = cph_all_E1129_model.predict_partial_hazard(data1) 

                survival_prob_day = cph_all_E1129_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1129_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100

        
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1129.png')
        
        
        
                table_data_E1129 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1129 = Table(table_data_E1129)
                table_E1129.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1129.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
                #image_path_E1129 = "lab_report_fig_E1129.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1129 = Image(image_path_E1129_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.2 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1129, image_E1129]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
        #elements.append(image1)
                elements.append(spacer)
                elements.append(spacer)
                # E1131
                header_text_E1131 = "Results of E1131 "
         # Create first large horizontal division
                header_paragraph_E1131 = Paragraph(header_text_E1131, style_sheet["Heading1"])
                elements.append(header_paragraph_E1131)
                num_rows = cph_all_E1131_data.shape[0]        
                survival_func = cph_all_E1131_model.predict_survival_function(cph_all_E1131_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1131_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
      
                partial_hazard = cph_all_E1131_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1131_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1131_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1131.png')
        
        
        
                table_data_E1131 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
        
        
                table_E1131 = Table(table_data_E1131)
                table_E1131.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1131.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
                #image_path_E1131 = "lab_report_fig_E1131.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1131 = Image(image_path_E1131_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1131, image_E1131]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
                elements.append(spacer)
                elements.append(spacer)
        
         # E1139
                header_text_E1139 = "Results of E1139 "
         # Create first large horizontal division
                header_paragraph_E1139 = Paragraph(header_text_E1139, style_sheet["Heading1"])
                elements.append(header_paragraph_E1139)
                num_rows = cph_all_E1139_data.shape[0]          
                survival_func = cph_all_E1139_model.predict_survival_function(cph_all_E1139_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1139_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1139_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1139_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1139_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1139.png')
        
                table_data_E1139 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1139 = Table(table_data_E1139)
                table_E1139.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1139.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)
              


         # Add an image
               # image_path_E1139 = "lab_report_fig_E1139.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1139 = Image(image_path_E1139_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1139, image_E1139]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
                
                elements.append(spacer)
                elements.append(spacer)
        
        # E1142
                header_text_E1142 = "Results of E1142 "
         # Create first large horizontal division
                header_paragraph_E1142 = Paragraph(header_text_E1142, style_sheet["Heading1"])
                elements.append(header_paragraph_E1142)
                num_rows = cph_all_E1142_data.shape[0]          
                survival_func = cph_all_E1142_model.predict_survival_function(cph_all_E1142_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1142_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1142_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1142_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1142_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1142.png')
        
                table_data_E1142 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1142 = Table(table_data_E1142)
                table_E1142.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1142.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
               # image_path_E1142 = "lab_report_fig_E1142.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1142 = Image(image_path_E1142_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1142, image_E1142]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
        
                elements.append(spacer)
                elements.append(spacer)
        
         # E1151
                header_text_E1151 = "Results of E1151 "
         # Create first large horizontal division
                header_paragraph_E1151 = Paragraph(header_text_E1151, style_sheet["Heading1"])
                elements.append(header_paragraph_E1151)
                num_rows = cph_all_E1151_data.shape[0]          
                survival_func = cph_all_E1151_model.predict_survival_function(cph_all_E1151_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1151_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1151_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1151_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1151_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1151.png')
        
                table_data_E1151 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1151 = Table(table_data_E1151)
                table_E1151.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1151.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
               # image_path_E1151 = "lab_report_fig_E1151.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1151 = Image(image_path_E1151_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1151, image_E1151]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
        
            
                elements.append(spacer)
                elements.append(spacer)
               
            
            # E1164
                header_text_E1164 = "Results of E1164 "
         # Create first large horizontal division
                header_paragraph_E1164 = Paragraph(header_text_E1164, style_sheet["Heading1"])
                elements.append(header_paragraph_E1164)
                num_rows = cph_all_E1164_data.shape[0]          
                survival_func = cph_all_E1164_model.predict_survival_function(cph_all_E1164_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1164_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1164_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1164_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1164_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1164.png')
        
                table_data_E1164 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1164 = Table(table_data_E1164)
                table_E1164.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1164.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
                #image_path_E1164 = "lab_report_fig_E1164.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1164 = Image(image_path_E1164_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1164, image_E1164]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
                elements.append(spacer)
                #elements.append(spacer)
        
        # E1165
                header_text_E1165 = "Results of E1165 "
         # Create first large horizontal division
                header_paragraph_E1165 = Paragraph(header_text_E1165, style_sheet["Heading1"])
                elements.append(header_paragraph_E1165)
                num_rows = cph_all_E1165_data.shape[0]          
                survival_func = cph_all_E1165_model.predict_survival_function(cph_all_E1165_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1165_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1165_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1165_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1165_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1165.png')
        
                table_data_E1165 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1165 = Table(table_data_E1165)
                table_E1165.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1165.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
               # image_path_E1165 = "lab_report_fig_E1165.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1165 = Image(image_path_E1165_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1165, image_E1165]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
                elements.append(spacer)
                elements.append(spacer)
                
        # E1171
                header_text_E1171 = "Results of E1171 "
         # Create first large horizontal division
                header_paragraph_E1171 = Paragraph(header_text_E1171, style_sheet["Heading1"])
                elements.append(header_paragraph_E1171)
                num_rows = cph_all_E1171_data.shape[0]          
                survival_func = cph_all_E1171_model.predict_survival_function(cph_all_E1171_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1171_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1171_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1171_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1171_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1171.png')
        
                table_data_E1171 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1171 = Table(table_data_E1171)
                table_E1171.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1171.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
               # image_path_E1171 = "lab_report_fig_E1171.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1171 = Image(image_path_E1171_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1171, image_E1171]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
                elements.append(spacer)
                elements.append(spacer)
        # E1172
                header_text_E1172 = "Results of E1172 "
         # Create first large horizontal division
                header_paragraph_E1172 = Paragraph(header_text_E1172, style_sheet["Heading1"])
                elements.append(header_paragraph_E1172)
                num_rows = cph_all_E1172_data.shape[0]          
                survival_func = cph_all_E1172_model.predict_survival_function(cph_all_E1172_data.iloc[0:num_rows], times=t)
                survival_means = survival_func.mean(axis=1)

                plt.figure()
                plt.plot(t,survival_means, label='Survival curve of the cohort')        
                
                b = cph_all_E1172_model.predict_survival_function(data1, times=t)
                plt.plot(t,b, label='your survival curve')
                age_diff = (number_input_age - number_input)*365
                partial_hazard = cph_all_E1172_model.predict_partial_hazard(data1) 
                survival_prob_day = cph_all_E1172_model.predict_survival_function(data1,times=age_diff)
                hazard = 1 - survival_prob_day
       
                time_today = [age_diff, age_diff + 365,age_diff + 730, age_diff + 1095,age_diff + 1461,age_diff + 1825,age_diff + 2191,age_diff + 2556,age_diff + 2922,age_diff + 3287,age_diff + 3652]
                survival_from_age_today = cph_all_E1172_model.predict_survival_function(data1,times=time_today)
                survival_from_age_today = (1 - survival_from_age_today)*100
                survival_from_age_today = survival_from_age_today.rename(columns={survival_from_age_today.index.name: 'Time'})
                survival_from_age_today.index = ['Now', '1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10 years']
        
        # get the current name of the first column
                old_name = survival_from_age_today.columns[0]

        # rename the first column
                survival_from_age_today = survival_from_age_today.rename(columns={old_name: 'Hazard probability in %'})
        
                survival_from_age_today['Hazard probability in %'] = round(survival_from_age_today['Hazard probability in %'],2)
        
                survival_from_age_today['Hazard probability in %'] = survival_from_age_today['Hazard probability in %'].apply('{:.2f}'.format)
        
   
                plt.plot(t,survival_means, label='Survival curve of the cohort')
        
                plt.legend()
        # Save the plot as an image
                plt.savefig('assets/lab_report_fig_E1172.png')
        
                table_data_E1172 = [['Year', 'Hazard probability in %'], ['Now',survival_from_age_today.iloc[0,0]], ['1 Year',survival_from_age_today.iloc[1,0] ],['2 Years', survival_from_age_today.iloc[2,0]],['3 Years', survival_from_age_today.iloc[3,0]],['4 Years', survival_from_age_today.iloc[4,0]],['5 Years',survival_from_age_today.iloc[5,0] ],['6 Years', survival_from_age_today.iloc[6,0]],['7 Years',survival_from_age_today.iloc[7,0] ],['8 Years', survival_from_age_today.iloc[8,0]],['9 Years', survival_from_age_today.iloc[9,0]],['10 Years', survival_from_age_today.iloc[10,0]]]
                table_E1172 = Table(table_data_E1172)
                table_E1172.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),('ALIGN', (0,0), (-1,-1), 'CENTER'),('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),('FONTSIZE', (0,0), (-1,0), 14),('BOTTOMPADDING', (0,0), (-1,0), 12),('BACKGROUND', (0,1), (-1,-1), colors.beige),('GRID', (0,0), (-1,-1), 1, colors.black),('LEFTPADDING', (0, 0), (-1, -1), 10),('RIGHTPADDING', (0, 0), (-1, -1), 10),('LEFT', (0, 0), (-1, -1), inch,)]))
                table_E1172.hAlign = 'LEFT'   
       # elements.append(table)

                spacer = Spacer(1, inch * 0.2)
                elements.append(spacer)


         # Add an image
                #image_path_E1172 = "lab_report_fig_E1172.png"
        #image1 = Image(image_path, width=inch*2, height=inch*2)
        
        # Create the Image object with a width and height of 2 inches
                image_E1172 = Image(image_path_E1172_lab, width=inch*3.5, height=inch*3.5)
        
                header_table = "Predicted risk for 10 years"
         # Create first large horizontal division
                header_table = Paragraph(header_table, style_sheet["Heading2"])
        
                header_graph = "Predicted risk as a graph "
         # Create first large horizontal division
                header_graph = Paragraph(header_graph, style_sheet["Heading2"])
        
                chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        
        
                elements.append(Table([[header_table, header_graph]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[1.5 * inch], style=chart_style))
        
                chart_style1 = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),('VALIGN', (0, 0), (-1, -1), 'CENTER')])

                elements.append(Table([[table_E1172, image_E1172]],
                             colWidths=[4 * inch, 4 * inch,],
                             rowHeights=[2.5 * inch], style=chart_style1))
        
        
        
                spacer = Spacer(1, 0.25*inch)
       
        # Build the document by adding the elements
                doc.build(elements)
                now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                report_filename = f"report_lab_details_{now}.pdf"
        
         # File name of the report
        #report_name = "example_report.pdf"
        
        # Move the buffer's pointer to the beginning of the buffer
                buffer.seek(0)
        # generate the download link
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download Report</a>'

        # display the link using st.markdown
                st.markdown(href, unsafe_allow_html=True)
        # Set up the download link
               # href = f'<a href="data:application/pdf;base64,{base64.b64encode(buffer.read()).decode()}">Download Report</a>'
                #st.markdown(href, unsafe_allow_html=True)
     
     
       # doc.build([header_paragraph,datetime_paragraph,name_paragraph,age_paragraph,address_paragraph,[column1, column2]])
        # Display a link to the generated PDF
                #st.success("PDF generated! Download the PDF [here](/Users/anuradhamadurapperuma/Documents/report_lab1.pdf).")



    
    
    
# Define the actions for each option
if selected_option == "E119 - Diabetes":
    E119_page()
elif selected_option == "E1122 - Diabetic Nephropathy":
    E1122_page()
elif selected_option == "E1129 - Kidney Complications AKI":
    E1129_page()
elif selected_option == "E1131 - Background Retinopathy":
    E1131_page()
elif selected_option == "E1139 - Other Opthalmic Complications":
    E1139_page()
elif selected_option == "E1142 - Diabetic Polyneuropathy":
    E1142_page()
elif selected_option == "E1151 -PVD":
    E1151_page()
elif selected_option == "E1164 - Hypoglycaemia":
    E1164_page()
elif selected_option == "E1165 - Poor Control - Hyperglycaemia":
    E1165_page()
elif selected_option == "E1171 - Microvascular and other specified nonvascular complications":
    E1171_page()    
elif selected_option == "E1172 - Fatty Liver":
    E1172_page()
else:
    All_Comp_page()

 



