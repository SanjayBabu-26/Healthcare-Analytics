import streamlit as st
from  PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px 
import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from streamlit_lottie import st_lottie
import json
import requests


img = Image.open('img.png')

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)



st.set_page_config(page_title="Healthcare Analytics",page_icon=img, layout="wide")

video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="")>
		  Your browser does not support HTML5 video.
		</video>
        """

st.markdown(video_html, unsafe_allow_html=True)

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")
# lottie_wall = load_lottiefile("LottieFiles/Cancer.json")
# st.sidebar.image(st_lottie(lottie_wall), use_column_width=False,width=None)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["About","Data","Visualization","Accuracy","Prediction"],
        icons=["house","activity","clipboard-data","file-text","graph-up-arrow"],
        menu_icon=None,
        
        styles={
                "container": {"padding": "0!important", "background-color": "97DECE"},  #background color for the Website
                "icon": {"color": "black", "font-size": "20px"},   # icon color for the tabs
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "CBEDD5", # icon color after hover
                },
                "nav-link-selected": {"background-color": "62B6B7"},   # Colour when tab is selected  #ADD8E6
                "icon": {"color": "black", "font-size": "20px"},   # icon color
        },
    )
        

global numeric_columns
global non_numeric_columns
global Diabetes
global Lung_Cancer

Diabetes = pd.read_csv('Diabetes.csv')
Lung_Cancer = pd.read_csv("Lung Cancer.csv")

if selected=="About":
    # Navigation = option_menu(
    #     menu_title=None,
    #     options=["Diabetes","Lung Cancer"],
    #     icons=["clipboard-plus","clipboard-plus"],
    #     orientation="horizontal"
    #) 
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
                st.header("Diabetes")
                st.subheader('*Introduction*')
                st.write("Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood glucose. Hyperglycaemia, also called raised blood glucose or raised blood sugar, is a common effect of uncontrolled diabetes and over time leads to serious damage to many of the body's systems, especially the nerves and blood vessels.")
        with right_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Cancer_2.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )
        with right_column:
            st.write("In 2014, 8.5% of adults aged 18 years and older had diabetes. In 2019, diabetes was the direct cause of 1.5 million deaths and 48% of all deaths due to diabetes occurred before the age of 70 years. Another 460 000 kidney disease deaths were caused by diabetes, and raised blood glucose causes around 20% of cardiovascular deaths.Between 2000 and 2019, there was a 3% increase in age-standardized mortality rates from diabetes. In lower-middle-income countries, the mortality rate due to diabetes increased 13%. By contrast, the probability of dying from any one of the four main noncommunicable diseases (cardiovascular diseases, cancer, chronic respiratory diseases or diabetes) between the ages of 30 and 70 decreased by 22% globally between 2000 and 2019. ")
        with left_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Diabetes.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )


    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
                st.header("Lung Cancer")
                st.subheader('*Introduction*')
                st.write("Lung cancer is a type of cancer that begins in the lungs. Your lungs are two spongy organs in your chest that take in oxygen when you inhale and release carbon dioxide when you exhale. Lung cancer is the leading cause of cancer deaths worldwide. People who smoke have the greatest risk of lung cancer, though lung cancer can also occur in people who have never smoked. The risk of lung cancer increases with the length of time and number of cigarettes you've smoked. If you quit smoking, even after smoking for many years, you can significantly reduce your chances of developing lung cancer.")
        with right_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Cancer.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )
        
        with right_column:
            st.write("Lung cancer is the third most common cancer in the U.S. Health systems report over 200,000 new cases of lung cancer each year.")
        with left_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Lung.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )

    
    
if selected=="Data":
    Navigation = option_menu(
        menu_title=None,
        options=["Diabetes","Lung Cancer"],
        icons=["clipboard-plus","clipboard-plus"],
        orientation="horizontal"
    )
    if Navigation=="Diabetes":
        lottie_Cancer = load_lottiefile("LottieFiles/Diabetes.json")
        st_lottie(
            lottie_Cancer,
            height=400,
            width=400,
            
        )
        st.write(Diabetes)
    if Navigation=="Lung Cancer":
        lottie_Cancer = load_lottiefile("LottieFiles/Diabetes.json")
        st_lottie(
            lottie_Cancer,
            height=400,
            width=400,
            
        )
        st.write(Lung_Cancer)




if selected=="Visualization":
    Navigation = option_menu(
        menu_title=None,
        options=["Diabetes","Lung Cancer"],
        icons=["clipboard-plus","clipboard-plus"],
        orientation="horizontal"
    ) 

    

    if Navigation == "Diabetes":
        numeric_columns = list(Diabetes.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(Diabetes.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
        x_values = st.selectbox('X axis', options=numeric_columns)
        y_values = st.selectbox('Y axis', options=numeric_columns)
        # color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
        plot = px.scatter(data_frame=Diabetes, x=x_values, y=y_values, color=None)
        st.plotly_chart(plot)

    if Navigation == "Lung Cancer":
        numeric_columns = list(Lung_Cancer.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(Lung_Cancer.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
        x_values = st.selectbox('X axis', options=numeric_columns)
        y_values = st.selectbox('Y axis', options=numeric_columns)
        plot = px.scatter(data_frame=Lung_Cancer, x=x_values, y=y_values, color=None)
        st.plotly_chart(plot)

if selected == "Accuracy":
    dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Diabetes', 'Lung Cancer')
)


    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('K-Nearest Neighbour', 'Support Vector Machine', 'Random Forest')
    )

    st.write(f"## {dataset_name} Accuracy for {classifier_name}")
    if classifier_name=='K-Nearest Neighbour':
        st.write("K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.")
    if classifier_name=='Support Vector Machine':
        st.write("SVM algorithm is used to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.")
    else:
        st.write("Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.")
    
    
    def get_dataset(name):
        data = None
        if name == 'Diabetes':
            data = pd.read_csv('Diabetes.csv')
        elif name == 'Lung Cancer':
            data = pd.read_csv('Lung Cancer.csv')
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        return X, y

    X, y = get_dataset(dataset_name)
    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)
    #### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier: {classifier_name}')
    st.write(f'Accuracy: ', acc)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    #plt.show()
    st.pyplot(fig)






if selected=="Prediction":
    Navigation = option_menu(
        menu_title=None,
        options=["Diabetes","Lung Cancer"],
        icons=["clipboard-plus","clipboard-plus"],
        orientation="horizontal"
    ) 
    
    if Navigation=="Diabetes":
        df = pd.read_csv("Diabetes.csv")
        X = df[["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]]
        y = df["Outcome"]
        clf = LogisticRegression() 
        clf.fit(X, y)
        joblib.dump(clf, "clf.pkl")
        a = st.number_input("Enter the number of Pregnancies")
        b = st.number_input("Enter the Amount of Gulcose in Blood")
        c = st.number_input("Enter the Blood Pressure Level")
        d = st.number_input("Enter the Skin Thickness")
        e = st.number_input("Enter the Insulin Amount")
        f = st.number_input("Enter the Body Mass Index (BMI)")
        g = st.number_input("Enter the Diabetes Pedigree Function")
        h = st.number_input("Enter the Age")
        if st.button("Submit"):
            clf = joblib.load("clf.pkl")
            X = pd.DataFrame([[a, b, c, d, e, f, g, h]], 
                        columns = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"])
            prediction = clf.predict(X)[0]
            st.write("Based on the data you have uploaded the model has trained itself to come to conclusion.")
            if prediction==1:
                st.info(f"The person has Lung Cancer")
            else:
                st.info(f"The person is Healthy")

    if Navigation=="Lung Cancer":
        df = pd.read_csv("Lung Cancer.csv")
        X = df[["Gender","Age","Stoke","Yellow Fingers","Anxiety","Peer Pressure","Chronic Disease","Fatigue","Allegry","Wheezing","Alcohol Consumption","Coughing","Shortness of Breath","Swallowing Difficulty"]]
        y = df["Outcome"]
        clf = LogisticRegression() 
        clf.fit(X, y)
        joblib.dump(clf, "clf.pkl")
        a = st.selectbox(
            'Select Gender',
            ('Male', 'Female')
            )
        if a == 'Male':
            a = 1
        else:
            a = 0
        b = st.number_input("Enter the Age")
        c = st.number_input("Enter the Stoke Duration")
        d = st.number_input("Enter the Duration of Yellow Fingers")
        e = st.number_input("Enter the Anxiety Rate")
        f = st.selectbox(
            "Enter Peer Pressure if any",
            (1,0)
            )
        g = st.number_input("Enter the Chronic Disease rate")
        h = st.number_input("Enter the Fatigue rate")
        i = st.number_input("Enter the Allergy (if any)")
        j = st.number_input("Enter the Wheezing")
        k = st.selectbox(
            "Enter Alcohol Consumption (if any)",
            (1,0)
            )
        l = st.number_input("Enter the Coughing rate")
        m = st.selectbox(
            "Enter the Shortness of Breath",
            (0,1)
            )
        n = st.selectbox(
            "Enter Swallowing Difficulty",
            (0,1)
            )

        if st.button("Submit"):
            clf = joblib.load("clf.pkl")
            X = pd.DataFrame([[a, b, c, d, e, f, g, h,i,j,k,l,m,n]], 
                        columns = ["Gender","Age","Stoke","Yellow Fingers","Anxiety","Peer Pressure","Chronic Disease","Fatigue","Allegry","Wheezing","Alcohol Consumption","Coughing","Shortness of Breath","Swallowing Difficulty"])
            prediction = clf.predict(X)[0]
            st.write("Based on the data you have uploaded the model has trained itself to come to conclusion.")
            if prediction==1:
                st.info(f"The person has Lung Cancer")
            else:
                st.info(f"The person is Healthy")

        




    