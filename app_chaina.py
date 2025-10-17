import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("china")



SAMPLE_COUNT = st.number_input("Enter SAMPLE COUNT")
TRIAL_LABEL = st.number_input("Enter TRIAL LABEL")
Participant = st.number_input("Enter Participant")
Item = st.number_input("Enter Item")
Bin = st.number_input("Enter Bin")
Time = st.number_input("Enter Time")
Target = st.number_input("Enter Target")
Competitor = st.number_input("Enter Competitor")
Same_col_single_ani = st.number_input("Enter Same col single ani")
EYE_4_P = st.number_input("Enter EYE 4 P")
EYE_5_P = st.number_input("Enter EYE 5 P")
EYE_6_P = st.number_input("Enter EYE 6 P")
Sentence_type = st.radio("Enter Sentence type",('Y','N'))
Same_col_single_ani_condion = st.radio("Enter Same col single ani condion",('Y','N'))
Same_col_single_ani_cond = st.radio("Enter Same col single ani cond",('Y','N'))
Sentence_type_Same = st.number_input("Enter Sentence type Same")

Sentence_type = 1 if Sentence_type == 'Y' else 0
Same_col_single_ani_condion = 1 if Same_col_single_ani_condion == 'Y' else 0
Same_col_single_ani_cond = 1 if Same_col_single_ani_cond == 'Y' else 0


if st.button("Predict"):

    data = np.array([[SAMPLE_COUNT, TRIAL_LABEL, Participant, Item, Bin, Time,
                  Target, Competitor, Same_col_single_ani, EYE_4_P, EYE_5_P,
                  EYE_6_P, Sentence_type, Same_col_single_ani_condion,
                  Same_col_single_ani_cond, Sentence_type_Same]])

    scaled_input = scaler.transform(data)
    prediction = model.predict(scaled_input)[0]
    
    result = "known" if prediction == 1 else "unknown"
    st.success(result)


