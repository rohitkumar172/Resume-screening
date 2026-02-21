import streamlit as st
import pandas as pd
import io
from utils.file_handler import extract_text_from_pdf
from utils.similarity_checker import calculate_similarity_score
from utils.metadata_extractor import extract_email_phone_location  

st.set_page_config(page_title="Resume Screening Bot", layout="centered")

st.title("📑 Resume Screening AI Bot")
st.markdown("### Job Description")

jd_input = st.text_area("📝 Job Description (Max 1000 words)", height=250)
word_count = len(jd_input.split())

if word_count > 1000:
    st.warning(f"❗ Word limit exceeded ({word_count}/1000). Please reduce your job description.")
    jd_input = ""  # clear jd if over limit

resume_files = st.file_uploader("📎 Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if jd_input and resume_files:
    st.markdown("## 📊 Match Scores")

    results = []

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        score = calculate_similarity_score(jd_input, resume_text)
        email, phone, location = extract_email_phone_location(resume_text)

        results.append({
            "Resume": file.name,
            "Score": f"{score}%",
            "Email": email,
            "Contact Number": phone,
            "Location": location
        })

    df_results = pd.DataFrame(results)
    st.table(df_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)

    st.download_button(
        label="📥 Download Results as Excel",
        data=output,
        file_name="resume_scores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif jd_input and not resume_files:
    st.info("Upload at least one resume PDF.")
elif resume_files and not jd_input:
    st.info("Paste the job description above.")


