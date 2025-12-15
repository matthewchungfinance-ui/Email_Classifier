import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer
import time

st.set_page_config(
    page_title="📧 Email Reply Assistant",
    page_icon="📧",
    layout="wide"
)

st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; text-align: center;}
.pipeline-card {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                 padding: 1.5rem; border-radius: 15px; margin: 1rem 0; 
                 border-left: 6px solid #1f77b4; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
.metric-card {background: white; padding: 1rem; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipelines():
    """Load all 3 pipelines with custom classifier"""
    with st.spinner('🔄 Loading AI models (2-3 min)...'):
        # ✅ YOUR CUSTOM CLASSIFIER (replaced)
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        classifier = pipeline(
            "text-classification",
            model="byinab/custom-email-classifier",
            tokenizer=tok,
        )
        
        generator = pipeline("text-generation", model="Kunal7370944861/Email-Writer-AI")
        translator = pipeline("translation", model="DDDSSS/translation_en-zh")
    
    return classifier, generator, translator

# Load models safely
try:
    classifier, generator, translator = load_pipelines()
    st.success("✅ All 3 pipelines ready! (Custom Email Classifier Loaded)")
except Exception as e:
    st.error(f"❌ Model loading error: {str(e)}")
    st.stop()

def classify_email(text, classifier):
    result = classifier(text[:512])[0]
    return result["label"], float(result["score"])

def build_prompt(email_text, category):
    return f"""You are a helpful customer service agent.
Email category: {category}

Customer email:
{email_text}

Write a polite, concise reply template.
Reply:"""

def generate_reply(prompt, generator):
    outputs = generator(prompt, max_length=300, num_return_sequences=1, 
                       do_sample=True, temperature=0.7)
    full_text = outputs[0]["generated_text"]
    if "Reply:" in full_text:
        return full_text.split("Reply:", 1)[-1].strip()
    return full_text.replace(prompt, "").strip()

def translate_reply(text, translator):
    if not text.strip(): return ""
    return translator(text)[0]["translation_text"].strip()

# Header
st.markdown('<h1 class="main-header">🤖 Email Reply Assistant</h1>', unsafe_allow_html=True)
st.markdown("**AI-powered: Classify → Generate Reply → Translate to Chinese**")

# Sidebar - UPDATED with your custom classifier
with st.sidebar:
    st.header("🔧 Pipeline Status")
    st.success("✅ **Pipeline 1**: `byinab/custom-email-classifier`")
    st.success("✅ **Pipeline 2**: `Kunal7370944861/Email-Writer-AI`") 
    st.success("✅ **Pipeline 3**: `DDDSSS/translation_en-zh`")
    st.markdown("---")
    st.info("👈 **Paste email → Process → Copy replies!**")

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📨 **Input Email**")
    email_text = st.text_area(
        "Paste complete email here...",
        placeholder="Subject: Order Issue\n\nHello,\nMy package arrived damaged...",
        height=220
    )
    
    if st.button("🚀 **PROCESS EMAIL**", type="primary", use_container_width=True):
        if email_text.strip():
            st.session_state.processed = True
            st.session_state.email = email_text
        else:
            st.error("❌ Please paste an email first!")
    if st.button("🧹 **CLEAR**", use_container_width=True):
        st.rerun()

with col2:
    if 'processed' in st.session_state and st.session_state.processed:
        email_text = st.session_state.email
        
        # Pipeline 1: YOUR CUSTOM CLASSIFIER
        with st.container():
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown("### 🔢 **Pipeline 1: Custom Email Classifier**")
            label, score = classify_email(email_text, classifier)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️ Tag</h3>
                    <h2>{label}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Confidence</h3>
                    <h2>{score:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pipeline 2: English Reply
        with st.container():
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown("### ✉️ **Pipeline 2: English Reply**")
            prompt = build_prompt(email_text, label)
            reply_en = generate_reply(prompt, generator)
            st.text_area("**Reply Template**", reply_en, height=140, disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pipeline 3: Chinese Translation
        with st.container():
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown("### 🇨🇳 **Pipeline 3: Chinese Translation**")
            reply_zh = translate_reply(reply_en, translator)
            st.text_area("**中文回复**", reply_zh, height=140, disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download buttons
        col_c, col_d = st.columns(2)
        with col_c:
            st.download_button("📥 Download English", reply_en, "email_reply_en.txt", use_container_width=True)
        with col_d:
            st.download_button("📥 Download Chinese", reply_zh, "email_reply_zh.txt", use_container_width=True)
    else:
        st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
        st.info("🎯 **Paste your email above and click PROCESS**")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Powered by Streamlit + Transformers | Custom Email Classifier*")
