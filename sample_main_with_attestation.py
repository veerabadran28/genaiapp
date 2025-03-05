import streamlit as st
import pandas as pd
import os
import datetime
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Barclays GenAI Application",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to style the app with Barclays branding
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stButton button {
        background-color: #00AEEF;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
    }
    .stButton button:hover {
        background-color: #0077A3;
    }
    h1, h2, h3 {
        color: #00395D;
    }
    .attestation-section {
        background-color: #f5f7f9;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #00AEEF;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #666;
        font-size: 12px;
    }
    .stCheckbox label {
        font-weight: bold;
        color: #00395D;
    }
    .warning-text {
        color: #FF0000;
        font-weight: bold;
    }
    .genai-feature {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 3px solid #00AEEF;
    }
</style>
""", unsafe_allow_html=True)

# Functions for attestation process
def get_attestation_version():
    return f"1.0.{int(datetime.datetime.now().timestamp())}"

def hash_attestation_content(content_dict):
    content_str = str(content_dict)
    return hashlib.sha256(content_str.encode()).hexdigest()

def save_attestation(data):
    filename = "attestation_records.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=[
            'employee_id', 
            'department', 
            'accepted_date', 
            'attestation_version',
            'attestation_content_hash',
            'attestation_full_text'
        ])
    
    attestation_content = {}
    for section, content in attestation_sections.items():
        if isinstance(content, list):
            attestation_content[section] = content
        else:
            attestation_content[section] = [content]
    
    version = get_attestation_version()
    content_hash = hash_attestation_content(attestation_content)
    
    full_text = ""
    for section, items in attestation_content.items():
        full_text += f"{section}\n"
        if isinstance(items, list):
            for idx, item in enumerate(items, 1):
                full_text += f"{idx}. {item}\n"
        else:
            full_text += f"{items}\n"
    
    data['attestation_version'] = version
    data['attestation_content_hash'] = content_hash
    data['attestation_full_text'] = full_text
    
    new_record = pd.DataFrame([data])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(filename, index=False)
    
    return version, content_hash

def check_user_attestation(employee_id):
    """Check if a user has already completed attestation"""
    filename = "attestation_records.csv"
    if not os.path.exists(filename):
        return False
    
    df = pd.read_csv(filename)
    return employee_id in df['employee_id'].values

# Attestation data
attestation_sections = {
    "Introduction": """
Welcome to Barclays' GenAI Application. Before proceeding with access to this application, you must review and explicitly acknowledge the following terms, conditions, and responsibilities. This attestation is designed to ensure responsible use of generative AI technology within our organization, maintain data security, and comply with all applicable regulations and policies.
    """,
    
    "Data Privacy and Security": [
        "**Confidential Information**: I understand that any data I upload, process, or generate using this application may contain confidential, proprietary, or sensitive information belonging to Barclays or its clients.",
        "**No Regulatory Submissions**: I acknowledge that outputs from this GenAI application are for informational purposes only and MUST NOT be used for any regulatory submissions, filings, or official documentation without proper human verification and approval through established channels.",
        "**Data Protection**: I will comply with all Barclays data protection policies and will not upload any data classified higher than the application's designated security threshold.",
        "**Personal Data Prohibition**: I will not enter, upload, or process any personally identifiable information (PII), sensitive personal information (SPI), or protected health information (PHI) into this application under any circumstances, including but not limited to:\n   - Full names, addresses, phone numbers, or email addresses\n   - Government-issued identification numbers (SSN, passport, driver's license)\n   - Financial account numbers or payment card information\n   - Biometric data or health information\n   - Date of birth or age information\n   - Any other information that could directly or indirectly identify an individual",
        "**Data Minimization**: I will follow the principle of data minimization and only upload the minimum amount of information necessary to accomplish my business purpose.",
        "**Data Retention**: I understand that data uploaded to this application may be retained for a specified period in accordance with Barclays' data retention policies and may be used to improve the application's functionality.",
        "**Cross-Border Data Transfers**: I will adhere to Barclays' policies regarding cross-border data transfers and will not use this application to circumvent these policies.",
        "**Audit Trail**: I acknowledge that my interactions with this application may be logged for security, compliance, and audit purposes.",
        "**Data Anonymization**: When working with datasets that originally contained sensitive information, I will ensure that all data has been properly anonymized or pseudonymized according to Barclays standards before uploading.",
        "**Access Controls**: I will respect the access controls implemented within the application and will not attempt to access data or functionality beyond my authorized permissions."
    ],
    
    # Additional attestation sections would be included here
    # For brevity, I've only included the first two sections in this example
}

# Initialize session state for user login and attestation status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'has_attested' not in st.session_state:
    st.session_state.has_attested = False
if 'attestation_complete' not in st.session_state:
    st.session_state.attestation_complete = False

# App title
st.image("https://logos-world.net/wp-content/uploads/2020/11/Barclays-Logo.png", width=200)
st.title("Barclays GenAI Application")

# Login section (if not logged in)
if not st.session_state.logged_in:
    st.subheader("Login")
    with st.form("login_form"):
        employee_id = st.text_input("Employee ID")
        department = st.text_input("Department")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if employee_id and department:
                st.session_state.logged_in = True
                st.session_state.user_id = employee_id
                st.session_state.department = department
                st.session_state.has_attested = check_user_attestation(employee_id)
                st.experimental_rerun()
            else:
                st.error("Please enter both Employee ID and Department")

# Main application flow after login
elif st.session_state.logged_in:
    # Check if user needs to complete attestation
    if not st.session_state.has_attested and not st.session_state.attestation_complete:
        st.subheader("User Attestation Required")
        
        # Display attestation sections
        for section_title, section_content in attestation_sections.items():
            st.markdown(f"""<div class='attestation-section'>
            <h3>{section_title}</h3>
            """, unsafe_allow_html=True)
            
            if isinstance(section_content, list):
                # Create numbered list with proper formatting
                for idx, item in enumerate(section_content, 1):
                    # Create numbered list for prohibited uses
                    if "**Prohibited Uses**" in item:
                        base_text = item.split(":\n")[0] + ":"
                        sub_items = item.split(":\n")[1].strip().split("\n")
                        formatted_item = f"{base_text}\n"
                        
                        for i, sub_item in enumerate(sub_items, 1):
                            formatted_item += f"\n    {i}. {sub_item.strip('- ')}"
                        
                        st.markdown(f"{idx}. {formatted_item}")
                    # Format Personal Data Prohibition with numbered sub-points
                    elif "**Personal Data Prohibition**" in item:
                        base_text = item.split(":\n")[0] + ":"
                        sub_items = item.split(":\n")[1].strip().split("\n")
                        formatted_item = f"{base_text}\n"
                        
                        for i, sub_item in enumerate(sub_items, 1):
                            formatted_item += f"\n    {i}. {sub_item.strip('- ')}"
                        
                        st.markdown(f"{idx}. {formatted_item}")
                    else:
                        st.markdown(f"{idx}. {item}")
            else:
                st.markdown(section_content)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Checkbox for agreement
        agree = st.checkbox("I have read, understood, and agree to abide by all the terms and conditions outlined above.")
        
        # Submit button
        if st.button("Submit Attestation"):
            if not agree:
                st.error("You must agree to the terms and conditions to proceed.")
            else:
                # Record the attestation
                user_data = {
                    'employee_id': st.session_state.user_id,
                    'department': st.session_state.department,
                    'accepted_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save the attestation with the complete content
                version, content_hash = save_attestation(user_data)
                
                st.session_state.attestation_complete = True
                st.session_state.has_attested = True
                st.session_state.attestation_version = version
                st.session_state.attestation_hash = content_hash
                
                st.success("Attestation completed successfully. You may now access the GenAI application.")
                st.experimental_rerun()
    
    # Show main application content after attestation is complete
    else:
        # User information display
        st.sidebar.markdown(f"""
        ### User Information
        - **Employee ID**: {st.session_state.user_id}
        - **Department**: {st.session_state.department}
        """)
        
        if st.sidebar.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
        
        # Main application content
        st.header("Welcome to the Barclays GenAI Application")
        st.markdown("""
        This application provides access to various generative AI tools and features 
        to help you with your work at Barclays. Select a feature below to get started.
        """)
        
        # Application features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='genai-feature'>", unsafe_allow_html=True)
            st.subheader("📄 Document Upload & Analysis")
            st.markdown("""
            Upload documents for AI-assisted analysis. Suitable for:
            - Financial reports
            - Research papers
            - Market analyses
            - Contract reviews
            """)
            if st.button("Open Document Analysis"):
                st.session_state.active_feature = "document_analysis"
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='genai-feature'>", unsafe_allow_html=True)
            st.subheader("💻 Code Generation")
            st.markdown("""
            Generate code snippets and solutions for:
            - Data analysis scripts
            - Automation tasks
            - API integrations
            - Database queries
            """)
            if st.button("Open Code Generator"):
                st.session_state.active_feature = "code_generation"
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='genai-feature'>", unsafe_allow_html=True)
            st.subheader("🤖 RAG-Enhanced Chatbot")
            st.markdown("""
            Chat with our AI assistant that has access to:
            - Barclays knowledge base
            - Financial regulations
            - Internal policies
            - Market data
            """)
            if st.button("Open Chatbot"):
                st.session_state.active_feature = "chatbot"
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='genai-feature'>", unsafe_allow_html=True)
            st.subheader("📊 Data Visualization")
            st.markdown("""
            Create interactive visualizations from your data:
            - Charts and graphs
            - Dashboards
            - Trend analysis
            - Comparative reports
            """)
            if st.button("Open Data Visualizer"):
                st.session_state.active_feature = "data_viz"
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature-specific content
        if 'active_feature' in st.session_state:
            st.divider()
            
            if st.session_state.active_feature == "document_analysis":
                st.header("Document Upload & Analysis")
                uploaded_file = st.file_uploader("Upload a document for analysis", type=["pdf", "docx", "txt"])
                if uploaded_file:
                    st.info("Document uploaded successfully. Analysis would be performed here.")
                    
            elif st.session_state.active_feature == "code_generation":
                st.header("Code Generation")
                code_request = st.text_area("Describe the code you need", height=150)
                language = st.selectbox("Select language", ["Python", "SQL", "JavaScript", "Java", "C#"])
                if st.button("Generate Code"):
                    st.code("# This is where the generated code would appear\ndef example_function():\n    return 'Hello, Barclays!'", language=language.lower())
                    
            elif st.session_state.active_feature == "chatbot":
                st.header("RAG-Enhanced Chatbot")
                user_query = st.text_input("Ask a question")
                if user_query:
                    st.markdown("**AI Assistant**: This is where the chatbot response would appear. The response would be generated based on your query and relevant information from Barclays' knowledge base.")
                    
            elif st.session_state.active_feature == "data_viz":
                st.header("Data Visualization")
                viz_upload = st.file_uploader("Upload data for visualization", type=["csv", "xlsx"])
                if viz_upload:
                    st.info("Data uploaded successfully. Visualization options would appear here.")
                    chart_type = st.selectbox("Select visualization type", ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart"])
                    st.image("https://images.dummyapi.io/photo-1541963463532-d68292c34b19?crop=faces&h=500&w=800")

# Footer
st.markdown("""
<div class="footer">
    <p>Barclays GenAI Application | © {0} Barclays Bank PLC. All rights reserved.</p>
    <p>For technical support, please contact: genai.support@barclays.com</p>
</div>
""".format(datetime.datetime.now().year), unsafe_allow_html=True)
