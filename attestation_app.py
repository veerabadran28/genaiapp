import streamlit as st
import datetime
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Barclays GenAI Application User Attestation",
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
</style>
""", unsafe_allow_html=True)

# Barclays logo
st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/7e/Barclays_logo.svg/391px-Barclays_logo.svg.png", width=200)

# Application title
st.title("Barclays GenAI Application User Attestation")

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
    
    "Acceptable Use Policy": [
        "**Business Purpose**: I will use this application solely for legitimate Barclays business purposes and not for personal use.",
        "**Unauthorized Access**: I will not share my access credentials or provide unauthorized access to the application to any other individual.",
        "**Content Guidelines**: I will not use the application to generate, store, or distribute inappropriate, offensive, or harmful content.",
        "**Intellectual Property**: I understand that I must respect all applicable intellectual property rights when using this application.",
        "**Resource Usage**: I will use the application's resources responsibly and will not engage in activities that could degrade system performance for other users.",
        "**Prompt Engineering**: I will create responsible prompts that do not attempt to circumvent the application's safety measures or ethical guidelines.",
        "**Citation and Attribution**: When using outputs from this application in work products, I will provide appropriate context about the AI-assisted nature of the content when relevant.",
        "**Third-Party Integration**: I will not connect unauthorized third-party applications, services, or APIs to this GenAI application without proper approval from IT Security.",
        "**No Circumvention**: I will not attempt to bypass, disable, or interfere with any security features or access controls built into the application.",
        "**Session Management**: I will log out of the application when not in use and will not leave active sessions unattended.",
        "**Prohibited Uses**: I will not use the application to:\n    - Generate content that could violate Barclays' code of conduct or ethics policies\n    - Create materials that could be construed as financial advice without appropriate disclosures\n    - Automate decision-making processes that require human judgment\n    - Process or analyze competitor data in ways that could violate competition laws"
    ],
    
    "Output Verification": [
        "**Critical Review**: I understand that the outputs generated by this application may contain inaccuracies, hallucinations, or inappropriate content. I will critically review all outputs before use.",
        "**Professional Judgment**: I will apply my professional judgment and expertise when using the outputs generated by this application.",
        "**Bias Awareness**: I recognize that AI systems may exhibit biases in their outputs. I will be vigilant for potential biases and take appropriate steps to mitigate their impact.",
        "**Decision Accountability**: I understand that I remain accountable for decisions made using insights from this application, regardless of whether the decision was influenced by AI-generated content. Any modification, editing, or alteration of AI-generated outputs will be solely my responsibility.",
        "**Output Disclosure**: When sharing outputs from this application with others, I will provide appropriate context about the AI-assisted nature of the content and any limitations that may apply.",
        "**Fact Verification**: I will verify any factual claims, references, or citations generated by the application through authoritative sources before relying on them for business purposes.",
        "**Calculation Validation**: I will independently validate any complex calculations, financial projections, or numerical analyses produced by the application.",
        "**Non-Overreliance**: I acknowledge the risk of \"automation bias\" and will not over-rely on the application's outputs without appropriate human oversight.",
        "**Contextual Understanding**: I recognize that the application may not fully understand the specific context of my query and will ensure that outputs are appropriate for the intended business context.",
        "**Documentation of Verification**: For high-importance use cases, I will document the steps taken to verify and validate outputs from this application.",
        "**Model Limitations**: I understand that the application has inherent limitations in terms of reasoning, recency of knowledge, and specialized domain expertise, and will take these limitations into account when using outputs."
    ],
    
    "Security Incident Reporting": [
        "**Security Incidents**: I will promptly report any suspected security incidents, data breaches, or misuse of the application through the established Barclays security incident reporting process.",
        "**Unusual Outputs**: I will report any concerning, unexpected, or potentially harmful outputs generated by the application to the appropriate IT or AI governance team.",
        "**Vulnerability Disclosure**: I will report any discovered vulnerabilities in the application without attempting to exploit them further.",
        "**Prompt Injection Awareness**: I understand the risks associated with prompt injection attacks and will report any instances where the application appears to have been manipulated to produce unauthorized outputs."
    ],
    
    "Compliance with Laws and Policies": [
        "**Regulatory Compliance**: I will use this application in compliance with all applicable laws, regulations, and Barclays policies.",
        "**Updates to Terms**: I understand that these terms may be updated periodically, and I agree to review and comply with the most current version.",
        "**Training Completion**: I confirm that I have completed all required training related to the responsible use of AI technologies at Barclays.",
        "**Conflict Resolution**: If I encounter a situation where the use of this application may conflict with Barclays policies or regulatory requirements, I will seek guidance from my manager or the appropriate compliance team.",
        "**Model-Specific Limitations**: I understand that different AI models within this application may have different capabilities and limitations, and I will use each in accordance with its specific usage guidelines.",
        "**Ethical AI Use**: I will use this application in accordance with Barclays' AI ethics principles and will not attempt to generate outputs that could cause harm or reputational damage."
    ],
    
    "Application Features Acknowledgment": [
        "**Document Upload**: I will only upload documents that I am authorized to access and process.",
        "**Code Generation**: I will review all generated code for security vulnerabilities and compliance with Barclays coding standards before implementation.",
        "**RAG-Enhanced Chatbot**: I understand that the chatbot's responses are generated based on both AI models and retrieved information from authorized knowledge bases.",
        "**Document Generation**: I will verify all AI-generated documents for accuracy and compliance before sharing.",
        "**Data Download**: I will handle all downloaded data in accordance with Barclays' data classification and handling procedures.",
        "**Visualization Tools**: I recognize that visual representations of data must be validated for accuracy before being used in decision-making processes.",
        "**Model Selection**: When multiple AI models are available within the application, I will select the appropriate model based on my specific use case and the sensitivity of the data involved.",
        "**Knowledge Management**: I understand that information I provide to the application may be used to improve its knowledge base, and I will follow appropriate protocols when contributing to this knowledge.",
        "**Collaborative Use**: When using the application collaboratively with other Barclays employees, I will ensure all participants understand and adhere to these attestation requirements.",
        "**Usage Metrics**: I understand that my usage patterns and interaction frequency with the application may be monitored for resource allocation and system improvement purposes."
    ],
    
    "Additional Responsibilities": [
        "**Training and Education**: I commit to staying informed about updates to Barclays' AI usage policies and completing any additional training required for responsible use of this application.",
        "**Feedback Provision**: I will provide feedback on the application's performance, usability, and outputs when requested to assist in improving its functionality and safety features.",
        "**Ethical Considerations**: I will consider the ethical implications of my use of this application, including potential impacts on customers, colleagues, and society.",
        "**Resource Limitations**: I understand that there may be usage quotas or resource limitations applied to this application and will respect these constraints.",
        "**Scheduled Maintenance**: I acknowledge that the application may be unavailable during scheduled maintenance windows and will plan my work accordingly.",
        "**Alternative Procedures**: I will maintain knowledge of alternative procedures for accomplishing my work tasks in the event that this application is unavailable.",
        "**Continuous Learning**: I recognize that best practices for using GenAI tools are evolving, and I commit to continuously improving my skills and knowledge in this area.",
        "**Chain of Responsibility**: I understand that I am responsible for outputs generated through my account, even if the specific prompts were suggested or created by another person."
    ],
    
    "Application-Specific Security Protocols": [
        "**Data Classification**: I will adhere to Barclays' data classification framework when using this application and will ensure that all data used is appropriately classified and handled.",
        "**Secure Environment**: I will only use this application in secure physical and digital environments that meet Barclays' security standards.",
        "**Approved Devices**: I will only access this application from approved Barclays devices that have current security patches and antivirus protection.",
        "**Network Security**: I will only access this application through secure, approved networks and will not attempt to access it through public Wi-Fi or unauthorized networks.",
        "**Suspicious Activity**: I will report any suspicious activities or behaviors exhibited by the application immediately to the IT Security team.",
        "**Screen Privacy**: I will ensure that application outputs are not visible to unauthorized individuals when working in public or shared spaces.",
        "**AWS Service Integration**: I understand that this application utilizes AWS GenAI services and will comply with all relevant AWS service-specific security protocols as communicated by Barclays."
    ]
}

# User session state
if 'agreed' not in st.session_state:
    st.session_state.agreed = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'employee_id': '',
        'department': '',
        'accepted_date': ''
    }

# Function to save attestation record
def save_attestation(data):
    # In a real application, this would save to a database
    # For demonstration, we'll just create a CSV if it doesn't exist
    filename = "attestation_records.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['employee_id', 'department', 'accepted_date'])
    
    new_record = pd.DataFrame([data])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(filename, index=False)
    return True

# Main application flow
if not st.session_state.agreed:
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
    
    # User information form
    st.subheader("User Information")
    col1, col2 = st.columns(2)
    
    with col1:
        employee_id = st.text_input("Employee ID *", key="employee_id")
    
    with col2:
        department = st.text_input("Department *", key="department")
    
    # Checkbox for agreement
    agree = st.checkbox("I have read, understood, and agree to abide by all the terms and conditions outlined above.")
    
    # Submit button
    if st.button("Submit Attestation"):
        if not agree:
            st.error("You must agree to the terms and conditions to proceed.")
        elif not employee_id or not department:
            st.error("Please fill in all required fields marked with *.")
        else:
            # Record the attestation
            st.session_state.user_data = {
                'employee_id': employee_id,
                'department': department,
                'accepted_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save the attestation
            save_attestation(st.session_state.user_data)
            
            st.session_state.agreed = True
            st.experimental_rerun()
else:
    # Display confirmation screen
    st.success("Thank you for completing the attestation!")
    st.markdown(f"""
    ### Attestation Confirmation
    
    Your attestation has been recorded with the following details:
    - **Employee ID**: {st.session_state.user_data['employee_id']}
    - **Department**: {st.session_state.user_data['department']}
    - **Acceptance Date**: {st.session_state.user_data['accepted_date']}
    
    You may now proceed to the Barclays GenAI Application.
    """)
    
    if st.button("Continue to Application"):
        st.markdown("Redirecting to the GenAI application...")
        # In a real implementation, this would redirect to the actual application
        st.balloons()

# Footer
st.markdown("""
<div class="footer">
    <p>Barclays GenAI Application User Attestation System | © {0} Barclays Bank PLC. All rights reserved.</p>
    <p>For technical support, please contact: genai.support@barclays.com</p>
</div>
""".format(datetime.datetime.now().year), unsafe_allow_html=True)
