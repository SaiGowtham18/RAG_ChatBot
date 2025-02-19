from langchain_openai import EnhancedChatModel
import streamlit as st

# function to load llm  and get output response
def custom_qna_model(input):
    llm = EnhancedChatModel(temperature=0.7, model='gpt-3.5-turbo')
    response = llm.invoke(input)
    return response.content

def q_and_a_main():
    st.header('Q&A Bot Powered by OpenAI and LangChain')
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #708090; padding: 10px; text-align: center;">&copy; 2024 Customized AI Chatbot. All Rights Reserved.</div>',
                    unsafe_allow_html=True
                    )
    st.write("Capabilities: Uses ChatGPT-3.5 to generate answers. It has no memory and each question is handled individually")
    
    input = st.text_input('Ask a question: ', key ='input')
    submit = st.button('Generate Answer')

    if submit:
        with st.spinner("Generating..."):
            output_response = custom_qna_model(input)
            st.write(output_response)
    
if __name__ == '__main__':
    q_and_a_main()

