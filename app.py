import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

##Function to get response from Llama 2 model

def getllamaresponse(input_text,no_words,blog_style):
    ##calling llama2 model using ctransformers
    llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main and download the base one with 7b parameter
    ##Prompt Template

    template = """
                Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
                """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                                           template=template)

    #generate the response from llama 2 model
    response = llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs ")
input_text=st.text_input("Enter the blog topic:")
col1,col2=st.columns([5,5])             
#create 2 additiona columns for additional inputs as blog style and no of words      
with col1:
    no_words=st.text_input("No of Words")
with col2:
    blog_style=st.selectbox("Writing the blog for", ('REsearchers', 'Data Scientist', 'Common people'), index=0)    

submit=st.button("Generate")

if submit:
    st.write(getllamaresponse(input_text,blog_style,no_words))


#to run streamlit stream run app.py    