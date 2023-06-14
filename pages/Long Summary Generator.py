# Bring in deps
import streamlit as st 
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (

    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


#Apis and embeddings
OPENAI_API_KEY=st.secrets["openai_key"]  # platform.openai.com
ENV = 'northamerica-northeast1-gcp'


embed = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

index_name = 'index-test'
pinecone.init(
    api_key=CONE_API_KEY,
    environment=ENV
)


llm = ChatOpenAI(temperature=.9, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")



st.title('🦜🔗 Script Analizer - Long Summaries')
namespace_options = ["Resilience Road", "Innovation & Leadership","Lindsay Hadley Podcast","The Jay davis Show"]
name_space = st.selectbox("Pick a Client show", namespace_options)

# Streamlit UI
st.title("Upload an episode")
loader=st.file_uploader("upload an episode script", type="txt")

if loader is not None:

    content = loader.getvalue().decode("utf-8")
    #st.write(content) 
    #documents = [{'page_content': content}]
    #hunk_size = 10000
    #chunk_overlap = 500
    #texts = split_text_in_chunks(content, chunk_size, chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=700)
    texts = text_splitter.create_documents([content])
    st.write(f'Now you have {len(texts)} documents')
    # Notifica al usuario que el archivo se ha cargado correctamente
    st.success("Success!")




#chain = load_qa_chain(llm, chain_type="stuff", verbose = True)

#data_base = Pinecone.from_existing_index(index_name, embed,namespace='Jay-davis',text_key="text")
#docs2=data_base.as_retriever()

#query = "summarize the episode where {guest_name} is the guest of the {name_space} podcast show "
#docs = data_base.similarity_search(query, include_metadata=True)


template_chat="""
This is a podcast transcript for the podcast show {host_podcast}. Each paragraph starts with a timestamp, then who is speakeing followed by if they are the host, co-host or guest.
You are a going to act as a genius that helps {host_name}, as the host of the {host_podcast}. To identify the episode you are going to search between the episodes for the guest whos name is {guest_name}.
Your goal is to write summary of the complete episode, using the most relevant information 
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"

"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template_chat)
human_template="{text}" # Simply just pass the text as a human message
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(messages=[system_message_prompt, human_message_prompt])


template_combined="""
You are a helpful assistant that helps {host_name}, a High profile podcaster, you are going to summarize information from the transcript where {guest_name} was invtitated to the show.
Your goal is to write a summary from the perspective of {host_name}that will highlight key points that will be relevant for his listeners
Do not respond with anything outside of the call transcript. If you don't know, say, "I don't know"

Respond with the following format
-700 Words at least
- A verbose summary
- You may do a few paragraphs to describe the transcript if needed
Please respond the summary above.

YOUR RESPONSE:
"""
system_message_prompt_combine = SystemMessagePromptTemplate.from_template(template_combined)
human_template="{text}" # Simply just pass the text as a human message
human_message_prompt_combine = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_combine, human_message_prompt_combine])


host_name = st.text_input('Input the host name')
guest_name = st.text_input('Input the guest name')

chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt,
                             combine_prompt=chat_prompt_combine
                            
                            )

if st.button("Submit"):
    output = chain.run({
                    "input_documents": texts,
                    "host_podcast": name_space, \
                    "host_name" : host_name, \
                    "guest_name":guest_name
                   })
    st.write("you answers are : ")
    st.write(output)
