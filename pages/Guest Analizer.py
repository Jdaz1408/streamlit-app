from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
import streamlit as st

# YouTube
from langchain.document_loaders import YoutubeLoader



def pull_from_website(url):
    st.write("Getting webpages...")

    # Doing a try in case it doesn't work
    try:
        response = requests.get(url)
    except:
        # In case it doesn't work
        print ("Whoops, error")
        return
    
    # Put your response in a beautiful soup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get your text
    text = soup.get_text()

    # Convert your html to markdown. This reduces tokens and noise
    text = md(text)
     
    return text

def get_video_transcripts(url):
    st.write("Getting YouTube Videos...")
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    documents = loader.load()
    transcript = ' '.join([doc.page_content for doc in documents])
    return transcript


def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key, max_tokens=2000, model_name='gpt-4')
    return llm

# A function that will be called only if the environment's openai_api_key isn't set

OPENAI_API_KEY=st.secrets["openai_key"]

def split_text(user_information):
    # First we make our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=700)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([user_information])

    return docs

# Get URLs from a string
def parse_urls(urls_string):
    """Split the string by comma and strip leading/trailing whitespaces from each URL."""
    return [url.strip() for url in urls_string.split(',')]

# Get information from those URLs
def get_content_from_urls(urls, content_extractor):
    """Get contents from multiple urls using the provided content extractor function."""
    return "\n".join(content_extractor(url) for url in urls)


response_types = {
    'Interview Questions' : """
        Your goal is to generate interview questions that we can ask them
        Please respond with list of a few interview questions based on the topics above
    """,
    '1-Page Summary' : """
        Your goal is to generate a 1 page summary about them
        Please respond with a few short paragraphs that would prepare someone to talk to this person
    """
}


map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {persons_name}.
Information will include tweets, interview transcripts, and blog posts about {persons_name}
Use specifics from the research when possible

{response_type}

% START OF INFORMATION ABOUT {persons_name}:
{text}
% END OF INFORMATION ABOUT {persons_name}:

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name", "response_type"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given information about {persons_name}.
Do not make anything up, only use information which is in the person's context

{response_type}

% PERSON CONTEXT
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name", "response_type"])

st.set_page_config(page_title="Guest Analizer Tool", page_icon=":robot:")

st.markdown("## :older_man: Giorno The Guest Researcher")

# Output type selection by the user
output_type = st.radio(
    "Output Type:",
    ('Interview Questions', '1-Page Summary'))

# Collect information about the person you want to research
person_name = st.text_input(label="Person's Name", key="persons_name")
youtube_videos = st.text_input(label="YouTube URLs (Use , to seperate videos)", key="youtube_user_input")
webpages = st.text_input(label="Web Page URLs (Use , to seperate urls. Must include https://)", key="webpage_user_input")


button_ind = st.button("*Generate Output*", type='secondary', help="Click to generate output based on information")

# Checking to see if the button_ind is true. If so, this means the button was clicked and we should process the links
if button_ind:
    if not (youtube_videos or webpages):
        st.warning('Please provide links to parse', icon="⚠️")
        st.stop()

    if not OPENAI_API_KEY:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    #if OPENAI_API_KEY == 'YourAPIKeyIfNotSet':
        # If the openai key isn't set in the env, put a text box out there
        #OPENAI_API_KEY = get_openai_api_key()

    # Go get your data
    video_text = get_content_from_urls(parse_urls(youtube_videos), get_video_transcripts) if youtube_videos else ""
    website_data = get_content_from_urls(parse_urls(webpages), pull_from_website) if webpages else ""

    user_information = "\n".join([ video_text, website_data])

    user_information_docs = split_text(user_information)

    # Calls the function above
    llm = load_LLM(openai_api_key=OPENAI_API_KEY)

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template,
                                 # verbose=True
                                 )
    
    st.write("Sending to LLM...")

    # Here we will pass our user information we gathered, the persons name and the response type from the radio button
    output = chain({"input_documents": user_information_docs, # The seven docs that were created before
                    "persons_name": person_name,
                    "response_type" : response_types[output_type]
                    })

    st.markdown(f"#### Output:")
    st.write(output['output_text'])
