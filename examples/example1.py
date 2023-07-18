from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
# from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, AutoModel
from langchain.memory import ConversationBufferMemory


loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2"
)
vectorstore = Chroma.from_documents(texts, embeddings)


# model_id = 'trl-internal-testing/tiny-random-GPT2LMHeadModel'
model_id = 'hf-tiny-model-private/tiny-random-GPT2Model'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=local_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    memory=memory
)
query = "What did the president say about Ketanji Brown Jackson"
qa.run({"question": query})

print()