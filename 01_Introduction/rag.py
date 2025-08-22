import ollama
import json
import minsearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('./documents.json', 'rt') as f:
    docs_raw = json.load(f)
    
documents = []
for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
        
index = minsearch.Index(text_fields=['question',
                             'text',
                             'section'],
                keyword_fields=['course']
)

index.fit(documents)


def search(query, filter_dict=None, boost_dict=None, num_results=5):
    results = index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost_dict,
        num_results=num_results,
    )
    return results

def build_prompt(query: str, search_results: list | list[dict]) -> str:
    context = ""
    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\ntext: {doc['text']}\n\n"
    context = context.strip()
    
    prompt_template = f"""
You're a course teaching assistant. Your job is to help students with their 
questions and provide explanations on various topics.
Answer the question based on the CONTEXT. Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT doesn't contain the answer, output NONE.

CONTEXT:
{context.strip()}

QUESTION:
{query.strip()}
"""
    return prompt_template.strip()

def llm(prompt: str) -> str:
    response = ollama.chat(
        model = "gpt-oss:20b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response["message"]["content"].strip()

def rag():
    import time

    start_time = time.time()
    query = "how can i get the certificate?"
    logger.info(f"Searching for query: {query}")
    search_results = search(query, filter_dict={'course': 'machine-learning-zoomcamp'},
                            num_results=10, boost_dict={
                                "AWS": 2,
                                "Amazon": 2,
                                "Lambda": 2
                            })
    
    aug_prompt = build_prompt(query, search_results)

    response = llm(aug_prompt)
    logger.info(f"Response: {response}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    return response


if __name__ == "__main__":
    rag()