import re

# Fixed length chunking
def fixed_length_chunking(doc, chunk_size):
    return [doc[i : i + chunk_size] for i in range(0,len(doc), chunk_size)]

# Sentence based chunking
def sentence_based_chunking(doc):
    #use sentence delimiters as split
    #(?<=...): A positive lookbehind assertion. It ensures the match occurs only if preceded by the specified characters.
    #+: Matches one or more spaces.
    #r'': A raw string literal.
    return re.split(r'(?<=[.!?]) +', doc)

#Paragraph based chunking - assuming split by multiple \n charachter
def paragraph_based_chunking(doc):
    return doc.split("\n\n")

#Sliding Window chunking
def sliding_window_chunking(doc, step, chunk_size):
    return [doc[i:i+chunk_size] for i in range(0, len(doc) - chunk_size + 1, step)]


"""
Treebank Tokenizer?
1. Parsing and Grammar Analysis: Ideal for tasks requiring tokenization that aligns with grammar rules (e.g., syntactic parsing).
2. Contraction-Aware Tokenization: When you need to handle contractions and possessives accurately.
3. Consistency with Penn Treebank: Useful if you're working with datasets or models trained on Penn Treebank tokenization conventions.

"""
#Semantic based chunking (need a tokenizer that tokenizers based on the semantic meaning of the sentence)
from nltk.tokenize import TreebankWordTokenizer

#Semantic based chunking (need a tokenizer that tokenizers based on the semantic meaning of the sentence)
from nltk.tokenize import TreebankWordTokenizer
def semantic_chunking(doc, max_tokens):
    sentences = TreebankWordTokenizer(doc)
    return [" ".join(sentences[i:i + max_tokens]) for i in range(0, len(sentences), max_tokens)]

#Recursive chunking, it splits into chunks unless a condition is met, here we are defining just based on chunksize
def recursive_chunking(text, chunk_size):
    if len(text) <= chunk_size:
        return [text]
    else:
        return [text[:chunk_size]] + recursive_chunking(text[chunk_size:], chunk_size)

#context enriched chunking, ie. add relevant content as metadata which is the surrounding/neighbouring words
def context_enriched_chunking(text, chunk_size, context_size=1):
    sentences = text.split(". ")
    enriched_chunks = []
    for i in range(len(sentences)):
        context = sentences[max(0, i-context_size):min(len(sentences), i+context_size+1)]
        enriched_chunks.append(" ".join(context))
    return enriched_chunks

#Agentic chunking - use hugging face pretrained llm to determine the boundaries of chunking text
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
def agentic_chunking_hf(text, chunking_instructions, model_name="bigscience/bloom-560m"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Construct the prompt
    prompt = f"""
You are an intelligent agent tasked with splitting text into meaningful chunks.
Here are your instructions:
- {chunking_instructions}
- Ensure each chunk represents a clear and complete idea.

Input Text:
{text}

Provide the chunks as a JSON array, with each chunk as a separate string.
"""
    # Tokenize input text
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

    # Generate response
    output_ids = model.generate(inputs.input_ids, max_length=1024, num_beams=5, temperature=0.7)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the chunked output
    try:
        start_idx = response.find("[")
        end_idx = response.rfind("]")
        chunks = eval(response[start_idx:end_idx + 1])  # Convert JSON-like text to Python list
        return chunks
    except Exception as e:
        print(f"Error parsing model output: {e}")
        return None

# Example Usage
if __name__ == "__main__":
    input_text = (     
        "Natural Language Processing is a fascinating field. "
        "It allows machines to understand and generate human-like text. "
        "Applications range from chatbots to translation tools. "
        "Deep learning has significantly advanced NLP in recent years."
    )

    instructions = "Split the text into chunks based on logical ideas or sentences."

    chunks = agentic_chunking_hf(input_text, instructions, model_name="bigscience/bloom-560m")

    if chunks:
        print("Generated Chunks:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk}")
