
# enchanted RAG with knowledge graph and knowledge table

In this project I built a dynamic RAG model based on information from a knowledge table and a knowledge graph.

With the help of the functions found in the *book_to_graph_func.py, book_to_table_func.py* files, I extract ideas and entities from the textual data, and produce a knowledge table and a knowledge graph that include all the entities that appear in the text files, as well as all the insights and ideas mentioned in them.

The functions I built convert the knowledge found in the text files from textual information to tabular information.
![image](https://github.com/YotamAflalo/projects/assets/81986483/be6842cd-4321-46a8-9791-42d783038790)

In the second part, with the  functions found in the *rag_graph.py,prompt.py and context_retriver.py* files, I put together an improved RAG mechanism, which is able to respond while basing both on a standard retrieval mechanism, and also on the information found in the knowledge graph and the knowledge table.

This method makes it possible to receive answers that are based on many sources of information (instead of being based on K specific sources, as is customary in classic RAG). 
In addition, my model gives accurate and complete answers **even without the classic retrieval component**, thus saving on the costs of using the language model.
![image](https://github.com/YotamAflalo/projects/assets/81986483/ef53f74f-596a-424a-8d76-1562ceee96d0)

## Roadmap

- install dependencies

- Collect books, research papers, articles, etc.

- insert LLM api and index keys.

- create knowledge table with books_to_table_func.py (for instractions see "example-How_Not_to_Die_to_table.ipynb).

- create knowledge graph with books_to_graph_func.py (for instractions see "book_to_graph.ipynb").

- Optional: for "normal retriver" make index (for instractions see create_the_index.ipynb).

- constract the enchanted-RAG with rag_graph.py.

- enjoy!!


## Installation

install dependencies using requirements.txt
```bash
  ! pip install -r requirements.txt
```
    
